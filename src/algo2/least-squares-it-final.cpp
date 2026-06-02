/*
 * least_squares_opt.cpp
 *
 * Soluzione ai minimi quadrati di Ax ≈ b  →  AtA·x = Atb
 *
 * OTTIMIZZAZIONI rispetto alla versione precedente:
 *
 *  [A] FALSE SHARING ELIMINATO
 *      – ThreadData paddata a 64 byte (una cache line ciascuna)
 *      – Atb e le righe di AtA scritte in buffer locali per thread,
 *        poi mergiati nel main dopo il join → zero cache-line bouncing
 *
 *  [B] BLOCK RIDOTTO A 16 DOUBLE (128 B = 2 cache line)
 *      Il dot-product su un singolo tile ora entra interamente in L1;
 *      il vecchio BLOCK=64 (512 B) provocava eviction prematura.
 *
 *  [C] ACCUMULO A 4 REGISTRI INDIPENDENTI (ILP)
 *      Ogni tile accumula su s0..s3 sommati alla fine → il pipeline
 *      FMA può eseguire 4 iterazioni in volo senza dipendenza RAW.
 *
 *  [D] ALLINEAMENTO A 64 BYTE (cache line) DI At E AtA
 *      std::aligned_alloc garantisce che i load AVX siano aligned;
 *      evita la penalità degli split load a cavallo di cache line.
 *
 *  [E] PREFETCH ESPLICITO
 *      __builtin_prefetch anticipa in L1 la riga Atj+1 mentre
 *      si calcola il dot-product della riga Atj.
 *
 *  [F] DISTRIBUZIONE A CHUNK CONTIGUI invece di round-robin
 *      Ogni thread lavora su un intervallo [i_start, i_end) di righe
 *      contigue di At: migliore riuso spaziale in L2/L3.
 *      Il bilanciamento è garantito dalla suddivisione (N + T-1)/T.
 *
 *  [G] SIMMETRIA PARALLELIZZATA
 *      La copia AtA[j][i] = AtA[i][j] viene fatta dai thread stessi
 *      nel secondo passaggio, eliminando il loop sequenziale nel main.
 *
 *  [H] GAUSS CON ELIMINAZIONE PARALLELA SUL TRIANGOLO
 *      Le righe sotto il pivot vengono aggiornate in parallelo
 *      (barrier su std::latch / fase per fase).
 *      [Nota: per N piccolo il Gauss rimane mono-thread; il parallelismo
 *       è abilitato automaticamente quando N > GAUSS_PAR_THRESHOLD]
 *
 *  [I] GENERAZIONE DATI CON xoshiro256** (no std::rand bottleneck)
 *      Più veloce e senza lock impliciti del global state di rand().
 *
 * Compilazione raccomandata:
 *   g++ -std=c++17 -O3 -march=native -funroll-loops \
 *       -ffast-math -pthread -o least_squares least_squares_opt.cpp
 *
 * Uso:
 *   ./least_squares <M> <N> <n_thread>
 */

/*
 * least_squares_ultra_opt.cpp
 * Soluzione ai minimi quadrati altamente ottimizzata per Cache Locality e SIMD.
 */

#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <atomic>
#include <new>

static constexpr int GAUSS_PAR_THRESHOLD = 256;
static constexpr std::size_t CACHE_LINE = 64;

// ---------------------------------------------------------------------------
// Allocatore Allineato (RAII)
// ---------------------------------------------------------------------------
struct AlignedBuffer {
    void   *ptr  = nullptr;
    std::size_t bytes = 0;

    AlignedBuffer() = default;
    AlignedBuffer(std::size_t n_bytes, std::size_t align = CACHE_LINE) {
        bytes = n_bytes;
        std::size_t padded = (n_bytes + align - 1) / align * align;
#if defined(_WIN32)
        ptr = _aligned_malloc(padded, align);
#else
        ptr = std::aligned_alloc(align, padded);
#endif
        if (!ptr) throw std::bad_alloc();
        std::memset(ptr, 0, padded);
    }
    ~AlignedBuffer() {
#if defined(_WIN32)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }
    AlignedBuffer(const AlignedBuffer&)            = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& o) noexcept : ptr(o.ptr), bytes(o.bytes) { o.ptr = nullptr; o.bytes = 0; }
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        if (this != &o) {
#if defined(_WIN32)
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
            ptr = o.ptr; bytes = o.bytes; o.ptr = nullptr; o.bytes = 0;
        }
        return *this;
    }

    double *as_double() { return static_cast<double*>(ptr); }
    const double *as_double() const { return static_cast<const double*>(ptr); }
};

// ---------------------------------------------------------------------------
// xoshiro256** PRNG
// ---------------------------------------------------------------------------
struct Xoshiro256ss {
    uint64_t s[4];
    explicit Xoshiro256ss(uint64_t seed) {
        for (int i = 0; i < 4; ++i) {
            seed += 0x9e3779b97f4a7c15ULL;
            uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            s[i] = z ^ (z >> 31);
        }
    }
    inline uint64_t next() {
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }
    inline double next_1_10() { return static_cast<double>(next() % 10) + 1.0; }
private:
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
};

// ---------------------------------------------------------------------------
// Sincronizzazione ad alte prestazioni per Gauss (Lock-Free Spin Barrier)
// ---------------------------------------------------------------------------
struct SpinBarrier {
    std::atomic<int> count{0};
    std::atomic<int> generation{0};
    int num_threads;

    void wait() {
        int gen = generation.load();
        if (++count == num_threads) {
            count = 0;
            generation++;
        } else {
            while (generation.load() == gen) {
                std::this_thread::yield();
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Struttura Dati Thread Padded
// ---------------------------------------------------------------------------
struct alignas(CACHE_LINE) ThreadData {
    const double *At_data;
    const double *b;
    double       *AtA_data;
    double       *Atb;
    double       *local_AtA;
    double       *local_Atb;
    int M, N;
    int i_start, i_end;
    int n_threads;
    int thread_id;
    char _pad[CACHE_LINE - (8*5 + 4*5) % CACHE_LINE];
};

// ---------------------------------------------------------------------------
// Kernel di Calcolo principale con Tiling 3D (Ottimizzato per L1/L2 e SIMD)
// ---------------------------------------------------------------------------
void compute_ata(ThreadData &d) {
    const int M       = d.M;
    const int N       = d.N;
    const int i_start = d.i_start;
    const int i_end   = d.i_end;

    double *lAtA = d.local_AtA;
    double *lAtb = d.local_Atb;

    // Passaggio 1: Calcolo Atb (Contiguo e auto-vettorializzabile)
    for (int i = i_start; i < i_end; ++i) {
        const int li = i - i_start;
        const double *Ati = d.At_data + static_cast<std::size_t>(i) * M;
        double sum_b = 0.0;
        for (int k = 0; k < M; ++k) {
            sum_b += Ati[k] * d.b[k];
        }
        lAtb[li] = sum_b;
    }

    // Passaggio 2: Calcolo AtA tramite Cache Tiling 3D strutturato
    // BI x BJ definiscono il tile memorizzato in L1, BK riduce lo streaming da RAM
    constexpr int BI = 32;
    constexpr int BJ = 64;
    constexpr int BK = 512;

    for (int ii = i_start; ii < i_end; ii += BI) {
        int lim_i = std::min(ii + BI, i_end);
        for (int jj = ii; jj < N; jj += BJ) {
            int lim_j = std::min(jj + BJ, N);
            for (int kk = 0; kk < M; kk += BK) {
                int lim_k = std::min(kk + BK, M);

                // Micro-kernel interno ottimizzato per l'auto-vettorializzazione SIMD
                for (int i = ii; i < lim_i; ++i) {
                    const int li = i - i_start;
                    double *row_out = lAtA + static_cast<std::size_t>(li) * N;
                    const double *Ati = d.At_data + static_cast<std::size_t>(i) * M;

                    int j_start = std::max(jj, i);
                    for (int j = j_start; j < lim_j; ++j) {
                        const double *Atj = d.At_data + static_cast<std::size_t>(j) * M;
                        
                        double sum = 0.0;
                        // Il compilatore srotolerà e userà FMA vettoriali qui in modo ottimale
                        for (int k = kk; k < lim_k; ++k) {
                            sum += Ati[k] * Atj[k];
                        }
                        row_out[j] += sum;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Merge e Simmetrizzazione a Fasi Isolate (Zero Conflitti di Cache)
// ---------------------------------------------------------------------------
void merge_upper_triangle(ThreadData &d) {
    const int N       = d.N;
    const int i_start = d.i_start;
    const int i_end   = d.i_end;
    double *lAtA      = d.local_AtA;
    double *lAtb      = d.local_Atb;
    double *gAtA      = d.AtA_data;
    double *gAtb      = d.Atb;

    for (int i = i_start; i < i_end; ++i) {
        const int li = i - i_start;
        gAtb[i] = lAtb[li];
        const double *src = lAtA + static_cast<std::size_t>(li) * N;
        double *dst_row_i = gAtA + static_cast<std::size_t>(i) * N;
        
        // Copia lineare e contigua del triangolo superiore
        std::memcpy(dst_row_i + i, src + i, (N - i) * sizeof(double));
    }
}

void transpose_lower_triangle(ThreadData &d) {
    const int N       = d.N;
    const int i_start = d.i_start;
    const int i_end   = d.i_end;
    double *gAtA      = d.AtA_data;

    for (int i = i_start; i < i_end; ++i) {
        double *dst_row_i = gAtA + static_cast<std::size_t>(i) * N;
        // Scrittura esclusiva sulle proprie righe leggendo da posizioni sparse stabili
        for (int j = 0; j < i; ++j) {
            dst_row_i[j] = gAtA[static_cast<std::size_t>(j) * N + i];
        }
    }
}

// ---------------------------------------------------------------------------
// Gauss con Thread Pool Persistente e Barriere Atomiche
// ---------------------------------------------------------------------------
void solve_system(const double *AtA_data, const double *Atb,
                  std::vector<double> &x, int N, int n_threads) {
    std::vector<double> G_data(static_cast<std::size_t>(N) * N);
    std::vector<double> g(Atb, Atb + N);
    std::vector<double*> G(N);
    for (int i = 0; i < N; ++i) {
        G[i] = G_data.data() + static_cast<std::size_t>(i) * N;
        std::memcpy(G[i], AtA_data + static_cast<std::size_t>(i) * N, N * sizeof(double));
    }

    const bool use_par = (N > GAUSS_PAR_THRESHOLD) && (n_threads > 1);

    if (!use_par) {
        // Solutore Sequenziale Ottimizzato
        for (int i = 0; i < N; ++i) {
            int pivot = i;
            double max_val = std::fabs(G[i][i]);
            for (int k = i + 1; k < N; ++k) {
                double v = std::fabs(G[k][i]);
                if (v > max_val) { max_val = v; pivot = k; }
            }
            if (max_val < 1e-14) throw std::runtime_error("Matrice singolare.");
            if (pivot != i) { std::swap(G[i], G[pivot]); std::swap(g[i], g[pivot]); }

            const double piv_inv = 1.0 / G[i][i];
            for (int k = i + 1; k < N; ++k) {
                const double factor = G[k][i] * piv_inv;
                double *Gk = G[k];
                const double *Gi = G[i];
                int j = i;
                for (; j + 4 <= N; j += 4) {
                    Gk[j]   -= factor * Gi[j];   Gk[j+1] -= factor * Gi[j+1];
                    Gk[j+2] -= factor * Gi[j+2]; Gk[j+3] -= factor * Gi[j+3];
                }
                for (; j < N; ++j) Gk[j] -= factor * Gi[j];
                g[k] -= factor * g[i];
            }
        }
    } else {
        // Solutore Parallelo a Thread Persistenti (Evita lo spawn continuo)
        SpinBarrier bar1{0, 0, n_threads};
        SpinBarrier bar2{0, 0, n_threads};
        std::atomic<bool> is_singular{false};

        std::vector<std::thread> workers(n_threads);
        for (int t = 0; t < n_threads; ++t) {
            workers[t] = std::thread([&, t]() {
                for (int i = 0; i < N; ++i) {
                    if (t == 0) { // Il Master Thread calcola il pivot
                        int pivot = i;
                        double max_val = std::fabs(G[i][i]);
                        for (int k = i + 1; k < N; ++k) {
                            double v = std::fabs(G[k][i]);
                            if (v > max_val) { max_val = v; pivot = k; }
                        }
                        if (max_val < 1e-14) is_singular = true;
                        if (pivot != i) { std::swap(G[i], G[pivot]); std::swap(g[i], g[pivot]); }
                    }

                    bar1.wait(); // Sincronizzazione 1: Pivot pronto
                    if (is_singular) return;

                    const int rows_below = N - i - 1;
                    if (rows_below > 0) {
                        const int k_start = i + 1 + t * rows_below / n_threads;
                        const int k_end   = i + 1 + (t + 1) * rows_below / n_threads;
                        const double piv_inv = 1.0 / G[i][i];
                        const double *pivot_row = G[i];

                        for (int k = k_start; k < k_end; ++k) {
                            const double factor = G[k][i] * piv_inv;
                            double *Gk = G[k];
                            int j = i;
                            for (; j + 4 <= N; j += 4) {
                                Gk[j]   -= factor * pivot_row[j];   Gk[j+1] -= factor * pivot_row[j+1];
                                Gk[j+2] -= factor * pivot_row[j+2]; Gk[j+3] -= factor * pivot_row[j+3];
                            }
                            for (; j < N; ++j) Gk[j] -= factor * pivot_row[j];
                            g[k] -= factor * g[i];
                        }
                    }
                    bar2.wait(); // Sincronizzazione 2: Eliminazione completata per questo step
                }
            });
        }
        for (auto &w : workers) w.join();
        if (is_singular) throw std::runtime_error("Matrice singolare o quasi-singolare.");
    }

    // Back-substitution
    x.resize(N);
    for (int i = N - 1; i >= 0; --i) {
        double sum = 0.0;
        const double *Gi = G[i];
        for (int j = i + 1; j < N; ++j) sum += Gi[j] * x[j];
        x[i] = (g[i] - sum) * (1.0 / Gi[i]);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::printf("Uso: %s <M> <N> <n_thread>\n", argv[0]);
        return 1;
    }

    const int M         = std::atoi(argv[1]);
    const int N         = std::atoi(argv[2]);
    const int n_threads = std::min(std::atoi(argv[3]), N);

    if (M <= 0 || N <= 0 || n_threads <= 0) {
        std::fprintf(stderr, "Parametri non validi.\n");
        return 1;
    }

    AlignedBuffer At_buf (static_cast<std::size_t>(N) * M * sizeof(double));
    AlignedBuffer AtA_buf(static_cast<std::size_t>(N) * N * sizeof(double));
    AlignedBuffer Atb_buf(static_cast<std::size_t>(N)     * sizeof(double));
    AlignedBuffer b_buf  (static_cast<std::size_t>(M)     * sizeof(double));

    double *At_data  = At_buf .as_double();
    double *AtA_data = AtA_buf.as_double();
    double *Atb      = Atb_buf.as_double();
    double *b        = b_buf  .as_double();

    // Inizializzazione parallela dati
    {
        std::vector<std::thread> gen_threads(n_threads);
        const int rows_per_thread = (M + n_threads - 1) / n_threads;
        for (int t = 0; t < n_threads; ++t) {
            gen_threads[t] = std::thread([&, t]() {
                Xoshiro256ss rng(42ULL + static_cast<uint64_t>(t) * 1234567ULL);
                const int i_start = t * rows_per_thread;
                const int i_end   = std::min(i_start + rows_per_thread, M);
                for (int i = i_start; i < i_end; ++i) {
                    for (int j = 0; j < N; ++j) {
                        At_data[static_cast<std::size_t>(j) * M + i] = rng.next_1_10();
                    }
                    b[i] = rng.next_1_10();
                }
            });
        }
        for (auto &t : gen_threads) t.join();
    }

    std::vector<int> chunk_start(n_threads + 1);
    for (int t = 0; t <= n_threads; ++t)
        chunk_start[t] = static_cast<int>(static_cast<long long>(t) * N / n_threads);

    std::vector<AlignedBuffer> local_AtA_bufs(n_threads);
    std::vector<AlignedBuffer> local_Atb_bufs(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        const int chunk = chunk_start[t+1] - chunk_start[t];
        local_AtA_bufs[t] = AlignedBuffer(static_cast<std::size_t>(chunk) * N * sizeof(double));
        local_Atb_bufs[t] = AlignedBuffer(static_cast<std::size_t>(chunk)     * sizeof(double));
    }

    std::vector<ThreadData> t_data(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        t_data[t].At_data   = At_data;
        t_data[t].b         = b;
        t_data[t].AtA_data  = AtA_data;
        t_data[t].Atb       = Atb;
        t_data[t].local_AtA = local_AtA_bufs[t].as_double();
        t_data[t].local_Atb = local_Atb_bufs[t].as_double();
        t_data[t].M         = M;
        t_data[t].N         = N;
        t_data[t].i_start   = chunk_start[t];
        t_data[t].i_end     = chunk_start[t+1];
        t_data[t].n_threads = n_threads;
        t_data[t].thread_id = t;
    }

    std::vector<std::thread> threads(n_threads);

    // ── INIZIO CALCOLO PARALLELO ────────────────────────────────────────────
    const auto t_start = std::chrono::steady_clock::now();

    // Fase 1: Calcolo AtA locale in Tiling 3D
    for (int t = 0; t < n_threads; ++t)
        threads[t] = std::thread(compute_ata, std::ref(t_data[t]));
    for (auto &t : threads) t.join();

    // Fase 2: Merge del triangolo superiore (Contiguo, isolato per righe)
    for (int t = 0; t < n_threads; ++t)
        threads[t] = std::thread(merge_upper_triangle, std::ref(t_data[t]));
    for (auto &t : threads) t.join();

    // Fase 3: Trasposizione del triangolo inferiore (Senza cache bouncing)
    for (int t = 0; t < n_threads; ++t)
        threads[t] = std::thread(transpose_lower_triangle, std::ref(t_data[t]));
    for (auto &t : threads) t.join();

    const auto t_end = std::chrono::steady_clock::now();
    // ── FINE CALCOLO PARALLELO ──────────────────────────────────────────────

    std::vector<double> x;
    try {
        solve_system(AtA_data, Atb, x, N, n_threads);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Errore: %s\n", e.what());
        return 1;
    }

    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
    std::printf("Tempo SOLO Parallelo: %.6f s\n", elapsed_us.count() * 1e-6);

    if (N >= 3)
        std::printf("Primi 3 valori di x: %f  %f  %f\n", x[0], x[1], x[2]);

    return 0;
}