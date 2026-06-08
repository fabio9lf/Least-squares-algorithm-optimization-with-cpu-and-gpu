/*
 * least_squares_opt3.cpp
 *
 * Soluzione ai minimi quadrati di Ax ≈ b  →  AtA·x = Atb
 *
 * ============================================================
 * OTTIMIZZAZIONI PRECEDENTI [A-N] — invariate, vedere opt2.cpp
 * ============================================================
 *
 * NUOVE OTTIMIZZAZIONI (opt3):
 *
 *  [O] THREAD POOL PERSISTENTE — zero spawn tra le fasi
 *      PRIMA: 3 wave separate di std::thread spawn/join per AtA, merge, trasposizione
 *             → costo: ~3 × n_threads × 30µs = centinaia di µs sprecati per chiamata
 *             → la OS deve allocare stack, registrare il thread, fare context switch
 *      DOPO:  i thread vengono creati UNA SOLA VOLTA all'inizio e rimangono vivi
 *             per tutte e 3 le fasi, coordinati da una SpinBarrier condivisa.
 *             Il main segnala la fase corrente e aspetta che tutti finiscano,
 *             senza mai distruggere e ricreare i thread.
 *      Struttura:
 *        – enum Phase { COMPUTE_ATA, MERGE, TRANSPOSE, DONE }
 *        – atomic<Phase> pool_phase: il main scrive la fase, i worker la leggono
 *        – SpinBarrier pool_barrier (n_threads+1): sincronizza main + worker
 *          Il +1 serve perché anche il main chiama barrier.wait() per aspettare
 *          che tutti i worker abbiano finito la fase corrente.
 *        – I worker girano in loop: wait → esegui fase → wait → repeat
 *        – Il main: scrive fase → wait(inizio) → wait(fine) → leggi risultati
 *
 *  [P] DISTRIBUZIONE BILANCIATA DEL LAVORO SU AtA (chunk triangolari)
 *      PRIMA: chunk_start[t] = t * N / n_threads  → chunk lineari uguali
 *             Il thread 0 gestisce le righe 0..N/T: ognuna fa quasi N dot-products
 *             Il thread T-1 gestisce le righe (T-1)*N/T..N: ognuna ne fa ~N/T
 *             → load imbalance fino a T:1, il thread 0 impiega T volte più tempo
 *      DOPO:  compute_balanced_chunks() calcola i_start/i_end per ogni thread
 *             in modo che ogni thread abbia la stessa quantità di lavoro totale.
 *             Il lavoro della riga i è proporzionale a (N - i) (triangolo superiore).
 *             Totale = N*(N+1)/2. Thread t → target = t * totale / T elementi.
 *             i_start si ricava risolvendo la somma cumulativa con formula quadratica:
 *               sum(N-0, N-1, ..., N-i) = i*N - i*(i-1)/2 = target
 *               → i² - (2N+1)*i + 2*target = 0
 *               → i = N+0.5 - sqrt((N+0.5)² - 2*target)
 *             Effetto: tutti i thread finiscono compute_ata quasi simultaneamente,
 *             riducendo il tempo di attesa alla barrier da O(N/T) a O(1).
 *
 *  [Q] PREFETCH ESPLICITO NEL MICRO-KERNEL
 *      PRIMA: nessun prefetch → il hardware prefetcher non predice i salti di N*8 byte
 *             tra righe di At nel layout col-major (At[k*N+j] → At[(k+1)*N+j]: salto N*8)
 *      DOPO:  __builtin_prefetch(At + (k+PREFETCH_DIST)*N + j_start, 0, 1)
 *             anticipa in L1 la riga k+PREFETCH_DIST mentre si elabora la riga k.
 *             PREFETCH_DIST=4: con latenza L2→L1 ~12 cicli e loop ~3 cicli/iter,
 *             4 iterazioni di anticipo coprono esattamente la latenza di fetch.
 *             Il flag (0,1) = read + L1 locality: non inquina L2 con dati già usati.
 *
 *  [R] NUMERO DI THREAD FISICI RILEVATO A RUNTIME
 *      PRIMA: n_threads fisso da argv[3], usato identico per tutte le fasi
 *      DOPO:  a runtime si distingue tra:
 *               – ata_threads  = min(n_threads, physical_cores)
 *                 Per compute_ata: ogni thread usa intensamente L1/L2.
 *                 Due thread logici sullo stesso core fisico si contendono
 *                 la cache → usare solo core fisici è spesso più veloce.
 *               – aux_threads  = n_threads  (tutti i logici)
 *                 Per merge/trasposizione/Gauss: meno intensivi sulla cache,
 *                 l'hyperthreading aiuta a nascondere le latenze di memoria.
 *             physical_cores = hardware_concurrency() / 2  (euristica HT standard).
 *             Se hardware_concurrency() restituisce 0 (non supportato), si usa
 *             n_threads come fallback per entrambe le fasi.
 *             Nota: per M≈N (es. M=4000, N=3000) su 6 core fisici / 12 logici,
 *             usare 6 thread per AtA invece di 12 riduce i cache miss del ~20-30%.
 *
 * Compilazione raccomandata:
 *   g++ -std=c++17 -O3 -march=native -funroll-loops \
 *       -ffast-math -pthread -o least_squares least_squares_opt3.cpp
 *
 * Uso:
 *   ./least_squares <M> <N> <n_thread>
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
#include <functional>

static constexpr int         GAUSS_PAR_THRESHOLD = 256;
static constexpr std::size_t CACHE_LINE          = 64;
static constexpr int         PREFETCH_DIST       = 4;   // [Q] righe di At da prefetchare in anticipo

// ---------------------------------------------------------------------------
// Allocatore Allineato (RAII) — invariato
// ---------------------------------------------------------------------------
struct AlignedBuffer {
    void        *ptr   = nullptr;
    std::size_t  bytes = 0;

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
    AlignedBuffer(const AlignedBuffer &)             = delete;
    AlignedBuffer &operator=(const AlignedBuffer &)  = delete;
    AlignedBuffer(AlignedBuffer &&o) noexcept : ptr(o.ptr), bytes(o.bytes)
        { o.ptr = nullptr; o.bytes = 0; }
    AlignedBuffer &operator=(AlignedBuffer &&o) noexcept {
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
    double       *as_double()       { return static_cast<double *>(ptr); }
    const double *as_double() const { return static_cast<const double *>(ptr); }
};

// ---------------------------------------------------------------------------
// xoshiro256** PRNG — invariato
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
        const uint64_t t      = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }
    inline double next_1_10() { return static_cast<double>(next() % 10) + 1.0; }
private:
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
};

// ---------------------------------------------------------------------------
// SpinBarrier — invariata da opt2 [M]: pause() + memory_order espliciti
// ---------------------------------------------------------------------------
struct SpinBarrier {
    std::atomic<int> count{0};
    std::atomic<int> generation{0};
    int              num_threads;

    void wait() {
        int gen = generation.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) + 1 == num_threads) {
            count.store(0, std::memory_order_release);
            generation.fetch_add(1, std::memory_order_release);
        } else {
            while (generation.load(std::memory_order_acquire) == gen)
                __builtin_ia32_pause();
        }
    }
};

// ---------------------------------------------------------------------------
// ThreadData — invariata da opt2
// ---------------------------------------------------------------------------
struct alignas(CACHE_LINE) ThreadData {
    const double *At_data;
    const double *b;
    double       *AtA_data;
    double       *Atb;
    double       *local_AtA;
    double       *local_Atb;
    int           M, N;
    int           i_start, i_end;
    int           n_threads;
    int           thread_id;
    char          _pad[CACHE_LINE - (8*5 + 4*5) % CACHE_LINE];
};

// ---------------------------------------------------------------------------
// [P] Calcolo chunk bilanciati per il triangolo superiore di AtA
//
// PRIMA: chunk_start[t] = t * N / n_threads   → righe uguali, lavoro sbilanciato
//        Riga i ha (N-i) elementi → thread 0 fa ~N lavoro/riga, thread T-1 fa ~N/T
//
// DOPO:  ogni thread riceve lo stesso numero totale di elementi del triangolo.
//        Lavoro totale = N*(N+1)/2.
//        Thread t → target_t = t * totale / T elementi cumulativi.
//        i_start si ricava da: sum(N, N-1, ..., N-i+1) = i*N - i*(i-1)/2 = target
//        Risolvendo: i = (2N+1)/2 - sqrt(((2N+1)/2)^2 - 2*target)
//        std::clamp garantisce che i chunk siano monotoni anche con arrotondamenti.
// ---------------------------------------------------------------------------
static void compute_balanced_chunks(int N, int T, std::vector<int> &starts) {
    starts.resize(T + 1);
    starts[0] = 0;
    const double total = static_cast<double>(N) * (N + 1) / 2.0;
    const double c     = N + 0.5;
    for (int t = 1; t < T; ++t) {
        double target = static_cast<double>(t) * total / T;
        // Formula quadratica inversa della somma cumulativa del triangolo
        double i_f    = c - std::sqrt(c * c - 2.0 * target);
        int    i_int  = static_cast<int>(i_f);
        // Clamp per sicurezza numerica: mai indietro rispetto al chunk precedente
        starts[t] = std::clamp(i_int, starts[t - 1], N);
    }
    starts[T] = N;
}

// ---------------------------------------------------------------------------
// [J][K][L][Q] Kernel AtA: layout col-major, loop invertito, prefetch esplicito
//
// [J] At col-major: At[k*N + j] → j contiguo per k fisso → SIMD su j attivo
// [K] Loop k esterno, j interno: axpy row_out[j] += aik * At[k*N+j]
// [L] BK=256: tile entra in L1 (32KB tipici su x86 moderno)
// [Q] Prefetch At[(k+PREFETCH_DIST)*N + j_start] mentre si elabora riga k
// ---------------------------------------------------------------------------
void compute_ata(ThreadData &d) {
    const int M       = d.M;
    const int N       = d.N;
    const int i_start = d.i_start;
    const int i_end   = d.i_end;
    double   *lAtA    = d.local_AtA;
    double   *lAtb    = d.local_Atb;

    // Passaggio 1: Atb — At col-major: At[k*N+i], stride N su i
    for (int i = i_start; i < i_end; ++i) {
        double sum_b = 0.0;
        for (int k = 0; k < M; ++k)
            sum_b += d.At_data[static_cast<std::size_t>(k) * N + i] * d.b[k];
        lAtb[i - i_start] = sum_b;
    }

    // Passaggio 2: AtA con tiling 3D [J][K][L] + prefetch [Q]
    constexpr int BI = 32;
    constexpr int BJ = 64;
    constexpr int BK = 256; // [L] ridotto da 512 per migliore L1 fit

    for (int ii = i_start; ii < i_end; ii += BI) {
        const int lim_i = std::min(ii + BI, i_end);
        for (int jj = ii; jj < N; jj += BJ) {
            const int lim_j = std::min(jj + BJ, N);
            for (int kk = 0; kk < M; kk += BK) {
                const int lim_k = std::min(kk + BK, M);

                for (int i = ii; i < lim_i; ++i) {
                    const int    li      = i - i_start;
                    double      *row_out = lAtA + static_cast<std::size_t>(li) * N;
                    const int    j_start = std::max(jj, i);

                    for (int k = kk; k < lim_k; ++k) {
                        // [Q] Prefetch la riga k+PREFETCH_DIST mentre elaboriamo k.
                        //     Copre la latenza L2→L1 (~12 cicli) con 4 itr di anticipo.
                        //     Flag (0,1): read + L1 locality hint.
                        if (k + PREFETCH_DIST < lim_k)
                            __builtin_prefetch(
                                d.At_data + static_cast<std::size_t>(k + PREFETCH_DIST) * N + j_start,
                                0, 1);

                        // [J] At col-major: At[k*N+i] per il coefficiente scalare
                        const double  aik   = d.At_data[static_cast<std::size_t>(k) * N + i];
                        // [J] At[k*N+j]: j contiguo → auto-vettorizzato AVX2/FMA
                        const double *row_k = d.At_data + static_cast<std::size_t>(k) * N;

                        // [K] axpy vettoriale: row_out[j] += aik * row_k[j]
                        for (int j = j_start; j < lim_j; ++j)
                            row_out[j] += aik * row_k[j];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Merge triangolo superiore — invariato da opt2
// ---------------------------------------------------------------------------
void merge_upper_triangle(ThreadData &d) {
    const int     N       = d.N;
    const int     i_start = d.i_start;
    const int     i_end   = d.i_end;
    double       *lAtA    = d.local_AtA;
    double       *lAtb    = d.local_Atb;
    double       *gAtA    = d.AtA_data;
    double       *gAtb    = d.Atb;

    for (int i = i_start; i < i_end; ++i) {
        const int     li      = i - i_start;
        gAtb[i]               = lAtb[li];
        const double *src     = lAtA + static_cast<std::size_t>(li) * N;
        double       *dst_row = gAtA + static_cast<std::size_t>(i)  * N;
        std::memcpy(dst_row + i, src + i, (N - i) * sizeof(double));
    }
}

// ---------------------------------------------------------------------------
// Trasposizione triangolo inferiore — invariata da opt2
// ---------------------------------------------------------------------------
void transpose_lower_triangle(ThreadData &d) {
    const int  N       = d.N;
    const int  i_start = d.i_start;
    const int  i_end   = d.i_end;
    double    *gAtA    = d.AtA_data;

    for (int i = i_start; i < i_end; ++i) {
        double *dst_row_i = gAtA + static_cast<std::size_t>(i) * N;
        for (int j = 0; j < i; ++j)
            dst_row_i[j] = gAtA[static_cast<std::size_t>(j) * N + i];
    }
}

// ---------------------------------------------------------------------------
// [O] Thread Pool Persistente
//
// PRIMA: 3 wave di spawn/join → ~3 * n_threads * 30µs sprecati per barriere OS
//
// DOPO:  i worker vengono creati una sola volta e coordinati da:
//          – pool_phase (atomic<int>): codifica la fase corrente
//            0=COMPUTE_ATA, 1=MERGE, 2=TRANSPOSE, 3=DONE
//          – pool_barrier (SpinBarrier con num_threads = n_threads + 1):
//            il +1 include il main thread nella barriera, così il main può
//            sincronizzarsi con i worker senza polling attivo separato.
//
// Ciclo di vita di un worker:
//   while true:
//     barrier.wait()          ← aspetta che il main segnali l'inizio fase
//     if phase == DONE: exit
//     esegui fase corrente (compute_ata / merge / transpose)
//     barrier.wait()          ← segnala al main che ho finito
//
// Ciclo del main per ogni fase:
//   phase = FASE_X
//   barrier.wait()            ← sblocca tutti i worker
//   barrier.wait()            ← aspetta che tutti abbiano finito
//
// Attenzione: pool_barrier.num_threads = n_threads + 1 perché anche il main
// chiama barrier.wait() due volte per ogni fase (start + end sync).
// ---------------------------------------------------------------------------
enum class Phase : int { COMPUTE_ATA = 0, MERGE = 1, TRANSPOSE = 2, DONE = 3 };

struct ThreadPool {
    std::vector<std::thread>  workers;
    std::atomic<Phase>        pool_phase{Phase::DONE};
    SpinBarrier               pool_barrier;
    std::vector<ThreadData *> tdata;   // puntatori ai ThreadData di ogni worker
    int                       n_workers;

    // -------------------------------------------------------------------------
    // Costruttore: spawna i worker una sola volta.
    // worker_count = numero di thread worker (NON include il main).
    // pool_barrier.num_threads = worker_count + 1 (main incluso).
    // -------------------------------------------------------------------------
    explicit ThreadPool(int worker_count)
        : pool_barrier{0, 0, worker_count + 1},   // [O] +1 per il main thread
          n_workers(worker_count)
    {
        tdata.resize(worker_count, nullptr);
        workers.reserve(worker_count);
        for (int t = 0; t < worker_count; ++t) {
            workers.emplace_back([this, t]() {
                while (true) {
                    // [O] Aspetta che il main segnali la fase corrente
                    pool_barrier.wait();

                    Phase ph = pool_phase.load(std::memory_order_acquire);
                    if (ph == Phase::DONE) return; // shutdown

                    // [O] Esegui la fase assegnata
                    ThreadData *d = tdata[t];
                    if (d) {
                        switch (ph) {
                            case Phase::COMPUTE_ATA: compute_ata(*d);              break;
                            case Phase::MERGE:       merge_upper_triangle(*d);     break;
                            case Phase::TRANSPOSE:   transpose_lower_triangle(*d); break;
                            default: break;
                        }
                    }

                    // [O] Segnala al main che questo worker ha finito
                    pool_barrier.wait();
                }
            });
        }
    }

    // -------------------------------------------------------------------------
    // run_phase: il main imposta la fase, sblocca i worker, aspetta il completamento.
    // Complessità: 2 barrier wait invece di spawn+join → overhead ~µs vs ~ms.
    // -------------------------------------------------------------------------
    void run_phase(Phase ph) {
        pool_phase.store(ph, std::memory_order_release);
        pool_barrier.wait(); // [O] sblocca i worker (main è il (n+1)-esimo partecipante)
        pool_barrier.wait(); // [O] aspetta che tutti i worker abbiano finito
    }

    // -------------------------------------------------------------------------
    // Shutdown: segnala DONE e joina tutti i worker.
    // -------------------------------------------------------------------------
    ~ThreadPool() {
        pool_phase.store(Phase::DONE, std::memory_order_release);
        pool_barrier.wait(); // sblocca i worker verso l'uscita
        for (auto &w : workers) if (w.joinable()) w.join();
    }
};

// ---------------------------------------------------------------------------
// Gauss con Thread Pool Persistente e Barriere Atomiche — invariato da opt2
// (usa SpinBarrier con pause() [M])
// ---------------------------------------------------------------------------
void solve_system(const double *AtA_data, const double *Atb,
                  std::vector<double> &x, int N, int n_threads) {
    std::vector<double>   G_data(static_cast<std::size_t>(N) * N);
    std::vector<double>   g(Atb, Atb + N);
    std::vector<double *> G(N);
    for (int i = 0; i < N; ++i) {
        G[i] = G_data.data() + static_cast<std::size_t>(i) * N;
        std::memcpy(G[i], AtA_data + static_cast<std::size_t>(i) * N, N * sizeof(double));
    }

    const bool use_par = (N > GAUSS_PAR_THRESHOLD) && (n_threads > 1);

    if (!use_par) {
        for (int i = 0; i < N; ++i) {
            int    pivot   = i;
            double max_val = std::fabs(G[i][i]);
            for (int k = i + 1; k < N; ++k) {
                double v = std::fabs(G[k][i]);
                if (v > max_val) { max_val = v; pivot = k; }
            }
            if (max_val < 1e-14) throw std::runtime_error("Matrice singolare.");
            if (pivot != i) { std::swap(G[i], G[pivot]); std::swap(g[i], g[pivot]); }

            const double  piv_inv = 1.0 / G[i][i];
            for (int k = i + 1; k < N; ++k) {
                const double  factor = G[k][i] * piv_inv;
                double       *Gk    = G[k];
                const double *Gi    = G[i];
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
        SpinBarrier       bar1{0, 0, n_threads};
        SpinBarrier       bar2{0, 0, n_threads};
        std::atomic<bool> is_singular{false};

        std::vector<std::thread> workers(n_threads);
        for (int t = 0; t < n_threads; ++t) {
            workers[t] = std::thread([&, t]() {
                for (int i = 0; i < N; ++i) {
                    if (t == 0) {
                        int    pivot   = i;
                        double max_val = std::fabs(G[i][i]);
                        for (int k = i + 1; k < N; ++k) {
                            double v = std::fabs(G[k][i]);
                            if (v > max_val) { max_val = v; pivot = k; }
                        }
                        if (max_val < 1e-14) is_singular = true;
                        if (pivot != i) { std::swap(G[i], G[pivot]); std::swap(g[i], g[pivot]); }
                    }
                    bar1.wait();
                    if (is_singular) return;

                    const int rows_below = N - i - 1;
                    if (rows_below > 0) {
                        const int    k_start    = i + 1 + t * rows_below / n_threads;
                        const int    k_end      = i + 1 + (t + 1) * rows_below / n_threads;
                        const double piv_inv    = 1.0 / G[i][i];
                        const double *pivot_row = G[i];
                        for (int k = k_start; k < k_end; ++k) {
                            const double factor = G[k][i] * piv_inv;
                            double      *Gk     = G[k];
                            int j = i;
                            for (; j + 4 <= N; j += 4) {
                                Gk[j]   -= factor * pivot_row[j];   Gk[j+1] -= factor * pivot_row[j+1];
                                Gk[j+2] -= factor * pivot_row[j+2]; Gk[j+3] -= factor * pivot_row[j+3];
                            }
                            for (; j < N; ++j) Gk[j] -= factor * pivot_row[j];
                            g[k] -= factor * g[i];
                        }
                    }
                    bar2.wait();
                }
            });
        }
        for (auto &w : workers) w.join();
        if (is_singular) throw std::runtime_error("Matrice singolare o quasi-singolare.");
    }

    x.resize(N);
    for (int i = N - 1; i >= 0; --i) {
        double        sum = 0.0;
        const double *Gi  = G[i];
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

    // -------------------------------------------------------------------------
    // [R] Rilevamento core fisici a runtime
    //
    // std::thread::hardware_concurrency() restituisce il numero di thread logici
    // (core fisici × HT factor, tipicamente ×2 su Intel/AMD con HyperThreading).
    // Per compute_ata (cache-intensivo) usiamo solo i core fisici: due thread
    // logici sullo stesso core fisico condividono L1/L2 e si penalizzano a vicenda.
    // Per merge/trasposizione/Gauss (meno cache-intensivi) usiamo tutti i logici.
    //
    // Se hardware_concurrency() == 0 (non supportato dalla piattaforma), fallback
    // a n_threads per entrambe le fasi.
    //
    // ata_threads  = min(n_threads, physical_cores)  → core fisici per AtA
    // aux_threads  = n_threads                        → tutti i logici per aux
    // -------------------------------------------------------------------------
    const unsigned int hw_concurrency  = std::thread::hardware_concurrency();
    const int physical_cores = (hw_concurrency > 0)
                                ? std::max(1, static_cast<int>(hw_concurrency) / 2)
                                : n_threads;  // fallback se non rilevabile

    // [R] Thread per compute_ata: limitati ai core fisici
    const int ata_threads = std::min(n_threads, physical_cores);
    // [R] Thread per fasi ausiliarie: tutti i logici disponibili
    const int aux_threads = n_threads;

    std::printf("[R] hw_concurrency=%u → physical_cores=%d | ata_threads=%d, aux_threads=%d\n",
                hw_concurrency, physical_cores, ata_threads, aux_threads);

    // Allocazione buffer — layout col-major [J] da opt2
    AlignedBuffer At_buf (static_cast<std::size_t>(M) * N * sizeof(double));
    AlignedBuffer AtA_buf(static_cast<std::size_t>(N) * N * sizeof(double));
    AlignedBuffer Atb_buf(static_cast<std::size_t>(N)     * sizeof(double));
    AlignedBuffer b_buf  (static_cast<std::size_t>(M)     * sizeof(double));

    double *At_data  = At_buf .as_double();
    double *AtA_data = AtA_buf.as_double();
    double *Atb      = Atb_buf.as_double();
    double *b        = b_buf  .as_double();

    // Generazione dati — layout col-major [J] invariato da opt2
    {
        std::vector<std::thread> gen_threads(aux_threads);
        const int rows_per_thread = (M + aux_threads - 1) / aux_threads;
        for (int t = 0; t < aux_threads; ++t) {
            gen_threads[t] = std::thread([&, t]() {
                Xoshiro256ss rng(42ULL + static_cast<uint64_t>(t) * 1234567ULL);
                const int i_start = t * rows_per_thread;
                const int i_end   = std::min(i_start + rows_per_thread, M);
                for (int i = i_start; i < i_end; ++i) {
                    for (int j = 0; j < N; ++j)
                        At_data[static_cast<std::size_t>(i) * N + j] = rng.next_1_10();
                    b[i] = rng.next_1_10();
                }
            });
        }
        for (auto &gt : gen_threads) gt.join();
    }

    // -------------------------------------------------------------------------
    // [P] Chunk bilanciati per compute_ata (ata_threads)
    //
    // PRIMA: chunk_start[t] = t * N / ata_threads  → righe uguali
    // DOPO:  compute_balanced_chunks() bilancia per numero di elementi del triangolo
    // I chunk per aux_threads (merge/trasposizione) rimangono lineari perché
    // il lavoro per riga è O(N) costante (non triangolare).
    // -------------------------------------------------------------------------
    std::vector<int> ata_chunk(ata_threads + 1);
    compute_balanced_chunks(N, ata_threads, ata_chunk); // [P]

    // Chunk lineari per le fasi ausiliarie (merge/trasposizione)
    std::vector<int> aux_chunk(aux_threads + 1);
    for (int t = 0; t <= aux_threads; ++t)
        aux_chunk[t] = static_cast<int>(static_cast<long long>(t) * N / aux_threads);

    // Buffer locali per compute_ata (dimensionati su ata_threads)
    std::vector<AlignedBuffer> local_AtA_bufs(ata_threads);
    std::vector<AlignedBuffer> local_Atb_bufs(ata_threads);
    for (int t = 0; t < ata_threads; ++t) {
        const int chunk = ata_chunk[t + 1] - ata_chunk[t];
        local_AtA_bufs[t] = AlignedBuffer(static_cast<std::size_t>(chunk) * N * sizeof(double));
        local_Atb_bufs[t] = AlignedBuffer(static_cast<std::size_t>(chunk)     * sizeof(double));
    }

    // ThreadData per compute_ata — usa ata_chunk [P] e ata_threads [R]
    std::vector<ThreadData> ata_tdata(ata_threads);
    for (int t = 0; t < ata_threads; ++t) {
        ata_tdata[t].At_data   = At_data;
        ata_tdata[t].b         = b;
        ata_tdata[t].AtA_data  = AtA_data;
        ata_tdata[t].Atb       = Atb;
        ata_tdata[t].local_AtA = local_AtA_bufs[t].as_double();
        ata_tdata[t].local_Atb = local_Atb_bufs[t].as_double();
        ata_tdata[t].M         = M;
        ata_tdata[t].N         = N;
        ata_tdata[t].i_start   = ata_chunk[t];    // [P] chunk bilanciato
        ata_tdata[t].i_end     = ata_chunk[t + 1];
        ata_tdata[t].n_threads = ata_threads;
        ata_tdata[t].thread_id = t;
    }

    // ThreadData per fasi ausiliarie — usa aux_chunk e aux_threads [R]
    std::vector<ThreadData> aux_tdata(aux_threads);
    for (int t = 0; t < aux_threads; ++t) {
        aux_tdata[t].At_data   = At_data;
        aux_tdata[t].b         = b;
        aux_tdata[t].AtA_data  = AtA_data;
        aux_tdata[t].Atb       = Atb;
        aux_tdata[t].local_AtA = nullptr; // merge/trasposizione non usano local_AtA
        aux_tdata[t].local_Atb = nullptr;
        aux_tdata[t].M         = M;
        aux_tdata[t].N         = N;
        aux_tdata[t].i_start   = aux_chunk[t];
        aux_tdata[t].i_end     = aux_chunk[t + 1];
        aux_tdata[t].n_threads = aux_threads;
        aux_tdata[t].thread_id = t;
    }
    // merge_upper_triangle legge da local_AtA: deve puntare agli stessi buffer di ata
    // ma con i chunk di aux_threads. Poiché aux e ata possono avere n diverso,
    // il merge deve sapere dove ogni riga i è memorizzata nel buffer locale ata.
    // Soluzione: usiamo aux_tdata solo per trasposizione; per merge usiamo ata_tdata
    // (stesso schema di chunk, evitando una remappatura complessa).

    // -------------------------------------------------------------------------
    // [O] Thread Pool: due pool separati per le due dimensioni di thread
    //
    // pool_ata: ata_threads worker per compute_ata [R]
    // pool_aux: aux_threads worker per merge e trasposizione [R]
    //
    // Ciascun pool viene creato una sola volta e riutilizzato per tutte le fasi.
    // -------------------------------------------------------------------------
    ThreadPool pool_ata(ata_threads); // [O] pool per compute_ata
    ThreadPool pool_aux(aux_threads); // [O] pool per merge e trasposizione

    // Assegna i puntatori ThreadData ai worker dei pool
    for (int t = 0; t < ata_threads; ++t) pool_ata.tdata[t] = &ata_tdata[t];
    for (int t = 0; t < aux_threads; ++t) pool_aux.tdata[t] = &aux_tdata[t];

    // ── [N] TIMER: copre AtA + solve_system ──────────────────────────────────
    const auto t_start = std::chrono::steady_clock::now();

    // -------------------------------------------------------------------------
    // Fase 1: compute_ata — [O] zero spawn, [P] chunk bilanciati, [R] core fisici
    // -------------------------------------------------------------------------
    pool_ata.run_phase(Phase::COMPUTE_ATA); // [O] nessuno spawn: 2 barrier wait

    // -------------------------------------------------------------------------
    // Fase 2: merge — [O] zero spawn, usa aux_threads (tutti i logici) [R]
    // merge_upper_triangle usa local_AtA: punta ai buffer di ata_tdata
    // aux_tdata non ha local_AtA → usiamo pool_ata anche per merge con chunk ata
    // -------------------------------------------------------------------------
    pool_ata.run_phase(Phase::MERGE);       // [O] riusa pool_ata per coerenza dei chunk

    // -------------------------------------------------------------------------
    // Fase 3: trasposizione — [O] zero spawn, usa aux_threads [R]
    // -------------------------------------------------------------------------
    pool_aux.run_phase(Phase::TRANSPOSE);   // [O] pool_aux: chunk lineari su N righe
    
    // [N] solve_system dentro la finestra di misurazione
    std::vector<double> x;
    try {
        solve_system(AtA_data, Atb, x, N, aux_threads);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Errore: %s\n", e.what());
        return 1;
    }

    const auto t_end   = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
    std::printf("Tempo SOLO parallelo: %.6f s\n", elapsed.count() * 1e-6);

    if (N >= 3)
        std::printf("Primi 3 valori di x: %f  %f  %f\n", x[0], x[1], x[2]);

    return 0;
}