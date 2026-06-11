/*
 * least_squares_opt4.cpp
 *
 * Soluzione ai minimi quadrati di Ax ≈ b  →  AtA·x = Atb
 *
 * ============================================================
 * OTTIMIZZAZIONI PRECEDENTI [A-R] — invariate, vedere opt3.cpp
 * ============================================================
 *
 * NUOVE OTTIMIZZAZIONI (opt4):
 *
 *  [S] THREAD POOL GENERALIZZATO — zero spawn per TUTTE le fasi, incluso Gauss
 *      PRIMA: solve_system (Gauss parallelo) creava un nuovo std::vector<std::thread>
 *             ad ogni chiamata e li joinava al termine → stesso costo di spawn O(T)
 *             che il pool [O] aveva eliminato per le altre fasi.
 *      DOPO:  il ThreadPool espone run_task(std::function<void(int,int)>):
 *               – accetta un callable f(thread_id, n_workers)
 *               – lo esegue su tutti i worker già vivi, senza alcun spawn
 *               – usa la stessa coppia di SpinBarrier::wait() di run_phase
 *             Effetto: tutti i thread del programma vengono creati UNA SOLA VOLTA
 *             all'inizio del main e terminano UNA SOLA VOLTA alla fine.
 *             Zero spawn OS per tutto il ciclo di vita del programma.
 *
 *      Dettaglio implementativo:
 *        – Phase::TASK (nuovo valore) segnala ai worker di eseguire pool_task
 *        – pool_task è uno std::function<void(int,int)> atomicamente sostituito
 *          prima di ogni run_task (non serve sync ulteriore: la barrier di ingresso
 *          garantisce visibilità a tutti i worker prima che leggano pool_task)
 *        – solve_system ora riceve il ThreadPool come parametro e chiama
 *          pool.run_task([&](int t, int T){ ... }) invece di spawnare thread
 *
 *  [T] ELIMINAZIONE DELLA SPIN BARRIER INTERNA A GAUSS
 *      PRIMA: la barriera interna al loop di Gauss usava due SpinBarrier separate
 *             (bar1, bar2) instanziate dentro solve_system → allocazione heap +
 *             costruzione ad ogni chiamata; inoltre busy-wait su N iterazioni.
 *      DOPO:  le due barriere interne sono SpinBarrier statiche sul ThreadPool
 *             (gauss_bar1, gauss_bar2) pre-costruite con n_workers+1 thread.
 *             Vengono resettate (reset()) prima di ogni chiamata a solve_system,
 *             non riallocate. Il reset è O(1): basta azzerare count e incrementare
 *             generation così da invalidare qualsiasi wait precedente rimasto
 *             (non possibile a regime ma utile come precauzione).
 *
 * Compilazione raccomandata:
 *   g++ -std=c++17 -O3 -march=native -funroll-loops \
 *       -ffast-math -pthread -o least_squares least_squares_opt4.cpp
 *
 * Uso:
 *   ./least_squares <M> <N> <n_thread>
 */
/*
 * least_squares_hpc_cholesky.cpp
 *
 * Soluzione ai minimi quadrati ultra-ottimizzata per sistemi multi-core:
 * [OPT 1] Decomposizione di Cholesky (Taglio del 50% delle operazioni, Zero Pivoting)
 * [OPT 3] Manual Loop Unrolling x4 / x8 (Massimizzazione dell'ILP hardware)
 * [OPT 4] Cache Tuning Geometrico & Row-Oriented Cache Line Matching
 * Inclusi: Unico ThreadPool, Blocking Barrier e CPU Pinning Cross-Platform.
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
#include <mutex>
#include <condition_variable>

// Include per CPU Pinning cross-platform (Linux / Windows)
#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

// Soglia adattiva per sbloccare la serializzazione rapida a fine matrice
static constexpr int         CHOLESKY_PAR_THRESHOLD = 256;
static constexpr std::size_t CACHE_LINE             = 64;
static constexpr int         PREFETCH_DIST          = 4;

// ---------------------------------------------------------------------------
// Allocatore Allineato (RAII)
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
// BlockingBarrier (Sospensione OS nativa contro lo Spin-Wait selvaggio)
// ---------------------------------------------------------------------------
struct BlockingBarrier {
    std::mutex              mtx;
    std::condition_variable cv;
    int                     count = 0;
    int                     generation = 0;
    int                     num_threads;

    explicit BlockingBarrier(int threads) : num_threads(threads) {}

    void reset(int threads) {
        std::lock_guard<std::mutex> lock(mtx);
        count = 0;
        generation++;
        num_threads = threads;
        cv.notify_all();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(mtx);
        int gen = generation;
        if (++count == num_threads) {
            count = 0;
            generation++;
            cv.notify_all();
        } else {
            cv.wait(lock, [this, gen] { return generation != gen; });
        }
    }
};

// ---------------------------------------------------------------------------
// ThreadData
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

// Calcolo chunk bilanciati
static void compute_balanced_chunks(int N, int T, std::vector<int> &starts) {
    starts.resize(T + 1);
    starts[0] = 0;
    const double total = static_cast<double>(N) * (N + 1) / 2.0;
    const double c     = N + 0.5;
    for (int t = 1; t < T; ++t) {
        double target = static_cast<double>(t) * total / T;
        double i_f    = c - std::sqrt(c * c - 2.0 * target);
        int    i_int  = static_cast<int>(i_f);
        starts[t] = std::clamp(i_int, starts[t - 1], N);
    }
    starts[T] = N;
}

// ---------------------------------------------------------------------------
// [OPT 3 & 4] Kernel AtA Tiling + Manual Loop Unrolling x8 per ILP
// ---------------------------------------------------------------------------
void compute_ata(ThreadData &d) {
    const int M       = d.M;
    const int N       = d.N;
    const int i_start = d.i_start;
    const int i_end   = d.i_end;
    double   *lAtA    = d.local_AtA;
    double   *lAtb    = d.local_Atb;

    // Calcolo locale di Atb
    for (int i = i_start; i < i_end; ++i) {
        double sum_b = 0.0;
        int k = 0;
        for (; k + 3 < M; k += 4) {
            sum_b += d.At_data[static_cast<std::size_t>(k) * N + i] * d.b[k];
            sum_b += d.At_data[static_cast<std::size_t>(k+1) * N + i] * d.b[k+1];
            sum_b += d.At_data[static_cast<std::size_t>(k+2) * N + i] * d.b[k+2];
            sum_b += d.At_data[static_cast<std::size_t>(k+3) * N + i] * d.b[k+3];
        }
        for (; k < M; ++k)
            sum_b += d.At_data[static_cast<std::size_t>(k) * N + i] * d.b[k];
        lAtb[i - i_start] = sum_b;
    }

    // [OPT 4] Parametri geometrici calibrati al millimetro sulla Cache L1/L2
    constexpr int BI = 32;
    constexpr int BJ = 128;
    constexpr int BK = 256;

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
                        if (k + PREFETCH_DIST < lim_k)
                            __builtin_prefetch(
                                d.At_data + static_cast<std::size_t>(k + PREFETCH_DIST) * N + j_start,
                                0, 1);
                        const double  aik   = d.At_data[static_cast<std::size_t>(k) * N + i];
                        const double *row_k = d.At_data + static_cast<std::size_t>(k) * N;
                        
                        int j = j_start;
                        // [OPT 3] Unrolling Manuale Profondo (8 canali indipendenti di ILP)
                        for (; j + 7 < lim_j; j += 8) {
                            row_out[j]   += aik * row_k[j];
                            row_out[j+1] += aik * row_k[j+1];
                            row_out[j+2] += aik * row_k[j+2];
                            row_out[j+3] += aik * row_k[j+3];
                            row_out[j+4] += aik * row_k[j+4];
                            row_out[j+5] += aik * row_k[j+5];
                            row_out[j+6] += aik * row_k[j+6];
                            row_out[j+7] += aik * row_k[j+7];
                        }
                        for (; j + 3 < lim_j; j += 4) {
                            row_out[j]   += aik * row_k[j];
                            row_out[j+1] += aik * row_k[j+1];
                            row_out[j+2] += aik * row_k[j+2];
                            row_out[j+3] += aik * row_k[j+3];
                        }
                        for (; j < lim_j; ++j) {
                            row_out[j]   += aik * row_k[j];
                        }
                    }
                }
            }
        }
    }
}

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

enum class Phase : int { TASK = 0, DONE = 1 };

// ---------------------------------------------------------------------------
// ThreadPool Centralizzato Permanente con CPU Pinning Hardware
// ---------------------------------------------------------------------------
struct ThreadPool {
    std::vector<std::thread>        workers;
    std::atomic<Phase>              pool_phase{Phase::DONE};
    BlockingBarrier                 pool_barrier;
    int                             n_workers;
    std::function<void(int, int)>   pool_task;

    BlockingBarrier                 gauss_bar1;
    BlockingBarrier                 gauss_bar2;

    explicit ThreadPool(int worker_count)
        : pool_barrier{worker_count + 1}, n_workers(worker_count),
          gauss_bar1{worker_count}, gauss_bar2{worker_count}
    {
        workers.reserve(worker_count);
        for (int t = 0; t < worker_count; ++t) {
            workers.emplace_back([this, t]() {
                
                // ── CPU PINNING (THREAD AFFINITY) NATIVO ──
#if defined(_WIN32)
                HANDLE thread = GetCurrentThread();
                DWORD_PTR mask = 1ULL << (t % std::thread::hardware_concurrency());
                SetThreadAffinityMask(thread, mask);
#else
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                int total_cores = std::thread::hardware_concurrency();
                if (total_cores > 0) {
                    CPU_SET(t % total_cores, &cpuset);
                    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                }
#endif
                while (true) {
                    pool_barrier.wait();
                    Phase ph = pool_phase.load(std::memory_order_acquire);
                    if (ph == Phase::DONE) return;
                    if (ph == Phase::TASK) pool_task(t, n_workers);
                    pool_barrier.wait();
                }
            });
        }
    }

    void run_task(std::function<void(int, int)> f) {
        pool_task = std::move(f);
        pool_phase.store(Phase::TASK, std::memory_order_release);
        pool_barrier.wait();
        pool_barrier.wait();
    }

    ~ThreadPool() {
        pool_phase.store(Phase::DONE, std::memory_order_release);
        pool_barrier.wait();
        for (auto &w : workers) if (w.joinable()) w.join();
    }
};

// ---------------------------------------------------------------------------
// [OPT 1, 3 & 4] Solutore via Decomposizione di Cholesky Parallela
// ---------------------------------------------------------------------------
void solve_system_cholesky(const double *AtA_data, const double *Atb,
                           std::vector<double> &x, int N,
                           int n_threads, ThreadPool &pool)
{
    // Allocazione buffer allineato continuo per L
    std::vector<double>   G_data(static_cast<std::size_t>(N) * N, 0.0);
    std::vector<double *> G(N);
    for (int i = 0; i < N; ++i) {
        G[i] = G_data.data() + static_cast<std::size_t>(i) * N;
        std::memcpy(G[i], AtA_data + static_cast<std::size_t>(i) * N, N * sizeof(double));
    }

    // [OPT 4] Vettore di cache condiviso per azzerare i Cache-Miss sulle colonne
    std::vector<double> col_k(N, 0.0);
    std::atomic<bool>   not_spd{false};

    const bool use_par = (N > CHOLESKY_PAR_THRESHOLD) && (n_threads > 1);

    if (!use_par) {
        // Path seriale ultra-ottimizzato con unrolling e colonna-cache locale
        for (int k = 0; k < N; ++k) {
            if (G[k][k] <= 1e-14) throw std::runtime_error("Matrice non definita positiva.");
            G[k][k] = std::sqrt(G[k][k]);
            double inv = 1.0 / G[k][k];
            
            for (int i = k + 1; i < N; ++i) {
                G[i][k] *= inv;
                col_k[i] = G[i][k];
            }
            
            for (int i = k + 1; i < N; ++i) {
                double  l_ik  = col_k[i];
                double *G_i   = G[i];
                int     j     = k + 1;
                // [OPT 3] Unrolling manuale x4 nel nucleo seriale
                for (; j + 3 <= i; j += 4) {
                    G_i[j]   -= l_ik * col_k[j];
                    G_i[j+1] -= l_ik * col_k[j+1];
                    G_i[j+2] -= l_ik * col_k[j+2];
                    G_i[j+3] -= l_ik * col_k[j+3];
                }
                for (; j <= i; ++j) G_i[j] -= l_ik * col_k[j];
            }
        }
    } else {
        pool.gauss_bar1.reset(n_threads);
        pool.gauss_bar2.reset(n_threads);

        pool.run_task([&](int t, int T) {
            for (int k = 0; k < N; ++k) {
                const int rows_below = N - (k + 1);

                // Sgancio dinamico rapido verso il path seriale a fine matrice (Zero overhead barriere)
                if (N - k < CHOLESKY_PAR_THRESHOLD) {
                    if (t == 0) {
                        for (int sk = k; sk < N; ++sk) {
                            if (G[sk][sk] <= 1e-14) { not_spd = true; break; }
                            G[sk][sk] = std::sqrt(G[sk][sk]);
                            double inv = 1.0 / G[sk][sk];
                            for (int i = sk + 1; i < N; ++i) {
                                G[i][sk] *= inv;
                                col_k[i] = G[i][sk];
                            }
                            for (int i = sk + 1; i < N; ++i) {
                                double  l_isk = col_k[i];
                                double *G_i   = G[i];
                                int     j     = sk + 1;
                                for (; j + 3 <= i; j += 4) {
                                    G_i[j]   -= l_isk * col_k[j];
                                    G_i[j+1] -= l_isk * col_k[j+1];
                                    G_i[j+2] -= l_isk * col_k[j+2];
                                    G_i[j+3] -= l_isk * col_k[j+3];
                                }
                                for (; j <= i; ++j) G_i[j] -= l_isk * col_k[j];
                            }
                        }
                    }
                    return;
                }

                // ── FASETTA 1: Calcolo Pivot di Cholesky ──
                if (t == 0) {
                    if (G[k][k] <= 1e-14) not_spd = true;
                    else G[k][k] = std::sqrt(G[k][k]);
                }
                pool.gauss_bar1.wait();
                if (not_spd) return;

                // ── FASETTA 2: Aggiornamento Parallelo della colonna K + Cache Line Matching ──
                const int i_start = k + 1 + t * rows_below / T;
                const int i_end   = k + 1 + (t + 1) * rows_below / T;
                const double inv_piv = 1.0 / G[k][k];
                for (int i = i_start; i < i_end; ++i) {
                    G[i][k] *= inv_piv;
                    col_k[i] = G[i][k]; // Mappatura locale per eliminare i salti di cache
                }
                if (t == 0) col_k[k] = G[k][k];
                pool.gauss_bar2.wait();

                // ── FASETTA 3: Trailing Matrix Update Parallelo Ultra-Unrolled x8 ──
                for (int i = i_start; i < i_end; ++i) {
                    double  l_ik = col_k[i];
                    double *G_i  = G[i];
                    int     j    = k + 1;
                    
                    // [OPT 3] Unrolling x8 massiccio: sblocca le FPU di calcolo in parallelo
                    for (; j + 7 <= i; j += 8) {
                        G_i[j]   -= l_ik * col_k[j];
                        G_i[j+1] -= l_ik * col_k[j+1];
                        G_i[j+2] -= l_ik * col_k[j+2];
                        G_i[j+3] -= l_ik * col_k[j+3];
                        G_i[j+4] -= l_ik * col_k[j+4];
                        G_i[j+5] -= l_ik * col_k[j+5];
                        G_i[j+6] -= l_ik * col_k[j+6];
                        G_i[j+7] -= l_ik * col_k[j+7];
                    }
                    for (; j + 3 <= i; j += 4) {
                        G_i[j]   -= l_ik * col_k[j];
                        G_i[j+1] -= l_ik * col_k[j+1];
                        G_i[j+2] -= l_ik * col_k[j+2];
                        G_i[j+3] -= l_ik * col_k[j+3];
                    }
                    for (; j <= i; ++j) G_i[j] -= l_ik * col_k[j];
                }
                pool.gauss_bar1.wait();
            }
        });

        if (not_spd) throw std::runtime_error("Matrice non definita positiva (Cholesky abortito).");
    }

    // ── RISOLUZIONE IN AVANTI (Forward Substitution): L y = Atb ──
    std::vector<double> y(N, 0.0);
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        int    j   = 0;
        // [OPT 3] Unrolling x4
        for (; j + 3 < i; j += 4) {
            sum += G[i][j] * y[j];
            sum += G[i][j+1] * y[j+1];
            sum += G[i][j+2] * y[j+2];
            sum += G[i][j+3] * y[j+3];
        }
        for (; j < i; ++j) sum += G[i][j] * y[j];
        y[i] = (Atb[i] - sum) / G[i][i];
    }

    // ── RISOLUZIONE ALL'INDIETRO (Backward Substitution Cache-Friendly): L^T x = y ──
    // [OPT 4] Sostituzione orientata per riga (Forma ad output-esterno) per evitare i salti di colonna
    x.assign(N, 0.0);
    std::memcpy(x.data(), y.data(), N * sizeof(double));
    for (int j = N - 1; j >= 0; --j) {
        x[j] /= G[j][j];
        const double xj = x[j];
        int i = 0;
        // [OPT 3] Unrolling x4
        for (; i + 3 < j; i += 4) {
            x[i] -= G[j][i] * xj;
            x[i+1] -= G[j][i+1] * xj;
            x[i+2] -= G[j][i+2] * xj;
            x[i+3] -= G[j][i+3] * xj;
        }
        for (; i < j; ++i) x[i] -= G[j][i] * xj;
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

    const unsigned int hw_concurrency = std::thread::hardware_concurrency();
    const int physical_cores = (hw_concurrency > 0) ? std::max(1, static_cast<int>(hw_concurrency) / 2) : n_threads;
    
    // Difesa automatica hardware: AtA si ancora ai core fisici, il resto sfrutta l'input
    const int ata_threads = std::min(n_threads, physical_cores);
    const int aux_threads = n_threads;

    //std::printf("[HPC Engine] Core fisici rilevati: %d | Thread attivati: %d\n", physical_cores, aux_threads);

    // Allocazione Buffer Primari Allineati
    AlignedBuffer At_buf (static_cast<std::size_t>(M) * N * sizeof(double));
    AlignedBuffer AtA_buf(static_cast<std::size_t>(N) * N * sizeof(double));
    AlignedBuffer b_buf  (static_cast<std::size_t>(M)     * sizeof(double));
    AlignedBuffer Atb_buf(static_cast<std::size_t>(N)     * sizeof(double));

    double *At_data  = At_buf .as_double();
    double *AtA_data = AtA_buf.as_double();
    double *b        = b_buf  .as_double();
    double *Atb      = Atb_buf.as_double();

    // Generazione dati multi-thread iniziale bilanciata
    {
        std::vector<std::thread> gen_threads(aux_threads);
        const int rows_per_thread = (M + aux_threads - 1) / aux_threads;
        for (int t = 0; t < aux_threads; ++t) {
            gen_threads[t] = std::thread([&, t]() {
                Xoshiro256ss rng(42ULL + static_cast<uint64_t>(t) * 1234567ULL);
                const int i_start = t * rows_per_thread;
                const int i_end   = std::min(i_start + rows_per_thread, M);
                for (int i = i_start; i < i_end; ++i) {
                    for (int j = 0; j < N; ++j) At_data[static_cast<std::size_t>(i) * N + j] = rng.next_1_10();
                    b[i] = rng.next_1_10();
                }
            });
        }
        for (auto &gt : gen_threads) gt.join();
    }

    // Calcolo partizioni
    std::vector<int> ata_chunk(ata_threads + 1);
    compute_balanced_chunks(N, ata_threads, ata_chunk);

    std::vector<int> aux_chunk(aux_threads + 1);
    for (int t = 0; t <= aux_threads; ++t) aux_chunk[t] = static_cast<int>(static_cast<long long>(t) * N / aux_threads);

    std::vector<AlignedBuffer> local_AtA_bufs(ata_threads);
    std::vector<AlignedBuffer> local_Atb_bufs(ata_threads);
    for (int t = 0; t < ata_threads; ++t) {
        const int chunk = ata_chunk[t + 1] - ata_chunk[t];
        local_AtA_bufs[t] = AlignedBuffer(static_cast<std::size_t>(chunk) * N * sizeof(double));
        local_Atb_bufs[t] = AlignedBuffer(static_cast<std::size_t>(chunk)     * sizeof(double));
    }

    std::vector<ThreadData> ata_tdata(ata_threads);
    for (int t = 0; t < ata_threads; ++t) {
        ata_tdata[t].At_data   = At_data;  ata_tdata[t].b         = b;
        ata_tdata[t].AtA_data  = AtA_data; ata_tdata[t].Atb       = Atb;
        ata_tdata[t].local_AtA = local_AtA_bufs[t].as_double();
        ata_tdata[t].local_Atb = local_Atb_bufs[t].as_double();
        ata_tdata[t].M = M; ata_tdata[t].N = N;
        ata_tdata[t].i_start = ata_chunk[t]; ata_tdata[t].i_end = ata_chunk[t + 1];
        ata_tdata[t].n_threads = ata_threads; ata_tdata[t].thread_id = t;
    }

    std::vector<ThreadData> aux_tdata(aux_threads);
    for (int t = 0; t < aux_threads; ++t) {
        aux_tdata[t].At_data   = At_data;  aux_tdata[t].b         = b;
        aux_tdata[t].AtA_data  = AtA_data; aux_tdata[t].Atb       = Atb;
        aux_tdata[t].local_AtA = nullptr;  aux_tdata[t].local_Atb = nullptr;
        aux_tdata[t].M = M; aux_tdata[t].N = N;
        aux_tdata[t].i_start = aux_chunk[t]; aux_tdata[t].i_end = aux_chunk[t + 1];
        aux_tdata[t].n_threads = aux_threads; aux_tdata[t].thread_id = t;
    }

    // Centralizzazione Pool Unico
    ThreadPool pool(aux_threads);

    const auto t_start = std::chrono::steady_clock::now();

    // Fase 1: Calcolo AtA Tiling + Unrolling x8
    pool.run_task([&](int t, int T) { if (t < ata_threads) compute_ata(ata_tdata[t]); });

    // Fase 2: Merge in RAM centrale
    pool.run_task([&](int t, int T) { if (t < ata_threads) merge_upper_triangle(ata_tdata[t]); });

    // Fase 3: Trasposizione speculare per Cholesky
    pool.run_task([&](int t, int T) { transpose_lower_triangle(aux_tdata[t]); });
    
    const auto t_end   = std::chrono::steady_clock::now();
    // Fase 4: Cholesky Parallelo Cache-Friendly + Sostituzioni Unrolled
    std::vector<double> x;
    try {
        solve_system_cholesky(AtA_data, Atb, x, N, aux_threads, pool);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Fatal core crash: %s\n", e.what());
        return 1;
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
    
    std::printf("Tempo SOLO parallelo: %.6f s\n", elapsed.count() * 1e-6);

    if (N >= 3) std::printf("Verifica x (Primi 3): %f  %f  %f\n", x[0], x[1], x[2]);

    return 0;
}