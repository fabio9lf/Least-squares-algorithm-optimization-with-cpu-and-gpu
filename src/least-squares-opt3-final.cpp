/*
 * least-squares-opt4.cpp
 * ============================================================
 * Ottimizzazioni rispetto a opt3-3:
 *
 *  1. PADDING ANTI-FALSE-SHARING su thread_data_t
 *  2. VETTORE v2[] = 2·v[i] PRECALCOLATO
 *  3. PARALLELIZZAZIONE dot(v,y) + riduzione tra thread
 *  4. DISTRIBUZIONE COLONNE INTERLEAVING (round-robin)
 *  5. PREFETCH ESPLICITO colonna j+PREFETCH_DIST
 *  6. INNER LOOP UNROLLED x8
 *  7. TIMING CORRETTO (prima del barrier finale)
 *  8. BACK SUBSTITUTION con accumulatore in registro
 *
 * Compilazione:
 *   g++ -O3 -march=native -ffast-math -o least-squares-opt4 least-squares-opt4.cpp -lpthread -lm
 * ============================================================
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <chrono>

using namespace std;

#define CACHE_LINE    64
#define PREFETCH_DIST  4

// ============================================================
// [OTT. 1] Forward declaration per permettere l'auto-riferimento.
// Il typedef viene completato dopo la definizione della struct.
// Senza forward declaration il compilatore non conosce ancora
// "thread_data_t" quando incontra il campo all_threads.
// ============================================================
struct thread_data_s;
typedef struct thread_data_s thread_data_t;

struct alignas(CACHE_LINE) thread_data_s {
    // HOT: scritti ad ogni iterazione → nella prima cache line
    double   partial_doty;   // Parziale locale di dot(v, y)
    double   timer_ms;       // Accumulatore timer (solo thread 0)

    // Puntatori condivisi (read-only dopo init)
    float*         R;
    float*         y;
    float*         v;
    float*         v2;           // [OTT. 2] 2·v[i] precalcolato
    thread_data_t* all_threads;  // Puntatore all'array per la riduzione

    int  M, N;
    int  n_threads;
    int  thread_id;
    pthread_barrier_t* barrier;
};

// ============================================================
// Calcolo vettore di Householder + v2 = 2·v  (thread 0)
// ============================================================
static inline void compute_householder_vector(
    const float* __restrict__ R,
    float* __restrict__       v,
    float* __restrict__       v2,
    int k, int M)
{
    const int base = k * M;
    float norm_x = 0.0f;
    for (int i = k; i < M; i++) { float val = R[base + i]; norm_x += val * val; }
    norm_x = sqrtf(norm_x);

    memset(v, 0, (size_t)k * sizeof(float));

    const float alpha = (R[base + k] > 0.0f) ? -norm_x : norm_x;
    v[k] = R[base + k] - alpha;
    for (int i = k + 1; i < M; i++) v[i] = R[base + i];

    float norm_v = 0.0f;
    for (int i = k; i < M; i++) norm_v += v[i] * v[i];
    norm_v = sqrtf(norm_v);

    if (norm_v > 1e-12f) {
        const float inv = 1.0f / norm_v;
        for (int i = k; i < M; i++) v[i] *= inv;
    }

    // [OTT. 2] Precalcolo v2: evita "2.0f * v[i]" per ogni colonna × ogni riga
    for (int i = k; i < M; i++) v2[i] = 2.0f * v[i];
}

// ============================================================
// WORKER THREAD
// ============================================================
void* qr_worker(void* arg) {
    thread_data_t* data       = (thread_data_t*)arg;
    const int      M          = data->M;
    const int      N          = data->N;
    const int      tid        = data->thread_id;
    const int      n_threads  = data->n_threads;
    float* __restrict__ R     = data->R;
    float* __restrict__ y     = data->y;
    float* __restrict__ v     = data->v;
    float* __restrict__ v2    = data->v2;

    for (int k = 0; k < N && k < M; k++) {

        // ---- Fase 1: Householder (sequenziale, thread 0) ----
        if (tid == 0)
            compute_householder_vector(R, v, v2, k, M);

        pthread_barrier_wait(data->barrier);

        // ---- Fase 2: Applica H_k alle colonne k+1..N-1 (PARALLELA) ----
        auto t_start = chrono::steady_clock::now();

        // [OTT. 4] Interleaving round-robin: carico bilanciato anche con
        // poche colonne residue (fine fattorizzazione, k ~ N)
        for (int j = (k + 1) + tid; j < N; j += n_threads) {

            // [OTT. 5] Prefetch della colonna j+PREFETCH_DIST
            if (j + PREFETCH_DIST < N)
                __builtin_prefetch(&R[(j + PREFETCH_DIST) * M + k], 1, 1);

            const int col = j * M;

            // [OTT. 6] Dot product unrolled x8
            int   i    = k;
            float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
            for (; i <= M - 8; i += 8) {
                d0 += v[i  ] * R[col + i  ];
                d1 += v[i+1] * R[col + i+1];
                d2 += v[i+2] * R[col + i+2];
                d3 += v[i+3] * R[col + i+3];
                d4 += v[i+4] * R[col + i+4];
                d5 += v[i+5] * R[col + i+5];
                d6 += v[i+6] * R[col + i+6];
                d7 += v[i+7] * R[col + i+7];
            }
            float dot = d0+d1+d2+d3+d4+d5+d6+d7;
            for (; i < M; i++) dot += v[i] * R[col + i];

            // Aggiornamento R(:,j) -= v2 * dot  (unrolled x8)
            i = k;
            for (; i <= M - 8; i += 8) {
                R[col+i  ] -= v2[i  ] * dot;
                R[col+i+1] -= v2[i+1] * dot;
                R[col+i+2] -= v2[i+2] * dot;
                R[col+i+3] -= v2[i+3] * dot;
                R[col+i+4] -= v2[i+4] * dot;
                R[col+i+5] -= v2[i+5] * dot;
                R[col+i+6] -= v2[i+6] * dot;
                R[col+i+7] -= v2[i+7] * dot;
            }
            for (; i < M; i++) R[col+i] -= v2[i] * dot;
        }

        // [OTT. 7] Timer campionato PRIMA del barrier (misura solo lavoro utile)
        auto t_end = chrono::steady_clock::now();
        if (tid == 0) {
            chrono::duration<double, milli> dur = t_end - t_start;
            data->timer_ms += dur.count();
        }

        // ---- Fase 3: aggiornamento di y (PARALLELIZZATO) ----
        // [OTT. 3a] Parziale locale di dot(v, y)
        {
            float local = 0.0f;
            for (int i = k + tid; i < M; i += n_threads)
                local += v[i] * y[i];
            data->partial_doty = (double)local;
        }

        pthread_barrier_wait(data->barrier);

        // [OTT. 3b] Thread 0: riduzione parziali + aggiornamento y
        if (tid == 0) {
            double doty = 0.0;
            thread_data_t* all = data->all_threads;
            for (int t = 0; t < n_threads; t++) doty += all[t].partial_doty;
            for (int i = k; i < M; i++) y[i] -= v2[i] * (float)doty;
        }

        pthread_barrier_wait(data->barrier);
    }

    if (tid == 0)
        printf("Tempo SOLO Parallelo: %.6f ms\n", data->timer_ms);

    return NULL;
}

// ============================================================
// [OTT. 8] Back substitution con acc in registro
// ============================================================
void back_substitution(const float* __restrict__ R,
                       const float* __restrict__ y,
                       float* __restrict__       x,
                       int M, int N)
{
    for (int i = N - 1; i >= 0; i--) {
        float acc = y[i];
        for (int j = i + 1; j < N; j++) acc -= R[j * M + i] * x[j];
        const float diag = R[i * M + i];
        x[i] = (fabsf(diag) > 1e-12f) ? acc / diag : 0.0f;
    }
}

// ============================================================
// ENTRY POINT PARALLELO
// ============================================================
void least_squares_parallel(float* A, float* b, float* x,
                             int M, int N, int n_threads)
{
    float *R = nullptr, *y = nullptr, *v = nullptr, *v2 = nullptr;
    if (posix_memalign((void**)&R,  CACHE_LINE, (size_t)M * N * sizeof(float)) != 0 ||
        posix_memalign((void**)&y,  CACHE_LINE, (size_t)M * sizeof(float))     != 0 ||
        posix_memalign((void**)&v,  CACHE_LINE, (size_t)M * sizeof(float))     != 0 ||
        posix_memalign((void**)&v2, CACHE_LINE, (size_t)M * sizeof(float))     != 0) {
        fprintf(stderr, "posix_memalign fallito\n"); exit(1);
    }

    // Trasposizione A (Row-Major) → R (Column-Major) con cache blocking
    const int BLK = 64;
    for (int i = 0; i < M; i += BLK)
        for (int j = 0; j < N; j += BLK) {
            int ei = (i+BLK < M) ? i+BLK : M;
            int ej = (j+BLK < N) ? j+BLK : N;
            for (int ii = i; ii < ei; ii++)
                for (int jj = j; jj < ej; jj++)
                    R[jj * M + ii] = A[ii * N + jj];
        }
    memcpy(y, b, (size_t)M * sizeof(float));

    // Alloca array di strutture allineato (ogni elemento su cache line propria)
    thread_data_t* t_data = (thread_data_t*)aligned_alloc(
        CACHE_LINE, (size_t)n_threads * sizeof(thread_data_t));
    if (!t_data) { fprintf(stderr, "aligned_alloc fallito\n"); exit(1); }
    memset(t_data, 0, (size_t)n_threads * sizeof(thread_data_t));

    pthread_t*        threads = (pthread_t*)malloc((size_t)n_threads * sizeof(pthread_t));
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, (unsigned)n_threads);

    for (int t = 0; t < n_threads; t++) {
        t_data[t].R           = R;
        t_data[t].y           = y;
        t_data[t].v           = v;
        t_data[t].v2          = v2;
        t_data[t].all_threads = t_data;
        t_data[t].M           = M;
        t_data[t].N           = N;
        t_data[t].n_threads   = n_threads;
        t_data[t].thread_id   = t;
        t_data[t].barrier     = &barrier;
        pthread_create(&threads[t], NULL, qr_worker, &t_data[t]);
    }
    for (int t = 0; t < n_threads; t++) pthread_join(threads[t], NULL);

    back_substitution(R, y, x, M, N);

    pthread_barrier_destroy(&barrier);
    free(threads); free(t_data);
    free(R); free(y); free(v); free(v2);
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 3) { printf("Utilizzo: %s M N [n_threads]\n", argv[0]); return 1; }
    int M         = atoi(argv[1]);
    int N         = atoi(argv[2]);
    int n_threads = (argc >= 4) ? atoi(argv[3]) : 1;

    float* A = (float*)malloc((size_t)M * N * sizeof(float));
    for (int i = 0; i < M * N; i++) A[i] = (float)(rand() % 100 + 1);
    float* b = (float*)malloc((size_t)M * sizeof(float));
    for (int i = 0; i < M; i++) b[i] = (float)(rand() % 100 + 1);
    float* x = (float*)malloc((size_t)N * sizeof(float));

    auto t0 = chrono::steady_clock::now();
    least_squares_parallel(A, b, x, M, N, n_threads);
    auto t1 = chrono::steady_clock::now();

    long long ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    printf("Thread: %d | Tempo TOTALE (Seq + Par): %lld ms\n", n_threads, ms);

    free(A); free(b); free(x);
    return 0;
}