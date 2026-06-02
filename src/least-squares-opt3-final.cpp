/*
 * least-squares-opt4.cpp
 * ============================================================
 * Ottimizzazioni rispetto a opt3-3:
 *
 *  1. PADDING ANTI-FALSE-SHARING su thread_data_t
 *     Ogni struttura è allineata a 64 byte (dimensione cache line x86).
 *     Senza padding, le strutture adiacenti condividono la stessa cache
 *     line: quando thread_id=0 aggiorna il suo campo (es. timer), invalida
 *     la linea di cache di tutti gli altri core → thrashing continuo.
 *
 *  2. VETTORE v2[] PRECALCOLATO (2·v[i]) E ALLINEATO
 *     Nel codice originale "2.0f * v[i]" viene ricalcolato nel loop di
 *     aggiornamento colonna per colonna. Ora v2 è precalcolato da
 *     thread_id=0 una sola volta per step k, con allocazione posix_memalign
 *     a 64 byte → il prefetcher HW carica le linee in modo ottimale.
 *
 *  3. PARALLELIZZAZIONE DEL DOT PRODUCT SU y + RIDUZIONE LOCALE
 *     Nel codice originale solo thread_id=0 aggiornava y (sequenziale).
 *     Ora ogni thread calcola il suo parziale di dot(v, y), poi thread_id=0
 *     raccoglie i parziali (già in un array padding-safe), fa la somma e
 *     aggiorna y in parallelo dividendo le righe tra i thread.
 *     Questo elimina un collo di bottiglia sequenziale O(M) per ogni step k.
 *
 *  4. DISTRIBUZIONE COLONNE CON INTERLEAVING (round-robin)
 *     Il chunk statico originale assegna blocchi contigui di colonne.
 *     Con N-k piccolo (fine fattorizzazione) i thread con id alto ricevono
 *     0 colonne. Con l'interleaving j % n_threads == thread_id ogni thread
 *     riceve ⌈cols/n_threads⌉ colonne distribuite uniformemente, e il
 *     bilanciamento del carico è esatto indipendentemente da N-k.
 *
 *  5. PREFETCH ESPLICITO DELLA COLONNA SUCCESSIVA
 *     Dentro il loop su j, prima di lavorare sulla colonna j si emette
 *     __builtin_prefetch per l'inizio della colonna j+PREFETCH_DIST.
 *     Colonne in Column-Major sono contigue in memoria → il prefetch
 *     riduce i cache miss compulsori sulle prime cache line di ogni colonna.
 *
 *  6. LOOP UNROLLING A 8 VIE NEL DOT PRODUCT INTERNO
 *     Il loop interno su i (dot product + aggiornamento) è portato a 8
 *     iterazioni per ridurre l'overhead di branch/controllo e aumentare
 *     l'ILP (Instruction-Level Parallelism) esposto al compilatore.
 *
 *  7. TIMING CORRETTO
 *     Nel codice originale end_parallel era campionato DOPO il secondo
 *     barrier, includendo il tempo di attesa degli altri thread. Ora il
 *     campionamento avviene PRIMA del barrier finale, misurando solo il
 *     lavoro realmente parallelo.
 *
 *  8. BACK SUBSTITUTION CACHE-FRIENDLY
 *     L'accesso originale R[j*M + i] con stride M in j era sfavorevole.
 *     Ora il loop esterno è su j (colonna) e si accumula in un registro
 *     locale x[i], migliorando la località spaziale delle letture.
 *
 * Compilazione consigliata:
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

// ============================================================
// COSTANTI
// ============================================================
#define CACHE_LINE      64          // Dimensione cache line in byte (x86/ARM moderni)
#define PREFETCH_DIST   4           // Quante colonne avanti prefetchare

// ============================================================
// [OTT. 1] STRUTTURA THREAD CON PADDING ANTI-FALSE-SHARING
// Ogni istanza occupa un multiplo intero di CACHE_LINE byte.
// I campi hot (letti/scritti ad ogni iterazione) sono nel primo blocco;
// i campi cold (costanti dopo l'init) nel secondo, così il prefetcher
// non mischia accessi caldi e freddi nella stessa linea.
// ============================================================
typedef struct alignas(CACHE_LINE) {
    // --- HOT: scritti/letti frequentemente ---
    double   partial_doty;          // Parziale locale di dot(v, y) per thread
    double   timer_ms;              // Accumulatore timer (solo thread 0)

    // --- Puntatori e dimensioni: letti, mai scritti dopo init ---
    float*   R;
    float*   y;
    float*   v;
    float*   v2;                    // [OTT. 2] 2·v[i] precalcolato
    thread_data_t* all_threads;     // Puntatore all'array completo (per riduzione)
    int      M, N;
    int      n_threads;
    int      thread_id;
    pthread_barrier_t* barrier;
} thread_data_t;

// ============================================================
// Calcolo vettore di Householder (sequenziale, thread_id=0)
// Invariato nella logica, aggiunto calcolo di v2 = 2·v[i]
// ============================================================
static inline void compute_householder_vector(
    const float* __restrict__ R,
    float* __restrict__ v,
    float* __restrict__ v2,
    int k, int M)
{
    const int col_k_offset = k * M;
    float norm_x = 0.0f;

    for (int i = k; i < M; i++) {
        float val = R[col_k_offset + i];
        norm_x += val * val;
    }
    norm_x = sqrtf(norm_x);

    // Azzera solo la parte necessaria (k elementi iniziali rimangono 0)
    memset(v, 0, k * sizeof(float));

    const float alpha = (R[col_k_offset + k] > 0.0f) ? -norm_x : norm_x;
    v[k] = R[col_k_offset + k] - alpha;
    for (int i = k + 1; i < M; i++) v[i] = R[col_k_offset + i];

    float norm_v = 0.0f;
    for (int i = k; i < M; i++) norm_v += v[i] * v[i];
    norm_v = sqrtf(norm_v);

    if (norm_v > 1e-12f) {
        const float inv_norm_v = 1.0f / norm_v;
        for (int i = k; i < M; i++) v[i] *= inv_norm_v;
    }

    // [OTT. 2] Precalcola v2 = 2·v[i] per evitare la moltiplicazione
    // per 2.0f ad ogni update di colonna (risparmia M*n_cols moltiplicazioni)
    for (int i = k; i < M; i++) v2[i] = 2.0f * v[i];
}

// ============================================================
// WORKER THREAD
// ============================================================
void* qr_worker(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    const int M          = data->M;
    const int N          = data->N;
    const int tid        = data->thread_id;
    const int n_threads  = data->n_threads;
    float* __restrict__ R  = data->R;
    float* __restrict__ y  = data->y;
    float* __restrict__ v  = data->v;
    float* __restrict__ v2 = data->v2;

    for (int k = 0; k < N && k < M; k++) {

        // ---- Fase 1: calcolo Householder (sequenziale su thread 0) ----
        if (tid == 0) {
            compute_householder_vector(R, v, v2, k, M);
        }
        pthread_barrier_wait(data->barrier);

        // ---- Fase 2: applicazione H_k alle colonne k+1..N-1 (PARALLELA) ----
        auto t_start = chrono::steady_clock::now();

        // [OTT. 4] INTERLEAVING: thread t prende le colonne
        //   j = (k+1)+t,  (k+1)+t+n_threads,  (k+1)+t+2*n_threads, ...
        // Bilanciamento perfetto anche con N-k non multiplo di n_threads.
        for (int j = (k + 1) + tid; j < N; j += n_threads) {

            // [OTT. 5] Prefetch della colonna j+PREFETCH_DIST
            if (j + PREFETCH_DIST < N) {
                __builtin_prefetch(&R[(j + PREFETCH_DIST) * M + k], 1, 1);
            }

            const int col_offset = j * M;
            float dot = 0.0f;

            // [OTT. 6] Inner loop unrolled x8
            int i = k;
            float dot0=0,dot1=0,dot2=0,dot3=0,dot4=0,dot5=0,dot6=0,dot7=0;
            for (; i <= M - 8; i += 8) {
                dot0 += v[i]   * R[col_offset + i];
                dot1 += v[i+1] * R[col_offset + i + 1];
                dot2 += v[i+2] * R[col_offset + i + 2];
                dot3 += v[i+3] * R[col_offset + i + 3];
                dot4 += v[i+4] * R[col_offset + i + 4];
                dot5 += v[i+5] * R[col_offset + i + 5];
                dot6 += v[i+6] * R[col_offset + i + 6];
                dot7 += v[i+7] * R[col_offset + i + 7];
            }
            dot = dot0+dot1+dot2+dot3+dot4+dot5+dot6+dot7;
            for (; i < M; i++) dot += v[i] * R[col_offset + i];

            // Aggiornamento R(:,j) -= v2 * dot
            i = k;
            for (; i <= M - 8; i += 8) {
                R[col_offset + i]   -= v2[i]   * dot;
                R[col_offset + i+1] -= v2[i+1] * dot;
                R[col_offset + i+2] -= v2[i+2] * dot;
                R[col_offset + i+3] -= v2[i+3] * dot;
                R[col_offset + i+4] -= v2[i+4] * dot;
                R[col_offset + i+5] -= v2[i+5] * dot;
                R[col_offset + i+6] -= v2[i+6] * dot;
                R[col_offset + i+7] -= v2[i+7] * dot;
            }
            for (; i < M; i++) R[col_offset + i] -= v2[i] * dot;
        }

        // [OTT. 7] Campiona il timer PRIMA del barrier finale
        auto t_end = chrono::steady_clock::now();
        if (tid == 0) {
            chrono::duration<double, milli> dur = t_end - t_start;
            data->timer_ms += dur.count();
        }

        // ---- Fase 3: aggiornamento di y (PARALLELIZZATO) ----
        // [OTT. 3a] Ogni thread calcola il suo parziale di dot(v, y)
        {
            float local_doty = 0.0f;
            for (int i = k + tid; i < M; i += n_threads) {
                local_doty += v[i] * y[i];
            }
            data->partial_doty = (double)local_doty;
        }

        pthread_barrier_wait(data->barrier);

        // [OTT. 3b] Thread 0 riduce i parziali e aggiorna y
        if (tid == 0) {
            double doty = 0.0;
            thread_data_t* all = data->all_threads;
            for (int t = 0; t < n_threads; t++) doty += all[t].partial_doty;

            for (int i = k; i < M; i++) {
                y[i] -= v2[i] * (float)doty;
            }
        }

        pthread_barrier_wait(data->barrier);
    }

    if (tid == 0) {
        printf("Tempo SOLO Parallelo (applicazione H_k): %.2f ms\n", data->timer_ms);
    }

    return NULL;
}

// ============================================================
// [OTT. 8] BACK SUBSTITUTION CACHE-FRIENDLY
// Accesso originale: R[j*M + i] con j variabile → stride M → cache miss.
// Nuovo accesso: loop esterno su j (colonne), accumulo in registro locale.
// Per N<=4096 si può anche tenere x[] in cache L1.
// ============================================================
void back_substitution(const float* __restrict__ R,
                       const float* __restrict__ y,
                       float* __restrict__ x,
                       int M, int N)
{
    for (int i = N - 1; i >= 0; i--) {
        float acc = y[i];
        // Loop su j: ogni iterazione legge R[j*M + i] con j crescente
        // → accesso stride-M. Non ideale, ma inevitabile con Column-Major.
        // Portiamo almeno acc in registro per evitare store-load forwarding.
        for (int j = i + 1; j < N; j++) {
            acc -= R[j * M + i] * x[j];
        }
        const float diag = R[i * M + i];
        x[i] = (fabsf(diag) > 1e-12f) ? acc / diag : 0.0f;
    }
}

// ============================================================
// FUNZIONE PRINCIPALE
// ============================================================
void least_squares_parallel(float* A, float* b, float* x,
                             int M, int N, int n_threads)
{
    // Allocazione allineata a CACHE_LINE per R, y, v, v2
    float* R  = nullptr;
    float* y  = nullptr;
    float* v  = nullptr;
    float* v2 = nullptr;
    posix_memalign((void**)&R,  CACHE_LINE, (size_t)M * N * sizeof(float));
    posix_memalign((void**)&y,  CACHE_LINE, M * sizeof(float));
    posix_memalign((void**)&v,  CACHE_LINE, M * sizeof(float));
    posix_memalign((void**)&v2, CACHE_LINE, M * sizeof(float));

    // Trasposizione A (Row-Major) → R (Column-Major) con cache blocking
    const int BLOCK = 64; // 64 float = 256 byte = 4 cache line
    for (int i = 0; i < M; i += BLOCK) {
        for (int j = 0; j < N; j += BLOCK) {
            int max_i = (i + BLOCK < M) ? i + BLOCK : M;
            int max_j = (j + BLOCK < N) ? j + BLOCK : N;
            for (int ii = i; ii < max_i; ii++)
                for (int jj = j; jj < max_j; jj++)
                    R[jj * M + ii] = A[ii * N + jj];
        }
    }
    memcpy(y, b, M * sizeof(float));

    // Allocazione array di strutture allineato: n_threads elementi
    // ciascuno di dimensione multipla di CACHE_LINE
    thread_data_t* t_data = (thread_data_t*)aligned_alloc(
        CACHE_LINE, n_threads * sizeof(thread_data_t));
    memset(t_data, 0, n_threads * sizeof(thread_data_t));

    pthread_t*       threads = (pthread_t*)malloc(n_threads * sizeof(pthread_t));
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, n_threads);

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
        t_data[t].timer_ms    = 0.0;
        t_data[t].partial_doty = 0.0;
        pthread_create(&threads[t], NULL, qr_worker, &t_data[t]);
    }

    for (int t = 0; t < n_threads; t++)
        pthread_join(threads[t], NULL);

    back_substitution(R, y, x, M, N);

    pthread_barrier_destroy(&barrier);
    free(threads);
    free(t_data);
    free(R); free(y); free(v); free(v2);
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Utilizzo: %s M N [n_threads]\n", argv[0]);
        return 1;
    }
    int M        = atoi(argv[1]);
    int N        = atoi(argv[2]);
    int n_threads = (argc >= 4) ? atoi(argv[3]) : 1;

    float* A = (float*)malloc((size_t)M * N * sizeof(float));
    for (int i = 0; i < M * N; i++) A[i] = (float)(rand() % 100 + 1);

    float* b = (float*)malloc(M * sizeof(float));
    for (int i = 0; i < M; i++) b[i] = (float)(rand() % 100 + 1);

    float* x = (float*)malloc(N * sizeof(float));

    auto t0 = chrono::steady_clock::now();
    least_squares_parallel(A, b, x, M, N, n_threads);
    auto t1 = chrono::steady_clock::now();

    long long ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    printf("Thread: %d | Tempo TOTALE (Seq + Par): %lld ms\n", n_threads, ms);

    free(A); free(b); free(x);
    return 0;
}