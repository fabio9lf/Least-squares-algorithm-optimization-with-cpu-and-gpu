#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

// ─────────────────────────────────────────────
//  Struttura condivisa dal thread pool
// ─────────────────────────────────────────────
typedef struct {
    // Dati della matrice (condivisi, read-only tranne R)
    double** R;
    double*  v;
    int      M, N;

    // Stato corrente dell'iterazione (scritto dal main, letto dai worker)
    volatile int     k;
    volatile int     active_threads; // quanti thread lavorano in questo passo k

    // Indice del thread (assegnato alla creazione)
    int thread_id;
    int total_threads;

    // Barriere di sincronizzazione
    pthread_barrier_t* barrier_start; // main → worker: "inizia il passo k"
    pthread_barrier_t* barrier_end;   // worker → main: "ho finito"

    // Flag di uscita
    volatile int stop;
} pool_thread_data_t;


// ─────────────────────────────────────────────
//  Worker del thread pool
// ─────────────────────────────────────────────
void* pool_worker(void* arg) {
    pool_thread_data_t* d = (pool_thread_data_t*)arg;

    while (1) {
        // Attendi che il main segnali un nuovo passo k
        pthread_barrier_wait(d->barrier_start);

        if (d->stop) break;

        int k   = d->k;
        int M   = d->M;
        int N   = d->N;
        int tid = d->thread_id;
        int nt  = d->active_threads; // thread effettivi per questo passo

        // Solo i thread che hanno lavoro da fare lo eseguono
        if (tid < nt) {
            int cols_to_process = N - k;
            int cols_per_thread = cols_to_process / nt;
            int start_col = k + tid * cols_per_thread;
            int end_col   = (tid == nt - 1) ? N : start_col + cols_per_thread;

            for (int j = start_col; j < end_col; j++) {
                double dot = 0.0;
                for (int i = k; i < M; i++)
                    dot += d->v[i] * d->R[i][j];
                for (int i = k; i < M; i++)
                    d->R[i][j] -= 2.0 * d->v[i] * dot;
            }
        }

        // Segnala al main che questo thread ha terminato
        pthread_barrier_wait(d->barrier_end);
    }

    return NULL;
}


// ─────────────────────────────────────────────
//  Householder vector (invariato)
// ─────────────────────────────────────────────
void compute_householder_vector(double** R, double* v, int k, int M) {
    double norm_x = 0.0;
    for (int i = k; i < M; i++)
        norm_x += R[i][k] * R[i][k];
    norm_x = sqrt(norm_x);

    for (int i = 0; i < M; i++) v[i] = 0.0;

    double alpha = (R[k][k] > 0) ? -norm_x : norm_x;
    v[k] = R[k][k] - alpha;
    for (int i = k + 1; i < M; i++)
        v[i] = R[i][k];

    // norm_v derivata algebricamente: evita il secondo loop
    // ||v||^2 = (R[k][k] - alpha)^2 + sum_{i>k} R[i][k]^2
    //         = R[k][k]^2 - 2*R[k][k]*alpha + alpha^2 + (norm_x^2 - R[k][k]^2)
    //         = 2*norm_x*(norm_x - R[k][k]*sign)   <- semplificato
    //   ma il calcolo diretto è già O(M), lasciamo per chiarezza:
    double norm_v = 0.0;
    for (int i = k; i < M; i++)
        norm_v += v[i] * v[i];
    norm_v = sqrt(norm_v);

    if (norm_v > 1e-12)
        for (int i = k; i < M; i++)
            v[i] /= norm_v;
}

void apply_householder_to_vector(double* v, double* y, int k, int M) {
    double doty = 0.0;
    for (int i = k; i < M; i++)
        doty += v[i] * y[i];
    for (int i = k; i < M; i++)
        y[i] -= 2.0 * v[i] * doty;
}


// ─────────────────────────────────────────────
//  Back substitution (invariata)
// ─────────────────────────────────────────────
void back_substitution(double** R, double* y, double* x, int N) {
    for (int i = N - 1; i >= 0; i--) {
        x[i] = 0.0;
        for (int j = i + 1; j < N; j++)
            x[i] += R[i][j] * x[j];
        if (fabs(R[i][i]) > 1e-12)
            x[i] = (y[i] - x[i]) / R[i][i];
        else
            x[i] = 0.0;
    }
}


// ─────────────────────────────────────────────────────────────
//  QR con thread pool — il main è uno degli Nthreads worker
//
//  Si creano solo (Nthreads - 1) thread secondari.
//  Le barrier sono inizializzate a Nthreads (non Nthreads+1)
//  perché il main contribuisce direttamente come thread 0.
// ─────────────────────────────────────────────────────────────
void qr_factorization(double** R, double* y, int M, int N, int Nthreads) {

    int workers = Nthreads - 1; // thread secondari da creare

    // Alloca strutture del pool sull'heap (no VLA)
    pthread_t*          threads = malloc(workers * sizeof(pthread_t));
    pool_thread_data_t* t_data  = malloc(Nthreads * sizeof(pool_thread_data_t));
    pthread_barrier_t*  b_start = malloc(sizeof(pthread_barrier_t));
    pthread_barrier_t*  b_end   = malloc(sizeof(pthread_barrier_t));
    double*             v       = malloc(M * sizeof(double));

    if (!threads || !t_data || !b_start || !b_end || !v) {
        fprintf(stderr, "Errore di allocazione nel thread pool\n");
        exit(1);
    }

    // Barrier dimensionate a Nthreads: main + (Nthreads-1) worker
    pthread_barrier_init(b_start, NULL, Nthreads);
    pthread_barrier_init(b_end,   NULL, Nthreads);

    // Inizializza t_data per tutti gli Nthreads slot
    for (int t = 0; t < Nthreads; t++) {
        t_data[t].R              = R;
        t_data[t].v              = v;
        t_data[t].M              = M;
        t_data[t].N              = N;
        t_data[t].k              = 0;
        t_data[t].active_threads = Nthreads;
        t_data[t].thread_id      = t;
        t_data[t].total_threads  = Nthreads;
        t_data[t].barrier_start  = b_start;
        t_data[t].barrier_end    = b_end;
        t_data[t].stop           = 0;
    }

    // Lancia solo i (Nthreads-1) thread secondari (id 1 .. Nthreads-1)
    for (int t = 0; t < workers; t++)
        pthread_create(&threads[t], NULL, pool_worker, &t_data[t + 1]);

    // ── Il main esegue il ruolo del thread 0 ──────────────────
    pool_thread_data_t* main_d = &t_data[0];

    for (int k = 0; k < N && k < M; k++) {

        // Solo il main calcola il vettore di Householder e aggiorna i t_data
        compute_householder_vector(R, v, k, M);

        int cols   = N - k;
        int active = (cols < Nthreads) ? cols : Nthreads;

        for (int t = 0; t < Nthreads; t++) {
            t_data[t].k              = k;
            t_data[t].active_threads = active;
        }

        // barrier_start: il main + i worker partono tutti insieme
        pthread_barrier_wait(b_start);

        // Il main esegue il suo slice di colonne (thread_id = 0)
        {
            int tid = 0;
            int nt  = active;
            if (tid < nt) {
                int cols_per_thread = cols / nt;
                int start_col = k + tid * cols_per_thread;
                int end_col   = (tid == nt - 1) ? N : start_col + cols_per_thread;

                for (int j = start_col; j < end_col; j++) {
                    double dot = 0.0;
                    for (int i = k; i < M; i++)
                        dot += v[i] * R[i][j];
                    for (int i = k; i < M; i++)
                        R[i][j] -= 2.0 * v[i] * dot;
                }
            }
        }

        // barrier_end: attende che tutti (main incluso) abbiano finito
        pthread_barrier_wait(b_end);

        // Solo il main aggiorna y (operazione sequenziale)
        apply_householder_to_vector(v, y, k, M);
    }

    // Segnala la fine ai worker secondari e fai join
    for (int t = 0; t < Nthreads; t++)
        t_data[t].stop = 1;
    pthread_barrier_wait(b_start); // sblocca l'ultimo ciclo → i worker vedono stop=1

    for (int t = 0; t < workers; t++)
        pthread_join(threads[t], NULL);

    pthread_barrier_destroy(b_start);
    pthread_barrier_destroy(b_end);
    free(b_start); free(b_end);
    free(threads); free(t_data); free(v);
}


// ─────────────────────────────────────────────
//  Least squares (invariata, con fix malloc check)
// ─────────────────────────────────────────────
void least_squares(double** A, double* b, double* x, int M, int N, int Nthreads) {
    double** R = malloc(M * sizeof(double*));
    double*  y = malloc(M * sizeof(double));
    if (!R || !y) { fprintf(stderr, "Errore allocazione R/y\n"); exit(1); }

    for (int i = 0; i < M; i++) {
        R[i] = malloc(N * sizeof(double));
        if (!R[i]) { fprintf(stderr, "Errore allocazione R[%d]\n", i); exit(1); }
        for (int j = 0; j < N; j++)
            R[i][j] = A[i][j];
        y[i] = b[i];
    }

    qr_factorization(R, y, M, N, Nthreads);
    back_substitution(R, y, x, N);

    for (int i = 0; i < M; i++) free(R[i]);
    free(R); free(y);
}


// ─────────────────────────────────────────────
//  main — usa clock_gettime per il tempo reale
// ─────────────────────────────────────────────
int main(int argc, char* argv[]) {
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    if (argc < 3) {
        fprintf(stderr, "Uso: %s M N [Nthreads]\n", argv[0]);
        return 1;
    }

    int M        = atoi(argv[1]);
    int N        = atoi(argv[2]);
    int Nthreads = (argc >= 4) ? atoi(argv[3]) : 1;

    if (N >= M) { fprintf(stderr, "M deve essere maggiore di N\n"); return 1; }
    if (Nthreads < 1) { fprintf(stderr, "Nthreads deve essere >= 1\n"); return 1; }

    srand(time(NULL));

    double** A = malloc(M * sizeof(double*));
    if (!A) { fprintf(stderr, "Errore allocazione A\n"); return 1; }
    for (int i = 0; i < M; i++) {
        A[i] = malloc(N * sizeof(double));
        if (!A[i]) { fprintf(stderr, "Errore allocazione A[%d]\n", i); return 1; }
        for (int j = 0; j < N; j++)
            A[i][j] = rand() % 100 + 1;
    }

    double* b = malloc(M * sizeof(double));
    double* x = malloc(N * sizeof(double));
    if (!b || !x) { fprintf(stderr, "Errore allocazione b/x\n"); return 1; }
    for (int i = 0; i < M; i++)
        b[i] = rand() % 100 + 1;

    least_squares(A, b, x, M, N, Nthreads);
/*
    for (int i = 0; i < N; i++)
        printf("x[%d] = %f\n", i, x[i]);*/

    free(b); free(x);
    for (int i = 0; i < M; i++) free(A[i]);
    free(A);

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed = (ts_end.tv_sec  - ts_start.tv_sec)
                   + (ts_end.tv_nsec - ts_start.tv_nsec) * 1e-9;
    printf("tempo di esecuzione: %.6f s\n", elapsed);
    return 0;
}