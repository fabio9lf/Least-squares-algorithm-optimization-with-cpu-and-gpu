#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

// ─────────────────────────────────────────────────────────────
//  Layout column-major: elemento (i,j) → data[i + j*M]
//  I loop interni di Householder iterano su i a j fisso
//  → accesso sequenziale in memoria → cache miss minimi.
// ─────────────────────────────────────────────────────────────
#define MAT(data, i, j, M)  (data)[(i) + (j)*(M)]



// ─────────────────────────────────────────────
//  Struttura condivisa dal thread pool
// ─────────────────────────────────────────────
typedef struct {
    double* R;           
    double* v;
    int     M, N;

    volatile int k;
    volatile int active_threads;

    int thread_id;
    int total_threads;

    pthread_barrier_t* barrier_start;
    pthread_barrier_t* barrier_end;

    volatile int stop;
} pool_thread_data_t;


struct timespec ts_start, ts_end;


// ─────────────────────────────────────────────
//  Worker del thread pool
// ─────────────────────────────────────────────
void* pool_worker(void* arg) {
    pool_thread_data_t* d = (pool_thread_data_t*)arg;

    while (1) {
        pthread_barrier_wait(d->barrier_start);
        if (d->stop) break;

        int     k   = d->k;
        int     M   = d->M;
        int     N   = d->N;
        int     tid = d->thread_id;
        int     nt  = d->active_threads;
        double* R   = d->R;
        double* v   = d->v;

        if (tid < nt) {
            int cols          = N - k;
            int cols_per_thread = cols / nt;
            int start_col     = k + tid * cols_per_thread;
            int end_col       = (tid == nt - 1) ? N : start_col + cols_per_thread;

            for (int j = start_col; j < end_col; j++) {
                // Puntatore diretto alla colonna j da riga k: accesso sequenziale
                double* col = &MAT(R, k, j, M);   // = R + k + j*M
                double dot  = 0.0;
                for (int i = 0; i < M - k; i++)
                    dot += v[k + i] * col[i];
                for (int i = 0; i < M - k; i++)
                    col[i] -= 2.0 * v[k + i] * dot;
            }
        }

        pthread_barrier_wait(d->barrier_end);
    }
    return NULL;
}


// ─────────────────────────────────────────────
//  Householder vector — column-major
// ─────────────────────────────────────────────
void compute_householder_vector(double* R, double* v, int k, int M, int N) {
    // Colonna k da riga k: puntatore diretto, accesso sequenziale
    double* col_k = &MAT(R, k, k, M);   // R + k + k*M

    double norm_x = 0.0;
    for (int i = 0; i < M - k; i++)
        norm_x += col_k[i] * col_k[i];
    norm_x = sqrt(norm_x);

    for (int i = 0; i < M; i++) v[i] = 0.0;

    double alpha = (col_k[0] > 0) ? -norm_x : norm_x;   // col_k[0] = R[k][k]
    v[k] = col_k[0] - alpha;
    for (int i = 1; i < M - k; i++)
        v[k + i] = col_k[i];

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
//  Back substitution — column-major
// ─────────────────────────────────────────────
void back_substitution(double* R, double* y, double* x, int N, int M) {
    for (int i = N - 1; i >= 0; i--) {
        double s = 0.0;
        for (int j = i + 1; j < N; j++)
            s += MAT(R, i, j, M) * x[j];
        double diag = MAT(R, i, i, M);
        x[i] = (fabs(diag) > 1e-12) ? (y[i] - s) / diag : 0.0;
    }
}


// ─────────────────────────────────────────────────────────────
//  QR con thread pool — il main è uno degli Nthreads worker
// ─────────────────────────────────────────────────────────────
void qr_factorization(double* R, double* y, int M, int N, int Nthreads) {

    int workers = Nthreads - 1;

    pthread_t*          threads = malloc(workers  * sizeof(pthread_t));
    pool_thread_data_t* t_data  = malloc(Nthreads * sizeof(pool_thread_data_t));
    pthread_barrier_t*  b_start = malloc(sizeof(pthread_barrier_t));
    pthread_barrier_t*  b_end   = malloc(sizeof(pthread_barrier_t));
    double*             v       = malloc(M * sizeof(double));

    if (!threads || !t_data || !b_start || !b_end || !v) {
        fprintf(stderr, "Errore di allocazione nel thread pool\n");
        exit(1);
    }

    pthread_barrier_init(b_start, NULL, Nthreads);
    pthread_barrier_init(b_end,   NULL, Nthreads);

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

    for (int t = 0; t < workers; t++)
        pthread_create(&threads[t], NULL, pool_worker, &t_data[t + 1]);


    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    // ── Main lavora come thread 0 ─────────────────────────────
    for (int k = 0; k < N && k < M; k++) {

        compute_householder_vector(R, v, k, M, N);

        int cols   = N - k;
        int active = (cols < Nthreads) ? cols : Nthreads;

        for (int t = 0; t < Nthreads; t++) {
            t_data[t].k              = k;
            t_data[t].active_threads = active;
        }

        pthread_barrier_wait(b_start);

        // Slice del main (thread_id = 0)
        {
            int nt  = active;
            int cols_per_thread = cols / nt;
            int start_col = k;                                         // tid=0
            int end_col   = (nt == 1) ? N : start_col + cols_per_thread;

            for (int j = start_col; j < end_col; j++) {
                double* col = &MAT(R, k, j, M);
                double dot  = 0.0;
                for (int i = 0; i < M - k; i++)
                    dot += v[k + i] * col[i];
                for (int i = 0; i < M - k; i++)
                    col[i] -= 2.0 * v[k + i] * dot;
            }
        }
        
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        pthread_barrier_wait(b_end);

        apply_householder_to_vector(v, y, k, M);
    }

    // Shutdown
    for (int t = 0; t < Nthreads; t++) t_data[t].stop = 1;
    pthread_barrier_wait(b_start);
    for (int t = 0; t < workers; t++) pthread_join(threads[t], NULL);

    pthread_barrier_destroy(b_start);
    pthread_barrier_destroy(b_end);
    free(b_start); free(b_end);
    free(threads); free(t_data); free(v);
}


// ─────────────────────────────────────────────
//  Least squares
// ─────────────────────────────────────────────
void least_squares(double* A, double* b, double* x, int M, int N, int Nthreads) {
    double* R = malloc(M * N * sizeof(double));
    double* y = malloc(M     * sizeof(double));
    if (!R || !y) { fprintf(stderr, "Errore allocazione R/y\n"); exit(1); }

    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++)
            MAT(R, i, j, M) = MAT(A, i, j, M);

    for (int i = 0; i < M; i++) y[i] = b[i];

    qr_factorization(R, y, M, N, Nthreads);
    back_substitution(R, y, x, N, M);

    free(R); free(y);
}


// ─────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────
int main(int argc, char* argv[]) {
    
    

    if (argc < 3) {
        fprintf(stderr, "Uso: %s M N [Nthreads]\n", argv[0]);
        return 1;
    }

    int M        = atoi(argv[1]);
    int N        = atoi(argv[2]);
    int Nthreads = (argc >= 4) ? atoi(argv[3]) : 1;

    if (N >= M)      { fprintf(stderr, "M deve essere maggiore di N\n"); return 1; }
    if (Nthreads < 1){ fprintf(stderr, "Nthreads deve essere >= 1\n");   return 1; }

    srand(time(NULL));

    double* A = malloc(M * N * sizeof(double));
    double* b = malloc(M     * sizeof(double));
    double* x = malloc(N     * sizeof(double));
    if (!A || !b || !x) { fprintf(stderr, "Errore allocazione\n"); return 1; }

    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++)
            MAT(A, i, j, M) = rand() % 100 + 1;

    for (int i = 0; i < M; i++)
        b[i] = rand() % 100 + 1;

    least_squares(A, b, x, M, N, Nthreads);
/*
    for (int i = 0; i < N; i++)
        printf("x[%d] = %f\n", i, x[i]);*/

    free(A); free(b); free(x);

    double elapsed = (ts_end.tv_sec  - ts_start.tv_sec)
                   + (ts_end.tv_nsec - ts_start.tv_nsec) * 1e-9;
    printf("tempo di esecuzione: %.6f s\n", elapsed);
    return 0;
}