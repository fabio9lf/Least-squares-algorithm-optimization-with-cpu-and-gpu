#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

// Struttura dati per i thread
typedef struct {
    double* restrict R;
    double* restrict y;
    double* restrict v;
    int M, N;
    int n_threads;
    int thread_id;
    pthread_barrier_t* barrier;
} thread_data_t;

void compute_householder_vector(double* restrict R, double* restrict v, int k, int M, int N) {
    double norm_x = 0.0;
    for (int i = k; i < M; i++) {
        double val = R[i * N + k];
        norm_x += val * val;
    }
    norm_x = sqrt(norm_x);
    
    for (int i = 0; i < M; i++) v[i] = 0.0;
    
    double alpha = (R[k * N + k] > 0) ? -norm_x : norm_x;
    v[k] = R[k * N + k] - alpha;
    
    for (int i = k + 1; i < M; i++) {
        v[i] = R[i * N + k];
    }
    
    double norm_v = 0.0;
    for (int i = k; i < M; i++) {
        norm_v += v[i] * v[i];
    }
    norm_v = sqrt(norm_v);
    
    if (norm_v > 1e-12) {
        for (int i = k; i < M; i++) {
            v[i] /= norm_v;
        }
    }
}

void* qr_worker(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    int M = data->M;
    int N = data->N;

    // Usiamo puntatori restrict locali per rassicurare il compilatore
    // e sbloccare la vettorizzazione SIMD
    double* restrict pR = data->R;
    double* restrict pv = data->v;
    double* restrict py = data->y;

    double* restrict local_dots = (double*)malloc(N * sizeof(double));

    for (int k = 0; k < N && k < M; k++) {
        
        if (data->thread_id == 0) {
            compute_householder_vector(pR, pv, k, M, N);
        }

        pthread_barrier_wait(data->barrier);

        int cols_to_process = N - (k + 1);
        if (cols_to_process > 0) {
            int chunk = (cols_to_process + data->n_threads - 1) / data->n_threads;
            int start_col = (k + 1) + data->thread_id * chunk;
            int end_col = start_col + chunk;
            if (end_col > N) end_col = N;

            int n_cols = end_col - start_col;
            
            if (n_cols > 0) {
                // Azzera array
                for(int j = 0; j < n_cols; j++) local_dots[j] = 0.0;

                // FASE 1: Prodotti scalari vettorizzati
                for (int i = k; i < M; i++) {
                    double vi = pv[i];
                    int row_offset = i * N + start_col;
                    
                    for (int j = 0; j < n_cols; j++) {
                        local_dots[j] += vi * pR[row_offset + j];
                    }
                }

                // FASE 2: Aggiornamento matrice con pre-calcolo
                for (int i = k; i < M; i++) {
                    // Pre-calcolo per evitare di ripeterlo nel ciclo più interno
                    double vi2 = pv[i] * 2.0; 
                    int row_offset = i * N + start_col;
                    
                    for (int j = 0; j < n_cols; j++) {
                        pR[row_offset + j] -= vi2 * local_dots[j];
                    }
                }
            }
        }

        if (data->thread_id == 0) {
            double doty = 0.0;
            for (int i = k; i < M; i++) {
                doty += pv[i] * py[i];   
            }
            for (int i = k; i < M; i++) {
                py[i] -= 2.0 * pv[i] * doty;
            }
        }

        pthread_barrier_wait(data->barrier);
    }
    
    free(local_dots);
    return NULL;
}

void back_substitution(double* restrict R, double* restrict y, double* restrict x, int N) {
    for (int i = N - 1; i >= 0; i--) {
        x[i] = 0.0;
        int row_offset = i * N;
        for (int j = i + 1; j < N; j++) {
            x[i] += R[row_offset + j] * x[j];
        }
        if (fabs(R[row_offset + i]) > 1e-12) {
            x[i] = (y[i] - x[i]) / R[row_offset + i];
        } else {
            x[i] = 0.0; 
        }
    }
}

void least_squares_parallel(double* A, double* b, double* x, int M, int N, int n_threads) {
    // Usiamo la malloc standard per massima compatibilità cross-platform
    double* R = malloc(M * N * sizeof(double));
    double* y = malloc(M * sizeof(double));
    double* v = malloc(M * sizeof(double));
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            R[i * N + j] = A[i * N + j];
        }
        y[i] = b[i];
    }

    pthread_t threads[n_threads];
    thread_data_t t_data[n_threads];
    pthread_barrier_t barrier;

    pthread_barrier_init(&barrier, NULL, n_threads);

    for (int t = 0; t < n_threads; t++) {
        t_data[t].R = R;
        t_data[t].y = y;
        t_data[t].v = v;
        t_data[t].M = M;
        t_data[t].N = N;
        t_data[t].n_threads = n_threads;
        t_data[t].thread_id = t;
        t_data[t].barrier = &barrier;
        pthread_create(&threads[t], NULL, qr_worker, &t_data[t]);
    }

    for (int t = 0; t < n_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    back_substitution(R, y, x, N);

    pthread_barrier_destroy(&barrier);
    free(R); free(y); free(v);
}

int main(int argc, char* argv[]) {
    if(argc < 3){
        printf("Utilizzo: %s M N [n_threads]\n", argv[0]);
        return 1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int n_threads = (argc == 4) ? atoi(argv[3]) : 1;

    struct timespec start, end;
    
    double *A = malloc(M * N * sizeof(double));
    double *b = malloc(M * sizeof(double));
    double *x = malloc(N * sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            A[i * N + j] = rand() % 100 + 1;
        }
        b[i] = rand() % 100 + 1;
    }
    
    // Per Windows/Mac usiamo ancora clock_gettime (su Windows MinGW funziona, su macOS è supportato dal 10.12 in poi)
    clock_gettime(CLOCK_MONOTONIC, &start);

    least_squares_parallel(A, b, x, M, N, n_threads);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Thread: %d | Tempo: %5.6f s\n", n_threads, time_taken);

    free(A); free(b); free(x);
    return 0;
}