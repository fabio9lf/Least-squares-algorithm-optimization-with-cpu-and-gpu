#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

// Struttura dati per i thread
typedef struct {
    double* R;      // OTTIMIZZAZIONE 2: Ora R è un puntatore singolo (Array 1D)
    double* y;
    double* v;
    int M, N;
    int n_threads;
    int thread_id;
    pthread_barrier_t* barrier;
} thread_data_t;

// Funzione sequenziale per il vettore di Householder
void compute_householder_vector(double* R, double* v, int k, int M, int N) {
    double norm_x = 0.0;
    for (int i = k; i < M; i++) {
        // Accesso 1D: Riga 'i', Colonna 'k' diventa 'i * N + k'
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

// Funzione parallela dei thread
void* qr_worker(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    int M = data->M;
    int N = data->N;

    // Alloca un array locale per immagazzinare i dot-product di tutte le colonne di questo thread.
    // Viene allocato una sola volta per thread, fuori dal ciclo.
    double* local_dots = (double*)malloc(N * sizeof(double));

    for (int k = 0; k < N && k < M; k++) {
        
        // 1. Solo il thread 0 calcola il vettore di Householder
        if (data->thread_id == 0) {
            compute_householder_vector(data->R, data->v, k, M, N);
        }

        // BARRIERA: Tutti i thread aspettano il thread 0
        pthread_barrier_wait(data->barrier);

        // 2. Divisione del lavoro
        int cols_to_process = N - (k + 1);
        if (cols_to_process > 0) {
            int chunk = (cols_to_process + data->n_threads - 1) / data->n_threads;
            int start_col = (k + 1) + data->thread_id * chunk;
            int end_col = start_col + chunk;
            if (end_col > N) end_col = N;

            int n_cols = end_col - start_col;
            
            if (n_cols > 0) {
                // Azzera l'array dei dot-product per questo step
                for(int j = 0; j < n_cols; j++) local_dots[j] = 0.0;

                // OTTIMIZZAZIONE 1: Calcolo Dot-Product scorrendo PER RIGHE (Cache Locality)
                for (int i = k; i < M; i++) {
                    double vi = data->v[i];
                    int row_offset = i * N; // Pre-calcola l'inizio della riga
                    
                    // Il ciclo interno scorre 'j'. La CPU legge la memoria in modo perfettamente sequenziale!
                    for (int j = 0; j < n_cols; j++) {
                        local_dots[j] += vi * data->R[row_offset + start_col + j];
                    }
                }

                // OTTIMIZZAZIONE 1: Aggiornamento Matrice scorrendo PER RIGHE
                for (int i = k; i < M; i++) {
                    double vi = data->v[i];
                    int row_offset = i * N;
                    for (int j = 0; j < n_cols; j++) {
                        data->R[row_offset + start_col + j] -= 2.0 * vi * local_dots[j];
                    }
                }
            }
        }

        // 3. Il thread 0 aggiorna il vettore 'y'
        if (data->thread_id == 0) {
            double doty = 0.0;
            for (int i = k; i < M; i++) {
                doty += data->v[i] * data->y[i];   
            }
            for (int i = k; i < M; i++) {
                data->y[i] -= 2.0 * data->v[i] * doty;
            }
        }

        // BARRIERA: Tutti devono finire lo step k
        pthread_barrier_wait(data->barrier);
    }
    
    free(local_dots); // Pulizia memoria thread-local
    return NULL;
}

// Back Substitution adattata per Array 1D
void back_substitution(double* R, double* y, double* x, int N) {
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
    // Allocazione contigua (Array 1D) al posto della 2D
    double* R = malloc(M * N * sizeof(double));
    double* y = malloc(M * sizeof(double));
    double* v = malloc(M * sizeof(double));
    
    // Copia i dati
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
    
    // Inizializzazione della matrice A come Array 1D contiguo
    double *A = malloc(M * N * sizeof(double));
    for (int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            A[i * N + j] = rand() % 100 + 1;
        }
    }
    
    double *b = malloc(M * sizeof(double));
    for(int i = 0; i < M; i++) b[i] = rand() % 100 + 1;
    
    double *x = malloc(N * sizeof(double));

    clock_gettime(CLOCK_MONOTONIC, &start);

    least_squares_parallel(A, b, x, M, N, n_threads);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Thread: %d | Tempo: %5.6f s\n", n_threads, time_taken);

    free(A); free(b); free(x);
    return 0;
}