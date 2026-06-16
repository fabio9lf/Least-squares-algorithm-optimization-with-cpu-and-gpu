#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <chrono> // Sostituisce time.h per allinearsi alla Slide 10

using namespace std;

// Struttura dati per i thread
typedef struct {
    double** R;
    double* y;
    double* v;
    int M, N;
    int n_threads;
    int thread_id;
    pthread_barrier_t* barrier;
} thread_data_t;

// Funzione sequenziale (eseguita solo dal thread 0)
void compute_householder_vector(double** R, double* v, int k, int M) {
    double norm_x = 0.0;
    for (int i = k; i < M; i++) {
        norm_x += R[i][k] * R[i][k];
    }
    norm_x = sqrt(norm_x);
    
    for (int i = 0; i < M; i++) v[i] = 0.0;
    
    double alpha = (R[k][k] > 0) ? -norm_x : norm_x;
    v[k] = R[k][k] - alpha;
    
    for (int i = k + 1; i < M; i++) {
        v[i] = R[i][k];
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
    
    // Variabile per accumulare il tempo passato SOLO in parallelo
    double accumulo_tempo_parallelo_ms = 0.0;

    for (int k = 0; k < data->N && k < data->M; k++) {
        
        // 1. Solo il thread 0 calcola il vettore di Householder (Fase Sequenziale)
        if (data->thread_id == 0) {
            compute_householder_vector(data->R, data->v, k, data->M);
        }

        // BARRIERA: Tutti i thread aspettano che il thread 0 abbia finito
        pthread_barrier_wait(data->barrier);

        // ==========================================
        // --- INIZIO CRONOMETRO PARTE PARALLELA ---
        // ==========================================
        auto start_parallel = chrono::steady_clock::now();

        // 2. Divisione del lavoro: ogni thread aggiorna un subset di colonne
        int cols_to_process = data->N - (k + 1);
        if (cols_to_process > 0) {
            int chunk = (cols_to_process + data->n_threads - 1) / data->n_threads;
            int start_col = (k + 1) + data->thread_id * chunk;
            int end_col = start_col + chunk;
            if (end_col > data->N) end_col = data->N;

            for (int j = start_col; j < end_col; j++) {
                double dot = 0.0;
                for (int i = k; i < data->M; i++) {
                    dot += data->v[i] * data->R[i][j];
                }
                for (int i = k; i < data->M; i++) {
                    data->R[i][j] -= 2.0 * data->v[i] * dot;
                }
            }
        }

        // 3. Il thread 0 aggiorna anche il vettore dei termini noti y
        if (data->thread_id == 0) {
            double doty = 0.0;
            for (int i = k; i < data->M; i++) {
                doty += data->v[i] * data->y[i];   
            }
            for (int i = k; i < data->M; i++) {
                data->y[i] -= 2.0 * data->v[i] * doty;
            }
        }

        // BARRIERA: Tutti devono finire lo step k prima di passare a k+1
        pthread_barrier_wait(data->barrier);

        // ==========================================
        // --- FINE CRONOMETRO PARTE PARALLELA ---
        // ==========================================
        auto end_parallel = chrono::steady_clock::now();
        
        // Accumuliamo il tempo misurato (lo fa solo il Thread 0 per evitare doppioni)
        if (data->thread_id == 0) {
            chrono::duration<double, std::milli> duration = end_parallel - start_parallel;
            accumulo_tempo_parallelo_ms += duration.count();
        }
    }
    
    // A fine lavoro, stampiamo il tempo puro parallelo
    if (data->thread_id == 0) {
        printf("Tempo SOLO Parallelo: %.2f ms\n", accumulo_tempo_parallelo_ms);
        fflush(stdout); // <-- AGGIUNTO: Forza l'output per lo script Python
    }
    
    return NULL;
}

void back_substitution(double** R, double* y, double* x, int N) {
    for (int i = N - 1; i >= 0; i--) {
        x[i] = 0.0;
        for (int j = i + 1; j < N; j++) {
            x[i] += R[i][j] * x[j];
        }
        if (fabs(R[i][i]) > 1e-12) {
            x[i] = (y[i] - x[i]) / R[i][i];
        } else {
            x[i] = 0.0; 
        }
    }
}

void least_squares_parallel(double** A, double* b, double* x, int M, int N, int n_threads) {
    // Aggiunti cast espliciti (double**) e (double*) per il C++
    double** R = (double**)malloc(M * sizeof(double*));
    double* y = (double*)malloc(M * sizeof(double));
    double* v = (double*)malloc(M * sizeof(double));
    for (int i = 0; i < M; i++) {
        R[i] = (double*)malloc(N * sizeof(double));
        for (int j = 0; j < N; j++) R[i][j] = A[i][j];
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
    for(int i = 0; i < M; i++) free(R[i]);
    free(R); free(y); free(v);
}

int main(int argc, char* argv[]) {
    if(argc < 3){
        printf("Utilizzo: %s M N [n_threads]\n", argv[0]);
        return 1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int n_threads = (argc >= 4) ? atoi(argv[3]) : 1;
    
    // Inizializzazione dati
    double **A = (double**)malloc(M * sizeof(double*));
    for (int i = 0; i < M; i++) {
        A[i] = (double*)malloc(N * sizeof(double));
        for(int j = 0; j < N; j++) A[i][j] = rand() % 100 + 1;
    }
    double *b = (double*)malloc(M * sizeof(double));
    for(int i = 0; i < M; i++) b[i] = rand() % 100 + 1;
    double *x = (double*)malloc(N * sizeof(double));

    // Utilizzo di chrono anche nel main (come da Slide 10)
    auto start = chrono::steady_clock::now();

    least_squares_parallel(A, b, x, M, N, n_threads);

    auto end = chrono::steady_clock::now();
    chrono::duration<double, std::milli> duration = end - start;

    // <-- COMMENTATO: non intasare l'output di Python durante il benchmark
    // printf("Thread: %d | Tempo TOTALE (Seq + Par): %.2f ms\n", n_threads, duration.count());

    // Pulizia
    for(int i = 0; i < M; i++) free(A[i]);
    free(A); free(b); free(x);
    return 0;
}
