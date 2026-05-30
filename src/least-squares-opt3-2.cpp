#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <chrono>

using namespace std; // Per evitare di scrivere std::chrono ovunque

// Struttura dati per i thread
typedef struct {
    float* R;      // Matrice 1D, mappata in formato Column-Major
    float* y;
    float* v;
    int M, N;
    int n_threads;
    int thread_id;
    pthread_barrier_t* barrier;
} thread_data_t;

// Funzione sequenziale per il vettore di Householder
void compute_householder_vector(float* R, float* v, int k, int M, int N) {
    float norm_x = 0.0;
    int col_k_offset = k * M; // Offset della colonna 'k' in RAM

    for (int i = k; i < M; i++) {
        // Accesso Column-Major: Riga 'i', Colonna 'k' diventa 'k * M + i'
        float val = R[col_k_offset + i];
        norm_x += val * val;
    }
    norm_x = sqrt(norm_x);
    
    for (int i = 0; i < M; i++) v[i] = 0.0;
    
    float alpha = (R[col_k_offset + k] > 0) ? -norm_x : norm_x;
    v[k] = R[col_k_offset + k] - alpha;
    
    for (int i = k + 1; i < M; i++) {
        v[i] = R[col_k_offset + i];
    }
    
    float norm_v = 0.0;
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

    // Copia locale di v, privata per ogni thread (sta in L1/L2)
    float* v_local = (float*)malloc(M * sizeof(float));

    double accumulo_tempo_parallelo_ms = 0.0;

    for (int k = 0; k < N && k < M; k++) {
        
        if (data->thread_id == 0) {
            compute_householder_vector(data->R, data->v, k, M, N);
        }

        pthread_barrier_wait(data->barrier);

        // Ogni thread copia v in locale UNA VOLTA per step k
        // v[0..k-1] sono zero, copiamo solo la parte utile
        for (int i = k; i < M; i++) {
            v_local[i] = data->v[i];
        }

        auto start_parallel = chrono::steady_clock::now();

        int cols_to_process = N - (k + 1);
        if (cols_to_process > 0) {
            int chunk = (cols_to_process + data->n_threads - 1) / data->n_threads;
            int start_col = (k + 1) + data->thread_id * chunk;
            int end_col = start_col + chunk;
            if (end_col > N) end_col = N;

            for (int j = start_col; j < end_col; j++) {
                int col_offset = j * M;
                float dot = 0.0;
                
                // Ora v_local è in L1 — niente più miss
                for (int i = k; i < M; i++) {
                    dot += v_local[i] * data->R[col_offset + i];
                }
                for (int i = k; i < M; i++) {
                    data->R[col_offset + i] -= 2.0f * v_local[i] * dot;
                }
            }
        }

        if (data->thread_id == 0) {
            float doty = 0.0;
            for (int i = k; i < M; i++) {
                doty += data->v[i] * data->y[i];   
            }
            for (int i = k; i < M; i++) {
                data->y[i] -= 2.0f * data->v[i] * doty;
            }
        }

        pthread_barrier_wait(data->barrier);

        auto end_parallel = chrono::steady_clock::now();
        if (data->thread_id == 0) {
            chrono::duration<double, std::milli> duration = end_parallel - start_parallel;
            accumulo_tempo_parallelo_ms += duration.count();
        }
    }
    
    if (data->thread_id == 0) {
        printf("Tempo SOLO Parallelo: %.2f ms\n", accumulo_tempo_parallelo_ms);
    }

    free(v_local);
    return NULL;
}

// Back Substitution adattata per Array 1D Column-Major
void back_substitution(float* R, float* y, float* x, int M, int N) {
    for (int i = N - 1; i >= 0; i--) {
        x[i] = 0.0;
        for (int j = i + 1; j < N; j++) {
            // Accesso elemento R[i][j] => R[j * M + i]
            x[i] += R[j * M + i] * x[j];
        }
        // Accesso diagonale R[i][i] => R[i * M + i]
        if (fabs(R[i * M + i]) > 1e-12) {
            x[i] = (y[i] - x[i]) / R[i * M + i];
        } else {
            x[i] = 0.0; 
        }
    }
}

void least_squares_parallel(float* A, float* b, float* x, int M, int N, int n_threads) {
    // Allocazione contigua con cast espliciti (float*)
    float* R = (float*)malloc(M * N * sizeof(float));
    float* y = (float*)malloc(M * sizeof(float));
    float* v = (float*)malloc(M * sizeof(float));
    
    // TRASPOSIZIONE INIZIALE:
    // Legge 'A' per riga come gliela passa il main (A[i * N + j])
    // Ma SALVA in 'R' per colonna (R[j * M + i])
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            R[j * M + i] = A[i * N + j];
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

    // Passiamo anche M alla back_substitution perché serve per il calcolo dell'offset Column-Major
    back_substitution(R, y, x, M, N);

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
    
    // Inizializzazione della matrice A come Array 1D standard (Row-Major)
    float *A = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            A[i * N + j] = rand() % 100 + 1;
        }
    }
    
    float *b = (float*)malloc(M * sizeof(float));
    for(int i = 0; i < M; i++) b[i] = rand() % 100 + 1;
    
    float *x = (float*)malloc(N * sizeof(float));

    // --- INIZIO MISURAZIONE DEL TEMPO COME NELLE SLIDE ---
    chrono::steady_clock::time_point start = chrono::steady_clock::now();

    least_squares_parallel(A, b, x, M, N, n_threads);

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    chrono::milliseconds duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    // --- FINE MISURAZIONE DEL TEMPO ---

    // Stampa il risultato in millisecondi interi
    printf("Thread: %d | Tempo TOTALE (Seq + Par): %lld ms\n", n_threads, (long long)duration.count());

    free(A); free(b); free(x);
    return 0;
}