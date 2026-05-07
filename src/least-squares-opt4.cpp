#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <chrono>

using namespace std;

// Struttura dati per i thread
typedef struct {
    float* R;      // Matrice 1D, Column-Major. Ora ha dimensione M x (N+1)!
    float* v;
    int M, N;
    int n_threads;
    int thread_id;
    pthread_barrier_t* barrier;
} thread_data_t;

// Funzione sequenziale per il vettore di Householder
void compute_householder_vector(float* R, float* v, int k, int M, int N) {
    float norm_x = 0.0f; // float esplicito
    int col_k_offset = k * M; 

    for (int i = k; i < M; i++) {
        float val = R[col_k_offset + i];
        norm_x += val * val;
    }
    norm_x = sqrtf(norm_x); // sqrtf è la versione super-veloce per i float
    
    for (int i = 0; i < M; i++) v[i] = 0.0f;
    
    float alpha = (R[col_k_offset + k] > 0.0f) ? -norm_x : norm_x;
    v[k] = R[col_k_offset + k] - alpha;
    
    for (int i = k + 1; i < M; i++) {
        v[i] = R[col_k_offset + i];
    }
    
    float norm_v = 0.0f;
    for (int i = k; i < M; i++) {
        norm_v += v[i] * v[i];
    }
    norm_v = sqrtf(norm_v);
    
    // Altra ottimizzazione: pre-calcolo la divisione
    if (norm_v > 1e-12f) {
        float inv_norm = 1.0f / norm_v; // Le moltiplicazioni sono più veloci delle divisioni!
        for (int i = k; i < M; i++) {
            v[i] *= inv_norm;
        }
    }
}

// Funzione parallela dei thread
void* qr_worker(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    int M = data->M;
    int N = data->N;
    
    // Il trucco: la matrice ha una colonna in più (il vettore 'y' è integrato)
    int total_cols = N + 1; 

    // Variabile per accumulare tutto il tempo passato a calcolare in parallelo
    double accumulo_tempo_parallelo_ms = 0.0;

    for (int k = 0; k < N && k < M; k++) {
        
        // ==========================================
        // 1. FASE SEQUENZIALE (Il timer è SPENTO qui)
        // ==========================================
        if (data->thread_id == 0) {
            compute_householder_vector(data->R, data->v, k, M, N);
        }

        // BARRIERA: Tutti i thread aspettano il thread 0
        pthread_barrier_wait(data->barrier);

        // ==========================================
        // --- INIZIO CRONOMETRO PARTE PARALLELA ---
        // ==========================================
        auto start_parallel = chrono::steady_clock::now();

        // 2. FASE PARALLELA
        // Ci dividiamo TUTTE le colonne rimanenti, inclusa la ex-colonna 'y'
        int cols_to_process = total_cols - (k + 1);
        if (cols_to_process > 0) {
            int chunk = (cols_to_process + data->n_threads - 1) / data->n_threads;
            int start_col = (k + 1) + data->thread_id * chunk;
            int end_col = start_col + chunk;
            if (end_col > total_cols) end_col = total_cols;

            for (int j = start_col; j < end_col; j++) {
                int col_offset = j * M; 
                float dot = 0.0f;
                
                for (int i = k; i < M; i++) {
                    dot += data->v[i] * data->R[col_offset + i];
                }

                // Ottimizzazione matematica: precalcolo la costante!
                float dot_x_2 = dot * 2.0f; 
                
                for (int i = k; i < M; i++) {
                    // Solo pura matematica a 32-bit, vettorizzazione perfetta
                    data->R[col_offset + i] -= data->v[i] * dot_x_2;
                }
            }
        }

        // BARRIERA: Tutti devono finire lo step parallelo
        pthread_barrier_wait(data->barrier);

        // ==========================================
        // --- FINE CRONOMETRO PARTE PARALLELA ---
        // ==========================================
        auto end_parallel = chrono::steady_clock::now();
        
        // Calcoliamo quanto è durata questa specifica fase parallela e la sommiamo
        if (data->thread_id == 0) {
            chrono::duration<double, std::milli> duration = end_parallel - start_parallel;
            accumulo_tempo_parallelo_ms += duration.count();
        }
    }
    
    // Finito tutto il ciclo, il thread 0 stampa il tempo totale parallelo accumulato
    if (data->thread_id == 0) {
        printf("Tempo SOLO Parallelo: %.2f ms\n", accumulo_tempo_parallelo_ms);
    }

    return NULL;
}

void back_substitution(float* R, float* x, int M, int N) {
    int y_col_offset = N * M; // 'y' è parcheggiato nell'ultima colonna

    for (int i = N - 1; i >= 0; i--) {
        x[i] = 0.0f;
        for (int j = i + 1; j < N; j++) {
            x[i] += R[j * M + i] * x[j];
        }
        if (fabsf(R[i * M + i]) > 1e-12f) { // fabsf per float
            x[i] = (R[y_col_offset + i] - x[i]) / R[i * M + i];
        } else {
            x[i] = 0.0f; 
        }
    }
}

void least_squares_parallel(float* A, float* b, float* x, int M, int N, int n_threads) {
    // Alloco (N + 1) colonne per includere 'y' direttamente in 'R'
    float* R = (float*)malloc(M * (N + 1) * sizeof(float));
    float* v = (float*)malloc(M * sizeof(float));
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            R[j * M + i] = A[i * N + j];
        }
        // Copio i termini noti (b) direttamente nell'ultima colonna di R
        R[N * M + i] = b[i];
    }

    pthread_t threads[n_threads];
    thread_data_t t_data[n_threads];
    pthread_barrier_t barrier;

    pthread_barrier_init(&barrier, NULL, n_threads);

    for (int t = 0; t < n_threads; t++) {
        t_data[t].R = R;
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

    back_substitution(R, x, M, N);

    pthread_barrier_destroy(&barrier);
    free(R); free(v);
}

int main(int argc, char* argv[]) {
    if(argc < 3){
        printf("Utilizzo: %s M N [n_threads]\n", argv[0]);
        return 1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int n_threads = (argc == 4) ? atoi(argv[3]) : 1;
    
    float *A = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            A[i * N + j] = (float)(rand() % 100 + 1);
        }
    }
    
    float *b = (float*)malloc(M * sizeof(float));
    for(int i = 0; i < M; i++) b[i] = (float)(rand() % 100 + 1);
    
    float *x = (float*)malloc(N * sizeof(float));

    chrono::steady_clock::time_point start = chrono::steady_clock::now();

    least_squares_parallel(A, b, x, M, N, n_threads);

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    chrono::milliseconds duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    printf("Thread: %d | Tempo TOTALE (Seq + Par): %lld ms\n", n_threads, (long long)duration.count());

    free(A); free(b); free(x);
    return 0;
}