#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <immintrin.h> // Header per AVX2 e FMA

// Struttura per il Thread Pool
typedef struct {
    float *R, *v;
    int M, k, N;
    int start_col, end_col;
    pthread_mutex_t mutex;
    pthread_cond_t cond_start;
    pthread_cond_t cond_finished;
    int run;       
    int finished;  
    int terminate; 
} worker_t;

// --- WORKER THREAD CON OTTIMIZZAZIONE SIMD AVX2 ---
void* thread_pool_worker(void* arg) {
    worker_t* self = (worker_t*)arg;
    while (1) {
        pthread_mutex_lock(&self->mutex);
        while (!self->run && !self->terminate) 
            pthread_cond_wait(&self->cond_start, &self->mutex);
        
        if (self->terminate) {
            pthread_mutex_unlock(&self->mutex);
            break;
        }

        int M = self->M;
        int k = self->k;
        float* v = self->v;
        float* R = self->R;

        // Aggiornamento delle colonne assegnate a questo thread
        for (int j = self->start_col; j < self->end_col; j++) {
            int col_offset = j * M;
            
            // 1. Prodotto scalare vettorizzato (AVX2)
            __m256 acc_vec = _mm256_setzero_ps();
            int i = k;
            for (; i <= M - 8; i += 8) {
                __m256 v_vec = _mm256_loadu_ps(&v[i]);
                __m256 r_vec = _mm256_loadu_ps(&R[col_offset + i]);
                acc_vec = _mm256_fmadd_ps(v_vec, r_vec, acc_vec); 
            }
            
            // Riduzione orizzontale della somma vettoriale
            float temp[8];
            _mm256_storeu_ps(temp, acc_vec);
            float dot = 0;
            for(int n=0; n<8; n++) dot += temp[n];
            for (; i < M; i++) dot += v[i] * R[col_offset + i]; 

            // 2. Aggiornamento colonna vettorizzato (R = R - 2*dot*v)
            float scalar = 2.0f * dot;
            __m256 scalar_vec = _mm256_set1_ps(scalar);
            i = k;
            for (; i <= M - 8; i += 8) {
                __m256 v_vec = _mm256_loadu_ps(&v[i]);
                __m256 r_vec = _mm256_loadu_ps(&R[col_offset + i]);
                __m256 update = _mm256_mul_ps(v_vec, scalar_vec);
                __m256 res = _mm256_sub_ps(r_vec, update);
                _mm256_storeu_ps(&R[col_offset + i], res);
            }
            for (; i < M; i++) R[col_offset + i] -= v[i] * scalar;
        }

        self->run = 0;
        self->finished = 1;
        pthread_cond_signal(&self->cond_finished);
        pthread_mutex_unlock(&self->mutex);
    }
    return NULL;
}

// --- FUNZIONI DI SUPPORTO ---
void compute_householder_vector(float* R, float* v, int k, int M) {
    float norm_x = 0.0;
    for (int i = k; i < M; i++) norm_x += R[k * M + i] * R[k * M + i];
    norm_x = sqrtf(norm_x);
    
    for (int i = 0; i < M; i++) v[i] = 0.0f;
    float alpha = (R[k * M + k] > 0) ? -norm_x : norm_x;
    v[k] = R[k * M + k] - alpha;
    for (int i = k + 1; i < M; i++) v[i] = R[k * M + i];
    
    float norm_v = 0.0;
    for (int i = k; i < M; i++) norm_v += v[i] * v[i];
    norm_v = sqrtf(norm_v);
    
    if (norm_v > 1e-10) {
        for (int i = k; i < M; i++) v[i] /= norm_v;
    }
}

void apply_householder_to_vector(float* v, float* y, int k, int M) {
    float doty = 0.0;
    for (int i = k; i < M; i++) doty += v[i] * y[i];
    for (int i = k; i < M; i++) y[i] -= 2.0f * v[i] * doty;
}

void back_substitution(float* R, float* y, float* x, int N, int M) {
    for (int i = N - 1; i >= 0; i--) {
        float sum = 0.0;
        for (int j = i + 1; j < N; j++) sum += R[j * M + i] * x[j];
        if (fabsf(R[i * M + i]) > 1e-10) 
            x[i] = (y[i] - sum) / R[i * M + i];
        else 
            x[i] = 0.0f;
    }
}

// --- QR FACTORIZATION CON THREAD POOL ---
void qr_factorization(float* R, float* y, int M, int N, int Nthreads) {
    pthread_t threads[Nthreads];
    worker_t workers[Nthreads];
    float* v = (float*)_mm_malloc(M * sizeof(float), 32); 

    for (int t = 0; t < Nthreads; t++) {
        workers[t].terminate = 0; workers[t].run = 0; workers[t].finished = 0;
        pthread_mutex_init(&workers[t].mutex, NULL);
        pthread_cond_init(&workers[t].cond_start, NULL);
        pthread_cond_init(&workers[t].cond_finished, NULL);
        pthread_create(&threads[t], NULL, thread_pool_worker, &workers[t]);
    }

    for (int k = 0; k < N && k < M; k++) {
        compute_householder_vector(R, v, k, M);

        int cols_to_process = N - (k + 1);
        if (cols_to_process > 0) {
            int cols_per_thread = (cols_to_process + Nthreads - 1) / Nthreads;
            for (int t = 0; t < Nthreads; t++) {
                pthread_mutex_lock(&workers[t].mutex);
                workers[t].R = R; workers[t].v = v; workers[t].k = k; workers[t].M = M;
                workers[t].start_col = (k + 1) + t * cols_per_thread;
                workers[t].end_col = (workers[t].start_col + cols_per_thread > N) ? N : workers[t].start_col + cols_per_thread;
                
                if (workers[t].start_col < N) {
                    workers[t].run = 1; workers[t].finished = 0;
                    pthread_cond_signal(&workers[t].cond_start);
                } else workers[t].finished = 1;
                pthread_mutex_unlock(&workers[t].mutex);
            }

            for (int t = 0; t < Nthreads; t++) {
                pthread_mutex_lock(&workers[t].mutex);
                while (!workers[t].finished) pthread_cond_wait(&workers[t].cond_finished, &workers[t].mutex);
                pthread_mutex_unlock(&workers[t].mutex);
            }
        }
        apply_householder_to_vector(v, y, k, M);
    }

    for (int t = 0; t < Nthreads; t++) {
        pthread_mutex_lock(&workers[t].mutex);
        workers[t].terminate = 1;
        pthread_cond_signal(&workers[t].cond_start);
        pthread_mutex_unlock(&workers[t].mutex);
        pthread_join(threads[t], NULL);
        pthread_mutex_destroy(&workers[t].mutex);
        pthread_cond_destroy(&workers[t].cond_start);
        pthread_cond_destroy(&workers[t].cond_finished);
    }
    _mm_free(v);
}

// --- MAIN ---
int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Utilizzo: %s M N [Nthreads]\n", argv[0]);
        return 1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int Nthreads = (argc == 4) ? atoi(argv[3]) : 4;

    srand(time(NULL));

    // Memoria allineata per AVX (fondamentale per le performance)
    float *R = (float*)_mm_malloc(M * N * sizeof(float), 32);
    float *y = (float*)_mm_malloc(M * sizeof(float), 32);
    float *x = (float*)_mm_malloc(N * sizeof(float), 32);

    // Inizializzazione dati
    for (int i = 0; i < M * N; i++) R[i] = (float)(rand() % 10 + 1);
    for (int i = 0; i < M; i++) y[i] = (float)(rand() % 10 + 1);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    qr_factorization(R, y, M, N, Nthreads);
    back_substitution(R, y, x, N, M);

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    printf("--- Risultati ---\n");
    printf("Ultimo coefficiente x[%d]: %f\n", N-1, x[N-1]);
    printf("Tempo di esecuzione: %.6f secondi\n", elapsed);
    printf("Thread utilizzati: %d\n", Nthreads);

    _mm_free(R); _mm_free(y); _mm_free(x);
    return 0;
}