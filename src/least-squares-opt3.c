#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

typedef struct {
    float *R, *v, *y;
    int M, N, Nthreads, tid;
    pthread_barrier_t *barrier;
} thread_data_t;

// Calcola il vettore di Householder (Eseguito solo dal Thread 0)
void compute_householder_inplace(float* R, float* v, int k, int M) {
    float norm_x = 0.0;
    int off = k * M;
    for (int i = k; i < M; i++) {
        norm_x += R[off + i] * R[off + i];
    }
    norm_x = sqrt(norm_x);

    float alpha = (R[off + k] > 0) ? -norm_x : norm_x;
    v[k] = R[off + k] - alpha;
    for (int i = k + 1; i < M; i++) {
        v[i] = R[off + i];
    }

    float norm_v = 0.0;
    for (int i = k; i < M; i++) {
        norm_v += v[i] * v[i];
    }
    norm_v = sqrt(norm_v);

    if (norm_v > 1e-12) {
        for (int i = k; i < M; i++) v[i] /= norm_v;
    }
}

void* qr_worker(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    int M = data->M;
    int N = data->N;

    for (int k = 0; k < N && k < M; k++) {
        // 1. Sincronizzazione: Il Thread 0 calcola il vettore v, gli altri aspettano
        if (data->tid == 0) {
            compute_householder_inplace(data->R, data->v, k, M);
        }
        pthread_barrier_wait(data->barrier);

        // 2. Aggiornamento di y (Termine noto) - Lo facciamo fare solo al thread 0
        // (Per matrici molto sottili M >> N, si potrebbe parallelizzare anche questo)
        if (data->tid == 0) {
            float doty = 0.0;
            for (int i = k; i < M; i++) doty += data->v[i] * data->y[i];
            for (int i = k; i < M; i++) data->y[i] -= 2.0 * data->v[i] * doty;
        }

        // 3. Parallelizzazione dell'aggiornamento delle colonne di R
        int start_col = k + 1;
        int cols_remaining = N - start_col;

        if (cols_remaining > 0) {
            int chunk = (cols_remaining + data->Nthreads - 1) / data->Nthreads;
            int my_start = start_col + data->tid * chunk;
            int my_end = my_start + chunk;
            if (my_end > N) my_end = N;

            for (int j = my_start; j < my_end; j++) {
                float dot = 0.0;
                int off = j * M;
                
                // --- LOOP UNROLLING FATTORIZZATO ---
                int i = k;
                for (; i <= M - 4; i += 4) {
                    dot += data->v[i] * data->R[off+i] + 
                           data->v[i+1] * data->R[off+i+1] +
                           data->v[i+2] * data->R[off+i+2] + 
                           data->v[i+3] * data->R[off+i+3];
                }
                for (; i < M; i++) dot += data->v[i] * data->R[off+i];

                float s = 2.0 * dot;
                i = k;
                for (; i <= M - 4; i += 4) {
                    data->R[off+i]   -= data->v[i]   * s;
                    data->R[off+i+1] -= data->v[i+1] * s;
                    data->R[off+i+2] -= data->v[i+2] * s;
                    data->R[off+i+3] -= data->v[i+3] * s;
                }
                for (; i < M; i++) data->R[off+i] -= data->v[i] * s;
            }
        }
        // Sincronizzazione: Tutti devono finire le loro colonne prima del prossimo passo k
        pthread_barrier_wait(data->barrier);
    }
    return NULL;
}

void back_substitution(float* R, float* y, float* x, int N, int M) {
    for (int i = N - 1; i >= 0; i--) {
        float sum = 0.0;
        for (int j = i + 1; j < N; j++) {
            sum += R[j * M + i] * x[j];
        }
        if (fabs(R[i * M + i]) > 1e-12)
            x[i] = (y[i] - sum) / R[i * M + i];
        else
            x[i] = 0.0;
    }
}

void least_squares(float* A, float* b, float* x, int M, int N, int Nthreads) {
    float* R = malloc(M * N * sizeof(float));
    float* y = malloc(M * sizeof(float));
    float* v = calloc(M, sizeof(float));

    // Trasposizione Row-Major (A) -> Column-Major (R)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            R[j * M + i] = A[i * N + j];
        }
        y[i] = b[i];
    }

    pthread_t threads[Nthreads];
    thread_data_t t_data[Nthreads];
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, Nthreads);

    for (int t = 0; t < Nthreads; t++) {
        t_data[t] = (thread_data_t){R, v, y, M, N, Nthreads, t, &barrier};
        pthread_create(&threads[t], NULL, qr_worker, &t_data[t]);
    }

    for (int t = 0; t < Nthreads; t++) pthread_join(threads[t], NULL);

    back_substitution(R, y, x, N, M);

    pthread_barrier_destroy(&barrier);
    free(R); free(y); free(v);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Uso: %s M N [Nthreads]\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int Nthreads = (argc == 4) ? atoi(argv[3]) : 1;

    float *A = malloc(M * N * sizeof(float));
    float *b = malloc(M * sizeof(float));
    float *x = malloc(N * sizeof(float));

    for (int i = 0; i < M * N; i++) A[i] = (rand() % 100) + 1;
    for (int i = 0; i < M; i++) b[i] = (rand() % 100) + 1;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    least_squares(A, b, x, M, N, Nthreads);

    gettimeofday(&end, NULL);
    float elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    printf("Ultimo x: %f\n", x[N-1]);
    printf("Tempo di esecuzione reale: %5.6f s\n", elapsed);

    free(A); free(b); free(x);
    return 0;
}