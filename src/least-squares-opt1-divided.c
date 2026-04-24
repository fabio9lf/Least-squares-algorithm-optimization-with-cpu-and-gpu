#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

typedef struct {
    double** R;
    double* v;
    int M, k, N;
    int start_col, end_col;
} thread_data_t;

void compute_householder_vector(double** R, double* v, int k, int M) {
    double norm_x = 0.0;
    for (int i = k; i < M; i++) {
        norm_x += R[i][k] * R[i][k];
    }
    norm_x = sqrt(norm_x);
    
    // Inizializza v
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

void* update_columns_thread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;

    for (int j = data->start_col; j < data->end_col; j++) {
        double dot = 0.0;
        for (int i = data->k; i < data->M; i++) {
            dot += data->v[i] * data->R[i][j];
        }

        for (int i = data->k; i < data->M; i++) {
            data->R[i][j] -= 2.0 * data->v[i] * dot;
        }
    }
    return NULL;
}

void apply_householder_to_vector(double* v, double* y, int k, int M) {
    double doty = 0.0;
    for (int i = k; i < M; i++) {
        doty += v[i] * y[i];   
    }
    for (int i = k; i < M; i++) {
        y[i] -= 2.0 * v[i] * doty;
    }
}

// =========================
// back substitution Rx = y
// =========================
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

void qr_factorization(double** R, double* y, int, int, int);

void least_squares(double** A, double* b, double* x, int M, int N, int Nthreads) {

    double** R = malloc(M * sizeof(double*));
    double* y = malloc(M * sizeof(double));

    for (int i = 0; i < M; i++){
        R[i] = malloc(N * sizeof(double));
        for (int j = 0; j < N; j++){
            R[i][j] = A[i][j];
        }
    }

    for (int i = 0; i < M; i++)
        y[i] = b[i];

    // =========================
    // QR con Householder
    // =========================
    qr_factorization(R, y, M, N, Nthreads);
    
    back_substitution(R, y, x, N);

    for(int i = 0;i < M;i++)
        free(R[i]);
    free(R);
    free(y);
}

void qr_factorization(double** R, double* y, int M, int N, int Nthreads){
    pthread_t threads[Nthreads];
    thread_data_t t_data[Nthreads];
    double* v = malloc(M * sizeof(double));

    for(int k = 0; k < N && k < M;k++){
        
        compute_householder_vector(R, v, k, M);

        int cols_to_process = N - k;
        int cols_per_thread = cols_to_process / Nthreads;

        for (int t = 0; t < Nthreads; t++) {
            t_data[t].R = R;
            t_data[t].v = v;
            t_data[t].k = k;
            t_data[t].M = M;
            t_data[t].start_col = k + t * cols_per_thread;
            t_data[t].end_col = (t == Nthreads - 1) ? N : k + (t + 1) * cols_per_thread;
            
            pthread_create(&threads[t], NULL, update_columns_thread, &t_data[t]);
        }

        for (int t = 0; t < Nthreads; t++) {
            pthread_join(threads[t], NULL);
        }
        
        apply_householder_to_vector(v, y, k, M);
    }
    free(v);
}

int main(int argc, char* argv[]) {
    clock_t start = clock();

    int M = 0, N = 0, Nthreads = 1;

    if(argc == 1 || argc == 2){
        printf("Devi passare la dimensione della matrice!");
        exit(1);
    }
    if(argc == 4){
        Nthreads = atoi(argv[3]);
    }
    M = atoi(argv[1]);
    N = atoi(argv[2]);

    if(N >= M){
        printf("M deve essere maggiore di N!");
        exit(1);
    }


    srand(time(NULL));


    double **A = malloc(M * sizeof(double*));

    for (int i = 0; i < M; i++){
        A[i] = malloc(N * sizeof(double));
        for(int j = 0; j < N; j++){
            A[i][j] = rand() % 100 + 1;
        }
    }
    double *b = malloc(M * sizeof(double));
    for(int i = 0; i < M;i++){
        b[i] = rand() % 100 + 1;
    }
    double *x = malloc(N * sizeof(double));

    least_squares(A, b, x, M, N, Nthreads);
        
    for (int i = 0; i < N; i++)
        printf("%f\n", x[i]);

    free(b);
    free(x);
    for(int i = 0; i < M; i++){
        free(A[i]);
    }
    free(A);

    clock_t end = clock();

    double time = (double)((end - start)/CLOCKS_PER_SEC);
    printf("tempo di esecuzione: %5.6f", time);
    return 0;
}