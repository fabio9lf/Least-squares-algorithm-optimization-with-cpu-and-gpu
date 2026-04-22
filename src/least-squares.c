#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

  // righe (N >= M colonne nel LS classico)
void qr_factorization(double** R, double* y, int, int);
void least_squares(double** A, double* b, double* x, int M, int N) {

    double** R = malloc(M * sizeof(double*));
    double* y = malloc(M * sizeof(double));

    // copia A in R
    for (int i = 0; i < M; i++){
        R[i] = malloc(N * sizeof(double));
        for (int j = 0; j < N; j++){
            R[i][j] = A[i][j];
        }
    }

    // copia b in y
    for (int i = 0; i < M; i++)
        y[i] = b[i];

    // =========================
    // QR con Householder
    // =========================
    qr_factorization(R, y, M, N);
    // =========================
    // back substitution Rx = y
    // =========================
    for (int i = N - 1; i >= 0; i--) {
        x[i] = 0.0;

        for (int j = i + 1; j < N; j++)
            x[i] += R[i][j] * x[j];

        if (fabs(R[i][i]) > 1e-12)
            x[i] = (y[i] - x[i]) / R[i][i];
        else
            x[i] = 0.0;
    }

    for(int i = 0;i < M;i++)
        free(R[i]);
    free(R);
    free(y);
}

void qr_factorization(double** R, double* y, int M, int N){
    
    double* v = malloc(M * sizeof(double)); 
    for (int k = 0; k < N && k < M; k++) {

        // 1. estrai x = colonna k da k in giù
        double norm = 0.0;
        for (int i = k; i < M; i++)
            norm += R[i][k] * R[i][k];
        norm = sqrt(norm);

        if (norm == 0.0) continue;

        double sign = (R[k][k] >= 0) ? 1.0 : -1.0;

        // v = x + sign*|x| e1
        for (int i = k; i < M; i++)
            v[i] = R[i][k];
        v[k] += sign * norm;

        // normalizza v -> u
        double vnorm = 0.0;
        for (int i = k; i < M; i++)
            vnorm += v[i] * v[i];
        vnorm = sqrt(vnorm);

        if (vnorm == 0.0) continue;

        for (int i = k; i < M; i++)
            v[i] /= vnorm;

        // =========================
        // applica riflessione a R
        // R = R - 2 u (u^T R)
        // =========================
        for (int j = k; j < N; j++) {
            double dot = 0.0;
            for (int i = k; i < M; i++)
                dot += v[i] * R[i][j];

            for (int i = k; i < M; i++)
                R[i][j] -= 2.0 * v[i] * dot;
        }

        // applica anche a y
        double doty = 0.0;
        for (int i = k; i < M; i++)
            doty += v[i] * y[i];

        for (int i = k; i < M; i++)
            y[i] -= 2.0 * v[i] * doty;
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

    least_squares(A, b, x, M, N);

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