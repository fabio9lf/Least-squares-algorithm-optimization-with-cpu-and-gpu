#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Macro per mappare array 2D su array 1D
#define IDX(row, col, N) ((row) * (N) + (col))

// Macro per il controllo degli errori CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/* ============================================================
 * KERNEL 1: Calcolo della matrice simmetrica AtA
 * Riprende l'ottimizzazione 4 della tua CPU: si calcola solo
 * il triangolo superiore e si copia per simmetria nell'inferiore.
 * ============================================================ */
__global__ void compute_AtA(const double* A, double* AtA, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Controllo dei confini
    if (row < N && col < N) {
        // Calcolo solo per la metà superiore della matrice
        if (row <= col) {
            double sum = 0.0;
            // Prodotto scalare lungo la dimensione M
            for (int k = 0; k < M; k++) {
                sum += A[IDX(k, row, N)] * A[IDX(k, col, N)];
            }

            // Scrittura del risultato
            AtA[IDX(row, col, N)] = sum;

            // Copia simmetrica (Ottimizzazione della tua CPU)
            if (row != col) {
                AtA[IDX(col, row, N)] = sum;
            }
        }
    }
}

/* ============================================================
 * KERNEL 2: Calcolo del vettore Atb
 * ============================================================ */
__global__ void compute_Atb(const double* A, const double* b, double* Atb, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        double sum = 0.0;
        for (int k = 0; k < M; k++) {
            sum += A[IDX(k, row, N)] * b[k];
        }
        Atb[row] = sum;
    }
}

/* ============================================================
 * ELIMINAZIONE DI GAUSS (Eseguita su CPU)
 * Trascritta esattamente dalla tua versione, riadattata per
 * lavorare su array 1D piatti al posto di array di puntatori.
 * ============================================================ */
void solve_system(const double* AtA, const double* Atb, double* x, int N) {
    // Copia difensiva (Ottimizzazione 5 della tua CPU)
    double* G = (double*)malloc(N * N * sizeof(double));
    double* g = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N * N; i++) G[i] = AtA[i];
    for (int i = 0; i < N; i++) g[i] = Atb[i];

    // Eliminazione con pivot parziale (Ottimizzazione 3 della tua CPU)
    for (int i = 0; i < N; i++) {
        int pivot = i;
        double max_val = fabs(G[IDX(i, i, N)]);

        for (int k = i + 1; k < N; k++) {
            if (fabs(G[IDX(k, i, N)]) > max_val) {
                max_val = fabs(G[IDX(k, i, N)]);
                pivot = k;
            }
        }
        if (max_val < 1e-14) {
            fprintf(stderr, "Errore: matrice singolare.\n");
            exit(EXIT_FAILURE);
        }

        // Swap
        if (pivot != i) {
            for (int j = 0; j < N; j++) {
                double tmp = G[IDX(i, j, N)];
                G[IDX(i, j, N)] = G[IDX(pivot, j, N)];
                G[IDX(pivot, j, N)] = tmp;
            }
            double tv = g[i]; g[i] = g[pivot]; g[pivot] = tv;
        }

        // Eliminazione
        for (int k = i + 1; k < N; k++) {
            double factor = G[IDX(k, i, N)] / G[IDX(i, i, N)];
            for (int j = i; j < N; j++) G[IDX(k, j, N)] -= factor * G[IDX(i, j, N)];
            g[k] -= factor * g[i];
        }
    }

    // Back-substitution
    for (int i = N - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < N; j++) sum += G[IDX(i, j, N)] * x[j];
        x[i] = (g[i] - sum) / G[IDX(i, i, N)];
    }

    free(G);
    free(g);
}

int main(int argc, char* argv[]) {
    // Compatibilità con lo script Python
    // Accettiamo M e N. Se Python passa il numero di thread della CPU, 
    // lo ignoriamo e impostiamo la geometria GPU ottimale.
    if (argc < 3) {
        printf("Uso: %s <M> <N>\n", argv[0]);
        return 1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);

    // ==========================================
    // ALLOCAZIONE E INIZIALIZZAZIONE (CPU)
    // ==========================================
    srand(42);
    double* A = (double*)malloc(M * N * sizeof(double));
    double* b = (double*)malloc(M * sizeof(double));
    double* AtA = (double*)calloc(N * N, sizeof(double));
    double* Atb = (double*)calloc(N, sizeof(double));
    double* x = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < M; i++) {
        b[i] = (double)(rand() % 10 + 1);
        for (int j = 0; j < N; j++) {
            A[IDX(i, j, N)] = (double)(rand() % 10 + 1);
        }
    }

    // ==========================================
    // ALLOCAZIONE (GPU)
    // ==========================================
    double* d_A, * d_b, * d_AtA, * d_Atb;
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_AtA, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Atb, N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_A, A, M * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, M * sizeof(double), cudaMemcpyHostToDevice));

    // Configurazione Griglia CUDA (Hardcoded a blocchi classici 16x16)
    dim3 block2D(16, 16);
    dim3 grid2D((N + block2D.x - 1) / block2D.x, (N + block2D.y - 1) / block2D.y);

    dim3 block1D(256);
    dim3 grid1D((N + block1D.x - 1) / block1D.x);

    // ==========================================
    // MISURAZIONE TEMPO (Solo Parallelo)
    // ==========================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // Inizio Timer GPU

    // Lancio Kernel
    compute_AtA << <grid2D, block2D >> > (d_A, d_AtA, M, N);
    CUDA_CHECK(cudaGetLastError());

    compute_Atb << <grid1D, block1D >> > (d_A, d_b, d_Atb, M, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize()); // Aspettiamo che la GPU finisca

    cudaEventRecord(stop); // Fine Timer GPU
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // ==========================================
    // DOWNLOAD E SOLUZIONE
    // ==========================================
    CUDA_CHECK(cudaMemcpy(AtA, d_AtA, N * N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Atb, d_Atb, N * sizeof(double), cudaMemcpyDeviceToHost));

    solve_system(AtA, Atb, x, N);

    // Formattazione esatta richiesta dal tuo script Python
    printf("Tempo SOLO Parallelo: %.2f ms\n", ms);

    if (N >= 3)
        printf("Primi 3 valori di x: %f, %f, %f\n", x[0], x[1], x[2]);

    // Pulizia finale
    free(A); free(b); free(AtA); free(Atb); free(x);
    cudaFree(d_A); cudaFree(d_b); cudaFree(d_AtA); cudaFree(d_Atb);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}