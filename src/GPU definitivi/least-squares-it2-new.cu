#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define IDX(row, col, N) ((row) * (N) + (col))

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/* ============================================================
 * KERNEL 1: AtA (Memoria Globale, Zero AtomicAdd, Simmetria)
 * ============================================================ */
    __global__ void compute_AtA_global(const float* __restrict__ A, float* __restrict__ AtA, int M, int N) {
    // Ogni thread è responsabile in modo esclusivo di UN elemento della matrice AtA
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Controllo dei limiti e calcolo del solo triangolo superiore
    if (row < N && col < N && row <= col) {
        float sum = 0.0f;

        // Lettura diretta dalla memoria globale
#pragma unroll 4
        for (int k = 0; k < M; k++) {
            sum += A[IDX(k, row, N)] * A[IDX(k, col, N)];
        }

        // Singola scrittura non bloccante
        AtA[IDX(row, col, N)] = sum;

        // Copia simmetrica
        if (row != col) {
            AtA[IDX(col, row, N)] = sum;
        }
    }
}

/* ============================================================
 * KERNEL 2: Atb (Memoria Globale)
 * ============================================================ */
__global__ void compute_Atb_global(const float* __restrict__ A, const float* __restrict__ b, float* __restrict__ Atb, int M, int N) {
    // Ogni thread calcola un singolo elemento del vettore Atb
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        float sum = 0.0f;

#pragma unroll 4
        for (int k = 0; k < M; k++) {
            sum += A[IDX(k, row, N)] * b[k];
        }

        Atb[row] = sum;
    }
}

/* ============================================================
 * ELIMINAZIONE DI GAUSS (CPU, Singola Precisione)
 * ============================================================ */
void solve_system(const float* AtA, const float* Atb, float* x, int N) {
    float* G = (float*)malloc(N * N * sizeof(float));
    float* g = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N * N; i++) G[i] = AtA[i];
    for (int i = 0; i < N; i++) g[i] = Atb[i];

    for (int i = 0; i < N; i++) {
        int pivot = i;
        float max_val = fabsf(G[IDX(i, i, N)]);

        for (int k = i + 1; k < N; k++) {
            if (fabsf(G[IDX(k, i, N)]) > max_val) {
                max_val = fabsf(G[IDX(k, i, N)]);
                pivot = k;
            }
        }
        if (max_val < 1e-14f) {
            fprintf(stderr, "Errore: matrice singolare.\n");
            exit(EXIT_FAILURE);
        }

        if (pivot != i) {
            for (int j = 0; j < N; j++) {
                float tmp = G[IDX(i, j, N)];
                G[IDX(i, j, N)] = G[IDX(pivot, j, N)];
                G[IDX(pivot, j, N)] = tmp;
            }
            float tv = g[i]; g[i] = g[pivot]; g[pivot] = tv;
        }

        for (int k = i + 1; k < N; k++) {
            float factor = G[IDX(k, i, N)] / G[IDX(i, i, N)];
            for (int j = i; j < N; j++) G[IDX(k, j, N)] -= factor * G[IDX(i, j, N)];
            g[k] -= factor * g[i];
        }
    }

    for (int i = N - 1; i >= 0; i--) {
        float sum = 0.0f;
        for (int j = i + 1; j < N; j++) sum += G[IDX(i, j, N)] * x[j];
        x[i] = (g[i] - sum) / G[IDX(i, i, N)];
    }

    free(G);
    free(g);
}

int main(int argc, char* argv[]) {
    // Questo script accetta solo M e N. Configura autonomamente la griglia ottimale.
    if (argc < 3) {
        printf("Uso: %s <M> <N>\n", argv[0]);
        return 1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);

    // ==========================================
    // ALLOCAZIONE HOST E INIZIALIZZAZIONE
    // ==========================================
    srand(42);
    float* A = (float*)malloc(M * N * sizeof(float));
    float* b = (float*)malloc(M * sizeof(float));
    float* AtA = (float*)calloc(N * N, sizeof(float));
    float* Atb = (float*)calloc(N, sizeof(float));
    float* x = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < M; i++) {
        b[i] = (float)(rand() % 10 + 1);
        for (int j = 0; j < N; j++) {
            A[IDX(i, j, N)] = (float)(rand() % 10 + 1);
        }
    }

    // ==========================================
    // ALLOCAZIONE DEVICE
    // ==========================================
    float* d_A, * d_b, * d_AtA, * d_Atb;
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_AtA, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Atb, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, M * sizeof(float), cudaMemcpyHostToDevice));

    // ==========================================
    // GEOMETRIA CUDA FLESSIBILE 
    // Usiamo blocchi da 32x32 per la matrice e 256 per il vettore
    // ==========================================
    dim3 block2D(32, 32);
    dim3 grid2D((N + block2D.x - 1) / block2D.x, (N + block2D.y - 1) / block2D.y);

    dim3 block1D(256);
    dim3 grid1D((N + block1D.x - 1) / block1D.x);

    // ==========================================
    // TIMING
    // ==========================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Lancio Kernel in Memoria Globale
    compute_AtA_global << <grid2D, block2D >> > (d_A, d_AtA, M, N);
    CUDA_CHECK(cudaGetLastError());

    compute_Atb_global << <grid1D, block1D >> > (d_A, d_b, d_Atb, M, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // ==========================================
    // DOWNLOAD E CALCOLO FINALE
    // ==========================================
    CUDA_CHECK(cudaMemcpy(AtA, d_AtA, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Atb, d_Atb, N * sizeof(float), cudaMemcpyDeviceToHost));

    solve_system(AtA, Atb, x, N);

    // Output formattato per la tua Regex Python
    printf("Tempo SOLO Parallelo: %.2f ms\n", ms);

    if (N >= 3)
        printf("Primi 3 valori di x: %f, %f, %f\n", x[0], x[1], x[2]);

    free(A); free(b); free(AtA); free(Atb); free(x);
    cudaFree(d_A); cudaFree(d_b); cudaFree(d_AtA); cudaFree(d_Atb);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}