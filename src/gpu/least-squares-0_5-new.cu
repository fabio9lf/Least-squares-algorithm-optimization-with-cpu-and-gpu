/*
====================================================================
CUDA Least Squares - HYBRID DYNAMIC SCALING
--------------------------------------------------------------------
Questa versione unisce le prestazioni della memoria coalescente (float)
con la scalabilità dinamica del Grid-Stride Loop.
Permette di testare la saturazione dei CUDA Core variando i blocchi!
====================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define IDX(row, col, N) ((row) * (N) + (col))

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/* ============================================================
 * KERNEL 1: AtA - Grid-Stride Ibrido sulle Righe e Colonne
 * ============================================================ */
__global__ void compute_AtA_hybrid(const float* __restrict__ A, float* __restrict__ AtA, int M, int N) {

    // I Blocchi si dividono le Righe
    for (int row = blockIdx.x; row < N; row += gridDim.x) {

        // I Thread si dividono le Colonne (Garantisce Coalescing!)
        for (int col = threadIdx.x; col < N; col += blockDim.x) {

            if (row <= col) {
                float sum = 0.0f;
                // Prodotto scalare
#pragma unroll 4
                for (int k = 0; k < M; k++) {
                    sum += A[IDX(k, row, N)] * A[IDX(k, col, N)];
                }

                AtA[IDX(row, col, N)] = sum;

                // Simmetria
                if (row != col) {
                    AtA[IDX(col, row, N)] = sum;
                }
            }
        }
    }
}

/* ============================================================
 * KERNEL 2: Atb - Grid-Stride 1D Lineare
 * ============================================================ */
__global__ void compute_Atb_hybrid(const float* __restrict__ A, const float* __restrict__ b, float* __restrict__ Atb, int M, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int row = tid; row < N; row += stride) {
        float sum = 0.0f;
#pragma unroll 4
        for (int k = 0; k < M; k++) {
            sum += A[IDX(k, row, N)] * b[k];
        }
        Atb[row] = sum;
    }
}

// ================================================================
// SOLVER CPU (In Float)
// ================================================================
void solve_system(float* AtA, float* Atb, float* x, int N) {
    float* G = (float*)malloc(N * N * sizeof(float));
    float* g = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N * N; i++) G[i] = AtA[i];
    for (int i = 0; i < N; i++) g[i] = Atb[i];

    for (int i = 0; i < N; i++) {
        int pivot = i;
        float maxv = fabsf(G[IDX(i, i, N)]);
        for (int k = i + 1; k < N; k++) {
            float v = fabsf(G[IDX(k, i, N)]);
            if (v > maxv) { maxv = v; pivot = k; }
        }
        if (pivot != i) {
            for (int j = 0; j < N; j++) {
                float tmp = G[IDX(i, j, N)]; G[IDX(i, j, N)] = G[IDX(pivot, j, N)]; G[IDX(pivot, j, N)] = tmp;
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
    free(G); free(g);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Uso: %s M N NUM_BLOCKS THREADS_PER_BLOCK\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int NUM_BLOCKS = atoi(argv[3]);
    int THREADS = atoi(argv[4]);

    float* A = (float*)malloc(M * N * sizeof(float));
    float* b = (float*)malloc(M * sizeof(float));
    float* AtA = (float*)malloc(N * N * sizeof(float));
    float* Atb = (float*)malloc(N * sizeof(float));
    float* x = (float*)malloc(N * sizeof(float));

    srand(42);
    for (int i = 0; i < M; i++) {
        b[i] = (float)(rand() % 10 + 1);
        for (int j = 0; j < N; j++) A[IDX(i, j, N)] = (float)(rand() % 10 + 1);
    }

    float* d_A, * d_b, * d_AtA, * d_Atb;
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_AtA, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Atb, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, M * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid(NUM_BLOCKS);
    dim3 block(THREADS); // Lo script Python gli passerà 32 in base al grafico

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Lancio Kernel Dinamici
    compute_AtA_hybrid << <grid, block >> > (d_A, d_AtA, M, N);
    compute_Atb_hybrid << <grid, block >> > (d_A, d_b, d_Atb, M, N);

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(AtA, d_AtA, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Atb, d_Atb, N * sizeof(float), cudaMemcpyDeviceToHost));

    solve_system(AtA, Atb, x, N);

    // Stampa nel formato esatto per lo script python (usato nelle vecchie it1/it3)
    printf("\n[SUCCESS] Tempo esecuzione GPU: %.3f ms\n", ms);
    // Inserisco anche l'altro formato se il tuo script matcha questo
    printf("Tempo SOLO Parallelo (Kernel): %.3f ms\n", ms);

    cudaFree(d_A); cudaFree(d_b); cudaFree(d_AtA); cudaFree(d_Atb);
    free(A); free(b); free(AtA); free(Atb); free(x);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}