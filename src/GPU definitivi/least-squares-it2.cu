/*
====================================================================
CUDA Least Squares - DYNAMIC LAUNCH PARAMETERS
--------------------------------------------------------------------

Calcola:
    x = (A^T A)^-1 (A^T b)

OTTIMIZZAZIONI MANTENUTE / AGGIORNATE:
✔ kernel separati AtA / Atb
✔ Grid-Stride Loops (supporta qualsiasi num. di blocchi e thread)
✔ loop unrolling
✔ accessi global memory migliorati
✔ simmetria AtA
✔ CUDA timing

Compilazione:
    nvcc -O3 least_squares_dynamic.cu -o ls_cuda

Esecuzione:
    ./ls_cuda M N NUM_BLOCKS THREADS_PER_BLOCK

Esempio:
    ./ls_cuda 10000 512 64 256

====================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define IDX(i,j,N) ((i)*(N)+(j))

// ================================================================
// CUDA CHECK
// ================================================================
#define CUDA_CHECK(err) \
if(err != cudaSuccess){ \
    printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
}

// ================================================================
// Atb KERNEL (Grid-Stride 1D)
// ================================================================
__global__
void compute_atb(
    const double* __restrict__ A,
    const double* __restrict__ b,
    double* __restrict__ Atb,
    int M,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int row = tid; row < N; row += stride) {
        double sum = 0.0;

#pragma unroll 4
        for (int k = 0; k < M; k++) {
            sum += A[IDX(k, row, N)] * b[k];
        }

        Atb[row] = sum;
    }
}

// ================================================================
// AtA KERNEL (Grid-Stride 1D con simmetria)
// ================================================================
__global__
void compute_ata_upper(
    const double* __restrict__ A,
    double* __restrict__ AtA,
    int M,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total_elements = N * N;

    // Iteriamo su tutti gli elementi della matrice NxN usando il Grid-Stride Loop
    for (int idx = tid; idx < total_elements; idx += stride) {

        int row = idx / N;
        int col = idx % N;

        // Calcoliamo solo la parte triangolare superiore
        if (row <= col) {
            double sum = 0.0;

#pragma unroll 4
            for (int k = 0; k < M; k++) {
                sum += A[IDX(k, row, N)] * A[IDX(k, col, N)];
            }

            AtA[IDX(row, col, N)] = sum;

            // Applichiamo la simmetria direttamente
            if (row != col) {
                AtA[IDX(col, row, N)] = sum;
            }
        }
    }
}

// ================================================================
// GAUSS ELIMINATION (Eseguita su CPU)
// ================================================================
void solve_system(
    double* AtA,
    double* Atb,
    double* x,
    int N
) {
    double* G = (double*)malloc(N * N * sizeof(double));
    double* g = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N * N; i++) G[i] = AtA[i];
    for (int i = 0; i < N; i++)   g[i] = Atb[i];

    // ELIMINATION
    for (int i = 0; i < N; i++) {
        int pivot = i;
        double maxv = fabs(G[IDX(i, i, N)]);

        for (int k = i + 1; k < N; k++) {
            double v = fabs(G[IDX(k, i, N)]);
            if (v > maxv) {
                maxv = v;
                pivot = k;
            }
        }

        // SWAP
        if (pivot != i) {
            for (int j = 0; j < N; j++) {
                double tmp = G[IDX(i, j, N)];
                G[IDX(i, j, N)] = G[IDX(pivot, j, N)];
                G[IDX(pivot, j, N)] = tmp;
            }
            double tv = g[i];
            g[i] = g[pivot];
            g[pivot] = tv;
        }

        // ELIMINATION
        for (int k = i + 1; k < N; k++) {
            double factor = G[IDX(k, i, N)] / G[IDX(i, i, N)];
            for (int j = i; j < N; j++) {
                G[IDX(k, j, N)] -= factor * G[IDX(i, j, N)];
            }
            g[k] -= factor * g[i];
        }
    }

    // BACK SUBSTITUTION
    for (int i = N - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < N; j++) {
            sum += G[IDX(i, j, N)] * x[j];
        }
        x[i] = (g[i] - sum) / G[IDX(i, i, N)];
    }

    free(G);
    free(g);
}

// ================================================================
// MAIN
// ================================================================
int main(int argc, char** argv) {

    if (argc < 5) {
        printf("Usage: %s M N NUM_BLOCKS THREADS_PER_BLOCK\n", argv[0]);
        return 1;
    }

    // ============================================================
    // PARAMETERS
    // ============================================================
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);

    int NUM_BLOCKS = atoi(argv[3]);
    int THREADS_PER_BLOCK = atoi(argv[4]);

    printf("\n");
    printf("M = %d\n", M);
    printf("N = %d\n", N);
    printf("Grid Size  = %d blocks\n", NUM_BLOCKS);
    printf("Block Size = %d threads\n\n", THREADS_PER_BLOCK);

    // ============================================================
    // HOST MEMORY
    // ============================================================
    double* A = (double*)malloc(M * N * sizeof(double));
    double* b = (double*)malloc(M * sizeof(double));
    double* AtA = (double*)malloc(N * N * sizeof(double));
    double* Atb = (double*)malloc(N * sizeof(double));
    double* x = (double*)malloc(N * sizeof(double));

    srand(42);

    // INITIALIZATION
    for (int i = 0; i < M; i++) {
        b[i] = rand() % 10 + 1;
        for (int j = 0; j < N; j++) {
            A[IDX(i, j, N)] = rand() % 10 + 1;
        }
    }

    // ============================================================
    // DEVICE MEMORY
    // ============================================================
    double* d_A;
    double* d_b;
    double* d_AtA;
    double* d_Atb;

    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_AtA, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Atb, N * sizeof(double)));

    // ============================================================
    // COPY TO GPU
    // ============================================================
    CUDA_CHECK(cudaMemcpy(d_A, A, M * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, M * sizeof(double), cudaMemcpyHostToDevice));

    // ============================================================
    // GRID CONFIGURATION (User Defined)
    // ============================================================
    dim3 grid(NUM_BLOCKS);
    dim3 block(THREADS_PER_BLOCK);

    // ============================================================
    // CUDA TIMING
    // ============================================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // ============================================================
    // AtA KERNEL
    // ============================================================
    compute_ata_upper << <grid, block >> > (d_A, d_AtA, M, N);
    CUDA_CHECK(cudaGetLastError());

    // ============================================================
    // Atb KERNEL
    // ============================================================
    compute_atb << <grid, block >> > (d_A, d_b, d_Atb, M, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // ============================================================
    // COPY BACK
    // ============================================================
    CUDA_CHECK(cudaMemcpy(AtA, d_AtA, N * N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Atb, d_Atb, N * sizeof(double), cudaMemcpyDeviceToHost));

    // ============================================================
    // SOLVE SYSTEM
    // ============================================================
    solve_system(AtA, Atb, x, N);

    // ============================================================
    // OUTPUT
    // ============================================================
    printf("GPU Time: %.6f ms\n", ms);

    if (N >= 3) {
        printf("x[0] = %f\n", x[0]);
        printf("x[1] = %f\n", x[1]);
        printf("x[2] = %f\n", x[2]);
    }

    // ============================================================
    // CLEANUP
    // ============================================================
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_AtA);
    cudaFree(d_Atb);

    free(A);
    free(b);
    free(AtA);
    free(Atb);
    free(x);

    return 0;
}