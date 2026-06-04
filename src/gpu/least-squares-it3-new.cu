#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define IDX(row, col, N) ((row) * (N) + (col))
#define TILE 32

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/* ============================================================
 * KERNEL 1: AtA con Tiling in Shared Memory e Bank-Conflict Padding
 * ============================================================ */
__global__ void compute_AtA_tiled(const float* __restrict__ A, float* __restrict__ AtA, int M, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // row e col determinano quale elemento della matrice AtA (NxN) stiamo calcolando
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    // TILE + 1 previene i Bank Conflicts quando si legge per colonna
    __shared__ float s_A_left[TILE][TILE + 1];
    __shared__ float s_A_right[TILE][TILE + 1];

    float sum = 0.0f;

    // Avanziamo a mattonelle (Tile) lungo la dimensione M
    for (int k_step = 0; k_step < (M + TILE - 1) / TILE; k_step++) {
        int k_idx = k_step * TILE + ty; // Riga di A da caricare

        // 1. Caricamento Coalesced della mattonella SINISTRA
        int a_col_left = blockIdx.y * TILE + tx;
        if (k_idx < M && a_col_left < N)
            s_A_left[ty][tx] = A[IDX(k_idx, a_col_left, N)];
        else
            s_A_left[ty][tx] = 0.0f;

        // 2. Caricamento Coalesced della mattonella DESTRA
        int a_col_right = blockIdx.x * TILE + tx;
        if (k_idx < M && a_col_right < N)
            s_A_right[ty][tx] = A[IDX(k_idx, a_col_right, N)];
        else
            s_A_right[ty][tx] = 0.0f;

        // Aspettiamo che tutti i thread del blocco abbiano finito di caricare
        __syncthreads();

        // 3. Prodotto scalare parziale usando l'ultra-veloce Shared Memory
#pragma unroll
        for (int k_local = 0; k_local < TILE; k_local++) {
            sum += s_A_left[k_local][ty] * s_A_right[k_local][tx];
        }

        // Aspettiamo prima di sovrascrivere la Shared Memory con la mattonella successiva
        __syncthreads();
    }

    // Singola scrittura in memoria globale (zero atomicAdd!) e sfruttamento simmetria
    if (row < N && col < N && row <= col) {
        AtA[IDX(row, col, N)] = sum;
        if (row != col) {
            AtA[IDX(col, row, N)] = sum; // Copia simmetrica
        }
    }
}

/* ============================================================
 * KERNEL 2: Atb con Caching del vettore 'b' in Shared Memory
 * ============================================================ */
__global__ void compute_Atb_tiled(const float* __restrict__ A, const float* __restrict__ b, float* __restrict__ Atb, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // Colonna di A / Riga di Atb
    int tx = threadIdx.x;

    __shared__ float s_b[TILE];
    float sum = 0.0f;

    for (int k_step = 0; k_step < (M + TILE - 1) / TILE; k_step++) {
        // Carichiamo una porzione di 'b' in Shared Memory
        int k_idx = k_step * TILE + tx;
        if (k_idx < M)
            s_b[tx] = b[k_idx];
        else
            s_b[tx] = 0.0f;

        __syncthreads();

        if (row < N) {
#pragma unroll
            for (int k_local = 0; k_local < TILE; k_local++) {
                int k_actual = k_step * TILE + k_local;
                if (k_actual < M) {
                    sum += A[IDX(k_actual, row, N)] * s_b[k_local];
                }
            }
        }
        __syncthreads();
    }

    if (row < N) {
        Atb[row] = sum;
    }
}

/* ============================================================
 * ELIMINAZIONE DI GAUSS (Convertita in Float)
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
    if (argc < 3) {
        printf("Uso: %s <M> <N>\n", argv[0]);
        return 1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);

    // ==========================================
    // ALLOCAZIONE E INIZIALIZZAZIONE (Tutto in Float)
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
    // ALLOCAZIONE (GPU)
    // ==========================================
    float* d_A, * d_b, * d_AtA, * d_Atb;
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_AtA, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Atb, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, M * sizeof(float), cudaMemcpyHostToDevice));

    // Configurazione Griglia CUDA Dinamica basata su TILE 32x32
    dim3 block2D(TILE, TILE);
    dim3 grid2D((N + block2D.x - 1) / block2D.x, (N + block2D.y - 1) / block2D.y);

    dim3 block1D(TILE);
    dim3 grid1D((N + block1D.x - 1) / block1D.x);

    // ==========================================
    // MISURAZIONE TEMPO (Solo Parallelo)
    // ==========================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // Inizio Timer GPU

    // Lancio dei Kernel Ottimizzati
    compute_AtA_tiled << <grid2D, block2D >> > (d_A, d_AtA, M, N);
    CUDA_CHECK(cudaGetLastError());

    compute_Atb_tiled << <grid1D, block1D >> > (d_A, d_b, d_Atb, M, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // ==========================================
    // DOWNLOAD E SOLUZIONE
    // ==========================================
    CUDA_CHECK(cudaMemcpy(AtA, d_AtA, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Atb, d_Atb, N * sizeof(float), cudaMemcpyDeviceToHost));

    solve_system(AtA, Atb, x, N);

    // Formattazione per la regex dello script Python
    printf("Tempo SOLO Parallelo: %.2f ms\n", ms);

    if (N >= 3)
        printf("Primi 3 valori di x: %f, %f, %f\n", x[0], x[1], x[2]);

    // Pulizia finale
    free(A); free(b); free(AtA); free(Atb); free(x);
    cudaFree(d_A); cudaFree(d_b); cudaFree(d_AtA); cudaFree(d_Atb);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}