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

// Dimensione del tile per la shared memory (deve coincidere con blockDim)
#define TILE_SIZE 16

/* ============================================================
 * KERNEL 1 (OTTIMIZZATO): AtA con Shared Memory Tiling
 *
 * Strategia:
 *   - La matrice A (M x N) viene caricata a tile di TILE_SIZE righe.
 *   - Ogni blocco 2D (TILE_SIZE x TILE_SIZE) è responsabile di un
 *     sotto-blocco di AtA. I thread caricano cooperativamente le
 *     colonne necessarie di A in shared memory, riducendo gli accessi
 *     alla DRAM globale da O(M*N^2) a O(M*N^2 / TILE_SIZE).
 *   - Viene sfruttata la simmetria: si calcola solo il triangolo
 *     superiore (row <= col) e si replica per simmetria.
 * ============================================================ */
__global__ void compute_AtA_shared(const float* __restrict__ A,
                                    float* __restrict__ AtA,
                                    int M, int N)
{
    // Indici globali dell'elemento di AtA che questo thread produce
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Due tile di shared memory: una per la colonna "row", una per "col"
    // Ogni tile ha forma [TILE_SIZE (k-strip) x TILE_SIZE (feature)]
    __shared__ float tileA[TILE_SIZE][TILE_SIZE]; // strip di colonne per "row"
    __shared__ float tileB[TILE_SIZE][TILE_SIZE]; // strip di colonne per "col"

    float sum = 0.0f;

    // Scorre le strip da TILE_SIZE righe di A
    for (int tile = 0; tile < (M + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        int k = tile * TILE_SIZE + threadIdx.y; // riga globale di A per tileA
        int l = tile * TILE_SIZE + threadIdx.x; // riga globale di A per tileB

        // Caricamento cooperativo:
        //   tileA[threadIdx.y][threadIdx.x] = A[k, col_of_row_group]
        //   tileB[threadIdx.y][threadIdx.x] = A[l, col_of_col_group]
        int rowFeature = blockIdx.y * TILE_SIZE + threadIdx.x; // colonna feature per tileA
        int colFeature = blockIdx.x * TILE_SIZE + threadIdx.y; // colonna feature per tileB (nota: ruoli invertiti per coalescing)

        // tileA: per ogni k nel tile, colonna = rowFeature
        tileA[threadIdx.y][threadIdx.x] = (k < M && rowFeature < N)
                                           ? A[IDX(k, rowFeature, N)]
                                           : 0.0f;

        // tileB: per ogni l nel tile, colonna = colFeature  (accesso contiguo)
        tileB[threadIdx.x][threadIdx.y] = (l < M && colFeature < N)
                                           ? A[IDX(l, colFeature, N)]
                                           : 0.0f;

        __syncthreads();

        // Prodotto parziale: somma su TILE_SIZE valori
        if (row < N && col < N && row <= col) {
#pragma unroll
            for (int t = 0; t < TILE_SIZE; t++) {
                sum += tileA[t][threadIdx.y] * tileB[t][threadIdx.x];
            }
        }

        __syncthreads();
    }

    // Scrittura finale (triangolo superiore + simmetria)
    if (row < N && col < N && row <= col) {
        AtA[IDX(row, col, N)] = sum;
        if (row != col)
            AtA[IDX(col, row, N)] = sum;
    }
}

/* ============================================================
 * KERNEL 2 (OTTIMIZZATO): Atb con Shared Memory
 *
 * Strategia:
 *   - Ogni blocco 1D calcola TILE_SIZE elementi di Atb in parallelo.
 *   - I thread del blocco caricano cooperativamente una strip del
 *     vettore b in shared memory: ogni tile carica TILE_SIZE elementi
 *     consecutivi di b, evitando che ogni thread rilegga b dalla DRAM
 *     per ognuna delle N feature.
 *   - A viene letta a strisce di TILE_SIZE righe: gli accessi rimangono
 *     coalescenti (thread adiacenti leggono colonne adiacenti di A).
 * ============================================================ */
__global__ void compute_Atb_shared(const float* __restrict__ A,
                                    const float* __restrict__ b,
                                    float* __restrict__ Atb,
                                    int M, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // colonna di A (= feature)

    // Shared memory per la strip corrente di b
    __shared__ float b_tile[TILE_SIZE];

    float sum = 0.0f;

    for (int tile = 0; tile < (M + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Caricamento cooperativo di TILE_SIZE elementi di b
        int k = tile * TILE_SIZE + threadIdx.x;
        if (threadIdx.x < TILE_SIZE) {
            b_tile[threadIdx.x] = (k < M) ? b[k] : 0.0f;
        }
        __syncthreads();

        // Ogni thread accumula la propria colonna moltiplicata per b_tile
        if (col < N) {
#pragma unroll
            for (int t = 0; t < TILE_SIZE; t++) {
                int row = tile * TILE_SIZE + t;
                if (row < M)
                    sum += A[IDX(row, col, N)] * b_tile[t];
            }
        }
        __syncthreads();
    }

    if (col < N)
        Atb[col] = sum;
}

/* ============================================================
 * ELIMINAZIONE DI GAUSS (CPU, Singola Precisione) — invariata
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
    // ALLOCAZIONE HOST E INIZIALIZZAZIONE
    // ==========================================
    srand(42);
    float* A   = (float*)malloc(M * N * sizeof(float));
    float* b   = (float*)malloc(M * sizeof(float));
    float* AtA = (float*)calloc(N * N, sizeof(float));
    float* Atb = (float*)calloc(N, sizeof(float));
    float* x   = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < M; i++) {
        b[i] = (float)(rand() % 10 + 1);
        for (int j = 0; j < N; j++)
            A[IDX(i, j, N)] = (float)(rand() % 10 + 1);
    }

    // ==========================================
    // ALLOCAZIONE DEVICE
    // ==========================================
    float* d_A, *d_b, *d_AtA, *d_Atb;
    CUDA_CHECK(cudaMalloc(&d_A,   M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b,   M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_AtA, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Atb, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, M * sizeof(float),     cudaMemcpyHostToDevice));

    // ==========================================
    // GEOMETRIA CUDA
    //   Kernel AtA: griglia 2D di blocchi TILE_SIZE x TILE_SIZE
    //   Kernel Atb: griglia 1D di blocchi TILE_SIZE
    // ==========================================
    dim3 block2D(TILE_SIZE, TILE_SIZE);
    dim3 grid2D((N + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);

    dim3 block1D(TILE_SIZE);
    dim3 grid1D((N + TILE_SIZE - 1) / TILE_SIZE);

    // ==========================================
    // TIMING
    // ==========================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    compute_AtA_shared<<<grid2D, block2D>>>(d_A, d_AtA, M, N);
    CUDA_CHECK(cudaGetLastError());

    compute_Atb_shared<<<grid1D, block1D>>>(d_A, d_b, d_Atb, M, N);
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
    CUDA_CHECK(cudaMemcpy(Atb, d_Atb, N * sizeof(float),     cudaMemcpyDeviceToHost));

    solve_system(AtA, Atb, x, N);

    printf("Tempo SOLO Parallelo: %.2f ms\n", ms);

    if (N >= 3)
        printf("Primi 3 valori di x: %f, %f, %f\n", x[0], x[1], x[2]);

    free(A); free(b); free(AtA); free(Atb); free(x);
    cudaFree(d_A); cudaFree(d_b); cudaFree(d_AtA); cudaFree(d_Atb);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
