/*
====================================================================
CUDA Least Squares - PROFILING CON CONTATORI HARDWARE (clock64)
--------------------------------------------------------------------
Versione modularizzata con l'aggiunta della misurazione manuale
dei cicli di clock direttamente all'interno dei CUDA Core.
AUTOPSIA DEL CICLO: Separazione esatta tra tempo di fetch (VRAM)
e tempo di calcolo (ALU/Math).
====================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define IDX(i,j,N) ((i)*(N)+(j))

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("\nCUDA ERROR: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

// ================================================================
// 1. KERNEL CUDA
// ================================================================

// Kernel per il calcolo della matrice AtA (IL BOTTLENECK)
__global__ void compute_AtA(float* A, float* AtA, int M, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int total_elements_AtA = N * N;
    for (int idx = tid; idx < total_elements_AtA; idx += stride) {
        int row = idx / N;
        int col = idx % N;

        // ----------------------------------------------------
        // AUTOPSIA DEL CICLO CON CRONOMETRI SEPARATI
        // ----------------------------------------------------
        long long tempo_memoria = 0;
        long long tempo_matematica = 0;
        float sum = 0.0f;

        for (int k = 0; k < M; k++) {
            // 1. CRONOMETRO IL FETCH DELLA MEMORIA (Latenza VRAM)
            long long start_mem = clock64();
            float val_row = A[IDX(k, row, N)];
            float val_col = A[IDX(k, col, N)];
            tempo_memoria += (clock64() - start_mem);

            // 2. CRONOMETRO LA MATEMATICA NEI REGISTRI (FMA)
            long long start_math = clock64();
            sum += val_row * val_col;
            tempo_matematica += (clock64() - start_math);
        }

        AtA[idx] = sum;

        // Stampiamo la diagnosi solo per il primo thread (idx == 0)
        // altrimenti la console esploderebbe di messaggi!
        if (idx == 0) {
            printf("\n>>> AUTOPSIA DEL CICLO INTERNO (Thread 0) <<<\n");
            printf("Cicli totali spesi ad aspettare la VRAM (Memoria) : %lld\n", tempo_memoria);
            printf("Cicli totali spesi a fare moltiplicazioni (Math)  : %lld\n", tempo_matematica);
            printf("-------------------------------------------------\n");
        }
    }
}

// Kernel per il calcolo del vettore Atb
__global__ void compute_Atb(float* A, float* b, float* Atb, int M, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int row = tid; row < N; row += stride) {
        float sum = 0.0f;
        for (int k = 0; k < M; k++) {
            sum += A[IDX(k, row, N)] * b[k];
        }
        Atb[row] = sum;
    }
}

// ================================================================
// 2. FUNZIONI HOST (Modularizzazione C++)
// ================================================================

void init_host_data(float* A, float* b, int M, int N) {
    srand(42);
    for (int i = 0; i < M; i++) {
        b[i] = (float)(rand() % 10 + 1);
        for (int j = 0; j < N; j++) {
            A[IDX(i, j, N)] = (float)(rand() % 10 + 1);
        }
    }
}

float launch_cuda_computation(float* d_A, float* d_b, float* d_AtA, float* d_Atb, int M, int N, int num_blocks, int threads_per_block) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Lancio i due kernel separatamente
    compute_AtA << <num_blocks, threads_per_block >> > (d_A, d_AtA, M, N);
    CUDA_CHECK(cudaGetLastError());

    compute_Atb << <num_blocks, threads_per_block >> > (d_A, d_b, d_Atb, M, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

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
                float tmp = G[IDX(i, j, N)];
                G[IDX(i, j, N)] = G[IDX(pivot, j, N)];
                G[IDX(pivot, j, N)] = tmp;
            }
            float tv = g[i]; g[i] = g[pivot]; g[pivot] = tv;
        }

        for (int k = i + 1; k < N; k++) {
            float factor = G[IDX(k, i, N)] / G[IDX(i, i, N)];
            for (int j = i; j < N; j++) { G[IDX(k, j, N)] -= factor * G[IDX(i, j, N)]; }
            g[k] -= factor * g[i];
        }
    }

    for (int i = N - 1; i >= 0; i--) {
        float sum = 0.0f;
        for (int j = i + 1; j < N; j++) { sum += G[IDX(i, j, N)] * x[j]; }
        x[i] = (g[i] - sum) / G[IDX(i, i, N)];
    }

    free(G); free(g);
}

// ================================================================
// 3. MAIN (Punto di ingresso pulito)
// ================================================================
int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1000;
    int N = (argc > 2) ? atoi(argv[2]) : 500;
    int num_blocks = (argc > 3) ? atoi(argv[3]) : 256;
    int threads_per_block = (argc > 4) ? atoi(argv[4]) : 256;

    printf("\n--- CALCOLO MINIMI QUADRATI CUDA ---\n");
    printf("Matrice: %d x %d\n", M, N);
    printf("Configurazione Kernel: %d Blocchi, %d Thread/Blocco\n", num_blocks, threads_per_block);
    printf("------------------------------------\n");

    // 1. Allocazione Memoria HOST
    float* A = (float*)malloc(M * N * sizeof(float));
    float* b = (float*)malloc(M * sizeof(float));
    float* AtA = (float*)malloc(N * N * sizeof(float));
    float* Atb = (float*)malloc(N * sizeof(float));
    float* x = (float*)malloc(N * sizeof(float));

    // 2. Inizializzazione
    init_host_data(A, b, M, N);

    // 3. Allocazione Memoria DEVICE
    float* d_A, * d_b, * d_AtA, * d_Atb;
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_AtA, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Atb, N * sizeof(float)));

    // 4. Trasferimento H2D
    CUDA_CHECK(cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, M * sizeof(float), cudaMemcpyHostToDevice));

    // 5. Lancio Computazione GPU (Funzione Isolato)
    float ms = launch_cuda_computation(d_A, d_b, d_AtA, d_Atb, M, N, num_blocks, threads_per_block);

    // 6. Trasferimento D2H
    CUDA_CHECK(cudaMemcpy(AtA, d_AtA, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Atb, d_Atb, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 7. Risoluzione Sistema Lineare su CPU
    solve_system(AtA, Atb, x, N);

    printf("\n[SUCCESS] Tempo esecuzione GPU: %.3f ms\n", ms);
    if (N >= 3) {
        printf("Primi 3 valori di x: [%.3f, %.3f, %.3f]\n", x[0], x[1], x[2]);
    }

    // 8. Pulizia Memoria
    cudaFree(d_A); cudaFree(d_b);
    cudaFree(d_AtA); cudaFree(d_Atb);
    free(A); free(b); free(AtA); free(Atb); free(x);

    return 0;
}
