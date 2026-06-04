/*
====================================================================
CUDA Least Squares - VERSIONE AGGIORNATA (Grid-Stride Loop 1D)
--------------------------------------------------------------------
Calcola: x = (A^T A)^-1 (A^T b)
Modificato per accettare in input numero di blocchi e thread
per blocco, utilizzando un approccio flessibile (Grid-Stride Loop).
====================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define IDX(i,j,N) ((i)*(N)+(j))

// ================================================================
// CUDA ERROR CHECK (Macro per catturare errori silenziosi)
// ================================================================
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("\nCUDA ERROR: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

// ================================================================
// KERNEL CUDA: Calcolo parallelo con Grid-Stride Loop 1D
// ================================================================
__global__ void compute_ata_atb(float* A, float* b, float* AtA, float* Atb, int M, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // 1. Calcolo matrice simmetrica AtA (N x N elementi)
    int total_elements_AtA = N * N;
    for (int idx = tid; idx < total_elements_AtA; idx += stride) {
        int row = idx / N;
        int col = idx % N;

        float sum = 0.0f;
        for (int k = 0; k < M; k++) {
            sum += A[IDX(k, row, N)] * A[IDX(k, col, N)];
        }
        AtA[idx] = sum;
    }

    // 2. Calcolo vettore Atb (N elementi)
    for (int row = tid; row < N; row += stride) {
        float sum = 0.0f;
        for (int k = 0; k < M; k++) {
            sum += A[IDX(k, row, N)] * b[k];
        }
        Atb[row] = sum;
    }
}

// ================================================================
// SOLVER CPU: Eliminazione di Gauss con pivot parziale
// ================================================================
void solve_system(float* AtA, float* Atb, float* x, int N) {
    float* G = (float*)malloc(N * N * sizeof(float));
    float* g = (float*)malloc(N * sizeof(float));

    // Copia difensiva per non sovrascrivere i risultati originali
    for (int i = 0; i < N * N; i++) G[i] = AtA[i];
    for (int i = 0; i < N; i++) g[i] = Atb[i];

    // Eliminazione
    for (int i = 0; i < N; i++) {
        int pivot = i;
        float maxv = fabsf(G[IDX(i, i, N)]);

        for (int k = i + 1; k < N; k++) {
            float v = fabsf(G[IDX(k, i, N)]);
            if (v > maxv) {
                maxv = v;
                pivot = k;
            }
        }

        // Swap righe
        if (pivot != i) {
            for (int j = 0; j < N; j++) {
                float tmp = G[IDX(i, j, N)];
                G[IDX(i, j, N)] = G[IDX(pivot, j, N)];
                G[IDX(pivot, j, N)] = tmp;
            }
            float tv = g[i];
            g[i] = g[pivot];
            g[pivot] = tv;
        }

        // Eliminazione vera e propria
        for (int k = i + 1; k < N; k++) {
            float factor = G[IDX(k, i, N)] / G[IDX(i, i, N)];
            for (int j = i; j < N; j++) {
                G[IDX(k, j, N)] -= factor * G[IDX(i, j, N)];
            }
            g[k] -= factor * g[i];
        }
    }

    // Back substitution
    for (int i = N - 1; i >= 0; i--) {
        float sum = 0.0f;
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
    // Parametri dinamici da riga di comando
    int M = (argc > 1) ? atoi(argv[1]) : 1000;
    int N = (argc > 2) ? atoi(argv[2]) : 500;
    int num_blocks = (argc > 3) ? atoi(argv[3]) : 256;  // Numero totale di blocchi
    int threads_per_block = (argc > 4) ? atoi(argv[4]) : 256;  // Thread per blocco

    printf("\n--- CALCOLO MINIMI QUADRATI CUDA ---\n");
    printf("Matrice: %d x %d\n", M, N);
    printf("Configurazione Kernel: %d Blocchi, %d Thread/Blocco\n", num_blocks, threads_per_block);
    printf("------------------------------------\n");

    // Allocazione Memoria HOST
    float* A = (float*)malloc(M * N * sizeof(float));
    float* b = (float*)malloc(M * sizeof(float));
    float* AtA = (float*)malloc(N * N * sizeof(float));
    float* Atb = (float*)malloc(N * sizeof(float));
    float* x = (float*)malloc(N * sizeof(float));

    // Inizializzazione dati
    srand(42);
    for (int i = 0; i < M; i++) {
        b[i] = (float)(rand() % 10 + 1);
        for (int j = 0; j < N; j++) {
            A[IDX(i, j, N)] = (float)(rand() % 10 + 1);
        }
    }

    // Allocazione Memoria DEVICE (GPU)
    float* d_A, * d_b, * d_AtA, * d_Atb;
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_AtA, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Atb, N * sizeof(float)));

    // Trasferimento dati Host -> Device
    CUDA_CHECK(cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, M * sizeof(float), cudaMemcpyHostToDevice));

    // ============================================================
    // TIMING CUDA (Cronometri Nativi)
    // ============================================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // Partenza!

    // Lancio Kernel 1D
    compute_ata_atb << <num_blocks, threads_per_block >> > (d_A, d_b, d_AtA, d_Atb, M, N);

    // Controllo errori istantaneo dopo il lancio
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop);  // Fine!
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    // ============================================================

    // Trasferimento risultati Device -> Host
    CUDA_CHECK(cudaMemcpy(AtA, d_AtA, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Atb, d_Atb, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Risoluzione Sistema Lineare su CPU
    solve_system(AtA, Atb, x, N);

    // Stampa Risultati
    printf("\n[SUCCESS] Tempo esecuzione GPU: %.3f ms\n", ms);
    if (N >= 3) {
        printf("Primi 3 valori di x: [%.3f, %.3f, %.3f]\n", x[0], x[1], x[2]);
    }

    // Pulizia Memoria
    cudaFree(d_A); cudaFree(d_b);
    cudaFree(d_AtA); cudaFree(d_Atb);
    free(A); free(b); free(AtA); free(Atb); free(x);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}