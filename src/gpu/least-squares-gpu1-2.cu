/*
====================================================================
CUDA Least Squares - OPTIMIZED VERSION
--------------------------------------------------------------------
Calcola:
    x = (A^T A)^-1 (A^T b)

OTTIMIZZAZIONI:
✔ kernel separati AtA / Atb
✔ tiling shared memory
✔ simmetria AtA
✔ memoria lineare
✔ thread/block runtime
✔ timing CUDA

Compilazione:
    nvcc -O3 least_squares_optimized.cu -o ls_cuda

Esecuzione:
    ./ls_cuda M N BX BY

Esempio:
    ./ls_cuda 10000 512 16 16
====================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define IDX(i,j,N) ((i)*(N)+(j))
#define TILE 32

// ================================================================
// CUDA CHECK
// ================================================================
#define CUDA_CHECK(err) \
if(err != cudaSuccess){ \
    printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
}

// ================================================================
// Atb KERNEL
// Ogni thread calcola:
//      Atb[row]
// ================================================================
__global__
void compute_atb(
    const double *A,
    const double *b,
    double *Atb,
    int M,
    int N
){

    int row =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(row >= N)
        return;

    double sum = 0.0;

    for(int k=0; k<M; k++){

        sum +=
            A[IDX(k,row,N)] *
            b[k];
    }

    Atb[row] = sum;
}

// ================================================================
// AtA TILED KERNEL
// Shared memory optimization
// ================================================================
__global__
void compute_ata_tiled(
    const double *A,
    double *AtA,
    int M,
    int N
){

    __shared__ double sA1[TILE][TILE];
    __shared__ double sA2[TILE][TILE];

    int row =
        blockIdx.y * TILE +
        threadIdx.y;

    int col =
        blockIdx.x * TILE +
        threadIdx.x;

    if(row >= N || col >= N)
        return;

    // solo triangolo superiore
    if(col < row)
        return;

    double sum = 0.0;

    for(int t=0; t<M; t += TILE){

        int k = t + threadIdx.x;

        // ========================================================
        // LOAD SHARED MEMORY
        // ========================================================
        if(k < M){

            sA1[threadIdx.y][threadIdx.x] =
                A[IDX(k,row,N)];

            sA2[threadIdx.y][threadIdx.x] =
                A[IDX(k,col,N)];
        }
        else{

            sA1[threadIdx.y][threadIdx.x] = 0.0;
            sA2[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        // ========================================================
        // TILE DOT PRODUCT
        // ========================================================
        for(int kk=0; kk<TILE; kk++){

            sum +=
                sA1[threadIdx.y][kk] *
                sA2[threadIdx.y][kk];
        }

        __syncthreads();
    }

    AtA[IDX(row,col,N)] = sum;

    // simmetria
    if(row != col){

        AtA[IDX(col,row,N)] = sum;
    }
}

// ================================================================
// GAUSS ELIMINATION
// ================================================================
void solve_system(
    double *AtA,
    double *Atb,
    double *x,
    int N
){

    double *G =
        (double*)malloc(N*N*sizeof(double));

    double *g =
        (double*)malloc(N*sizeof(double));

    for(int i=0;i<N*N;i++)
        G[i] = AtA[i];

    for(int i=0;i<N;i++)
        g[i] = Atb[i];

    // ============================================================
    // ELIMINATION
    // ============================================================
    for(int i=0;i<N;i++){

        int pivot = i;

        double maxv =
            fabs(G[IDX(i,i,N)]);

        for(int k=i+1;k<N;k++){

            double v =
                fabs(G[IDX(k,i,N)]);

            if(v > maxv){

                maxv = v;
                pivot = k;
            }
        }

        // ========================================================
        // SWAP
        // ========================================================
        if(pivot != i){

            for(int j=0;j<N;j++){

                double tmp =
                    G[IDX(i,j,N)];

                G[IDX(i,j,N)] =
                    G[IDX(pivot,j,N)];

                G[IDX(pivot,j,N)] = tmp;
            }

            double tv = g[i];
            g[i] = g[pivot];
            g[pivot] = tv;
        }

        // ========================================================
        // ELIMINATION
        // ========================================================
        for(int k=i+1;k<N;k++){

            double factor =
                G[IDX(k,i,N)] /
                G[IDX(i,i,N)];

            for(int j=i;j<N;j++){

                G[IDX(k,j,N)] -=
                    factor *
                    G[IDX(i,j,N)];
            }

            g[k] -= factor * g[i];
        }
    }

    // ============================================================
    // BACK SUBSTITUTION
    // ============================================================
    for(int i=N-1;i>=0;i--){

        double sum = 0.0;

        for(int j=i+1;j<N;j++){

            sum +=
                G[IDX(i,j,N)] *
                x[j];
        }

        x[i] =
            (g[i] - sum) /
            G[IDX(i,i,N)];
    }

    free(G);
    free(g);
}

// ================================================================
// MAIN
// ================================================================
int main(int argc, char **argv){

    if(argc < 5){

        printf(
            "Usage: %s M N BX BY\n",
            argv[0]
        );

        return 1;
    }

    // ============================================================
    // PARAMETERS
    // ============================================================
    int M  = atoi(argv[1]);
    int N  = atoi(argv[2]);

    int BX = atoi(argv[3]);
    int BY = atoi(argv[4]);

    printf("\n");
    printf("M = %d\n", M);
    printf("N = %d\n", N);
    printf("Block = (%d,%d)\n\n", BX, BY);

    // ============================================================
    // HOST MEMORY
    // ============================================================
    double *A =
        (double*)malloc(M*N*sizeof(double));

    double *b =
        (double*)malloc(M*sizeof(double));

    double *AtA =
        (double*)malloc(N*N*sizeof(double));

    double *Atb =
        (double*)malloc(N*sizeof(double));

    double *x =
        (double*)malloc(N*sizeof(double));

    srand(42);

    // ============================================================
    // INITIALIZATION
    // ============================================================
    for(int i=0;i<M;i++){

        b[i] = rand()%10 + 1;

        for(int j=0;j<N;j++){

            A[IDX(i,j,N)] =
                rand()%10 + 1;
        }
    }

    // ============================================================
    // DEVICE MEMORY
    // ============================================================
    double *d_A;
    double *d_b;
    double *d_AtA;
    double *d_Atb;

    CUDA_CHECK(
        cudaMalloc(&d_A,
        M*N*sizeof(double))
    );

    CUDA_CHECK(
        cudaMalloc(&d_b,
        M*sizeof(double))
    );

    CUDA_CHECK(
        cudaMalloc(&d_AtA,
        N*N*sizeof(double))
    );

    CUDA_CHECK(
        cudaMalloc(&d_Atb,
        N*sizeof(double))
    );

    // ============================================================
    // COPY TO GPU
    // ============================================================
    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            A,
            M*N*sizeof(double),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_b,
            b,
            M*sizeof(double),
            cudaMemcpyHostToDevice
        )
    );

    // ============================================================
    // GRID CONFIGURATION
    // ============================================================
    dim3 blockA(TILE,TILE);

    dim3 gridA(
        (N + TILE - 1)/TILE,
        (N + TILE - 1)/TILE
    );

    int TB = BX * BY;

    dim3 blockB(TB);

    dim3 gridB(
        (N + TB - 1)/TB
    );

    printf(
        "Grid AtA = (%d,%d)\n",
        gridA.x,
        gridA.y
    );

    printf(
        "Grid Atb = (%d)\n\n",
        gridB.x
    );

    // ============================================================
    // TIMING
    // ============================================================
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // ============================================================
    // AtA
    // ============================================================
    compute_ata_tiled<<<
        gridA,
        blockA
    >>>(
        d_A,
        d_AtA,
        M,
        N
    );

    CUDA_CHECK(cudaGetLastError());

    // ============================================================
    // Atb
    // ============================================================
    compute_atb<<<
        gridB,
        blockB
    >>>(
        d_A,
        d_b,
        d_Atb,
        M,
        N
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0.0f;

    cudaEventElapsedTime(
        &ms,
        start,
        stop
    );

    // ============================================================
    // COPY BACK
    // ============================================================
    CUDA_CHECK(
        cudaMemcpy(
            AtA,
            d_AtA,
            N*N*sizeof(double),
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            Atb,
            d_Atb,
            N*sizeof(double),
            cudaMemcpyDeviceToHost
        )
    );

    // ============================================================
    // SOLVE SYSTEM
    // ============================================================
    solve_system(
        AtA,
        Atb,
        x,
        N
    );

    // ============================================================
    // OUTPUT
    // ============================================================
    printf(
        "GPU Time: %.6f ms\n",
        ms
    );

    if(N >= 3){

        printf(
            "x[0] = %f\n",
            x[0]
        );

        printf(
            "x[1] = %f\n",
            x[1]
        );

        printf(
            "x[2] = %f\n",
            x[2]
        );
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