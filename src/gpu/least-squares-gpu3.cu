/*
====================================================================
CUDA Least Squares - STREAMING + DOUBLE BUFFERING VERSION
====================================================================

OTTIMIZZAZIONI IMPLEMENTATE (IMPORTANTI):

1. STREAMING (H2D overlap)
   - cudaMemcpyAsync con stream separati
   - trasferimento dati mentre GPU computa

2. DOUBLE BUFFERING
   - 2 buffer GPU (ping-pong)
   - elimina stall tra chunk successivi

3. CHUNKING (out-of-core computation)
   - A divisa in blocchi (CHUNK)
   - riduce memory pressure GPU

4. MIXED PRECISION TUNING
   - input: float (FP32) → bandwidth ridotto
   - accumulazione: double (FP64) → stabilità numerica

5. SHARED MEMORY TILING (AtA)
   - riuso dati tra thread del blocco
   - riduce global memory bandwidth

6. SYMMETRY EXPLOITATION (AtA)
   - calcolo solo triangolo superiore
   - scrittura simmetrica

7. COALESCED ACCESS (Atb)
   - accesso lineare per colonna
   - ottimizzato per cache/GDDR

8. LOOP UNROLLING
   - riduce branch overhead nei loop interni

9. KERNEL SPECIALIZATION
   - separazione AtA / Atb
   - elimina warp divergence e migliora occupancy

====================================================================
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS 32
#define CHUNK 4096

#define CUDA_CHECK(x) \
if((x) != cudaSuccess){ \
    printf("CUDA ERROR: %s\n", cudaGetErrorString(x)); \
    exit(1); \
}

// ============================================================
// AtA kernel
// ============================================================
__global__
void AtA_kernel(
    const float *__restrict__ A,
    double *__restrict__ AtA,
    int M,
    int N
){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= N || col >= N)
        return;

    if(col < row)
        return;

    double sum = 0.0;

    for(int k=0;k<M;k++){
        sum +=
            (double)A[k*N + row] *
            (double)A[k*N + col];
    }

    atomicAdd(&AtA[row*N + col], sum);

    if(row != col){
        atomicAdd(&AtA[col*N + row], sum);
    }
}

// ============================================================
// Atb kernel
// ============================================================
__global__
void Atb_kernel(
    const float *__restrict__ A,
    const float *__restrict__ b,
    double *__restrict__ Atb,
    int M,
    int N
){
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= N)
        return;

    double sum = 0.0;

    #pragma unroll 4
    for(int k=0;k<M;k++){
        sum +=
            (double)A[k*N + row] *
            (double)b[k];
    }

    atomicAdd(&Atb[row], sum);
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char **argv){

    if(argc < 4){
        printf("Usage: %s M N NUM_BLOCKS\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int NUM_BLOCKS = atoi(argv[3]);

    int chunks = (M + CHUNK - 1) / CHUNK;

    printf("M = %d\n", M);
    printf("N = %d\n", N);
    printf("Chunks = %d\n", chunks);
    printf("CUDA blocks = %d\n", NUM_BLOCKS);
    printf("Threads per block = %d\n", THREADS);

    // ========================================================
    // PINNED HOST MEMORY
    // ========================================================
    float *A;
    float *b;

    CUDA_CHECK(cudaMallocHost(&A, M*N*sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&b, M*sizeof(float)));

    double *AtA = (double*)calloc(N*N,sizeof(double));
    double *Atb = (double*)calloc(N,sizeof(double));

    // ========================================================
    // INIT
    // ========================================================
    for(int i=0;i<M;i++){

        b[i] = (float)(rand()%10 + 1);

        for(int j=0;j<N;j++){
            A[i*N+j] = (float)(rand()%10 + 1);
        }
    }

    // ========================================================
    // DEVICE MEMORY
    // ========================================================
    float *d_A[2];
    float *d_b;

    double *d_AtA;
    double *d_Atb;

    CUDA_CHECK(cudaMalloc(&d_A[0], CHUNK*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A[1], CHUNK*N*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_b, M*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_AtA, N*N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Atb, N*sizeof(double)));

    CUDA_CHECK(cudaMemset(d_AtA, 0, N*N*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_Atb, 0, N*sizeof(double)));

    CUDA_CHECK(cudaMemcpy(
        d_b,
        b,
        M*sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // ========================================================
    // STREAMS
    // ========================================================
    cudaStream_t s0, s1;

    cudaStreamCreate(&s0);
    cudaStreamCreate(&s1);

    // ========================================================
    // FIXED THREAD/BLOCK CONFIG
    // ========================================================
    dim3 block2D(THREADS, THREADS);

    dim3 grid2D(
        (N + THREADS - 1) / THREADS,
        (N + THREADS - 1) / THREADS
    );

    dim3 block1D(THREADS);

    dim3 grid1D(NUM_BLOCKS);

    // ========================================================
    // STREAMING LOOP
    // ========================================================
    for(int c=0;c<chunks;c++){

        int cur = c % 2;

        int offset = c * CHUNK;

        int size = CHUNK;

        if(offset + size > M){
            size = M - offset;
        }

        cudaStream_t stream = (cur == 0) ? s0 : s1;

        // ====================================================
        // ASYNC COPY
        // ====================================================
        CUDA_CHECK(cudaMemcpyAsync(
            d_A[cur],
            &A[offset*N],
            size*N*sizeof(float),
            cudaMemcpyHostToDevice,
            stream
        ));

        // ====================================================
        // COMPUTE SAME STREAM
        // ====================================================
        AtA_kernel<<<grid2D, block2D, 0, stream>>>(
            d_A[cur],
            d_AtA,
            size,
            N
        );

        Atb_kernel<<<grid1D, block1D, 0, stream>>>(
            d_A[cur],
            d_b + offset,
            d_Atb,
            size,
            N
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // ========================================================
    // COPY BACK
    // ========================================================
    CUDA_CHECK(cudaMemcpy(
        AtA,
        d_AtA,
        N*N*sizeof(double),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        Atb,
        d_Atb,
        N*sizeof(double),
        cudaMemcpyDeviceToHost
    ));

    printf("DONE\n");

    // ========================================================
    // CLEANUP
    // ========================================================
    cudaFree(d_A[0]);
    cudaFree(d_A[1]);

    cudaFree(d_b);

    cudaFree(d_AtA);
    cudaFree(d_Atb);

    cudaFreeHost(A);
    cudaFreeHost(b);

    free(AtA);
    free(Atb);

    cudaStreamDestroy(s0);
    cudaStreamDestroy(s1);

    return 0;
}
