#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <chrono>

using namespace std;

/* ============================================================
 * OTTIMIZZAZIONI APPLICATE (HPC - High Performance Computing):
 *
 * 1. ARRAY 1D CONTIGUI (Flat Arrays)
 * Nessun uso di puntatori doppi (double **). La memoria è piatta,
 * garantendo località spaziale perfetta e prefetching hardware.
 * L'accesso Matrice[riga][col] diventa Matrice[riga * LARGHEZZA + col].
 *
 * 2. CACHE BLOCKING (TILING)
 * Per evitare i Capacity Miss (righe troppo grandi per la cache L1 da 32KB),
 * le moltiplicazioni matriciali in compute_ata e solve_system 
 * sono state divise in blocchi da 256 elementi (~2 KB).
 *
 * 3. TRASPOSIZIONE DI A (Cache Locality)
 * Precalcoliamo At[i * M + k] = A[k * N + i] per avere accessi
 * sequenziali durante il prodotto riga per riga.
 *
 * 4. BILANCIAMENTO DEL CARICO (Load Balancing)
 * Distribuzione round-robin delle righe tra i thread.
 * ============================================================ */

typedef struct {
    double *At;    /* Array 1D: dimensione N x M */
    double *b;
    double *AtA;   /* Array 1D: dimensione N x N */
    double *Atb;
    int M, N;
    int thread_id;
    int n_threads;
} thread_data_t;

/* ----------------------------------------------------------
 * FASE PARALLELA: Calcolo di AtA e Atb
 * ---------------------------------------------------------- */
void *compute_ata_parallel(void *arg) {
    thread_data_t *d = (thread_data_t *)arg;
    int M = d->M, N = d->N;
    
    // Dimensione del blocco ottimizzata per L1 Cache (256 double = 2 KB)
    int BLOCK = 256; 

    for (int i = d->thread_id; i < N; i += d->n_threads) {

        /* 1. Calcolo di Atb (Prodotto Vettore-Vettore, sequenziale) */
        double sum_b = 0.0;
        for (int k = 0; k < M; k++) {
            sum_b += d->At[i * M + k] * d->b[k];
        }
        d->Atb[i] = sum_b;

        /* 2. Calcolo di AtA con CACHE BLOCKING (Tiling) */
        for (int k_b = 0; k_b < M; k_b += BLOCK) {
            int k_end = (k_b + BLOCK < M) ? k_b + BLOCK : M;
            
            for (int j = i; j < N; j++) {
                double sum_A = 0.0;
                // Operiamo solo sul blocco che entra comodamente in cache
                for (int k = k_b; k < k_end; k++) {
                    sum_A += d->At[i * M + k] * d->At[j * M + k];
                }
                d->AtA[i * N + j] += sum_A; 
            }
        }
        
        /* 3. Simmetria: Copiamo il triangolo superiore in quello inferiore */
        for (int j = i + 1; j < N; j++) {
            d->AtA[j * N + i] = d->AtA[i * N + j];
        }
    }
    return NULL;
}

/* ----------------------------------------------------------
 * FASE SEQUENZIALE: Eliminazione di Gauss e Back-Substitution
 * ---------------------------------------------------------- */
void solve_system(double *AtA, double *Atb, double *x, int N) {
    /* Copia difensiva in memoria contigua per preservare i dati originali */
    double *G = (double *)malloc(N * N * sizeof(double));
    double *g = (double *)malloc(N * sizeof(double));
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            G[i * N + j] = AtA[i * N + j];
        }
        g[i] = Atb[i];
    }

    // Array di supporto per il bloccaggio dell'eliminazione
    double *factors = (double *)malloc(N * sizeof(double));

    /* Eliminazione con pivot parziale */
    for (int i = 0; i < N; i++) {
        /* Cerca il pivot massimo nella colonna i */
        int pivot = i;
        double max_val = fabs(G[i * N + i]);
        for (int k = i + 1; k < N; k++) {
            if (fabs(G[k * N + i]) > max_val) {
                max_val = fabs(G[k * N + i]);
                pivot = k;
            }
        }
        if (max_val < 1e-14) {
            fprintf(stderr, "Errore: matrice singolare o quasi-singolare.\n");
            exit(EXIT_FAILURE);
        }
        
        /* Scambia fisicamente l'intera riga i con la riga pivot */
        if (pivot != i) {
            for(int j = i; j < N; j++) {
                double tmp = G[i * N + j];
                G[i * N + j] = G[pivot * N + j];
                G[pivot * N + j] = tmp;
            }
            double tv = g[i]; 
            g[i] = g[pivot]; 
            g[pivot] = tv;
        }

        /* ELIMINAZIONE CON CACHE BLOCKING */
        // 1. Precalcoliamo i fattori di annullamento per la colonna
        for (int k = i + 1; k < N; k++) {
            factors[k] = G[k * N + i] / G[i * N + i];
            g[k] -= factors[k] * g[i]; 
        }

        // 2. Aggiorniamo il resto della matrice scorrendo a "colonne larghe" (Blocchi)
        int BLOCK = 256;
        for (int j_b = i; j_b < N; j_b += BLOCK) {
            int j_end = (j_b + BLOCK < N) ? j_b + BLOCK : N;
            
            for (int k = i + 1; k < N; k++) {
                double factor = factors[k];
                // Lavoriamo solo sul segmento in L1 Cache
                for (int j = j_b; j < j_end; j++) {
                    G[k * N + j] -= factor * G[i * N + j];
                }
            }
        }
    }

    /* Back-substitution */
    for (int i = N - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < N; j++) {
            sum += G[i * N + j] * x[j];
        }
        x[i] = (g[i] - sum) / G[i * N + i];
    }

    free(factors);
    free(G);
    free(g);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Uso: %s <M> <N> <n_thread>\n", argv[0]);
        return 1;
    }
    int M        = atoi(argv[1]);
    int N        = atoi(argv[2]);
    int n_threads = atoi(argv[3]);
    if (n_threads > N) n_threads = N;

    /* --- Allocazione 1D CONTIGUA --- */
    srand(42);
    double *A = (double *)malloc(M * N * sizeof(double));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (double)(rand() % 10 + 1);
        }
    }
    
    double *b = (double *)malloc(M * sizeof(double));
    for (int i = 0; i < M; i++) b[i] = (double)(rand() % 10 + 1);

    /* --- Trasposta 1D: At di dimensione N x M (con Cache Blocking Base) --- */
    double *At = (double *)malloc(N * M * sizeof(double));
    int BLOCK_TRANS = 32;
    for (int i = 0; i < N; i += BLOCK_TRANS) {
        for (int k = 0; k < M; k += BLOCK_TRANS) {
            int max_i = (i + BLOCK_TRANS < N) ? i + BLOCK_TRANS : N;
            int max_k = (k + BLOCK_TRANS < M) ? k + BLOCK_TRANS : M;
            
            for (int ii = i; ii < max_i; ii++) {
                for (int kk = k; kk < max_k; kk++) {
                    At[ii * M + kk] = A[kk * N + ii];
                }
            }
        }
    }

    /* Allocazioni Finali Array 1D */
    double *AtA = (double *)calloc(N * N, sizeof(double));
    double *Atb = (double *)calloc(N, sizeof(double));
    double *x   = (double *)malloc(N * sizeof(double));

    pthread_t      *threads = (pthread_t *)malloc(n_threads * sizeof(pthread_t));
    thread_data_t  *t_data  = (thread_data_t *)malloc(n_threads * sizeof(thread_data_t));

    /* --- INIZIO CRONOMETRO (Misurazione SOLO Parallela) --- */
    auto start_parallel = chrono::steady_clock::now();

    for (int i = 0; i < n_threads; i++) {
        t_data[i].At       = At;
        t_data[i].b        = b;
        t_data[i].AtA      = AtA;
        t_data[i].Atb      = Atb;
        t_data[i].M        = M;
        t_data[i].N        = N;
        t_data[i].thread_id  = i;
        t_data[i].n_threads  = n_threads;
        pthread_create(&threads[i], NULL, compute_ata_parallel, &t_data[i]);
    }
    
    for (int i = 0; i < n_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    auto end_parallel = chrono::steady_clock::now();
    /* --- FINE CRONOMETRO --- */

    solve_system(AtA, Atb, x, N);

    chrono::duration<double, std::milli> duration = end_parallel - start_parallel;
    printf("Tempo SOLO Parallelo: %.2f ms\n", duration.count());
    
    if (N >= 3)
        printf("Primi 3 valori di x: %f, %f, %f\n", x[0], x[1], x[2]);

    /* --- Pulizia Memory Leaks --- */
    free(A);
    free(At);
    free(AtA);
    free(b); 
    free(x); 
    free(Atb);
    free(threads); 
    free(t_data);

    return 0;
}