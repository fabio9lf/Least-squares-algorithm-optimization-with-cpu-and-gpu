#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

/* ============================================================
 * OTTIMIZZAZIONI APPLICATE:
 *
 * 1. TRASPOSIZIONE DI A (Cache Locality)
 *    Invece di accedere ad A[k][i] (salti in memoria per colonna),
 *    si precalcola At[i][k] = A[k][i], rendendo gli accessi
 *    interni sequenziali → meno cache miss.
 *
 * 2. BILANCIAMENTO DEL CARICO (Load Balancing)
 *    Il calcolo di AtA è triangolare: la riga i richiede N-i
 *    prodotti scalari. Distribuzione round-robin delle righe
 *    tra i thread invece di blocchi contigui → carico uniforme.
 *
 * 3. PIVOT PARZIALE nella Gauss
 *    Stabilità numerica: si cerca il massimo pivot prima di
 *    ogni eliminazione, evitando divisioni per valori piccoli.
 *
 * 4. SIMMETRIA DI AtA
 *    AtA è simmetrica → si calcola solo il triangolo superiore
 *    e si copia nell'inferiore: circa la metà del lavoro.
 *
 * 5. COPIA DIFENSIVA prima della Gauss
 *    solve_system lavora su copie di AtA e Atb, preservando
 *    i dati originali (il codice originale li distruggeva).
 * ============================================================ */

typedef struct {
    double **At;   /* Trasposta di A: At[i][k] = A[k][i] */
    double  *b;
    double **AtA;
    double  *Atb;
    int M, N;
    int thread_id;
    int n_threads;
} thread_data_t;

/* ----------------------------------------------------------
 * Thread: calcola le righe assegnate con distribuzione
 * round-robin per bilanciare il carico non uniforme.
 * ---------------------------------------------------------- */
void *compute_ata_parallel(void *arg) {
    thread_data_t *d = (thread_data_t *)arg;
    int M = d->M, N = d->N;

    /* Ogni thread prende le righe i ≡ thread_id (mod n_threads) */
    for (int i = d->thread_id; i < N; i += d->n_threads) {

        /* Atb[i] = At[i] · b  (accessi sequenziali grazie alla trasposta) */
        double sum_b = 0.0;
        for (int k = 0; k < M; k++)
            sum_b += d->At[i][k] * d->b[k];
        d->Atb[i] = sum_b;

        /* Triangolo superiore di AtA: j >= i
         * Sfrutta la simmetria: AtA[j][i] = AtA[i][j]          */
        for (int j = i; j < N; j++) {
            double sum_A = 0.0;
            for (int k = 0; k < M; k++)
                sum_A += d->At[i][k] * d->At[j][k];
            d->AtA[i][j] = sum_A;
            d->AtA[j][i] = sum_A;   /* simmetria */
        }
    }
    return NULL;
}

/* ----------------------------------------------------------
 * Eliminazione di Gauss con pivot parziale.
 * Lavora su copie di AtA e Atb per non distruggere i dati.
 * ---------------------------------------------------------- */
void solve_system(double **AtA, double *Atb, double *x, int N) {
    /* Copia difensiva */
    double **G = malloc(N * sizeof(double *));
    double  *g = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        G[i] = malloc(N * sizeof(double));
        for (int j = 0; j < N; j++) G[i][j] = AtA[i][j];
        g[i] = Atb[i];
    }

    /* Eliminazione con pivot parziale */
    for (int i = 0; i < N; i++) {
        /* Cerca il pivot massimo nella colonna i */
        int pivot = i;
        double max_val = fabs(G[i][i]);
        for (int k = i + 1; k < N; k++) {
            if (fabs(G[k][i]) > max_val) {
                max_val = fabs(G[k][i]);
                pivot = k;
            }
        }
        if (max_val < 1e-14) {
            fprintf(stderr, "Errore: matrice singolare o quasi-singolare.\n");
            exit(EXIT_FAILURE);
        }
        /* Scambia riga i con la riga pivot */
        if (pivot != i) {
            double *tmp = G[i]; G[i] = G[pivot]; G[pivot] = tmp;
            double  tv  = g[i]; g[i] = g[pivot]; g[pivot] = tv;
        }
        /* Eliminazione */
        for (int k = i + 1; k < N; k++) {
            double factor = G[k][i] / G[i][i];
            for (int j = i; j < N; j++) G[k][j] -= factor * G[i][j];
            g[k] -= factor * g[i];
        }
    }

    /* Back-substitution */
    for (int i = N - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < N; j++) sum += G[i][j] * x[j];
        x[i] = (g[i] - sum) / G[i][i];
    }

    for (int i = 0; i < N; i++) free(G[i]);
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

    /* --- Allocazione e inizializzazione di A e b --- */
    srand(42);
    double **A = malloc(M * sizeof(double *));
    for (int i = 0; i < M; i++) {
        A[i] = malloc(N * sizeof(double));
        for (int j = 0; j < N; j++) A[i][j] = (double)(rand() % 10 + 1);
    }
    double *b = malloc(M * sizeof(double));
    for (int i = 0; i < M; i++) b[i] = (double)(rand() % 10 + 1);

    /* --- Trasposta At[i][k] = A[k][i] (cache-friendly) --- */
    double **At = malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        At[i] = malloc(M * sizeof(double));
        for (int k = 0; k < M; k++) At[i][k] = A[k][i];
    }

    double **AtA = malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) AtA[i] = calloc(N, sizeof(double));
    double *Atb = calloc(N, sizeof(double));
    double *x   = malloc(N * sizeof(double));

    pthread_t      *threads = malloc(n_threads * sizeof(pthread_t));
    thread_data_t  *t_data  = malloc(n_threads * sizeof(thread_data_t));

    /* --- Misurazione parte parallela --- */
    struct timespec ts, te;
    clock_gettime(CLOCK_MONOTONIC, &ts);

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
    for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &te);
    /* ----------------------------------- */

    solve_system(AtA, Atb, x, N);

    double elapsed = (te.tv_sec - ts.tv_sec)
                   + (te.tv_nsec - ts.tv_nsec) / 1e9;
    printf("Tempo SOLO Parallelo: %.2f s\n", elapsed);
    if (N >= 3)
        printf("Primi 3 valori di x: %f, %f, %f\n", x[0], x[1], x[2]);

    /* --- Pulizia --- */
    for (int i = 0; i < M; i++) free(A[i]); free(A);
    for (int i = 0; i < N; i++) free(At[i]); free(At);
    for (int i = 0; i < N; i++) free(AtA[i]); free(AtA);
    free(b); free(x); free(Atb);
    free(threads); free(t_data);

    return 0;
}