#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

// ═══════════════════════════════════════════════════════════════════════════
//  LAYOUT MEMORIA — Column-major
//
//  Una matrice M×N è memorizzata in un singolo array contiguo di M*N float.
//  L'elemento (i,j) si trova all'indice  i + j*M  (colonna j parte da j*M).
//
//  Esempio 3×2:
//    Matrice:          Array in memoria:
//    [ a b ]           [ a, c, e,   b, d, f ]
//    [ c d ]             col 0       col 1
//    [ e f ]
//
//  Vantaggio per Householder: il loop hot itera su i a j fisso
//  → scorre la colonna j che è CONTIGUA in memoria → zero cache miss.
// ═══════════════════════════════════════════════════════════════════════════
#define MAT(data, i, j, M)  (data)[(i) + (j)*(M)]


// ═══════════════════════════════════════════════════════════════════════════
//  FASI DEL THREAD POOL
//
//  Ogni passo k della fattorizzazione QR è diviso in 3 fasi parallele.
//  Il main imposta t_data[t].phase prima di ogni barrier_start, così i
//  worker sanno cosa eseguire quando si svegliano.
//
//  Flusso per ogni passo k:
//
//    PHASE_NORM        → riduzione parallela: calcola ||col_k||²
//    PHASE_HOUSEHOLDER → aggiorna in parallelo le colonne k..N-1 di R
//    PHASE_APPLY_Y     → riduzione parallela: calcola dot(v, y[k..M-1])
//    PHASE_STOP        → segnale di uscita ai worker (fine algoritmo)
// ═══════════════════════════════════════════════════════════════════════════
#define PHASE_HOUSEHOLDER  0
#define PHASE_APPLY_Y      1
#define PHASE_NORM         2
#define PHASE_STOP        -1


// ═══════════════════════════════════════════════════════════════════════════
//  STRUTTURA DEL THREAD POOL  —  pool_thread_data_t
//
//  Un array t_data[Nthreads] è condiviso tra main e worker.
//  Ogni slot appartiene a un thread specifico (thread_id = indice nell'array).
//
//  Campi scritti SOLO dal main prima della barrier:
//    k, active_threads, phase
//
//  Campi scritti SOLO dal thread proprietario dopo la barrier:
//    partial  ← risultato parziale della riduzione (somma parziale di norm o dot)
//              Il main lo legge DOPO barrier_end, quindi non c'è race condition.
//
//  __attribute__((aligned(64))): forza ogni slot a iniziare su un confine
//  di cache line (64 byte). Senza questo, t_data[0] e t_data[1] potrebbero
//  condividere la stessa cache line → false sharing → ogni scrittura di
//  `partial` da parte di un thread invalida la cache line degli altri.
// ═══════════════════════════════════════════════════════════════════════════
typedef struct {
    float* R;              // puntatore all'array piatto della matrice (column-major)
    float* v;              // vettore di Householder corrente (condiviso in lettura)
    float* y;              // vettore del termine noto (aggiornato ogni passo k)
    int     M, N;           // dimensioni della matrice

    volatile int k;               // passo corrente della fattorizzazione
    volatile int active_threads;  // thread effettivi per questo passo (≤ Nthreads)
    volatile int phase;           // quale operazione eseguire (vedi PHASE_*)

    int thread_id;          // indice di questo thread nell'array t_data
    int total_threads;      // numero totale di thread nel pool

    pthread_barrier_t* barrier_start;  // main → worker: "inizia la fase"
    pthread_barrier_t* barrier_end;    // worker → main: "fase completata"

    // Risultato parziale della riduzione parallela.
    // In PHASE_NORM    : somma parziale di col_k[i]²
    // In PHASE_APPLY_Y : somma parziale di v[i]*y[i]
    // Il main somma tutti i partial[] dopo barrier_end per ottenere il totale.
    float partial;

    char _pad[8]; // padding extra (la struct è già >64 byte, l'allineamento fa il grosso)
} __attribute__((aligned(64))) pool_thread_data_t;


// ═══════════════════════════════════════════════════════════════════════════
//  WORKER DEL THREAD POOL  —  pool_worker()
//
//  Ogni thread secondario (id 1..Nthreads-1) esegue questo loop infinito.
//  Il thread 0 è il main, che esegue lo stesso lavoro ma nel loop di
//  qr_factorization() direttamente (evita una chiamata di funzione e
//  permette al main di fare il lavoro di coordinazione tra le fasi).
//
//  Ciclo di vita per ogni passo k:
//    1. barrier_start: il thread si blocca finché il main non ha impostato
//       phase/k/active_threads e ha chiamato barrier_start lui stesso.
//    2. Il thread legge `phase` e sceglie cosa fare.
//    3. barrier_end: il thread segnala di aver finito la sua porzione.
//       Il main raccoglie i risultati solo dopo che tutti hanno raggiunto
//       questa barrier.
// ═══════════════════════════════════════════════════════════════════════════
void* pool_worker(void* arg) {
    pool_thread_data_t* d = (pool_thread_data_t*)arg;

    while (1) {
        // ── Attende il segnale del main per iniziare la prossima fase ──
        pthread_barrier_wait(d->barrier_start);

        int phase = d->phase;
        if (phase == PHASE_STOP) break;  // fine algoritmo, esci dal loop

        // Copia locale delle variabili condivise (evita accessi ripetuti
        // alla struttura condivisa nel loop hot — il compilatore può
        // tenerle nei registri).
        int     k   = d->k;
        int     M   = d->M;
        int     N   = d->N;
        int     tid = d->thread_id;
        int     nt  = d->active_threads;
        float* R   = d->R;
        float* v   = d->v;
        float* y   = d->y;

        if (phase == PHASE_HOUSEHOLDER) {
            // ── FASE HOUSEHOLDER: aggiorna le colonne k..N-1 di R ────────
            //
            // Ogni thread processa un sottoinsieme di colonne [start_col, end_col).
            // Le colonne sono indipendenti tra loro → nessuna race condition.
            //
            // Per ogni colonna j: R[:,j] ← R[:,j] - 2·v·(vᵀ·R[:,j])
            // Con layout column-major, &MAT(R,k,j,M) punta all'inizio
            // del segmento utile della colonna j → accesso sequenziale.
            if (tid < nt) {
                int cols            = N - k;
                int cols_per_thread = cols / nt;
                int start_col       = k + tid * cols_per_thread;
                // L'ultimo thread prende anche le colonne rimaste dalla divisione intera
                int end_col         = (tid == nt - 1) ? N : start_col + cols_per_thread;

                for (int j = start_col; j < end_col; j++) {
                    float* col = &MAT(R, k, j, M); // puntatore diretto alla colonna j da riga k
                    float dot  = 0.0;
                    for (int i = 0; i < M - k; i++)
                        dot += v[k + i] * col[i];
                    for (int i = 0; i < M - k; i++)
                        col[i] -= 2.0 * v[k + i] * dot;
                }
            }

        } else if (phase == PHASE_APPLY_Y) {
            // ── FASE APPLY_Y: riduzione parallela di dot(v, y[k..M-1]) ──
            //
            // Invece di calcolare tutto il prodotto scalare sequenzialmente,
            // ogni thread calcola la somma parziale del suo segmento di righe.
            // Il main raccoglie i parziali dopo barrier_end.
            //
            // Ogni thread scrive SOLO nel suo slot d->partial → no race condition.
            int len             = M - k;
            int rows_per_thread = (len + nt - 1) / nt; // divisione con arrotondamento su
            int start_row       = k + tid * rows_per_thread;
            int end_row         = start_row + rows_per_thread;
            if (end_row > M) end_row = M; // clamp all'ultimo elemento valido

            float dot = 0.0;
            if (start_row < M) {
                for (int i = start_row; i < end_row; i++)
                    dot += v[i] * y[i];
            }
            d->partial = dot; // scrivo solo nel mio slot, nessun altro thread tocca questo campo

        } else if (phase == PHASE_NORM) {
            // ── FASE NORM: riduzione parallela di ||col_k[k..M-1]||² ────
            //
            // Stessa tecnica di PHASE_APPLY_Y: ogni thread somma il quadrato
            // degli elementi del suo segmento. Il main somma i parziali e
            // prende la radice quadrata per ottenere norm_x.
            float* col_k       = &MAT(R, k, k, M); // colonna k da riga k (contigua)
            int     len         = M - k;
            int rows_per_thread = (len + nt - 1) / nt;
            int start_row       = tid * rows_per_thread;
            int end_row         = start_row + rows_per_thread;
            if (end_row > len) end_row = len;

            float s = 0.0;
            for (int i = start_row; i < end_row; i++)
                s += col_k[i] * col_k[i];
            d->partial = s;
        }

        // ── Segnala al main che questa fase è completata ───────────────
        pthread_barrier_wait(d->barrier_end);
    }
    return NULL;
}


// ═══════════════════════════════════════════════════════════════════════════
//  FATTORIZZAZIONE QR CON THREAD POOL
//
//  Algoritmo di Householder: ad ogni passo k si costruisce un riflettore
//  H_k = I - 2·v·vᵀ che azzera gli elementi sotto la diagonale della
//  colonna k. Applicando H_0, H_1, ..., H_{N-1} si ottiene:
//    Q^T · A = R   (R triangolare superiore)
//    Q^T · b = y   (y aggiornato ad ogni passo)
//
//  Struttura di ogni passo k:
//
//    [SEQUENZIALE] PHASE_NORM:
//      Calcola norm_x = ||R[k:M, k]|| in parallelo (riduzione).
//
//    [SEQUENZIALE] Costruisce il vettore v:
//      v[k]   = R[k][k] - alpha,  alpha = -sign(R[k][k])·norm_x
//      v[k+i] = R[k+i][k]  per i > 0
//      Normalizza v usando norm_v calcolata ALGEBRICAMENTE (no loop):
//        norm_v² = 2·norm_x·|v[k]|
//
//    [PARALLELO] PHASE_HOUSEHOLDER:
//      Per ogni colonna j in [k, N): R[:,j] ← R[:,j] - 2·v·(vᵀ·R[:,j])
//      Distribuito tra i thread per colonne.
//
//    [PARALLELO] PHASE_APPLY_Y:
//      Calcola dot(v, y[k:M]) in parallelo, poi aggiorna y sequenzialmente.
//      (L'update di y non è parallelizzabile: y[i] -= 2·v[i]·doty dipende
//       da doty globale, ma doty si calcola in parallelo.)
//
//  Il main è il thread 0: esegue il suo slice di lavoro tra le barrier
//  invece di stare idle ad aspettare i worker.
//  Si creano solo Nthreads-1 thread secondari (id 1..Nthreads-1).
//  Le barrier hanno count=Nthreads (main incluso).
// ═══════════════════════════════════════════════════════════════════════════
void qr_factorization(float* R, float* y, int M, int N, int Nthreads) {

    int workers = Nthreads - 1; // thread secondari da creare (il main è il thread 0)

    pthread_t*          threads = malloc(workers  * sizeof(pthread_t));
    pool_thread_data_t* t_data  = malloc(Nthreads * sizeof(pool_thread_data_t));
    pthread_barrier_t*  b_start = malloc(sizeof(pthread_barrier_t));
    pthread_barrier_t*  b_end   = malloc(sizeof(pthread_barrier_t));
    float*             v       = malloc(M * sizeof(float)); // vettore Householder corrente

    if (!threads || !t_data || !b_start || !b_end || !v) {
        fprintf(stderr, "Errore di allocazione nel thread pool\n");
        exit(1);
    }

    // Le barrier hanno count = Nthreads perché anche il main partecipa
    pthread_barrier_init(b_start, NULL, Nthreads);
    pthread_barrier_init(b_end,   NULL, Nthreads);

    // Inizializza tutti gli slot di t_data (incluso lo slot 0 del main)
    for (int t = 0; t < Nthreads; t++) {
        t_data[t].R              = R;
        t_data[t].v              = v;
        t_data[t].y              = y;
        t_data[t].M              = M;
        t_data[t].N              = N;
        t_data[t].k              = 0;
        t_data[t].active_threads = Nthreads;
        t_data[t].phase          = PHASE_HOUSEHOLDER;
        t_data[t].thread_id      = t;
        t_data[t].total_threads  = Nthreads;
        t_data[t].barrier_start  = b_start;
        t_data[t].barrier_end    = b_end;
        t_data[t].partial        = 0.0;
        //t_data[t].stop           = 0;
    }

    // Crea solo i thread secondari (id 1..Nthreads-1)
    // Il main usa t_data[0] ma NON chiama pool_worker: lo fa inline sotto.
    for (int t = 0; t < workers; t++)
        pthread_create(&threads[t], NULL, pool_worker, &t_data[t + 1]);

    // ─────────────────────────────────────────────────────────────────────
    //  LOOP PRINCIPALE — passo k della fattorizzazione QR
    //  Il main esegue il lavoro del thread 0 tra le coppie di barrier.
    // ─────────────────────────────────────────────────────────────────────
    for (int k = 0; k < N && k < M; k++) {

        float* col_k = &MAT(R, k, k, M); // puntatore alla colonna k da riga k
        int     len   = M - k;            // numero di righe ancora attive
        int     nt    = Nthreads;

        // ── FASE 1: calcolo parallelo di norm_x = ||R[k:M, k]|| ──────────
        //
        // Ogni thread somma i quadrati del suo segmento di col_k e salva
        // il risultato parziale in t_data[t].partial.
        // Dopo barrier_end, il main somma tutti i partial[] e prende sqrt.
        for (int t = 0; t < Nthreads; t++) {
            t_data[t].k              = k;
            t_data[t].active_threads = nt;
            t_data[t].phase          = PHASE_NORM;
        }
        pthread_barrier_wait(b_start); // sblocca i worker per PHASE_NORM

        // Contributo del main (thread 0): primo segmento di col_k
        {
            int rows_per_thread = (len + nt - 1) / nt;
            int end_row = rows_per_thread < len ? rows_per_thread : len;
            float s = 0.0;
            for (int i = 0; i < end_row; i++)
                s += col_k[i] * col_k[i];
            t_data[0].partial = s;
        }

        pthread_barrier_wait(b_end); // attende che tutti i worker abbiano finito PHASE_NORM

        // Raccolta dei parziali: O(Nthreads) — trascurabile rispetto a O(M)
        float norm_x = 0.0;
        for (int t = 0; t < Nthreads; t++)
            norm_x += t_data[t].partial;
        norm_x = sqrt(norm_x);

        // ── Costruisce il vettore di Householder v ────────────────────────
        //
        // v è definito come:
        //   v[k]   = R[k][k] - alpha
        //   v[k+i] = R[k+i][k]   per i = 1..M-k-1
        //   v[i]   = 0            per i < k
        //
        // alpha = -sign(R[k][k]) · norm_x  (scelta del segno per stabilità numerica:
        //   evita cancellazione catastrofica quando R[k][k] ≈ norm_x)
        for (int i = 0; i < k; i++) v[i] = 0.0;

        float rkk   = col_k[0];                      // R[k][k] (primo elemento della colonna)
        float alpha = (rkk > 0) ? -norm_x : norm_x;
        v[k]         = rkk - alpha;
        for (int i = 1; i < len; i++)
            v[k + i] = col_k[i];

        // Normalizzazione di v: v ← v / ||v||
        //
        // OTTIMIZZAZIONE ALGEBRICA: ||v||² si calcola senza loop.
        //   ||v||² = (R[k][k] - alpha)² + Σ_{i>k} R[i][k]²
        //          = R[k][k]² - 2·R[k][k]·alpha + alpha² + (norm_x² - R[k][k]²)
        //          = 2·norm_x² - 2·R[k][k]·alpha
        //          = 2·norm_x·(norm_x - R[k][k]·sign)
        //          = 2·norm_x·|v[k]|          ← formula chiusa, O(1)
        //
        // Risparmio: elimina un intero loop O(M) presente nella versione originale.
        float norm_v = sqrt(2.0 * norm_x * fabs(v[k]));
        if (norm_v > 1e-12)
            for (int i = k; i < M; i++)
                v[i] /= norm_v;

        // ── FASE 2: aggiornamento parallelo delle colonne di R ────────────
        //
        // Applica la riflessione di Householder a tutte le colonne j in [k, N):
        //   R[:,j] ← R[:,j] - 2·v·(vᵀ·R[:,j])
        //
        // Le colonne sono indipendenti → distribuzione perfetta tra thread.
        // active = min(N-k, Nthreads): se ci sono meno colonne che thread,
        // alcuni thread non ricevono lavoro (tid >= active) e aspettano idle.
        int cols   = N - k;
        int active = (cols < Nthreads) ? cols : Nthreads;

        for (int t = 0; t < Nthreads; t++) {
            t_data[t].active_threads = active;
            t_data[t].phase          = PHASE_HOUSEHOLDER;
        }
        pthread_barrier_wait(b_start); // sblocca i worker per PHASE_HOUSEHOLDER

        // Contributo del main (thread 0): prime cols/active colonne
        {
            int cols_per_thread = cols / active;
            int start_col = k;  // thread 0 parte dalla colonna k
            int end_col   = (active == 1) ? N : start_col + cols_per_thread;

            for (int j = start_col; j < end_col; j++) {
                float* col = &MAT(R, k, j, M); // accesso sequenziale alla colonna j
                float dot  = 0.0;
                for (int i = 0; i < len; i++) dot += v[k + i] * col[i];
                for (int i = 0; i < len; i++) col[i] -= 2.0 * v[k + i] * dot;
            }
        }

        pthread_barrier_wait(b_end); // attende fine PHASE_HOUSEHOLDER

        // ── FASE 3: applica Householder a y — riduzione parallela ─────────
        //
        // y deve essere aggiornata come:  y[k:M] ← y[k:M] - 2·(vᵀ·y[k:M])·v[k:M]
        //
        // Il prodotto scalare doty = vᵀ·y[k:M] si calcola in parallelo
        // (riduzione come in PHASE_NORM): ogni thread contribuisce con la
        // somma parziale del suo segmento → partial[t].
        //
        // L'update y[i] -= 2·v[i]·doty rimane sequenziale nel main perché
        // dipende da doty globale, ma il costo della riduzione (la parte
        // più pesante con M grande) è ora distribuito.
        for (int t = 0; t < Nthreads; t++) {
            t_data[t].active_threads = Nthreads;
            t_data[t].phase          = PHASE_APPLY_Y;
        }
        pthread_barrier_wait(b_start); // sblocca i worker per PHASE_APPLY_Y

        // Contributo del main (thread 0): primo segmento di y[k..M-1]
        {
            int rows_per_thread = (len + nt - 1) / nt;
            int start_row = k;
            int end_row   = start_row + rows_per_thread;
            if (end_row > M) end_row = M;
            float dot = 0.0;
            for (int i = start_row; i < end_row; i++)
                dot += v[i] * y[i];
            t_data[0].partial = dot;
        }

        pthread_barrier_wait(b_end); // attende fine PHASE_APPLY_Y

        // Raccolta dei parziali e update finale di y
        float doty = 0.0;
        for (int t = 0; t < Nthreads; t++) doty += t_data[t].partial;
        for (int i = k; i < M; i++) y[i] -= 2.0 * v[i] * doty;
        // Nota: questo loop O(M) è sequenziale ma non parallelizzabile ulteriormente
        // perché ogni y[i] dipende da doty che era sconosciuto fino a un momento fa.
    }

    // ── Shutdown del thread pool ──────────────────────────────────────────
    // Imposta phase=STOP e sblocca i worker: useciranno dal loro while(1).
    for (int t = 0; t < Nthreads; t++) t_data[t].phase = PHASE_STOP;
    pthread_barrier_wait(b_start); // ultimo sblocco: i worker vedono PHASE_STOP ed escono
    for (int t = 0; t < workers; t++) pthread_join(threads[t], NULL);

    pthread_barrier_destroy(b_start);
    pthread_barrier_destroy(b_end);
    free(b_start); free(b_end);
    free(threads); free(t_data); free(v);
}


// ═══════════════════════════════════════════════════════════════════════════
//  BACK SUBSTITUTION  —  risolve Rx = y con R triangolare superiore
//
//  Algoritmo standard O(N²): per i = N-1 .. 0:
//    x[i] = (y[i] - Σ_{j>i} R[i][j]·x[j]) / R[i][i]
//
//  Con layout column-major, R[i][j] = data[i + j*M]: accesso per riga
//  (i fisso, j varia) è non sequenziale, ma la back substitution è
//  O(N²) contro O(M·N²) della fattorizzazione → non è il bottleneck.
// ═══════════════════════════════════════════════════════════════════════════
void back_substitution(float* R, float* y, float* x, int N, int M) {
    for (int i = N - 1; i >= 0; i--) {
        float s = 0.0;
        for (int j = i + 1; j < N; j++)
            s += MAT(R, i, j, M) * x[j];
        float diag = MAT(R, i, i, M);
        x[i] = (fabs(diag) > 1e-12) ? (y[i] - s) / diag : 0.0;
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  LEAST SQUARES  —  risolve min ||Ax - b||₂
//
//  Procedura:
//    1. Copia A → R  (non modifichiamo la matrice originale)
//    2. Fattorizza R = QR tramite riflessioni di Householder
//       (R viene sovrascritta con la matrice R triangolare superiore)
//    3. Risolve Rx = y  (y = Q^T·b, aggiornato durante la fattorizzazione)
// ═══════════════════════════════════════════════════════════════════════════
void least_squares(float* A, float* b, float* x, int M, int N, int Nthreads) {
    // Alloca R come copia di A: una sola malloc per M*N float contigui
    float* R = malloc(M * N * sizeof(float));
    float* y = malloc(M     * sizeof(float));
    if (!R || !y) { fprintf(stderr, "Errore allocazione R/y\n"); exit(1); }

    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++)
            MAT(R, i, j, M) = MAT(A, i, j, M);
    for (int i = 0; i < M; i++) y[i] = b[i];

    qr_factorization(R, y, M, N, Nthreads);
    back_substitution(R, y, x, N, M);

    free(R); free(y);
}


// ═══════════════════════════════════════════════════════════════════════════
//  MAIN
//
//  Uso: ./qr M N [Nthreads]
//    M        : numero di righe    (M > N richiesto per sistema sovradeterminato)
//    N        : numero di colonne
//    Nthreads : thread da usare (default 1; il main conta come thread 0)
//
//  Genera A e b casuali, risolve il sistema ai minimi quadrati,
//  stampa la soluzione x e il tempo wall-clock con clock_gettime.
//
//  Nota: clock() misurerebbe il tempo CPU cumulativo di tutti i thread
//  (valore gonfiato con molti thread). clock_gettime(CLOCK_MONOTONIC)
//  misura il tempo reale trascorso — corretto per benchmark paralleli.
// ═══════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    if (argc < 3) {
        fprintf(stderr, "Uso: %s M N [Nthreads]\n", argv[0]);
        return 1;
    }

    int M        = atoi(argv[1]);
    int N        = atoi(argv[2]);
    int Nthreads = (argc >= 4) ? atoi(argv[3]) : 1;

    if (N >= M)       { fprintf(stderr, "M deve essere maggiore di N\n"); return 1; }
    if (Nthreads < 1) { fprintf(stderr, "Nthreads deve essere >= 1\n");   return 1; }

    srand(time(NULL));

    // Alloca A e b come array piatti column-major (una malloc ciascuno)
    float* A = malloc(M * N * sizeof(float));
    float* b = malloc(M     * sizeof(float));
    float* x = malloc(N     * sizeof(float));
    if (!A || !b || !x) { fprintf(stderr, "Errore allocazione\n"); return 1; }

    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++)
            MAT(A, i, j, M) = rand() % 100 + 1;
    for (int i = 0; i < M; i++)
        b[i] = rand() % 100 + 1;

    least_squares(A, b, x, M, N, Nthreads);

    /*for (int i = 0; i < N; i++)
        printf("x[%d] = %f\n", i, x[i]);
*/
    free(A); free(b); free(x);

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    float elapsed = (ts_end.tv_sec  - ts_start.tv_sec)
                   + (ts_end.tv_nsec - ts_start.tv_nsec) * 1e-9;
    printf("tempo di esecuzione: %.6f s\n", elapsed);
    return 0;
}