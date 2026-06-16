/*
 * least_squares.cpp
 *
 * Soluzione ai minimi quadrati di Ax ≈ b mediante la normale AtA·x = Atb.
 *
 * OTTIMIZZAZIONI:
 *  1. Memoria contigua (no double** con N malloc separati)
 *  2. Trasposta At costruita insieme ad A in un solo loop
 *  3. Distribuzione round-robin delle righe tra i thread (load balancing)
 *  4. Solo triangolo superiore di AtA calcolato; simmetria completata dopo join
 *  5. Tiling sul loop interno (cache blocking)
 *  6. Fix race condition: AtA[j][i] scritto solo dopo il join
 *  7. Gauss con pivot parziale su copia difensiva contigua
 *  8. Misurazione tempo con std::chrono
 *
 * Compilazione:
 *   g++ -std=c++17 -O2 -march=native -pthread -o least_squares least_squares.cpp
 * Uso:
 *   ./least_squares <M> <N> <n_thread>
 */

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <algorithm>

// Dimensione del blocco per il tiling sul loop interno (in double)
static constexpr int BLOCK = 64;

// ---------------------------------------------------------------------------
// Struttura dati passata a ogni thread
// ---------------------------------------------------------------------------
struct ThreadData {
    const double *At_data;   // trasposta contigua: At[i][k] = At_data[i*M + k]
    const double *b;
    double       *AtA_data;  // contigua: AtA[i][j] = AtA_data[i*N + j]
    double       *Atb;
    int M, N;
    int thread_id;
    int n_threads;
};

// ---------------------------------------------------------------------------
// Kernel parallelo: calcola le righe assegnate con distribuzione round-robin
// ---------------------------------------------------------------------------
void compute_ata(const ThreadData &d) {
    const int M = d.M, N = d.N;

    for (int i = d.thread_id; i < N; i += d.n_threads) {
        const double *Ati = d.At_data + static_cast<std::size_t>(i) * M;

        // Atb[i] = At[i] · b
        double sum_b = 0.0;
        for (int k = 0; k < M; ++k)
            sum_b += Ati[k] * d.b[k];
        d.Atb[i] = sum_b;

        // Triangolo superiore di AtA con tiling su k
        for (int j = i; j < N; ++j) {
            const double *Atj = d.At_data + static_cast<std::size_t>(j) * M;
            double sum_A = 0.0;
            for (int k0 = 0; k0 < M; k0 += BLOCK) {
                const int k_end = std::min(k0 + BLOCK, M);
                for (int k = k0; k < k_end; ++k)
                    sum_A += Ati[k] * Atj[k];
            }
            // Scrivo solo il triangolo superiore;
            // la copia simmetrica avviene nel main dopo il join (no race)
            d.AtA_data[static_cast<std::size_t>(i) * N + j] = sum_A;
        }
    }
}

// ---------------------------------------------------------------------------
// Eliminazione di Gauss con pivot parziale
// Lavora su una copia contigua di AtA e Atb per non distruggere i dati.
// ---------------------------------------------------------------------------
void solve_system(const double *AtA_data, const double *Atb,
                  std::vector<double> &x, int N) {
    // Copia difensiva contigua
    std::vector<double> G_data(static_cast<std::size_t>(N) * N);
    std::vector<double> g(Atb, Atb + N);
    std::vector<double *> G(N);

    for (int i = 0; i < N; ++i) {
        G[i] = G_data.data() + static_cast<std::size_t>(i) * N;
        std::memcpy(G[i], AtA_data + static_cast<std::size_t>(i) * N,
                    N * sizeof(double));
    }

    // Eliminazione in avanti
    for (int i = 0; i < N; ++i) {
        // Ricerca del pivot massimo nella colonna i
        int    pivot   = i;
        double max_val = std::fabs(G[i][i]);
        for (int k = i + 1; k < N; ++k) {
            double v = std::fabs(G[k][i]);
            if (v > max_val) { max_val = v; pivot = k; }
        }
        if (max_val < 1e-14)
            throw std::runtime_error("Matrice singolare o quasi-singolare.");

        if (pivot != i) {
            std::swap(G[i], G[pivot]);
            std::swap(g[i], g[pivot]);
        }

        for (int k = i + 1; k < N; ++k) {
            const double factor = G[k][i] / G[i][i];
            for (int j = i; j < N; ++j)
                G[k][j] -= factor * G[i][j];
            g[k] -= factor * g[i];
        }
    }

    // Back-substitution
    x.resize(N);
    for (int i = N - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < N; ++j)
            sum += G[i][j] * x[j];
        x[i] = (g[i] - sum) / G[i][i];
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::printf("Uso: %s <M> <N> <n_thread>\n", argv[0]);
        return 1;
    }

    const int M         = std::atoi(argv[1]);
    const int N         = std::atoi(argv[2]);
    const int n_threads = std::min(std::atoi(argv[3]), N);

    std::srand(42);

    // --- Allocazione contigua di A, At, b ---
    std::vector<double> A_data (static_cast<std::size_t>(M) * N);
    std::vector<double> At_data(static_cast<std::size_t>(N) * M);
    std::vector<double> b      (M);

    // Costruiamo A e At in un unico loop
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double v = static_cast<double>(std::rand() % 10 + 1);
            A_data [static_cast<std::size_t>(i) * N + j] = v;
            At_data[static_cast<std::size_t>(j) * M + i] = v;  // trasposta
        }
        b[i] = static_cast<double>(std::rand() % 10 + 1);
    }

    std::vector<double> AtA_data(static_cast<std::size_t>(N) * N, 0.0);
    std::vector<double> Atb     (N, 0.0);

    // --- Lanci dei thread ---
    std::vector<ThreadData>  t_data (n_threads);
    std::vector<std::thread> threads(n_threads);

    // ── Inizio misurazione ──────────────────────────────────────────────────
    const auto t_start = std::chrono::steady_clock::now();

    for (int i = 0; i < n_threads; ++i) {
        t_data[i] = {
            At_data.data(), b.data(),
            AtA_data.data(), Atb.data(),
            M, N, i, n_threads
        };
        threads[i] = std::thread(compute_ata, std::cref(t_data[i]));
    }
    for (auto &t : threads) t.join();

    // Simmetria di AtA completata qui, dopo il join → nessuna race condition
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            AtA_data[static_cast<std::size_t>(j) * N + i] =
            AtA_data[static_cast<std::size_t>(i) * N + j];

    const auto t_end = std::chrono::steady_clock::now();
    // ── Fine misurazione ────────────────────────────────────────────────────

    // --- Risoluzione del sistema ---
    std::vector<double> x;
    try {
        solve_system(AtA_data.data(), Atb.data(), x, N);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Errore: %s\n", e.what());
        return 1;
    }

    // --- Output ---
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
    const double elapsed_s = elapsed_us.count() * 1e-6;

    std::printf("Tempo SOLO parallelo: %.6f s  (%lld µs)\n",
                elapsed_s,
                static_cast<long long>(elapsed_us.count()));

    return 0;
}