#include <stdio.h>
#include <math.h>

#define N 4
#define M 57   // righe (N >= M colonne nel LS classico)

void least_squares(double A[M][N], double b[M], double x[N]) {

    double R[M][N];
    double y[M];

    // copia A in R
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            R[i][j] = A[i][j];

    // copia b in y
    for (int i = 0; i < M; i++)
        y[i] = b[i];

    // =========================
    // QR con Householder
    // =========================
    for (int k = 0; k < N && k < M; k++) {

        // 1. estrai x = colonna k da k in giù
        double norm = 0.0;
        for (int i = k; i < M; i++)
            norm += R[i][k] * R[i][k];
        norm = sqrt(norm);

        if (norm == 0.0) continue;

        double sign = (R[k][k] >= 0) ? 1.0 : -1.0;

        // v = x + sign*|x| e1
        double v[M];
        for (int i = k; i < M; i++)
            v[i] = R[i][k];
        v[k] += sign * norm;

        // normalizza v -> u
        double vnorm = 0.0;
        for (int i = k; i < M; i++)
            vnorm += v[i] * v[i];
        vnorm = sqrt(vnorm);

        if (vnorm == 0.0) continue;

        for (int i = k; i < M; i++)
            v[i] /= vnorm;

        // =========================
        // applica riflessione a R
        // R = R - 2 u (u^T R)
        // =========================
        for (int j = k; j < N; j++) {
            double dot = 0.0;
            for (int i = k; i < M; i++)
                dot += v[i] * R[i][j];

            for (int i = k; i < M; i++)
                R[i][j] -= 2.0 * v[i] * dot;
        }

        // applica anche a y
        double doty = 0.0;
        for (int i = k; i < M; i++)
            doty += v[i] * y[i];

        for (int i = k; i < M; i++)
            y[i] -= 2.0 * v[i] * doty;
    }

    // =========================
    // back substitution Rx = y
    // =========================
    for (int i = N - 1; i >= 0; i--) {
        x[i] = 0.0;

        for (int j = i + 1; j < N; j++)
            x[i] += R[i][j] * x[j];

        if (fabs(R[i][i]) > 1e-12)
            x[i] = (y[i] - x[i]) / R[i][i];
        else
            x[i] = 0.0;
    }
}

int main() {
    double A[M][N] = {
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27},
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27},
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27},
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27},
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27},
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27},
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27},
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27},
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27},
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27},
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27},
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27},
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27},
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27},
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27},
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27},
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27},
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27},
        {1, 1, 1, 1},
        {1, 2, 3, 4},
        {1, 3, 9, 27}
    };

    double b[M] = {6, 10, 18};
    double x[N];

    least_squares(A, b, x);

    for (int i = 0; i < N; i++)
        printf("%f\n", x[i]);

    return 0;
}