#include <stdio.h>
#include <stdlib.h>

#define N 4
#define M 3

int main() {
    double A[M][N] = {
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