#include <limits.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define L 176


void calculate_total_energy(int d, float J0, float J1, float J2, int N, int** matrix, float* total_energy) {
    float sum = 0.0;
    total_energy[d] = 0.0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < N; j++) {   
            int jless = (j - 1 >= 0) ? j - 1 : N -1;
            int jplus = (j + 1 < N) ? j + 1 : 0;
            int iless = (i - 1 >= 0) ? i - 1 : L - 1;
            int iplus = (i + 1 < L) ? i + 1 : 0;
            int iplus2 = ((i+2  ) < L) ? i+2 : i+2-L;
            int iless2 = ((i-2) > -1) ? i - 2 : i - 2 + L;
            sum += -1.0 * matrix[i][j]*(J1*(matrix[iplus][j]+matrix[iless][j])+J2*(matrix[iless2][j] +matrix[iplus2][j]) + J0*(matrix[i][jless] + matrix[i][jplus]));
        }
    }
    total_energy[d] = sum / 2;
}
