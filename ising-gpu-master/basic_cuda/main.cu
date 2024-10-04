#include <limits.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define L 176
#define N_EQUILIBRIUM 1000
#define N_AVERAGE 10000

enum Color {BLACK, WHITE, GREEN};

void initialize_matrix(int N, int* matrix, float* randomMatrix) {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i*N+j] = 1;
            randomMatrix[i*N+j] = ((float) (rand() / (float)(RAND_MAX)));
        }
    }
}

void reinitialize_random_matrix(int N, float* randomMatrix) {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < N; j++) {
            randomMatrix[i*N+j] = ((float) ((rand()) /(float)(RAND_MAX)));
        }
    }
}

void write_info(float total_energy[], float total_energy_v, float av_energy, float variance) {
    FILE *fptr = fopen("energias.txt", "w");
    for (int i = 1+N_EQUILIBRIUM; i < 1+N_EQUILIBRIUM+N_AVERAGE; i++) {
        fprintf(fptr, "%f \n", total_energy[i]);
    }
    fclose(fptr);
    fptr = fopen("detalhes.txt", "w");
    fprintf(fptr, "\n");
    fprintf(fptr, "Soma das energias de todas as iterações: %f\n", total_energy_v);
    fprintf(fptr, "Energia média: %f\n", av_energy);
    fprintf(fptr, "Variância: %f \n", variance);
    fclose(fptr);
}

void write_values(char* filename, float t, float sh) {
    FILE *fptr3 = fopen(filename, "a");
    fprintf(fptr3, "%f, %f ", t,  sh);
    fprintf(fptr3, "\n");
    fclose(fptr3);
}

void calculate_total_energy(int d, float J0, float J1, float J2, int N, int* matrix, float* total_energy) {
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
            sum += -1.0 * matrix[i*N+j]*(J1*(matrix[iplus*N+j]+matrix[iless*N+j])+J2*(matrix[iless2*N+j] +matrix[iplus2*N+j]) + J0*(matrix[i*N+jless] + matrix[i*N+jplus]));
        }
    }
    total_energy[d] = sum / 2;
}

__global__ void flip_spins(enum Color color, float J0, float J1, float J2, float t, int N, int* matrix, float* randomMatrix, int n) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   int i = (idx) / N;
//   int j = idx % N;
int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int i = (idx) / N;
  int j = idx % N;

            if ((idx < n) && ((j%3) == ((i+color)%3)) && (i < (L-3)) && (j < (N - 3))) {
            

            int jless = (j - 1 >= 0) ? j - 1 : N -1;
            int jplus = (j + 1 < N) ? j + 1 : 0;
            int iless = (i - 1 >= 0) ? i - 1 : L - 1;
            int iplus = (i + 1 < L) ? i + 1 : 0;
            int iplus2 = ((i+2) < L) ? i+2 : i+2-L;
            int iless2 = ((i-2) > -1) ? i - 2 : i - 2 + L;
            
            float sum = J1*(matrix[iplus*N+j]+matrix[iless*N+j])+J2*(matrix[iless2*N+j] +matrix[iplus2*N+j]) + J0*(matrix[i*N+jless] + matrix[i*N+jplus]);
            int mij = matrix[i*N+j];
            float acceptance_ratio = exp(-2.0f * sum * mij / t);

            if (randomMatrix[i*N+j] < acceptance_ratio) {
            matrix[i*N+j] = -mij;
            }
        }
        
   
  
}

__global__ void flip_spins2(enum Color color, float J0, float J1, float J2, float t, int N, int* matrix, float* randomMatrix, int n) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   int i = (idx) / N;
//   int j = idx % N;
int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int i = (idx) / N;
  int j = idx % N;

            if ((idx < n) && ((j%3) == ((i+color)%3)) && !((i >= L-3) && (j >= (N - 3))) && ((i>=L-3) || (j>=N-3))) {
            

            int jless = (j - 1 >= 0) ? j - 1 : N -1;
            int jplus = (j + 1 < N) ? j + 1 : 0;
            int iless = (i - 1 >= 0) ? i - 1 : L - 1;
            int iplus = (i + 1 < L) ? i + 1 : 0;
            int iplus2 = ((i+2) < L) ? i+2 : i+2-L;
            int iless2 = ((i-2) > -1) ? i - 2 : i - 2 + L;
            
            float sum = J1*(matrix[iplus*N+j]+matrix[iless*N+j])+J2*(matrix[iless2*N+j] +matrix[iplus2*N+j]) + J0*(matrix[i*N+jless] + matrix[i*N+jplus]);
            int mij = matrix[i*N+j];
            float acceptance_ratio = exp(-2.0f * sum * mij / t);

            if (randomMatrix[i*N+j] < acceptance_ratio) {
            matrix[i*N+j] = -mij;
            }
        }
        
      
}

__global__ void flip_spins3(enum Color color, float J0, float J1, float J2, float t, int N, int* matrix, float* randomMatrix, int n) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   int i = (idx) / N;
//   int j = idx % N;
int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int i = (idx) / N;
  int j = idx % N;

            if ((idx < n) && ((j%3) == ((i+color)%3)) && ((i >= L-3) && (j >= (N - 3)))) {
            

            int jless = (j - 1 >= 0) ? j - 1 : N -1;
            int jplus = (j + 1 < N) ? j + 1 : 0;
            int iless = (i - 1 >= 0) ? i - 1 : L - 1;
            int iplus = (i + 1 < L) ? i + 1 : 0;
            int iplus2 = ((i+2) < L) ? i+2 : i+2-L;
            int iless2 = ((i-2) > -1) ? i - 2 : i - 2 + L;
            
            float sum = J1*(matrix[iplus*N+j]+matrix[iless*N+j])+J2*(matrix[iless2*N+j] +matrix[iplus2*N+j]) + J0*(matrix[i*N+jless] + matrix[i*N+jplus]);
            int mij = matrix[i*N+j];
            float acceptance_ratio = exp(-2.0f * sum * mij / t);

            if (randomMatrix[i*N+j] < acceptance_ratio) {
            matrix[i*N+j] = -mij;
            }
        }
        
      
}


int runc(float alpha, float t, float t_end, float step, char* filename, int N) {
    float* total_energy = (float *)malloc((N_EQUILIBRIUM +N_AVERAGE + 1) * sizeof(float));
    int *matrix = (int *)malloc(L*N*sizeof(int));
    float *randomMatrix = (float*) malloc(L*N*sizeof(float));
    float* total_energy2 = (float *)malloc((N_AVERAGE) * sizeof(float));
    float* squareOfDistanceFromMean = (float *)malloc((N_AVERAGE) * sizeof(float));

    int *d_matrix;
    float *d_randomMatrix;

    int bytes = sizeof(int) * L * N;
    int err;
    int cudaSuccess = 0;
    err = cudaMalloc((void **) &d_matrix, bytes);
    if (err != cudaSuccess) {
      fprintf(stdout, "Erro na alocação de d_matrix: \n");
    }

    int bytes2 = sizeof(float) * L * N;
    err = cudaMalloc((void **) &d_randomMatrix, bytes2);
    if (err != cudaSuccess) {
      fprintf(stdout, "Erro na alocação de d_randomMatrix: \n");
    }

    
    
    while (t < t_end) {
        srand((unsigned) time(NULL));
        clock_t start, end;
    
        /* Recording the starting clock tick.*/
        start = clock(); 
    
        float J0 = 1.0;
        float J1 = (1-alpha)* J0;
        float J2 = -alpha*J0;

        initialize_matrix(N, matrix, randomMatrix);
        calculate_total_energy(0, J0, J1, J2, N, matrix, total_energy);
        int T = 128;
        int BL = (L*N + (T-1)) / T;
        for (int i = 0; i < N_EQUILIBRIUM+N_AVERAGE; i++) {
            cudaMemcpy(d_matrix, matrix, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_randomMatrix, randomMatrix, bytes2, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            flip_spins<<<BL,T>>>(BLACK, J0, J1, J2, t, N, d_matrix, d_randomMatrix, L*N);
            cudaDeviceSynchronize();
            flip_spins2<<<BL,T>>>(BLACK, J0, J1, J2, t, N, d_matrix, d_randomMatrix, L*N);
            flip_spins3<<<BL,T>>>(BLACK, J0, J1, J2, t, N, d_matrix, d_randomMatrix, L*N);
            cudaDeviceSynchronize();
            reinitialize_random_matrix(N, randomMatrix);

            cudaMemcpy(d_randomMatrix, randomMatrix, bytes2, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            flip_spins<<<BL,T>>>(WHITE, J0, J1, J2, t, N, d_matrix, d_randomMatrix, L*N);
            cudaDeviceSynchronize();
            flip_spins2<<<BL,T>>>(WHITE, J0, J1, J2, t, N, d_matrix, d_randomMatrix, L*N);
            flip_spins3<<<BL,T>>>(WHITE, J0, J1, J2, t, N, d_matrix, d_randomMatrix, L*N);
            cudaDeviceSynchronize();
            reinitialize_random_matrix(N, randomMatrix);

            cudaMemcpy(d_randomMatrix, randomMatrix, bytes2, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            flip_spins<<<BL,T>>>(GREEN, J0, J1, J2, t, N, d_matrix, d_randomMatrix, L*N);
            cudaDeviceSynchronize();
            flip_spins2<<<BL,T>>>(GREEN, J0, J1, J2, t, N, d_matrix, d_randomMatrix, L*N);
            flip_spins3<<<BL,T>>>(GREEN, J0, J1, J2, t, N, d_matrix, d_randomMatrix, L*N);
            cudaDeviceSynchronize();
            reinitialize_random_matrix(N, randomMatrix);

            cudaMemcpy(matrix, d_matrix, bytes, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            calculate_total_energy(i+1, J0, J1, J2, N, matrix, total_energy);
        }
    
        for (int i = 1 + N_EQUILIBRIUM; i < 1+N_EQUILIBRIUM+N_AVERAGE; i++) {
            total_energy2[i-(1+N_EQUILIBRIUM)] = total_energy[i];
        }

        int p = 1; 
        while (p < N_AVERAGE) {
            for (int i = 0; i+p < N_AVERAGE; i+= 2*p) {
                total_energy2[i] = total_energy2[i] + total_energy2[i+p];
            }
            p *= 2;
        }
        float av_energy = total_energy2[0] / (N_AVERAGE);
        
        for (int i = 1+N_EQUILIBRIUM; i <  1+N_EQUILIBRIUM+N_AVERAGE; i++) {
            squareOfDistanceFromMean[i - (1+N_EQUILIBRIUM)] = (total_energy[i]-av_energy)*(total_energy[i]-av_energy);
        }

        p = 1;
        while (p < N_AVERAGE) {
            for (int i = 0; i+p < N_AVERAGE; i+= 2*p) {
                squareOfDistanceFromMean[i] = squareOfDistanceFromMean[i] + squareOfDistanceFromMean[i+p];
            }
            p *= 2;
        }
        squareOfDistanceFromMean[0] = squareOfDistanceFromMean[0] / (N_AVERAGE);

        float specific_heat = squareOfDistanceFromMean[0] / (t*t*L*N);
        write_info(total_energy, total_energy2[0], av_energy, squareOfDistanceFromMean[0]);
        write_values(filename, t, specific_heat);
        
        end = clock();

        double time_taken = ((end - start)+0.0) / (CLOCKS_PER_SEC); 
        printf("The program took %f seconds to execute", time_taken);
        
        FILE *fptr4 = fopen("time_taken.txt", "a");
        fprintf(fptr4, "%f, %f ", t,  specific_heat);
        fprintf(fptr4, "%f sec", time_taken);
        fprintf(fptr4, "\n");
        fclose(fptr4);
        
        t += step;
    }
    return 0;
}

int main(int argc, char* argv[]) {
    float alpha = atof(argv[1]);
    float t = atof(argv[2]);
    float t_end = atof(argv[3]);
    float step = atof(argv[4]);
    char* fileName = argv[5];
    int N = atoi(argv[6]);
    runc(alpha, t, t_end, step, fileName, N);
}
