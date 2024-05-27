#include <limits.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>


#define L 176
// #define J0 1.0f
// #define J1 0.0f
// #define J2 -1.0f // Valor de alpha = 1, com J_0=1 (paper Selke 1981)


//int rand();

//float alpha = 0.1f;
// #define TEMP 3.0f
#define N_EQUILIBRIUM 1000
#define N_AVERAGE 20000

// float total_energy[N_EQUILIBRIUM +N_AVERAGE + 1];

//int j_0=1, j_1 = 1, j_2 = 1;

enum Color {BLACK, WHITE, GREEN};

// void initialize_matrix(int N, int matrix[][N], float randomMatrix[][N]) {
void initialize_matrix(int N, int** matrix, float** randomMatrix) {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < N; j++) {
            //matrix[i][j] = 1.0 *    rand() / (INT_MAX / 2) == 0 ? -1 : 1;
            //matrix[i][j] = ((double)   (rand() / (RAND_MAX))) < 0.5 ? -1 : 1;
            matrix[i][j] = 1;
            randomMatrix[i][j] = ((float) (rand() / (float)(RAND_MAX)));
        }
    }

}

// void reinitialize_random_matrix(int N, float randomMatrix[][N]) {
void reinitialize_random_matrix(int N, float** randomMatrix) {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < N; j++) {
            //matrix[i][j] = 1.0 *    rand() / (INT_MAX / 2) == 0 ? -1 : 1;
            randomMatrix[i][j] = ((float) ((rand()) /(float)(RAND_MAX)));
        }
    }

}

void initialize_ordered(int N, int matrix[][N]) {

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = 1;
        
        }
    }

}

// void initialize_total_energy(int d, float J0, float J1, float J2, int N, int matrix[][N]) {
void initialize_total_energy(int d, float J0, float J1, float J2, int N, int** matrix, float* total_energy) {
    float sum = 0.0;
    total_energy[d] = 0.0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < N; j++) {   
	    int jless = (j - 1 >= 0) ? j - 1 : N -1;
        int jplus = (j + 1 < N) ? j + 1 : 0;
        int iless = (i - 1 >= 0) ? i - 1 : L - 1;
        int iplus = (i + 1 < L) ? i + 1 : 0;
        //int jless2 = (j-2>=0) ? j-2 : j-2+C;
        //int jplus2 = (j+2 < C) ? j+2 : j+2 -C;
        int iplus2 = ((i+2  ) < L) ? i+2 : i+2-L;
        int iless2 = ((i-2) > -1) ? i - 2 : i - 2 + L;

            sum += -1.0 * matrix[i][j]*(J1*(matrix[iplus][j]+matrix[iless][j])+J2*(matrix[iless2][j] +matrix[iplus2][j]) + J0*(matrix[i][jless] + matrix[i][jplus]));
        }
    }
    total_energy[d] = sum / 2;
    //total_energy *= 0.5;
}

// void flip_spins(enum Color color, float J0, float J1, float J2, float t, int N, int matrix[][N], float randomMatrix[][N]) {
void flip_spins(enum Color color, float J0, float J1, float J2, float t, int N, int** matrix, float** randomMatrix) {
    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < L; i++) {
        for (int j = (i+ color) % 3;j < N; j+=3) {
            int jless = (j - 1 >= 0) ? j - 1 : N -1;
            int jplus = (j + 1 < N) ? j + 1 : 0;
            int iless = (i - 1 >= 0) ? i - 1 : L - 1;
            int iplus = (i + 1 < L) ? i + 1 : 0;
            //int jless2 = (j-2>=0) ? j-2 : j-2+C;
            //int jplus2 = (j+2 < C) ? j+2 : j+2 -C;
            int iplus2 = ((i+2) < L) ? i+2 : i+2-L;
            int iless2 = ((i-2) > -1) ? i - 2 : i - 2 + L;
            // int iplus3 = ((i+3) < L) ? i + 3 : i+3 - L;
            // int iless3 = ((i-3) > -1) ? i - 3 : i - 3 + L;
            // int iplus4 = ((i+4) < L) ? i + 4 : i+4 - L;
            // int iless4 = ((i-4) > -1) ? i-4 : L + i-4;
            // int jplus4 = ((j+4) < C) ? j + 4 : j +4 - C;
            // int jless4 = ((j-4) > -1) ? j - 4 : j - 4 + C;
            // int jplus3 = ((j+3) < C) ? j + 3 : j + 3 -C;
            // int jless3 = ((j-3) > -1) ? j-3: j-3+C;  

            float sum = J1*(matrix[iplus][j]+matrix[iless][j])+J2*(matrix[iless2][j] +matrix[iplus2][j]) + J0*(matrix[i][jless] + matrix[i][jplus]);
            int mij = matrix[i][j];
            float acceptance_ratio = exp(-2.0f * sum * mij / t);
            //int random = (float) rand() / (RAND_MAX) < 0.5 ? -1 : 1;
            
            if (randomMatrix[i][j] < acceptance_ratio) {
                matrix[i][j] = -mij;
		        //total_energy += 2.0f * sum * mij;
            }
        }
    }
    //printf("opa");
}


// void write_matrix(FILE *fptr, FILE *fptr2, int i, int N, int matrix[][N]) {
//     //int num;
    

//     // fptr = fopen("arquivo.txt", "w");
//     // if(fptr == NULL)
//     // {
//     //     printf("Error!");   
//     //     exit(1);             
//     // } 
//     for (int i = 0; i < L; i++) {
//         for (int j = 0; j < N; j++) {
//             //fprintf(fptr, "%d ", (int)matrix[i][j]);
//             if(matrix[i][j] == 1) {
//                 //printf("opa");
//             }
//         }
//         //fprintf(fptr, "\n");
//     }
//     fprintf(fptr, "\n");
//     //fclose(fptr);

//     fprintf(fptr2, "%d, %f ", i, total_energy[i]);
//     fprintf(fptr2, "\n");
//     //fprintf(fptr2, "%f %d %f %f", total_energy, J0, J1, J2);
    
    
// }

void write_info(float total_energy[], float total_energy_v, float av_energy, float variance) {
    FILE *fptr = fopen("vetor.txt", "a");
    for (int i = 1+N_EQUILIBRIUM; i < 1+N_EQUILIBRIUM+N_AVERAGE; i++) {
        fprintf(fptr, "%f ", total_energy[i]);
    }
    fprintf(fptr, "\n");
    fprintf(fptr, "Energia total: %f\n", total_energy_v);
    fprintf(fptr, "Energia médida: %f\n", av_energy);
    fprintf(fptr, "Variância: %f \n", variance);
}

void write_values(char* filename, float t, float sh) {
    FILE *fptr3 = fopen(filename, "a");
    fprintf(fptr3, "%f, %f ", t,  sh);
    fprintf(fptr3, "\n");
    fclose(fptr3);
}




int runc(float alpha, float t, float t_end, float step, char* filename, int N) {
    float* total_energy = (float *)malloc((N_EQUILIBRIUM +N_AVERAGE + 1) * sizeof(float));

    int **matrix = (int **)malloc(L*sizeof(int*));
    for (int i = 0; i < L; i++) {
        matrix[i] = (int*)malloc(N * sizeof(int));
    }
    float **randomMatrix = (float**) malloc(L*sizeof(float*));
    for (int i = 0 ; i < L; i++) {
        randomMatrix[i] = (float*)malloc(N*sizeof(float));
    }
    float* total_energy2 = (float *)malloc((N_AVERAGE) * sizeof(float));
    float* variance = (float *)malloc((N_AVERAGE) * sizeof(float));
    
    while (t < t_end) {
        srand((unsigned) time(NULL));
        clock_t start, end;
        
    
        /* Recording the starting clock tick.*/
        start = clock(); 
    
    

        
        //int random = rand();
        float J0 = 1.0;

        float J1 = (1-alpha)* J0;
        float J2 = -alpha*J0;

        // char[]
        // char[] filename = "valores%d.txt", i;
        
        
        //int matrix[L][N];
        
        // float randomMatrix[L][N];
        
        initialize_matrix(N, matrix, randomMatrix);
        //initialize_ordered();
        initialize_total_energy(0, J0, J1, J2, N, matrix, total_energy);
        //printf("%f", total_energy);
        //FILE *fptr, *fptr2, *fptr3;
        // fptr = fopen("arquivo_eq.txt", "w");
        // fptr2 = fopen("energia_eq.txt", "w");
        
        for (int i = 0; i < N_EQUILIBRIUM+N_AVERAGE; i++) {
            //write_matrix(fptr, fptr2, i); 
            flip_spins(BLACK, J0, J1, J2, t, N, matrix, randomMatrix);
            flip_spins(WHITE, J0, J1, J2, t, N, matrix, randomMatrix);
            flip_spins(GREEN, J0, J1, J2, t, N, matrix, randomMatrix);
            initialize_total_energy(i+1, J0, J1, J2, N, matrix, total_energy);
            reinitialize_random_matrix(N, randomMatrix);
        }
        // fclose(fptr);
        // fclose(fptr2);
        // fclose(fptr3);

        // fptr = fopen("arquivo_av.txt", "w");
        // fptr2 = fopen("energia_av.txt", "w");
        // for (int i = 0; i < N_AVERAGE; i++) {
        //     //write_matrix(fptr, fptr2, N_EQUILIBRIUM+i);
        //     flip_spins(BLACK, J0, J1, J2, t, N, matrix, randomMatrix);
        //     flip_spins(WHITE, J0, J1, J2, t, N, matrix, randomMatrix);
        //     flip_spins(GREEN, J0, J1, J2, t, N, matrix, randomMatrix);
        //     initialize_total_energy(1+N_EQUILIBRIUM+i, J0, J1, J2, N, matrix);
        //     reinitialize_random_matrix(N, randomMatrix);
        // }
        // fclose(fptr);
        // fclose(fptr2);
        //fclose(fptr3);

        float av_energy = 0;
        // float total_energy2[N_AVERAGE];
        
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

        av_energy = total_energy2[0] / (N_AVERAGE);
        // float variance[N_AVERAGE];
        
        for (int i = 1+N_EQUILIBRIUM; i <  1+N_EQUILIBRIUM+N_AVERAGE; i++) {
            variance[i - (1+N_EQUILIBRIUM)] = (total_energy[i]-av_energy)*(total_energy[i]-av_energy);
        }

        p = 1;
        while (p < N_AVERAGE) {
            for (int i = 0; i+p < N_AVERAGE; i+= 2*p) {
                variance[i] = variance[i] + variance[i+p];
            }
            p *= 2;
        }

        variance[0] = variance[0] / (N_AVERAGE);
        float specific_heat = variance[0] / (t*t*L*N);
        write_info(total_energy, total_energy2[0], av_energy, variance[0]);
        write_values(filename, t, specific_heat);
        //TEMP : 1.5f -> specific_heat: 0.233231202
        //TEMP : 2.0f -> specific_heat: 0.868345141
        //TEMP : 2.5f -> specific_heat: 0.987424552
        //TEMP : 3.0f -> specific_heat: 0.42233637
        //printf("%f \n", specific_heat);

        end = clock();

        double time_taken = ((end - start)+0.0) / (CLOCKS_PER_SEC); 
        printf("The program took %f seconds to execute", time_taken);
        //cout << fixed << time_taken << setprecision(5) << " seconds \n";

        //write_matrix();
        
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
    // float alpha = 0.1f;
    // float t = 2.26918531421f;
    // float t_end = t + 0.1;
    // float step = 1;
    // char* fileName= "haha.txt";
    // int N =5120;

    float alpha = atof(argv[1]);
    float t = atof(argv[2]);
    float t_end = atof(argv[3]);
    float step = atof(argv[4]);
    char* fileName = argv[5];
    int N = atoi(argv[6]);
    runc(alpha, t, t_end, step, fileName, N);
}
