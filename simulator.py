#include <limits.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

from random import seed
from random import gauss
import math
import time

TCRIT= 2.26918531421
L=5120
C=5120
J0 =1
J1 =1
J2 =1

matrix = L * [None * C]
randomMatrix = L * [None * C]


alpha = 0.1
inv_temp = 1.0 / (0.1 * TCRIT)
n_iterations = 1000

colorV = {"BLACK":0, "WHITE":1, "GREEN":2}

def initialize_matrix():

    for i in range(L):
        matrix.append([])
        for j in range(C): 
            matrix[i][j] = -1 if gauss(0,1) < 0.5 else 1
            randomMatrix[i][j] = -1 if gauss(0,1) < 0.5 else 1

def flip_spins(color):
    r = colorV[color]
    for i in range(L):
        for j in range(i%3+r, C, 3):
            jless = j-1 if (j - 1 >= 0) else C -1
            jplus = j+1 if (j + 1 < C) else 0
            iless = i - 1 if (i - 1 >= 0) else L - 1
            iplus = i + 1 if (i + 1 < L) else 0
            jless2 = j-2 if (j-2>=0) else j-2+C
            jplus2 = j+2 if (j+2 < C) else j+2 -C
            iplus2 = i+2 if ((i+2) < C) else i+2-L
            iless2 = i-2 if ((i-2) > -1) else i - 2 + L

            sum = J1*(matrix[i][iplus]+matrix[i][iless])+J2*(matrix[iless2][j] +matrix[iplus2][j]) + J0*(matrix[i][jless] + matrix[i][jplus])
            mij = matrix[i][j]
            acceptance_ratio = math.exp(-2.0 * inv_temp * sum * mij)
            if (randomMatrix[i][j] < acceptance_ratio):
                matrix[i][j] = -mij


def write_matrix():

    fptr = open("arquivo.txt", "w");
    for i in range(L): 
        for j in range(C):
            fptr.write(str(int(matrix[i][j])) + " ")
        fptr.write("\n")
    fptr.close()
    
    

def main():
    st = time.time()
  

    initialize_matrix()
    for i in range(n_iterations):
        flip_spins(colorV.BLACK)
        flip_spins(colorV.WHITE)
        flip_spins(colorV.GREEN)
    
 
    et = time.time()
 
    time_taken = ((et - st)) 
    print("The program took %f seconds to execute", time_taken)
    
    write_matrix()

    
main()