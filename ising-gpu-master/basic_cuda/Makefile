CUDA_HOME=/usr/local/cuda
CUDACC=$(CUDA_HOME)/bin/nvcc
CC=gcc
LD=$(CUDACC)
LDFLAGS=-lcurand
CFLAGS=-c -O3 -g -I$(CUDA_HOME)/include 
CUDACFLAGS= -std=c++17 -c -O3 -lineinfo -arch=sm_70 -Xptxas=-v -I../external/cub -I../..
GFLAGS  = -std=c99 -Wall -Werror

all: ising_basic

ising_basic: main.o
	$(LD) -o ising_basic ../../energy.o main.o $(LDFLAGS) -ccbin g++ -m64 -gencode arch=compute_70,code=sm_70 

energy.o: energy.c
	gcc $(GFLAGS) -g -c $< -o $@

%.o: %.cu
	nvcc -c -gencode arch=compute_70,code=sm_70 -ccbin g++ -m64 $(CUDACFLAGS) $<

clean:
	rm *.o ising_basic
