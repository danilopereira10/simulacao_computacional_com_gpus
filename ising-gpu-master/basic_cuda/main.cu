/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <chrono>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>
#include <time.h>

#include <cuda_fp16.h>
#include <curand.h>
#include <cublas_v2.h>

#include <cub/cub.cuh>
#define CUB_CHUNK_SIZE ((1ll<<31) - (1ll<<28))

#include "cudamacro.h"

#define TCRIT 2.26918531421f
#define THREADS  128

enum Color {BLACK, WHITE, GREEN};

// // Initialize lattice spins
// __global__ void init_spins(signed char* lattice,
//                            const float* __restrict__ randvals,
//                            const long long nx,
//                            const long long ny) {
//   const long long  tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
//   if (tid >= nx * ny) return;

//   float randval = randvals[tid];
//   signed char val = (randval < 0.5f) ? -1 : 1;
//   lattice[tid] = val;
// }

// Initialize lattice spins
__global__ void init_spins(signed char* lattice,
                           const float* __restrict__ randvals,
                           const long long nx,
                           const long long ny) {
  const long long  tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
  if (tid >= nx * ny) return;

  // float randval = randvals[tid];
  // signed char val = (randval < 0.5f) ? -1 : 1;
  lattice[tid] = 1;
}

__global__ void calculate_spin_energy(signed char* lattice,
                                 float* spin_energy,
                                  const long long nx,
                                  const long long ny,
                                  float j0, float j1, float j2) {
  const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
  const int i = tid / ny;
  const int j = tid % ny;

  if (i >= nx || j >= ny) return;


  int ipp = (i + 1 < nx) ? i + 1 : 0;
  int inn = (i - 1 >= 0) ? i - 1: nx - 1;
  int jpp = (j + 1 < ny) ? j + 1 : 0;
  int jnn = (j - 1 >= 0) ? j - 1: ny - 1;
  int jp2 = (j + 2 < ny) ? j + 2 : j + 2 - ny;
  int jm2 = (j - 2 >= 0) ? j - 2 : j - 2 + ny;

  spin_energy[i*ny+j] = (-lattice[i*ny+j]) * (j0 * (lattice[inn*ny+j] + lattice[ipp*ny+j]) + j1*(lattice[i*ny+jpp] + lattice[i*ny+jnn])
      + j2 * (lattice[i*ny+jp2] + lattice[i*ny+jm2]));
}

template<bool is_black>
__global__ void update_lattice(enum Color color, signed char* lattice,
                               const float* __restrict__ randvals,
                               const float inv_temp,
                               const long long nx,
                               const long long ny, float j0, float j1, float j2) {
  const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
  const int i = tid / ny;
  const int j = tid % ny;

  if (i >= nx || j >= ny) return;
  if ((j%3) != ((i+color)%3)) return;

  // Set stencil indices with periodicity
  int ipp = (i + 1 < nx) ? i + 1 : 0;
  int inn = (i - 1 >= 0) ? i - 1: nx - 1;
  int jpp = (j + 1 < ny) ? j + 1 : 0;
  int jnn = (j - 1 >= 0) ? j - 1: ny - 1;
  int jp2 = (j + 2 < ny) ? j + 2 : j + 2 - ny;
  int jm2 = ((j - 2) >= 0) ? j - 2 : j-2 + ny;
  

  // Compute sum of nearest neighbor spins
  float nn_sum = j0 * (lattice[inn*ny+j] + lattice[ipp*ny+j]) + j1*(lattice[i*ny+jpp] + lattice[i*ny+jnn])
      + j2 * (lattice[i*ny+jp2] + lattice[i*ny+jm2]);
  
  

  // Determine whether to flip spin
  signed char lij = lattice[i * ny + j];
  float acceptance_ratio = exp(-2.0f * inv_temp * nn_sum * lij);
  if (randvals[i*ny + j] < acceptance_ratio) {
    lattice[i * ny + j] = -lij;
  }
}

// Write lattice configuration to file
void write_lattice(signed char *lattice_b, signed char *lattice_w, std::string filename, long long nx, long long ny) {
  printf("Writing lattice to %s...\n", filename.c_str());
  signed char *lattice_h, *lattice_b_h, *lattice_w_h;
  lattice_h = (signed char*) malloc(nx * ny * sizeof(*lattice_h));
  lattice_b_h = (signed char*) malloc(nx * ny/2 * sizeof(*lattice_b_h));
  lattice_w_h = (signed char*) malloc(nx * ny/2 * sizeof(*lattice_w_h));

  CHECK_CUDA(cudaMemcpy(lattice_b_h, lattice_b, nx * ny/2 * sizeof(*lattice_b), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(lattice_w_h, lattice_b, nx * ny/2 * sizeof(*lattice_w), cudaMemcpyDeviceToHost));

  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny/2; j++) {
      if (i % 2) {
        lattice_h[i*ny + 2*j+1] = lattice_b_h[i*ny/2 + j];
        lattice_h[i*ny + 2*j] = lattice_w_h[i*ny/2 + j];
      } else {
        lattice_h[i*ny + 2*j] = lattice_b_h[i*ny/2 + j];
        lattice_h[i*ny + 2*j+1] = lattice_w_h[i*ny/2 + j];
      }
    }
  }

  std::ofstream f;
  f.open(filename);
  if (f.is_open()) {
    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
         f << (int)lattice_h[i * ny + j] << " ";
      }
      f << std::endl;
    }
  }
  f.close();

  free(lattice_h);
  free(lattice_b_h);
  free(lattice_w_h);
}

void update(enum Color color, signed char *lattice, float* randvals, curandGenerator_t rng, float inv_temp, long long nx, long long ny,
  float j0, float j1, float j2) {

  // Setup CUDA launch configuration
  int blocks = (nx * ny + THREADS - 1) / THREADS;

  // Update black
  CHECK_CURAND(curandGenerateUniform(rng, randvals, nx*ny));
  update_lattice<true><<<blocks, THREADS>>>(color, lattice, randvals, inv_temp, nx, ny, j0, j1, j2);

  // Update white
  CHECK_CURAND(curandGenerateUniform(rng, randvals, nx*ny/2));
  update_lattice<false><<<blocks, THREADS>>>(color, lattice  , randvals, inv_temp, nx, ny, j0, j1, j2);
}

static void usage(const char *pname) {

  const char *bname = rindex(pname, '/');
  if (!bname) {bname = pname;}
  else        {bname++;}

  fprintf(stdout,
          "Usage: %s [options]\n"
          "options:\n"
          "\t-x|--lattice-n <LATTICE_N>\n"
          "\t\tnumber of lattice rows\n"
          "\n"
          "\t-y|--lattice_m <LATTICE_M>\n"
          "\t\tnumber of lattice columns\n"
          "\n"
          "\t-w|--nwarmup <NWARMUP>\n"
          "\t\tnumber of warmup iterations\n"
          "\n"
          "\t-n|--niters <NITERS>\n"
          "\t\tnumber of trial iterations\n"
          "\n"
          "\t-a|--alpha <ALPHA>\n"
          "\t\tcoefficient of critical temperature\n"
          "\n"
          "\t-s|--seed <SEED>\n"
          "\t\tseed for random number generation\n"
          "\n"
          "\t-o|--write-lattice\n"
          "\t\twrite final lattice configuration to file\n\n",
          bname);
  exit(EXIT_SUCCESS);
}

void write_info(float total_energy[], float total_energy_v, float av_energy, float variance, int niters) {
    FILE *fptr = fopen("energias.txt", "w");
    for (int i = 0; i < niters; i++) {
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

int simulate(float alpha, float t, char* fileName, int ny, int niters) {
  // Defaults
  long long nx = 5120;
  // long long ny = 5120;
  // float alpha = 0.1f;
  int nwarmup = 100;
  // int niters = 1000;
  bool write = false;
  unsigned long long seed = 1234ULL;
  float j0, j1, j2;
  j0 = 1.0;
  j1 = (1-alpha)*j0;
  j2 = -alpha*j0;

  // while (1) {
  //   static struct option long_options[] = {
  //       {     "lattice-n", required_argument, 0, 'x'},
  //       {     "lattice-m", required_argument, 0, 'y'},
  //       {         "alpha", required_argument, 0, 'y'},
  //       {          "seed", required_argument, 0, 's'},
  //       {       "nwarmup", required_argument, 0, 'w'},
  //       {        "niters", required_argument, 0, 'n'},
  //       { "write-lattice",       no_argument, 0, 'o'},
  //       {          "help",       no_argument, 0, 'h'},
  //       {               0,                 0, 0,   0}
  //   };

  //   int option_index = 0;
  //   int ch = getopt_long(argc, argv, "x:y:a:s:w:n:oh", long_options, &option_index);
  //   if (ch == -1) break;

  //   switch(ch) {
  //     case 0:
  //       break;
  //     case 'x':
  //       nx = atoll(optarg); break;
  //     case 'y':
  //       ny = atoll(optarg); break;
  //     case 'a':
  //       alpha = atof(optarg); break;
  //     case 's':
  //       seed = atoll(optarg); break;
  //     case 'w':
  //       nwarmup = atoi(optarg); break;
  //     case 'n':
  //       niters = atoi(optarg); break;
  //     case 'o':
  //       write = true; break;
  //     case 'h':
  //       usage(argv[0]); break;
  //     case '?':
  //       exit(EXIT_FAILURE);
  //     default:
  //       fprintf(stderr, "unknown option: %c\n", ch);
  //       exit(EXIT_FAILURE);
  //   }
  // }

  // Check arguments
  if (nx % 2 != 0 || ny % 2 != 0) {
    fprintf(stderr, "ERROR: Lattice dimensions must be even values.\n");
    exit(EXIT_FAILURE);
  }

  // float inv_temp = 1.0f / (alpha*TCRIT);
  float inv_temp = 1.0f / t;

  // Setup cuRAND generator
  curandGenerator_t rng;
  CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, seed));
  float *randvals;
  CHECK_CUDA(cudaMalloc(&randvals, nx * ny * sizeof(*randvals)));

  // Setup black and white lattice arrays on device
  signed char *lattice;
  float *spin_energy;
  CHECK_CUDA(cudaMalloc(&spin_energy, nx*ny * sizeof(*spin_energy)));
  CHECK_CUDA(cudaMalloc(&lattice, nx * ny * sizeof(*lattice)));

  int blocks = (nx * ny + THREADS - 1) / THREADS;
  CHECK_CURAND(curandGenerateUniform(rng, randvals, nx*ny));
  init_spins<<<blocks, THREADS>>>(lattice, randvals, nx, ny);
  // calculate_spin_energy<<<blocks, THREADS>>>(lattice, spin_energy, nx, ny, j0, j1, j2);
  float total_energy[niters];

  // float* squareOfDistanceToMean;
  // CHECK_CUDA(cudaMalloc(&squareOfDistanceToMean, niters*sizeof(float)));
  clock_t start, end;
  start = clock();
  // Warmup iterations
  printf("Starting warmup...\n");
  for (int i = 0; i < nwarmup; i++) {
    update(Color::BLACK, lattice, randvals, rng, inv_temp, nx, ny, j0, j1, j2);
    update(Color::WHITE, lattice, randvals, rng, inv_temp, nx, ny, j0, j1, j2);
    update(Color::GREEN, lattice, randvals, rng, inv_temp, nx, ny, j0, j1, j2);
  }

  CHECK_CUDA(cudaDeviceSynchronize());

  printf("Starting trial iterations...\n");
  auto t0 = std::chrono::high_resolution_clock::now();
  float av_energy = 0;
  for (int i = 0; i < niters; i++) {
    update(Color::BLACK, lattice, randvals, rng, inv_temp, nx, ny, j0, j1, j2);
    update(Color::WHITE, lattice, randvals, rng, inv_temp, nx, ny, j0, j1, j2);
    update(Color::GREEN, lattice, randvals, rng, inv_temp, nx, ny, j0, j1, j2);
    calculate_spin_energy<<<blocks,THREADS>>>(lattice, spin_energy, nx, ny, j0, j1, j2);

    CHECK_CUDA(cudaDeviceSynchronize());
    double* devsum;
    int nchunks = (nx * ny + CUB_CHUNK_SIZE - 1)/ CUB_CHUNK_SIZE;
    CHECK_CUDA(cudaMalloc(&devsum,  nchunks * sizeof(*devsum)));
    size_t cub_workspace_bytes = 0;
    void* workspace = NULL;
    CHECK_CUDA(cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, spin_energy, devsum, CUB_CHUNK_SIZE));
    CHECK_CUDA(cudaMalloc(&workspace, cub_workspace_bytes));
    for (int j = 0; j < nchunks; j++) {
      CHECK_CUDA(cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, &spin_energy[j*CUB_CHUNK_SIZE], devsum + j,
                              std::min((long long) CUB_CHUNK_SIZE, nx * ny - j * CUB_CHUNK_SIZE)));
    }

    double* hostsum;
    hostsum = (double*)malloc(nchunks * sizeof(*hostsum));
    CHECK_CUDA(cudaMemcpy(hostsum, devsum, nchunks * sizeof(*devsum), cudaMemcpyDeviceToHost));
    double fullsum = 0.0;
    for (int j = 0; j < nchunks; j++) {
      fullsum += hostsum[j];
    }
    
    CHECK_CUDA(cudaFree(devsum));
    CHECK_CUDA(cudaFree(workspace));
    CHECK_CUDA(cudaFree(hostsum));
    total_energy[i] = fullsum;
    av_energy += fullsum;
    if (i % 1000 == 0) printf("Completed %d/%d iterations...\n", i+1, niters);
  }
  av_energy /= niters;
  float variance = 0;
  for (int i = 0; i < niters; i++) {
    variance += (total_energy[i]-av_energy)*(total_energy[i]-av_energy);
  }
  variance /= niters;
  float specific_heat = variance / (t*t*nx*ny);

  


  write_info(total_energy, av_energy * niters, av_energy, variance, niters);
  write_values(fileName, t, specific_heat);
  end = clock();
  double time_taken = ((end-start)+0.0) / CLOCKS_PER_SEC;

  FILE *fptr4 = fopen("time_taken.txt", "a");
  fprintf(fptr4, "%f, %f ", t,  specific_heat);
  fprintf(fptr4, "%f sec", time_taken);
  fprintf(fptr4, "\n");
  fclose(fptr4);
  auto t1 = std::chrono::high_resolution_clock::now();

  double duration = (double) std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
  printf("REPORT:\n");
  printf("\tnGPUs: %d\n", 1);
  printf("\ttemperature: %f * %f\n", alpha, TCRIT);
  printf("\tseed: %llu\n", seed);
  printf("\twarmup iterations: %d\n", nwarmup);
  printf("\ttrial iterations: %d\n", niters);
  printf("\tlattice dimensions: %lld x %lld\n", nx, ny);
  printf("\telapsed time: %f sec\n", duration * 1e-6);
  printf("\tupdates per ns: %f\n", (double) (nx * ny) * niters / duration * 1e-3);

  // Reduce
  double* devsum;
  int nchunks = (nx * ny + CUB_CHUNK_SIZE - 1)/ CUB_CHUNK_SIZE;
  CHECK_CUDA(cudaMalloc(&devsum,  nchunks * sizeof(*devsum)));
  int cub_workspace_bytes = 0;
  void* workspace = NULL;
  CHECK_CUDA(cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, lattice, devsum, CUB_CHUNK_SIZE));
  CHECK_CUDA(cudaMalloc(&workspace, cub_workspace_bytes));
  for (int i = 0; i < nchunks; i++) {
    CHECK_CUDA(cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, &lattice[i*CUB_CHUNK_SIZE], devsum + i,
                           std::min((long long) CUB_CHUNK_SIZE, nx * ny - i * CUB_CHUNK_SIZE)));
  }

  double* hostsum;
  hostsum = (double*)malloc(nchunks * sizeof(*hostsum));
  CHECK_CUDA(cudaMemcpy(hostsum, devsum, nchunks * sizeof(*devsum), cudaMemcpyDeviceToHost));
  double fullsum = 0.0;
  for (int i = 0; i < nchunks; i++) {
    fullsum += hostsum[i];
  }
  std::cout << "\taverage magnetism (absolute): " << abs(fullsum / (nx * ny)) << std::endl;

  // if (write) write_lattice(lattice_b, lattice_w, "final.txt", nx, ny);

  return 0;
}

int main(int argc, char* argv[]) {
  float alpha = atof(argv[1]);
  float t = atof(argv[2]);
  char* fileName = argv[3];
  int C = atoi(argv[4]);
  int iterations = atoi(argv[5]);
  simulate(alpha, t, fileName, C, iterations);
}
