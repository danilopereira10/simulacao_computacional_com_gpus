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

#include <cuda_fp16.h>
#include <curand.h>
#include <cublas_v2.h>

#include <cub/cub.cuh>
#define CUB_CHUNK_SIZE ((1ll<<31) - (1ll<<28))

#include "cudamacro.h"
#include "cuda_runtime.h"
#include "thrust/device_vector.h"

#define TCRIT 2.26918531421f
#define THREADS  128

#define L 240
#define C 12
#define J0 1.0f
#define J1 0.0f
#define J2 -1.0f
#define N_EQUILIBRIUM 20000
#define N_AVERAGE 100000




enum Color {BLACK, WHITE, GREEN};

struct saxpy_functor
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const {
            return (x-a) * (x - a);
        }
};

// Initialize lattice spins
__global__ void init_spins(signed char* lattice,
                           const float* __restrict__ randvals,
                           const long long nx,
                           const long long ny) {
  const long long  tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
  if (tid >= nx * ny) return;

  float randval = randvals[tid];
  signed char val = (randval < 0.5f) ? -1 : 1;
  lattice[tid] = val;
}

__global__ void copy_lattice(const signed char* __restrict__ lattice, signed char* extra_lattice, const long long nx,
                              const long long ny) {
  const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
  const int i = tid / ny;
  const int j = tid % ny;
  
  if (i >= nx || j >= ny) return;

  extra_lattice[i*ny + j] = lattice[i*ny + j];
}

__global__ void initialize_spin_energy(thrust::device_vector<float> spin_energy, Color color, 
                               const signed char* __restrict__ two_lattice,
                               const signed char* __restrict__ three_lattice,
                               const long long nx,
                               const long long ny) {
  const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
  const int i = tid / ny;
  const int j = tid % ny;

  if (i >= nx || j >= ny) return;

  // Set stencil indices with periodicity
  int ipp = (i + 1 < nx) ? i + 1 : 0;
  int ip2 = (i + 2 < nx) ? i + 2 : i + 2 - nx;
  int inn = (i - 1 >= 0) ? i - 1: nx - 1;
  int in2 = (i - 2 >= 0) ? i - 2 : i - 2 + ny;
  int jpp = (j + 1 < ny) ? j + 1 : 0;
  int jnn = (j - 1 >= 0) ? j - 1: ny - 1;
  int j2 = (j == (ny-1)) ? jpp : j;
  int j3 = (j == 0) ? jnn : j;


  // Compute sum of nearest neighbor spins

  signed char nn_sum;
  nn_sum = J1*(three_lattice[inn * ny + j] + two_lattice[ipp * ny + j]) +  // vizinho 1 vertical
                      J2*(three_lattice[ip2 * ny + j] + two_lattice[in2 * ny + j]) +  // vizinho 2 vertical
                      J0*(two_lattice[i * ny + j2] + three_lattice[i * ny + j3]);   // vizinho 1 horizontal

  spin_energy[3*(i*ny + j) +color] = nn_sum;
}

//template<bool is_black>
__global__ void update_lattice(float* spin_energy, Color color, signed char* lattice,
                               const signed char* __restrict__ two_lattice,
                               const signed char* __restrict__ three_lattice,
                               const float* __restrict__ randvals,
                               const float inv_temp,
                               const long long nx,
                               const long long ny) {
  const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
  const int i = tid / ny;
  const int j = tid % ny;

  if (i >= nx || j >= ny) return;

  // Set stencil indices with periodicity
  int ipp = (i + 1 < nx) ? i + 1 : 0;
  int ip2 = (i + 2 < nx) ? i + 2 : i + 2 - nx;
  int inn = (i - 1 >= 0) ? i - 1: nx - 1;
  int in2 = (i - 2 >= 0) ? i - 2 : i - 2 + ny;
  int jpp = (j + 1 < ny) ? j + 1 : 0;
  int jnn = (j - 1 >= 0) ? j - 1: ny - 1;
  int j2 = (j == (ny-1)) ? jpp : j;
  int j3 = (j == 0) ? jnn : j;


  // Compute sum of nearest neighbor spins

  signed char nn_sum;
  nn_sum = J1*(three_lattice[inn * ny + j] + two_lattice[ipp * ny + j]) +  // vizinho 1 vertical
                      J2*(three_lattice[ip2 * ny + j] + two_lattice[in2 * ny + j]) +  // vizinho 2 vertical
                      J0*(two_lattice[i * ny + j2] + three_lattice[i * ny + j3]);   // vizinho 1 horizontal

  

  // Determine whether to flip spin
  signed char lij = lattice[i * ny + j];
  float acceptance_ratio = exp(-2.0f * inv_temp * nn_sum * lij);

  if (randvals[i*ny + j] < acceptance_ratio) { // se entrar significa que flipou
    lattice[i * ny + j] = -lij;
    spin_energy[3*(i*ny + j) +color] = nn_sum;
  }
}

// Write lattice configuration to file
void write_lattice(signed char *lattice_g, signed char *lattice_b, signed char *lattice_w, std::string filename, long long nx, long long ny) {
  printf("Writing lattice to %s...\n", filename.c_str());
  signed char *lattice_h, *lattice_g_h, *lattice_b_h, *lattice_w_h;
  lattice_h = (signed char*) malloc(nx * ny * sizeof(*lattice_h));
  lattice_g_h = (signed char*) malloc(nx * ny/3 * sizeof(*lattice_g_h));
  lattice_b_h = (signed char*) malloc(nx * ny/3 * sizeof(*lattice_b_h));
  lattice_w_h = (signed char*) malloc(nx * ny/3 * sizeof(*lattice_w_h));

  CHECK_CUDA(cudaMemcpy(lattice_g_h, lattice_g, nx * ny/3 * sizeof(*lattice_g), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(lattice_b_h, lattice_b, nx * ny/3 * sizeof(*lattice_b), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(lattice_w_h, lattice_w, nx * ny/3 * sizeof(*lattice_w), cudaMemcpyDeviceToHost));

  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny/3; j++) {
        if ((i%3) == 0) {
            lattice_h[i*ny + 3*j] = lattice_b_h[i*ny/3 + j];
            lattice_h[i*ny + 3*j+1] = lattice_w_h[i*ny/3 + j];
            lattice_h[i*ny + 3*j+2] = lattice_g_h[i*ny/3 + j];
        } else if ((i%3) == 1) {
            lattice_h[i*ny + 3*j] = lattice_g_h[i*ny/3 + j];
            lattice_h[i*ny + 3*j+1] = lattice_b_h[i*ny/3 + j];
            lattice_h[i*ny + 3*j+2] = lattice_w_h[i*ny/3 + j];
        } else {
            lattice_h[i*ny + 3*j] = lattice_w_h[i*ny/3 + j];
            lattice_h[i*ny + 3*j+1] = lattice_g_h[i*ny/3 + j];
            lattice_h[i*ny + 3*j+2] = lattice_b_h[i*ny/3+j];
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

void write_values(char* filename, float t, float sh) {
  std::ofstream f;
  f.open(filename, std::ios::app);
  if (f.is_open()) {
    f << t << ", " << sh << " ";
    f << std::endl;
    
  }
  f.close();
}

void update(float* total_energy, signed char *lattice_g, signed char *lattice_b, signed char *lattice_w, float* randvals, curandGenerator_t rng, float inv_temp, long long nx, long long ny) {

  // Setup CUDA launch configuration
  int blocks = (nx * ny/3 + THREADS - 1) / THREADS;

  // Update black
  //copy_lattice<<<blocks, THREADS>>>(lattice_b, extra_lattice, nx, ny/2);
  CHECK_CURAND(curandGenerateUniform(rng, randvals, nx*ny/3));
  update_lattice<<<blocks, THREADS>>>(total_energy, Color::BLACK, lattice_b, lattice_w, lattice_g, randvals, inv_temp, nx, ny/3);

  // Update white
  //copy_lattice<<<blocks, THREADS>>>(lattice_w, extra_lattice, nx, ny/2);
  CHECK_CURAND(curandGenerateUniform(rng, randvals, nx*ny/3));
  update_lattice<<<blocks, THREADS>>>(total_energy, Color::WHITE, lattice_w, lattice_g, lattice_b,  randvals, inv_temp, nx, ny/3);

  CHECK_CURAND(curandGenerateUniform(rng, randvals, nx*ny/3));
  update_lattice<<<blocks, THREADS>>>(total_energy, Color::GREEN, lattice_g, lattice_b, lattice_w, randvals, inv_temp, nx, ny/3);

  thrust::reduce(total_energy.begin(), total_energy.end());
  total_energy[0] /= -2;
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

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

int main(int argc, char **argv) {

  float alpha = atof(argv[1]);
  float t = atof(argv[2]);
  char* fileName = argv[3];
  long long ny = atoi(argv[4]);
  // Defaults
  long long nx = 240;
  //long long ny = 12;
  //float alpha = 0.1f;
  int nwarmup = N_EQUILIBRIUM;
  int niters = N_AVERAGE;
  bool write = false;
  unsigned long long seed = 1234ULL;


  // Check arguments
  if (nx % 3 != 0 || ny % 3 != 0) {
    fprintf(stderr, "ERROR: Lattice dimensions must be multiple of 3.\n");
    exit(EXIT_FAILURE);
  }

  float inv_temp = 1.0f / (alpha*TCRIT);

  // Setup cuRAND generator
  curandGenerator_t rng;
  CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, seed));
  float *randvals;
  CHECK_CUDA(cudaMalloc(&randvals, (nx * ny/3) * sizeof(*randvals)));

  // Setup black and white lattice arrays on device
  signed char *lattice_b, *lattice_w, *lattice_g;

  CHECK_CUDA(cudaMalloc(&lattice_b, (nx * ny/3) * sizeof(*lattice_b)));
  CHECK_CUDA(cudaMalloc(&lattice_w, (nx * ny/3) * sizeof(*lattice_w)));
  CHECK_CUDA(cudaMalloc(&lattice_g, (nx*ny/3) * sizeof(*lattice_g)))
  //CHECK_CUDA(cudaMalloc(&extra_lattice, nx * ny/2 * sizeof(*extra_lattice)));

  //CHECK_CUDA(cudaMalloc(&total_energy, ()))
  


  int blocks = (nx * ny/3 + THREADS - 1) / THREADS;
  CHECK_CURAND(curandGenerateUniform(rng, randvals, (nx*ny/3)));
  init_spins<<<blocks, THREADS>>>(lattice_b, randvals, nx, ny/3);
  CHECK_CURAND(curandGenerateUniform(rng, randvals, (nx*ny/3)));
  init_spins<<<blocks, THREADS>>>(lattice_w, randvals, nx, ny/3);
  CHECK_CURAND(curandGenerateUniform(rng, randvals, (nx*ny/3)));
  init_spins<<<blocks, THREADS>>>(lattice_g, randvals, nx, ny/3);


  thrust::device_vector<float> spin_energy(nx*ny);
  float *spin_energy_ptr = thrust::raw_pointer_cast(&spin_energy[0]);
  initialize_spin_energy<<<blocks, THREADS>>>(spin_energy_ptr, Color::WHITE, lattice_b, lattice_g, nx, ny/3);
  initialize_spin_energy<<<blocks, THREADS>>>(spin_energy_ptr, Color::BLACK, lattice_g, lattice_w, nx, ny/3);
  initialize_spin_energy<<<blocks, THREADS>>>(spin_energy_ptr, Color::GREEN, lattice_w, lattice_b, nx, ny/3);

  thrust::device_vector<float> total_energy(N_AVERAGE);
  

  // Warmup iterations
  printf("Starting warmup...\n");
  for (int i = 0; i < nwarmup; i++) {
    update(spin_energy_ptr, lattice_g, lattice_b, lattice_w, randvals, rng, inv_temp, nx, ny);
  }
  

  CHECK_CUDA(cudaDeviceSynchronize());

  printf("Starting trial iterations...\n");
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < niters; i++) {
    update(spin_energy_ptr, lattice_g, lattice_b, lattice_w, randvals, rng, inv_temp, nx, ny);
    total_energy[i] = thrust::reduce(spin_energy.begin(), spin_energy.end());
    if (i % 10000 == 0) printf("Completed %d/%d iterations...\n", i+1, niters);
  }
  float sum2 = thrust::reduce(total_energy.begin(), total_energy.end());
  sum2 /= N_AVERAGE;
  float variance = thrust::reduce(total_energy.begin(), total_energy.end(), 0, saxpy_functor(sum2));

  variance /= N_AVERAGE;
  float specific_heat = variance / (t * t * nx * ny);
  write_values(fileName, t, specific_heat);

  CHECK_CUDA(cudaDeviceSynchronize());
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
  int nchunks = (nx * ny/2 + CUB_CHUNK_SIZE - 1)/ CUB_CHUNK_SIZE;
  CHECK_CUDA(cudaMalloc(&devsum, 2 * nchunks * sizeof(*devsum)));
  size_t cub_workspace_bytes = 0;
  void* workspace = NULL;
  CHECK_CUDA(cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, lattice_b, devsum, CUB_CHUNK_SIZE));
  CHECK_CUDA(cudaMalloc(&workspace, cub_workspace_bytes));
  for (int i = 0; i < nchunks; i++) {
    CHECK_CUDA(cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, &lattice_b[i*CUB_CHUNK_SIZE], devsum + 2*i,
                           std::min((long long) CUB_CHUNK_SIZE, nx * ny/2 - i * CUB_CHUNK_SIZE)));
    CHECK_CUDA(cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, &lattice_w[i*CUB_CHUNK_SIZE], devsum + 2*i + 1,
                           std::min((long long) CUB_CHUNK_SIZE, nx * ny/2 - i * CUB_CHUNK_SIZE)));
  }

  double* hostsum;
  hostsum = (double*)malloc(2 * nchunks * sizeof(*hostsum));
  CHECK_CUDA(cudaMemcpy(hostsum, devsum, 2 * nchunks * sizeof(*devsum), cudaMemcpyDeviceToHost));
  double fullsum = 0.0;
  for (int i = 0; i < 2 * nchunks; i++) {
    fullsum += hostsum[i];
  }
  std::cout << "\taverage magnetism (absolute): " << abs(fullsum / (nx * ny)) << std::endl;

  if (write) write_lattice(lattice_g, lattice_b, lattice_w, "final.txt", nx, ny);
  write_lattice(lattice_g, lattice_b, lattice_w, "final.txt", nx, ny);

  return 0;
}
