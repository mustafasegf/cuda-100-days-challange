#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

#define N 1024
#define c0 0.5f
#define c1 0.1f
#define c2 0.1f
#define c3 0.1f
#define c4 0.1f
#define c5 0.05f
#define c6 0.05f

__global__ void stencil_kernel(float *in, float *out, unsigned int n) {
  unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= 1 && i < n - 1 && j >= 1 && j < n - 1 && k >= 1 && k < n - 1) {
    out[i * n * n + j * n + k] = c0 * in[i * n * n + j * n + k] +
                                 c1 * in[i * n * n + j * n + (k - 1)] +
                                 c2 * in[i * n * n + j * n + (k + 1)] +
                                 c3 * in[i * n * n + (j - 1) * n + k] +
                                 c4 * in[i * n * n + (j + 1) * n + k] +
                                 c5 * in[(i - 1) * n * n + j * n + k] +
                                 c6 * in[(i + 1) * n * n + j * n + k];
  }
}

#define FILTER_RADIUS 2
#define IN_TILE_DIM 16
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)

__global__ void stencil_kernel_tiled(float *in, float *out, unsigned int n) {
  int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
  int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
  int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

  __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
  if (i >= 0 && i < n - 1 && j >= 0 && j < n - 1 && k >= 0 && k < n) {
    in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * n * n + j * n + k];
  }

  __syncthreads();

  if (i >= 1 && i < n - 1 && j >= 1 && j < n - 1 && k >= 1 && k < n - 1) {
    if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1 && threadIdx.y >= 1 &&
        threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 &&
        threadIdx.x < IN_TILE_DIM - 1) {
      out[i * n * n + j * n + k] =
          c0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x] +
          c1 * in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] +
          c2 * in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] +
          c3 * in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
          c4 * in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
          c5 * in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] +
          c6 * in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
    }
  }
}

void fill_data(float *data, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    data[i] = static_cast<float>(rand()) / RAND_MAX;
  }
}

int main() {
  size_t total_elems = N * N * N;
  size_t total_bytes = total_elems * sizeof(float);

  float *h_input = (float *)malloc(total_bytes);
  float *h_output = (float *)malloc(total_bytes);
  fill_data(h_input, total_elems);

  float *d_input, *d_output;
  cudaMalloc(&d_input, total_bytes);
  cudaMalloc(&d_output, total_bytes);

  cudaMemcpy(d_input, h_input, total_bytes, cudaMemcpyHostToDevice);

  dim3 block(8, 8, 8);
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y,
            (N + block.z - 1) / block.z);

  auto start = std::chrono::high_resolution_clock::now();
  stencil_kernel<<<grid, block>>>(d_input, d_output, N);
  cudaMemcpy(h_output, d_output, total_bytes, cudaMemcpyDeviceToHost);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << std::fixed << std::setprecision(3)
            << "basic kernel: " << elapsed.count() << " ms" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  stencil_kernel_tiled<<<grid, block>>>(d_input, d_output, N);
  cudaMemcpy(h_output, d_output, total_bytes, cudaMemcpyDeviceToHost);
  end = std::chrono::high_resolution_clock::now();

  elapsed = end - start;
  std::cout << std::fixed << std::setprecision(3)
            << "tiled shared memory: " << elapsed.count() << " ms" << std::endl;

  cudaFree(d_input);
  cudaFree(d_output);
  free(h_input);
  free(h_output);

  return 0;
}
