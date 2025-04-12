#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#define eps 1e-2f

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_tiled_kernel(float *a, float *b, size_t n) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  size_t x = blockIdx.x * TILE_DIM + threadIdx.x;
  size_t y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (size_t i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    tile[threadIdx.y + i][threadIdx.x] = b[(y + i) * n + x];
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (size_t i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    a[(y + i) * n + x] = tile[threadIdx.x][threadIdx.y + i];
  }
}

__global__ void transpose_kernel(float *a, float *b, size_t n) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    b[col * n + row] = a[row * n + col];
  }
}

void generate_matrix(float *m, size_t n) {
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      m[i * n + j] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
    }
  }
}

void print_assert_matrix(float *matrix_gpu, float *matrix_cpu, size_t n) {
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      float gpu_val = matrix_gpu[i * n + j];
      float cpu_val = matrix_cpu[i * n + j];
      float diff = fabs(gpu_val - cpu_val);

      std::cout << std::fixed << std::setprecision(6);

      std::cout << "i=" << std::setw(2) << i << " j=" << std::setw(2) << j
                << " | GPU: " << std::setw(10) << gpu_val
                << " | CPU: " << std::setw(10) << cpu_val
                << " | DIFF: " << std::setw(10) << diff;

      if (diff >= eps) {
        std::cout << "  <-- ASSERT FAILED";
        std::cout << std::endl;
        assert(diff < eps);
      }

      std::cout << std::endl;
    }
  }
}

void transpose_cpu(float *a, float *b, int n) {
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      b[j * n + i] = a[i * n + j];
    }
  }
}

int main() {
  srand(0);
  size_t n = 1 << 12;
  size_t size = n * n * sizeof(float);

  float *a_h = (float *)malloc(size);
  float *b_h = (float *)malloc(size);
  float *b_ref = (float *)malloc(size);

  generate_matrix(a_h, n);

  float *a_d, *b_d;
  cudaMalloc((void **)&a_d, size);
  cudaMalloc((void **)&b_d, size);

  cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

  dim3 dimGrid((n + TILE_DIM - 1) / TILE_DIM,
               (n + TILE_DIM - 1) / TILE_DIM);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS);


  auto start = std::chrono::high_resolution_clock::now();
  transpose_kernel<<<dimGrid, dimBlock>>>(a_d, b_d, n);
  cudaMemcpy(b_h, b_d, size, cudaMemcpyDeviceToHost);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> duration = end - start;

  std::cout << std::fixed << std::setprecision(3) << std::showpoint
            << "gpu: " << duration.count() << " ms" << std::endl;

  cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
  start = std::chrono::high_resolution_clock::now();
  transpose_tiled_kernel<<<dimGrid, dimBlock>>>(a_d, b_d, n);
  cudaMemcpy(b_ref, b_d, size, cudaMemcpyDeviceToHost);
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;

  std::cout << std::fixed << std::setprecision(3) << std::showpoint
            << "gpu tiled: " << duration.count() << " ms" << std::endl;

  // print_assert_matrix(b_h, b_ref, n);
}
