#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#define eps 1e-2f

#define TILE_WIDTH 16

__global__ void matmul_tiled_kernel(float *a, float *b, float *c, int n) {

  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float sum = 0;

  for (int p = 0; p < n / TILE_WIDTH; ++p) {
    Mds[ty][tx] = a[Row * n + (p * TILE_WIDTH) + tx];
    Nds[ty][tx] = b[((p * TILE_WIDTH) + ty) * n + Col];
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
      sum += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }

  c[Row * n + Col] = sum;
}

__global__ void matmul_kernel(float *a, float *b, float *c, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= n || col >= n) {
    return;
  }

  float sum = 0.0;

  for (size_t i = 0; i < n; i++) {
    sum += a[row * n + i] * b[i * n + col];
  }

  c[row * n + col] = sum;
}

void generate_vector(float *m, int n) {
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      m[i * n + j] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
    }
  }
}

void print_assert_matrix(float *matrix_gpu, float *matrix_cpu, int n) {
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      float gpu_val = matrix_gpu[i * n + j];
      float cpu_val = matrix_cpu[i * n + j];
      float diff = fabs(gpu_val - cpu_val);

      std::cout << std::fixed << std::setprecision(6);
      std::cout << "i=" << std::setw(2) << i << " j=" << std::setw(2) << j
                << " | GPU: " << std::setw(10) << gpu_val
                << " | CPU: " << std::setw(10) << cpu_val
                << " | DIFF: " << std::setw(10) << diff;

      // if (diff >= eps) {
      //   std::cout << "  <-- ASSERT FAILED";
      //   std::cout << std::endl;
      //   assert(diff < eps);
      // }

      std::cout << std::endl;
    }
  }
}

void matmul_cpu(float *a, float *b, float *c, int n) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int k = 0; k < n; k++)
        sum += a[i * n + k] * b[k * n + j];
      c[i * n + j] = sum;
    }
}

int main() {
  srand(0);
  int n = 1 << 10;
  size_t size = n * n * sizeof(float);

  float *a_h = (float *)malloc(size);
  float *b_h = (float *)malloc(size);
  float *c_h = (float *)malloc(size);
  float *c_ref = (float *)malloc(size);

  generate_vector(a_h, n);
  generate_vector(b_h, n);

  float *a_d, *b_d, *c_d;
  cudaMalloc((void **)&a_d, size);
  cudaMalloc((void **)&b_d, size);
  cudaMalloc((void **)&c_d, size);

  cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

  size_t blockSize = 16;
  dim3 dimGrid((n + blockSize - 1) / blockSize,
               (n + blockSize - 1) / blockSize);
  dim3 dimBlock(blockSize, blockSize);

  auto start = std::chrono::high_resolution_clock::now();
  matmul_kernel<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, n);
  cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> duration = end - start;

  std::cout << std::fixed << std::setprecision(3) << std::showpoint
            << "gpu: " << duration.count() << " ms" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  matmul_tiled_kernel<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, n);
  cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;

  std::cout << std::fixed << std::setprecision(3) << std::showpoint
            << "gpu tiled: " << duration.count() << " ms" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  matmul_cpu(a_h, b_h, c_ref, n);
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;

  std::cout << std::fixed << std::setprecision(3) << std::showpoint
            << "cpu: " << duration.count() << " ms" << std::endl;

  print_assert_matrix(c_h, c_ref, n);
}
