#include <cstdlib>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <iomanip>

#define eps 1e-2f

__global__ void transpose_kernel(float *a, float *b, size_t n)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= n || col >= n)
  {
    return;
  }

  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = 0; j < n; j++)
    {
      b[j * n + i] = a[i * n + j];
    }
  }
}

void generate_matrix(float *m, size_t n)
{
  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = 0; j < n; j++)
    {
      m[i * n + j] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
    }
  }
}

void print_assert_matrix(float *matrix_gpu, float *matrix_cpu, size_t n)
{
  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = 0; j < n; j++)
    {
      float gpu_val = matrix_gpu[i * n + j];
      float cpu_val = matrix_cpu[i * n + j];
      float diff = fabs(gpu_val - cpu_val);

      std::cout << std::fixed << std::setprecision(6);

      std::cout << "i=" << std::setw(2) << i
                << " j=" << std::setw(2) << j
                << " | GPU: " << std::setw(10) << gpu_val
                << " | CPU: " << std::setw(10) << cpu_val
                << " | DIFF: " << std::setw(10) << diff;

      if (diff >= eps)
      {
        std::cout << "  <-- ASSERT FAILED";
        std::cout << std::endl;
        assert(diff < eps);
      }

      std::cout << std::endl;
    }
  }
}

void transpose_cpu(float *a, float *b, int n)
{
  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = 0; j < n; j++)
    {
      b[j * n + i] = a[i * n + j];
    }
  }
}

int main()
{
  srand(0);
  size_t n = 16;
  size_t size = n * n * sizeof(float);

  float *a_h = (float *)malloc(size);
  float *b_h = (float *)malloc(size);
  float *b_ref = (float *)malloc(size);

  generate_matrix(a_h, n);

  float *a_d, *b_d;
  cudaMalloc((void **)&a_d, size);
  cudaMalloc((void **)&b_d, size);

  cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

  size_t blockSize = 16;
  dim3 dimGrid((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);
  dim3 dimBlock(blockSize, blockSize);

  transpose_kernel<<<dimGrid, dimBlock>>>(a_d, b_d, n);
  cudaMemcpy(b_h, b_d, size, cudaMemcpyDeviceToHost);

  transpose_cpu(a_h, b_ref, n);

  print_assert_matrix(b_h, b_ref, n);
}