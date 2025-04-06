#include <cstdlib>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <iomanip>

#define eps 1e-2f

__global__ void vector_add_kernel(float *a, float *b, float *c, int n)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i += stride)
  {
    c[i] = a[i] + b[i];
  }
}

void generate_vector(float *m, int n)
{
  for (size_t i = 0; i < n; i++)
  {
    m[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
  }
}

void print_assert_matrix(float *matrix_gpu, float *matrix_cpu, int n)
{
  for (int i = 0; i < n; i++)
  {
    float gpu_val = matrix_gpu[i];
    float cpu_val = matrix_cpu[i];
    float diff = fabs(gpu_val - cpu_val);

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "i=" << std::setw(2) << i
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

void vector_add_cpu(float *a, float *b, float *c, int n)
{
  for (int i = 0; i < n; i++)
  {
    c[i] = a[i] + b[i];
  }
}

int main()
{
  srand(0);
  int n = 256;
  size_t size = n * sizeof(float);

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

  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
  printf("Device ID: %d\tNumber of SMs: %d\n", deviceId, numberOfSMs);

  size_t blockSize = 16;
  size_t dimGrid = 32 * numberOfSMs;

  vector_add_kernel<<<dimGrid, blockSize>>>(a_d, b_d, c_d, n);
  cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

  vector_add_cpu(a_h, b_h, c_ref, n);

  print_assert_matrix(c_h, c_ref, n);
}