#include <cassert>
#include <cstdlib>

__global__ void vecAddKernel(float *A, float *B, float *C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

void vecAdd(float *A, float *B, float *C, int n) {
  float *A_d, *B_d, *C_d;
  int size = n * sizeof(float);

  cudaMalloc((void **)&A_d, size);
  cudaMalloc((void **)&B_d, size);
  cudaMalloc((void **)&C_d, size);

  cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

  vecAddKernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

  cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

int main() {
  int n = 128;
  size_t size = n * sizeof(float);
  float *A_h = (float *)malloc(size);
  float *B_h = (float *)malloc(size);
  float *C_h = (float *)malloc(size);

  for (size_t i = 0; i < n; i++) {
    A_h[i] = 1;
    B_h[i] = 2;
  }

  vecAdd(A_h, B_h, C_h, n);

  for (size_t i = 0; i < n; i++) {
    assert(C_h[i] == 3);
  }

  return 0;
}
