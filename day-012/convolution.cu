#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

#define FILTER_RADIUS 2
#define TILE_DIM 32
#define FILTER_WIDTH (2 * FILTER_RADIUS + 1)

// extern __constant__ float F_const[FILTER_WIDTH * FILTER_WIDTH];
// extern __constant__ float F_c[FILTER_WIDTH * FILTER_WIDTH];

// Kernel 1: Basic 2D Convolution
__global__ void convolution_2D_basic_kernel(float *N, float *F, float *P, int r,
                                            int width, int height) {
  int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  int outRow = blockIdx.y * blockDim.y + threadIdx.y;
  float Pvalue = 0.0f;

  for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
    for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
      int inRow = outRow - r + fRow;
      int inCol = outCol - r + fCol;

      if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
        Pvalue += F[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
      }
    }
  }
  P[outRow * width + outCol] = Pvalue;
}

// Kernel 2: Using Constant Memory
__constant__ float F_const[FILTER_WIDTH * FILTER_WIDTH];
__global__ void convolution_2D_const_mem_kernel(float *N, float *P, int r,
                                                int width, int height) {
  int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  int outRow = blockIdx.y * blockDim.y + threadIdx.y;
  float Pvalue = 0.0f;

  for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
    for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
      int inRow = outRow - r + fRow;
      int inCol = outCol - r + fCol;

      if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
        Pvalue += F_const[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
      }
    }
  }
  P[outRow * width + outCol] = Pvalue;
}

// Kernel 3: Tiled Shared Memory + Constant Memory
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)
__constant__ float F_c[(2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1)];
__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P,
                                                      int width, int height) {
  int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
  int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

  __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];

  if (row >= 0 && row < height && col >= 0 && col < width)
    N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
  else
    N_s[threadIdx.y][threadIdx.x] = 0.0f;

  __syncthreads();

  int tileCol = threadIdx.x - FILTER_RADIUS;
  int tileRow = threadIdx.y - FILTER_RADIUS;

  if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 &&
      tileRow < OUT_TILE_DIM && col >= 0 && col < width && row >= 0 &&
      row < height) {
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
      for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
        Pvalue += F_c[fRow * (2 * FILTER_RADIUS + 1) + fCol] *
                  N_s[tileRow + fRow][tileCol + fCol];
      }
    }
    P[row * width + col] = Pvalue;
  }
}

// Kernel 4: Cached Global with Fallback
__global__ void convolution_cached_tiled_2D_const_mem_kernel(float *N, float *P,
                                                             int width,
                                                             int height) {
  int col = blockIdx.x * TILE_DIM + threadIdx.x;
  int row = blockIdx.y * TILE_DIM + threadIdx.y;

  __shared__ float N_s[TILE_DIM][TILE_DIM];

  if (row < height && col < width)
    N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
  else
    N_s[threadIdx.y][threadIdx.x] = 0.0f;

  __syncthreads();

  if (col < width && row < height) {
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
      for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
        int localRow = threadIdx.y - FILTER_RADIUS + fRow;
        int localCol = threadIdx.x - FILTER_RADIUS + fCol;
        if (localRow >= 0 && localRow < TILE_DIM && localCol >= 0 &&
            localCol < TILE_DIM) {
          Pvalue += F_c[fRow * (2 * FILTER_RADIUS + 1) + fCol] *
                    N_s[localRow][localCol];
        } else {
          int globalRow = row - FILTER_RADIUS + fRow;
          int globalCol = col - FILTER_RADIUS + fCol;
          if (globalRow >= 0 && globalRow < height && globalCol >= 0 &&
              globalCol < width) {
            Pvalue += F_c[fRow * (2 * FILTER_RADIUS + 1) + fCol] *
                      N[globalRow * width + globalCol];
          }
        }
      }
    }
    P[row * width + col] = Pvalue;
  }
}

void generate_input(float *input, int width, int height) {
  for (int i = 0; i < width * height; i++) {
    input[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
  }
}

void run_and_time_kernel(const char *label, int width, int height,
                         void (*kernel)(float *, float *, float *, int, int,
                                        int),
                         float *d_input, float *d_output, float *d_filter,
                         int r) {

  float *d_in, *d_out, *d_f;

  cudaMalloc(&d_in, width * height * sizeof(float));
  cudaMalloc(&d_out, width * height * sizeof(float));
  d_f = (float *)malloc(width * height * sizeof(float));

  cudaMemcpy(d_in, d_input, width * height * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, d_output, width * height * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_f, d_filter, FILTER_WIDTH * FILTER_WIDTH * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 blockSize(TILE_DIM, TILE_DIM);
  dim3 gridSize((width + TILE_DIM - 1) / TILE_DIM,
                (height + TILE_DIM - 1) / TILE_DIM);

  auto start = std::chrono::high_resolution_clock::now();
  kernel<<<gridSize, blockSize>>>(d_in, d_f, d_out, r, width, height);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << label << ": " << duration.count() << " ms\n";

  cudaMemcpy(d_output, d_out, width * height * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_f);
}

int main() {
  srand(0);
  const int width = 1 << 14;
  const int height = 1 << 14;

  const int r = FILTER_RADIUS;
  const size_t size = width * height * sizeof(float);
  const size_t filter_size = FILTER_WIDTH * FILTER_WIDTH * sizeof(float);

  float *h_input = (float *)malloc(size);
  float *h_output = (float *)malloc(size);
  float *h_filter = (float *)malloc(filter_size);

  generate_input(h_input, width, height);
  generate_input(h_filter, FILTER_WIDTH, FILTER_WIDTH);

  cudaMemcpyToSymbol(F_const, h_filter, filter_size);
  cudaMemcpyToSymbol(F_c, h_filter, filter_size);

  float *d_input, *d_output;
  cudaMalloc(&d_input, size);
  cudaMalloc(&d_output, size);
  cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

  dim3 blockSize(TILE_DIM, TILE_DIM);
  dim3 gridSize((width + TILE_DIM - 1) / TILE_DIM,
                (height + TILE_DIM - 1) / TILE_DIM);

  auto start = std::chrono::high_resolution_clock::now();
  convolution_2D_basic_kernel<<<gridSize, blockSize>>>(
      d_input, h_filter, d_output, r, width, height);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "Basic Kernel: " << duration.count() << " ms\n";

  cudaMemcpy(d_output, d_input, size, cudaMemcpyDeviceToDevice);
  start = std::chrono::high_resolution_clock::now();
  convolution_2D_const_mem_kernel<<<gridSize, blockSize>>>(d_input, d_output, r,
                                                           width, height);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "Constant Memory Kernel: " << duration.count() << " ms\n";

  cudaMemcpy(d_output, d_input, size, cudaMemcpyDeviceToDevice);
  start = std::chrono::high_resolution_clock::now();
  convolution_tiled_2D_const_mem_kernel<<<gridSize, blockSize>>>(
      d_input, d_output, width, height);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "Tiled Shared Memory Kernel: " << duration.count() << " ms\n";

  cudaMemcpy(d_output, d_input, size, cudaMemcpyDeviceToDevice);
  start = std::chrono::high_resolution_clock::now();
  convolution_cached_tiled_2D_const_mem_kernel<<<gridSize, blockSize>>>(
      d_input, d_output, width, height);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "Cached Shared+Global Kernel: " << duration.count() << " ms\n";

  cudaFree(d_input);
  cudaFree(d_output);
  free(h_input);
  free(h_output);
  free(h_filter);

  return 0;
}
