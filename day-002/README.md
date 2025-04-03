# Day 2

File: [colorToGrayscale.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-002/colorToGrayscale.cu)

Implemented simple bitmap 24 bit to 8bit bitmap grayscale conversion using 2d cuda kernel.

what i do: 
Doing the pmpl chapter 2 excercise
```cu
// 1. C
// i = blockIdx.x * blockDim.x + threadIdx.x;
// simple arithmatic to calculate the index

// 2. C
// i = (blockIdx.x * blockDim.x + threadIdx.x) / 2;
// we want to i times two, so we can calcualte the i + 1 too for the adjacent vec

__global__ void vecAdd(float* A, float* B, float* C, int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) / 2;
    // i is 0, 2, 4, 6, ...
    if (i < N) {
        C[i] = A[i] + B[i];
        C[i + 1] = A[i + 1] + B[i + 1];
    }
}

// 3. D
// i = blockIdx.x * 2 *blockDim.x + threadIdx.x;
// we want blockDim to skip every one calculation, so we can calcualte the i + blockDim.x too for the adjacent block

__global__ void vecAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * 2 *blockDim.x + threadIdx.x;
    // i is 0, 1, 2, 3... blockdim.x, * 2, blockdim.x * 2 + 1, ... 

    if (i < N) {
        C[i] = A[i] + B[i];
    }

    i += blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// 4. C
// thread block size = 1024, n = 8000. so block = ceil(8000 / 1024) = 8. 
// thread = 8 * 1024 = 8192

// 5. D
// because malloc expect the size of byte, so need to be mult by sizeof

// 6. D
// because cuda malloc expect pointer of void*, we need to cast to (void**) and use &A_d to get the pointer

// 7. C
// cudaMemcpy use, to, from, size, flag as their argument

// 8. C

// 9a. 128
// 9b. 200.064
// 9c. 1564
// 9d. 200.064
// 9e. 200.000

// 10. use `__host__` and `__device__` to compile the code twice for the host and device
```

What resource i use:
- excersice pmpp chapter 2
- first part pmpp chapter 3 for grayscale

## PMPP book chap 3
the general idea i got is thre's more than one way to organize the array. we can use up to 3 dimention in the threadIdx and in the blockIdx. the max amount of thread we can spawn is 1024. but theres not limit on how much block we can spawn. how we organize our data to 1, 2, or 3d depends on what the data looks like.
