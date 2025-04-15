# Writing cuda kerne in 100 days chalanges

jensen i need a job

Doing it with @wreckitral

## Day 1

File: [vec-add.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-001/vec-add.cu)

Implemtented a basic vector addition cuda kernel. Followed the pmpp book until chapter 2. Read about the reason of gpu and cpu being different in latency and throughput focus.

What i do:

- created a basic working cuda kernel code to add two vector
- read about thread, block, grid memorry hierarchy in cuda
- read about how to allocate memorry in device `cudaMalloc`, how to copy memory from host to device and vice verca `cudaMemcpy`. and learn how to free the memory `cudaFree`

What resource i use:

- read chapter 1 and 2 pmpp book

## Day 2

File: [colorToGrayscale.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-002/colorToGrayscale.cu)

Implemented simple bitmap 24 bit to 8bit bitmap grayscale conversion using 2d cuda kernel.

what i do:

- Doing the pmpl chapter 2 excercise
- read about thread can be 1, 2,or 3D. that also included block. thread in block can be up to 1024, where block can be much higher.

What resource i use:

- excersice pmpp chapter 2
- first part pmpp chapter 3 for grayscale

## Day 3

File: [blur.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-003/blur.cu)

Implemented simple blur kernel for 24bit bitmap color using 2d cuda kernel.

what i do:

- implemented modified blurKernel from chapter 3 pmpp.

### PMPP book chap 3

the general idea for the blur kernel is to read area around the index. sicne i want to use 24bit bitmap, i need to index the r, g, b seperatedly. then average the pixel amount.

## Day 4

File: [matmul.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-004/matmul.cu)

Implemented simple rectangular matmul in cuda

what i do:

- implemented matmul in cuda with assert validation with c++ cpu host code

### PMPP book chap 3

for the matmul, we do thread to data mapping. every data on the output is mapped to one thread. since the matmul require row and column from two matrix, we loop both of them to get all the data.

## Day 5

File: [vector-add-sm.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-005/vector-add-sm.cu)

Implemented vector addition but check the amount of sm i have and spawn the block size appropriately.

what i do:

- Implemented vector addition

### PMPP book chap 4

this chapter talks about artchitecture and scheduling. it talks about gpu organized to sm, sm have multiple procesing blocks that share control logic and memory resource. when grid is launched, the block assigned to sm. in each sm, per 32 thread is bundled as a warp. one warp execute the same instruction simultaneously. in cuda, we can't assume the timing any of the thread. if we want to synchronize all of the thread to wait for each other, we can use `__syncthreads()`.

## Day 6

File: [transpose-matrix.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-006/transpose-matrix.cu)

implemented matrix transpose kernel

what i do:

- Implemented matrix transpose

didn't have the energy to read pmpp book so implemented the simple kernel

## Day 7

File: [sobel.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-007/sobel.cu)

implemented sobel edge detection kernel

what i do:

- Implemented sobel edge detection kernel

didn't have the energy again today to read pmpp book so implemented the simple kernel


## Day 8

File: [matmul-tiled.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-008/matmul-tiled.cu)

implemented matmul with tiled kernel

what i do
- read pmpp book chapter 5
- implemented matmul wit shared memory

### PMPP book chap 5
this chapter discus about how memory have latency and there's different rank of memory. we need to choose memory to reduce the latency.


## Day 9

File: [blur-tiled.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-009/blur-tiled.cu)

implemented blur with tiled kernel

what i do
- implemented blur with shared memory

## Day 10

File: [matmul-thread-coarsening.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-010/matmul-thread-coarsening.cu)

implemented matmul with coarsed thread

what i do
- read pmpp chap 6
- implemented matmul with coarsed thread

### PMPP book chap 6
this chapter discuss about performance optimization. it talks about the memory access pattern and the speed of memory acess. it talks on what kind of optimization that we can do.

## Day 11

File: [transpose-shared-memory-bank.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-011/transpose-shared-memory-bank.cu)

implemented transpsoe matrix with shared memory bank


what i do
- implemented transpsoe matrix with shared memory bank


## Day 12

File: [convolution.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-012/convolution.cu)

implemented convolution in 4 different way

what i do
- read pmpp book chapter 7
- implemented convolution kernel
- implemented convolution with const memory
- implemented convolution with shared memory
- implemented convolution with shared memory + global kernel

### PMPP book chap 7
this chapter discus about how to do convolution kernel and different way to optimize it. it shows concretely how to optimize it steps by steps. talks more about how to make better data access pattern
