# Writing cuda kerne in 100 days chalanges

jensen i need a job

Doing it with @wreckitral

# Day 1

File: [vec-add.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-001/vec-add.cu)

Implemtented a basic vector addition cuda kernel. Followed the pmpp book until chapter 2. Read about the reason of gpu and cpu being different in latency and throughput focus.

What i do:

- created a basic working cuda kernel code to add two vector
- read about thread, block, grid memorry hierarchy in cuda
- read about how to allocate memorry in device `cudaMalloc`, how to copy memory from host to device and vice verca `cudaMemcpy`. and learn how to free the memory `cudaFree`

What resource i use:

- read chapter 1 and 2 pmpp book

# Day 2

File: [colorToGrayscale.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-002/colorToGrayscale.cu)

Implemented simple bitmap 24 bit to 8bit bitmap grayscale conversion using 2d cuda kernel.

what i do:

- Doing the pmpl chapter 2 excercise
- read about thread can be 1, 2,or 3D. that also included block. thread in block can be up to 1024, where block can be much higher.

What resource i use:

- excersice pmpp chapter 2
- first part pmpp chapter 3 for grayscale

# Day 3

File: [blur.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-003/blur.cu)

Implemented simple blur kernel for 24bit bitmap color using 2d cuda kernel.

what i do:

- implemented modified blurKernel from chapter 3 pmpp.

## PMPP book chap 3

the general idea for the blur kernel is to read area around the index. sicne i want to use 24bit bitmap, i need to index the r, g, b seperatedly. then average the pixel amount.

# Day 4

File: [matmul.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-004/matmul.cu)

Implemented simple rectangular matmul in cuda

what i do:

- implemented matmul in cuda with assert validation with c++ cpu host code

## PMPP book chap 3

for the matmul, we do thread to data mapping. every data on the output is mapped to one thread. since the matmul require row and column from two matrix, we loop both of them to get all the data.

# Day 5

File: [vector-add-sm.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-005/vector-add-sm.cu)

Implemented vector addition but check the amount of sm i have and spawn the block size appropriately.

what i do:

- Implemented vector addition

## PMPP book chap 4

this chapter talks about artchitecture and scheduling. it talks about gpu organized to sm, sm have multiple procesing blocks that share control logic and memory resource. when grid is launched, the block assigned to sm. in each sm, per 32 thread is bundled as a warp. one warp execute the same instruction simultaneously. in cuda, we can't assume the timing any of the thread. if we want to synchronize all of the thread to wait for each other, we can use `__syncthreads()`.
