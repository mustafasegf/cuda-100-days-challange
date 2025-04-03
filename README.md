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