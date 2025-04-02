# Day 1

File: [vec-add.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-001/vec-add.cu)

Implemtented a basic vector addition cuda kernel. Followed the pmpp book until chapter 2. Read about the reason of gpu and cpu being different in latency and throughput focus.

What i do:

- created a basic working cuda kernel code to add two vector
- read about thread, block, grid memorry hierarchy in cuda
- read about how to allocate memorry in device `cudaMalloc`, how to copy memory from host to device and vice verca `cudaMemcpy`. and learn how to free the memory `cudaFree`

What resource i use:

- read chapter 1 and 2 pmpp book

# PMPP book chap 1

The general idea i got is cpu is optimized for latency, and gpu is optimized for throughput.  
That way gpu can have more die area on more cores compared to cpu who use that resource for less latency.  
Gpu is throughput oriented.

# PMPP book chap 2

this chapter introduce to basic cuda programming like what the memorry hierarchy, how to malloc, how to copy, how to free. the general idea of what being executed when calling a cuda kernel.
