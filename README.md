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
