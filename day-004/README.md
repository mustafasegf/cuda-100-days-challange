# Day 4

File: [matmul.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-004/matmul.cu)

Implemented simple rectangular matmul in cuda

what i do:
- implemented matmul in cuda with assert validation with c++ cpu host code

## PMPP book chap 3
for the matmul, we do thread to data mapping. every data on the output is mapped to one thread. since the matmul require row and column from two matrix, we loop both of them to get all the data.