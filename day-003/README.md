# Day 3

File: [blur.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-003/blur.cu)

Implemented simple blur kernel for 24bit bitmap color using 2d cuda kernel.

what i do:

- implemented modified blurKernel from chapter 3 pmpp.

## PMPP book chap 3

the general idea for the blur kernel is to read area around the index. sicne i want to use 24bit bitmap, i need to index the r, g, b seperatedly. then average the pixel amount.
