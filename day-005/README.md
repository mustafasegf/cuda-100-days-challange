# Day 5

File: [vector-add-sm.cu](https://github.com/mustafasegf/cuda-100-days-challange/blob/master/day-005/vector-add-sm.cu)

Implemented vector addition but check the amount of sm i have and spawn the block size appropriately.

what i do:

- Implemented vector addition

## PMPP book chap 4

this chapter talks about artchitecture and scheduling. it talks about gpu organized to sm, sm have multiple procesing blocks that share control logic and memory resource. when grid is launched, the block assigned to sm. in each sm, per 32 thread is bundled as a warp. one warp execute the same instruction simultaneously. in cuda, we can't assume the timing any of the thread. if we want to synchronize all of the thread to wait for each other, we can use `__syncthreads()`.
