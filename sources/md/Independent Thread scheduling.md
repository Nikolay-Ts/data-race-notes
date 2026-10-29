#GPU #cuda #sync 

[source](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)

On GPUs with a compute capability lower than 7.0, [[GPU Memory Consistency#GPU Architecture|warps]] used a single PC shared all 32 threads in the warp alongside an active mask which specified which threads of the warp were active. As a result, threads form the same warp in divergent regions or states of execution were not able to share or signal info among themselves in fine-grained sharing of data guarded by locks or mutex, could lead to a dead lock. 

GPUs from 7.0 and onward, have Independent thread scheduling which allows full concurrency between threads, regardless of warp. With ITS, the GPU maintains execution state per thread, including a PC and call stack, and can **yield** execution at the per-thread granularity, either to make use of the execution or to allow one thread to wait for data to be produced by the other. A scheduler decides how to group active threads from the same warp into [[GPU Memory Consistency#GPU Architecture|SIMT]] units. 

Now threads can diverge on the sub-warp level.

However, ITS can break code that relies on implicit warp-sync behavior from previous GPU architectures. Warp-sync codes assumes that threads in the same warp execute in lockstep at every instruction. Thus, new kernels should not assume this and should use `__syncwarp()` where necessary to ensure correct behavior across GPU generations. 