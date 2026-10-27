#data-race #GPU #instrumentation #recompilation


[[super-collider]]

![[Screenshot From 2026-10-26 15-53-22.png]]

compilation process to use SuperCollider. 

## Introduction

GPUs are becoming more and more popular in everyday compute, however data-races are still a crux for programmers. A data-race occurs when two or more threads can concurrently access the same memory without synchronization tools and one of the threads must commit a write operation. 

Cuda organizes kernels into a grid of blocks which are themselves groups of 32 threads called [[GPU Memory Consistency#GPU Architecture|warps]]. Each kernel can execute concurrently with other kernels. The [[CUDA Programming Model#Thread Hierarchy|thread hierarchy]] is mirrored with multiple levels of memory hierarchy, starting from global -> shared -> local, which means that a data race can occur anywher:
- Between threads within a warp 
- Within warps in the same block
- Threads form different blocks 
- Threads in different kernels
- Threads in different GPUs
- Threads in GPUs and CPUs

SOTA dynamic GPU race detectors operate by tracking mem sync operation performed by every thread in the kernel. Tools such as [[iGuard]], [[Compute Sanitizer]] do it. However track all of that state is a tricky proposition. Many cuda kernel launch tens of thousands of threads collectively accessing multiple gigabytes of data. 

Without a shock, iGuard and [[MSc/M1/data-races/lit-notes/races/HiRace]] required up to 17x and 19x memory to store all of the metadata in order to track the mem. 

This paper introduces SuperCollider which takes a stateless and sync agnostic approach to detect races in cuda applications, inspired by DataCollider. DataCollider applies a transformation to all `weak loads` and `store` in a device-side cuda code. The core idea: a compile pass expands every weak memory operation in the program (excluding strong mem and atomics) with two accesses to the same address separated by a random delay followed by an equality check of the values fetch from mem. The idea is simple: **if back-to-back loads separated by a random delay return different values, then we can say that there is a data race**

Because weak mem loads are the majority of GPU operations, SuperCollider is very effective. SuperCollider also, as a result, allows for cross-kernel data-race detection, across OS, between CPU and GPU and across multiple GPUs. 

## Methodology

Because there are so many threads with so many data access points, you cannot do random sampling. Instead, they chose a `redudant-read` approach and extend it to hand lost updates, leveraging peculiarities of [[GPU Memory Consistency#GPU Architecture|SIMT]] and async mem operations. SuperCollider checks every weak mem spot. 

### Read Instrumentation: Clobbered Read

```
// pre compile 
mul vr0 = vr1, 4
add vr2 = %adrr, vr0
read.global vr3 = [vr2] // weak load
mul vr4 = vr3, 3.0 

// instrumented code
mul vr0 = vr1, 4
add vr2 = %adrr, vr0
read.global vr3 = [vr2] // weak load
nanosleep.rand 
readl.global.sys.strong vr5 [vr2]
if vr3 != vr5 {
	notify clobbered read error 
}
mul vr4 = vr3, 3.0
```

As seen, for all weak operations to the global, shared, or generic me spaces, the instrumentation preserves the original operation and  introduces a strong, system-scoped duplicate load. This prevents the compiler from reordering the duplicate but also bypasses the cache hierarchy . 

The instrumentation also inserts a random sleep between the weak and strong load. Intuitively, the longer the sleep window, the more likely the technique is to expose latent races. 

The compiler uses `generic` addresses, when it cannot disambiguate the memory space to which a pointer maps. Therefore, reads with a generic addr cant target local, shared or global mem. Local mem cannot lead to data-races, but it is not harmless to check.

### Write Instrumentation: Lost Updates 

```
// universal lost update 
write.globa [%addr] = vr0 // original store
nanosleep.rand // shuffle warps
read.global.sys.strong vr1 [%addr]
if vr0 != vr1 {
	notify lost update
}
```

This is what the compiler inserts for weak write operations to the global, shared or generic memory space. It  mirrors the instrumentation for reads, but the approach checks whether another thread overwrites this thread's update. 

One way to also check is to include an additional write of garbage data to cover the case where two threads do have overwrites but the value they write is the same. This is mentioned in the next case 

### Write Instrumentation: Intra-warp Lost Updates 

```
// intra-warp lost update instrumentation 
match vro = %addr
popcount vr1 = vr0 
if vr1 > 1 {
	notify lost_update
}

write.global [%addr] = vr2
```

GPUs rely on SIMT execution to efficiently amortize the work of an instruction over a warp. The snippet shows how SuperCollider leverages it to ensure that multiple threads in a warp do not write to the same addr.

The check use the `match` warp-collective operation to compare addresses across active threads and records if more than one thread write to the same addr. 

### Handling Async Copies

Hopper introduced `Tensor Memory Accelerator` (TMA), an async copy engine that allows for high-speed, bulk data movement between the GPU's off-chip global mem and on-chip shared mem without using any registers. 

This is exposed via `memcpy_async` and in PTX with `cp.async` & `cp.async.bukm`. Because these instructions are async w.r.t the issuing thread, the executing thread cannot access the shared me destination without intervening synchronization. Hence, unlike regular weak mem access that maintains per-thread [[1 - Foundations of Distributed Algorithms#Sequential Consistency|sequential consistency]], async copies can create intra-thread races in addition to inter-thread races. Also, because these instructions touch both global and shared me, races may exists at either the source or target and the instrumentation needs to insert redundant reads for both.   

Their implementation only covers bulk copies from share to global 

### Rejigerring Thread Scheduling 

SuperCollider is blind to syncing protocols. Unlike state-tracking race detectors, if a race condition does not physically manifest at runtime, SuperCollider will not catch it. Therefore, SuperCollider uses techniques to randomly perturb sub-warp, warp and block schedule to expose latent races at runtime. 

#### Warp & Sub-Warp Shuffling 

The nanosleep performes two tasks:
1. Putting the active threads to sleep for some random amount of time
2. Switching the GPU's scheduler to another ready sub-warp or warp 
The Volta architecture introduced the `threads-are-threads` programming model used by subsequent NVIDIA architectures. 

In this model, when a convergent group of threads in a warp sleep, the scheduler considers scheduling other sets of "ready" threads, either from the same or another warp, which can create conflicting sub-warps or warps, potentially exposing a data-race. 

#### Block Shuffling 

Consider the following example with a race between the first and last elements.

```c
__global__ void 
racy_add(const float *A, const float *B, float *C, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int ix = i == (N-1) ? 0 : i;
	if (i < N)  {
		C[ix] = A[i] + B[i] + 0.0f;
	}
}
```

On one end, consider a GPU with a single SM that executes thread blocks serially. SuperCollider would never expose the race for sufficiently large $N$, because the race would happen between different blocks. Even though in practice we could predict it, according to NVIDIA, thread blocks are required to execute independently, in any order or series or in parallel (i.e a well formed cuda kernel should have the same result regardless of the block execution). 

SuperCollider injects a kernel header that allows users to perturb block schedules which we call `block shuffling`. The approach pseudo-randomly alters the architecture's thread block schedule by rewriting `blockIdx.x` 
$$\text{blockIdx.x}_\text{new} = (a\cdot \text{blockIdx.x}+b) \text{ mod } N_{x} $$
where $N_x$ is the grid size in the x dimension, $a$ and $N_x$ are co-prime and all blocks aggree on $a$ & $b$. Block shuffling is then performed as a compiler transformation and does not require source code modification. However, it requires all code referencing `blockIdx.x` to be recompiled for correctness. 

### Summary 

SuperCollider is very scalable and in the absence of races is totally stateless apart from the short-lived thread-local variables. SuperCollider can seamlessly detect races within a thread, between threads and within a warp, within a block and within a kernel. 

## Implementation & Usage 

To use it, you must use a custom runtime and the other uses CUDA-GDB. Both rely on a modified compiler and the code must be recompiled. 

When compiling use the `--sanitize=supercollider` 
