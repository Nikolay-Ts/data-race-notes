
#data-race #GPU 

[[asplos21_predatar]]

## Introduction

Modern GPUs now also support fin-grained inter-thread communication. synchronizing these using CUDA is becoming more difficult than on the CPU. Currently some methods are to use memory barriers and atomic operations with augmented scope to protect concurrent accesses of shared variables. 

A data race is where you have concurrent accesses to the same global or shared memory with incorrect synchronization such that both accesses are not `atomic` and at least one of them is a `write`. 

The difficulty comes with the fact that data-races are `non-deterministic` and are not easy to reproduce always. The goal is then to be able to detect as many data-races as possible by observing only a few (ideally one) executions. 

Techniques such as the [[Happens-Before Relation|happens before relation]] has a limited predictability since the number of races detected is impacted by spurious ordering among concurrent events. Most current GPU data-race detectors are not predictive  For example [[ScoRD]] uses locket system analysis, thus having a limited predictive capability but misses some classes of predictable data-races.

This paper introduces  a predictive partial order relation `GWCP` for race detection on the GPU. They claim that GWCP has no false positives. GWCP builds upon WCP to account for GPUs.

This paper implements *PreDataR*, a dynamic analysis that tracks the GWCP relations using a vector clock. This can lead to prohibitive runtime and memory overheads however. 

Threads within the same warp cannot have a data-race since they all share the same program counter. However, interwarp races can arise, meaning two threads from different warps, could write to the same address concurrently.

Example of interwarp DR

```c
__global__
void interwarp(unsigned *data) {
	data[0] = threadIdx.x;
}

__global__
void interblk_scop(unsigned *data) {
	atomicExch_block(data, threadIdx.x);
}
```