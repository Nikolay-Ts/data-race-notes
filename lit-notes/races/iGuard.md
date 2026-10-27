#data-race #GPU #instrumentation

[[SOSP21_iGUARD]]
## Intro 

### Background

Current GPU vendors have made features such as independent thread scheduling to make GPUs faster but in some case they can also introduce synchronisation errors. 

iGuard is meant to be a tool to detect these race conditions while trying to minimise the run-time overhead. For these reasons it is directly on the GPU. 

Note: a GPU program is referred to as a kernel. 

The most basic kernels would use bulk-synchronisation where the data would get evenly dispersed among a max of 1024 equal-sized threadblocks (the GPU itself has hundreds of thousands of threads). The data in each threadblock is shared among the threads and is usually synchornised using a threadblock barrier `syncthreads` in CUDA.  

However this fails for more advanced GPU kernels and features (e.g Graph processing). A modern way to do this is to introduce a scope of the threadblock data such that only local threads can see it. This has major boost in performance but in a producer consumer case it leads to a race. GPUs schedule threads in a threadblock in groups of 32-64 threads called a `warp`. Usually the progress is lockstep (each unit advances) together , but threads can deadlock if they use independent locks. 

However with the introduction of `Independent Thread Scheduling`, it allows now threads in a warp to make independent progress. This means that application that relied on lockstep synchronisation would now need to introduce warp-level barriers `syncwrap`

### Limitations of Current Race Detectors

current ones can detect races in scratchpad memory but not in the GPU's global memory. Other tools do not support advanced features (e.g Barracuda does not support atomic operations). 

iGUARD uses thread-fences to create locks even though the GPU does not support it. It can infer if the threads are coarse grained (one lock per warp) or fine grained. It then uses the [[Lockset Algorithm|lockset algorithm]] to detect race conditions. 

## Background 

Current GPUs have a [[CUDA Programming Model#Thread Hierarchy|hierarchy]]  to scale hundreds of thousands of concurrent threads. A Streaming Multiprocessor (SM) is a basic execution block. Modern GPUs have around 100 and this block is usually SIMD. The SIMD units of a SM block share an L1 cache and scratchpad. The hardware decides the cache but the software decides the scratchpad. 

Note: All SM share an L2 cache. 

The global memory is accessible to all threads and is usually HBM or GDDR memory. 

## Race Conditions

### Race Conditions due to Scopes 

For CPUs a race condition occurs if we have a concurrent read and write and we need sync. For GPUs, sync alone cannot prevent a race of insufficient scope. In modern GPUs both atomics and fence can be qualified with a scope. These two operations can be combine to create scoped lock/unlock.

Consider a graph coloring problem where each thread assigns a color, one vertex at a time. Usually the number of vertices far exceeds the threads among a thread blocks which then perform multiple iterations to color all vertices in their partitions. Because the coloring depends on the number of edges, each block may take a different time to complete. 

```c
__device__ int getWork(...) {
	if (tid != 0) return -1; // leader thread
	
	currHead[blockId] = 
		atomicAdd_block(&nextHead[blockId],NTHREADS);
	
	if (currHead[blockId] < partitionEnd(blockId)) 
		return currHead[blockId];
		
	int victimBlock = getPartitionToStealFrom();
	if (victimBlcok == -1) return -1; // no work left
	
	currHead[blockId] = 
		atomicAdd(&nextHead[victimBlock], NTHREADS);
		
	if (currHead[blockId] < partionEnd[victimBlock])
		return currHead[blockId];
		
	return -1; 
} 
```

the kernel `getWork` shows how the leader thread obtains the next set of vertices to be colored. One may think that the block-scope atomic is sufficient (which in the common case is, when no stealing occurs). However if a thread block tries to steal work form the victim Block's partition when the leader of the victim block was also assigning itself the next vertex set, a subtle scoped race arises. The thread stealing work may not see the update as it falls outside the block scope of the victim block. 

##### Barriers & Scoped Races

Note that thread block barriers include the effect of a block-sopped fence. A barrier additionally waits for all threads in a block to reach it and thus can be slower. However, a race preventable by a block-scope fence is also prevented by a barrier. 

### Races Due to ITS

Any program that relied on lockstep execution and does not use `__syncwarp()` would lead to a race condition on modern GPUs due to [[MSc/M1/data-races/lit-notes/general/Independent Thread Scheduling|Independent Thread Scheduling]]. 

Here is an example

```c
__global__ void reductionKernel(...) {
	if (blockSize >= 4 and tid < 2)
		sdata[tid] = mySum = mySum + sdata[tid+2];
		
	// __syncwarp(); <- needed to avoid a race 
	
	if (blocklSize >= 2 && tid == 0)
		sdata[tid] = mySum = mySum + sdata[tid+1]; // race!
}
```

This is done because without the `__syncwarp`, the first thread may move on and already check and modify `sdata` for the second time. 

### Races due To Cooperative Groups (CG)

- [ ] Finish reading this sections🔼 

## Design Overview of iGuard\

#### In-GPU Race Detection

All of the iGuard happens on the GPU without needed any hardware modifications. IGUARD uses [[NVBit]] which is a dynamic tool that enables inspection and modification of cuda kernel assembly code (SASS) on the GPU without needing to recompile. iGUARD performs two primary tasks via instrumentation:
1. Collects and updates metadata for active syn operations and accesses to global mem locs. The metadata is then used for race detection. It instruments sync operations (fence and barriers) and keeps track of their issuing thread warp and scope qualifiers. All loads, stores and atomics are instrumented to update the metadata
2. It implements race detection using information of the current instruction & metadata of the address of the access. Race detection happens on: loads, stores, atomics. 

An in GPU detector needs to consider thousands of threads and their interactions (i.e via vector clocks) which is infeasible. Thus the mem metadata tracks reads & writes accessors, alongside relevant sync information, including scopes, for each unit of glob mem (4 bytes by default). Similar to [[ScoRD]], the [[Happens-Before Relation]] is used to check if there have been conflicting accesses which are not synced. 

This is also unrealistic, instead the thread and warp ID of the last writer & last accesssor is kept. This may lead to a false negative if a writer correctly is synced with the last reader, but based on the experiments, this is unlikely in practice. iGUARD also tracks if threads accessing a location are within a thread-block or span across thread-blocks to determine inappropriate use of scope. 

To catch [[MSc/M1/data-races/sources/md/Independent Thread scheduling|ITS]] related races, iGUARD tracks thread IDs beside warpIDs in the metadata and tracks warp-level barriers. It will report conflicting mem access within the same warp if they are or not separated by a sync (`__syncthread()` or `__syncwarp()`). 

#### Inferring Locking Protocols 

While there are no explicit locks in cuda, guides suggest using atomics and mem fences to create them. iGUARD can infer these instructions without needing programmer annotations. However, there is a new challenge since it should support ITS. under ITS some kernels employ per-thread lock within a warp (e.g. matrix multiplication), while others take one lock per warp, where a leader performs the sync. 

To satisfy the goal of not having to need dev interventions, iGUARD infers the protocols used by a kernel at runtime by assuming a warp-level lock by default. However, it monitors if two or more threads from the same warp attempt to acquire the lock concurrently, which would mean that it is per-thread locking instead. Instead of the happens before relation, iGUARD uses the [lockset](https://dl.acm.org/doi/pdf/10.1145/781498.781528) technique which can detect races that do not manifest during and execution. However this is limited to locks/unlocks, thus the sync metadata is augmented with lock tables to infer locks locking protocols. The mem metadata is also augmented with a summary of locks held while accessing a loc. 

#### Metadata Management 

iGUARD requires 16 bytes of metadata per 4 bytes (4x mem overhead) and a total of ~2 MB of sync metadata. If the entire metadata resides on the GPU, it will constrain the kernel. Thus [[CUDA Runtime#Unified Virtual Address Space|UVM]] mem is used. With UVM, memory is pinned (reserved) on the GPU. It is moved between the host and device on demand by the driver. Since only a small fraction is update & used for the race detection at a given time, UVM keeps the appropriate amount of data in device mem. 

#### Optimizing Metadata Accesses

The challenge for in GPU race detection is the **serialization** of metadata accesses. Thousands of GPU threads can concurrently access a shared var. Accesses to the corresponding metadata needs to be serialized for the detection correctness. However, serialization can hamstring GPU software, thus iGUARD opportunistically coalesces metadata access due to react detection to reduce serialization. It also employees dynamically adjusted exponential backoff. It adjust the backoff length based on the number of concurrent threads at runtime to limit metadata to limit metadata contention while avoid not needed waits.

Note: this is what keeps the performance overheads so minimal for kernels with frequent acc to shared vars. 

#### Race Reporting 

iGUARD reports identities of instruction the address of the data participating in a race, & the cause. iGUARD allocates a 1 MB buffer to accumulate information of races without stopping execution on detecting a race. This buffer is sent to the CPU when full or when the program ends. It may happen that a kernel live-locks due to a race 

## Implementation of iGUARD

### Metadata Layout & Allocation 

the MD is similar to [[ScoRD]] but is extended for [[MSc/M1/data-races/sources/md/Independent Thread scheduling|ITS]] & better for race detection accuracy & performance 

#### Sync the MD 

The active sync status of threads and thread-blocks are maintained in the sync MD. iGUARD keeps counters for diff sync ops to identify the latest sync op performed by  individual threads, warps & blocks. Specifically a block barrier counter is kept in *each* block and is incremented upon countering `__syncthreads()`. As well a warp barrier counter is kept for each warp and is incremented the same way. iGUARD keeps two thread-fence counters
1. One for block-scope 
2. One for device scope
for each thread. 

The thread fences are per thread since cuda defines the semantics of thread-fences to be per thread. Under ITS each thread may have executed different thread-fences at a given point. Finally, a lock table is kept to infer locks and locking protocols. The syn MD takes ~2 MB. 

#### Mem MD 

iGUARD keeps MD for each unit of global me. It tracks the identity of the accessor and access type to a given loc and sync information. The sync info is copied from the sync MD of the acc at the time of access. Instead of keeping all identities of accs, iGUARD keeps the MD of the last writer and last acc.

#### Allocating MD 

while the sync MD needs only 2 MB, the MD incurs 4x overhead. To combat this `cudaMallocMannaged()` is used in order to alloc the virtual address without allocating physical mem. When the detector access the MD, page faults are triggered,; the UVM driver then allocs the physical mem for the page containing the MD. The driver also migrates this page between the host and device mem based on access patterns. 

A drawback of UVM is the overhead of page faults incurred on the first access of the MD and the cost of migrating pages. iGUARD ensures that the overhead caused by UVM are incurred only if unavoidable. It keeps track of free mem on the device by tracking the amount of mem reserved by the kernel. 

### MD Updates 

iGUARD instruments sync ops using the [[NVBit]]  tool. Look at the following code snippet

```c
__device__ void instr_barrier(uint8_t **syncMD) {
	if (threadId < WARP_SIZE) {
		unsigned mask = __activemask();
		unsigned chosen = ((mask-1) & mask) ^ mask; 
		
		if ((1<<threadId) & chosen) // leader
			++syncMD[blkBarID][blockId];
 	}
	__syncthread();
}
```

This is an instrumentation code for updating sync MD on encountering `__syncthreads()`. One of the active threads, increments the corresponding block barrier counter. All threads in the block are synced thereafter. On encountering the `__syncwarp`, the warp the warp counter is similarly incremented.

```c
__device__ void instr_fence(
	scope_t scope,
	uint8_t **syncMD, 
	uint64_t *locks
) {
	switch (scope) {
		case DEV: ++syncMD[DevFenceId][threadId];break;
		case BLK: ++syncMD[blkFenceId][threadId];break
	}
	activateLocks(scope, locks);	
}
```

This code snippet shows code on encountering a mem fence operation. iGUARD increments the corresponding thread-block or device scope fence counter for the corresponding thread. This is also part of the lock/unlock procedure. 

![[Pasted image 20261027131548.png]]


- [ ] finish reading the paper after understanding NVBIT ⏫ 

