#cuda #Memory-Model

[[CUDA_C_Programming_Guide]]

## Device Memory 

The cuda runtime assumes a heterogeneous system with a host (CPU) and a device (GPU). Kernels work with device memory which is separate from host memory, which means that you must do own mallocs and transfers from host to device and back. Device memory can be either
- Linear memory 
- CUDA arrays 

cuda arrays are opaque memory layouts optimized for texture fetching. Linear memory is allocated in a single unified address space which means that different entities can reference one another via pointers. The size of the address space depends on the host and compute capability of the GPU. 

For N-dimensional arrays it is recommended to use `cudaMallocPitch()` and `cudaMalloc3D()` instead of `cudaMalloc()` as the former add appropriate padding for alignment

### Device Memory L2 Access Management 

When a cuda kernel calls global data repeatedly, such accesses can be considered to be `persiting`. One the other hand, if is accesses once, it is considered to be `streaming`

#### L2 Cache Set-Aside for Persisting Accesses 

A portion of the L2 cache can be set aside for persisting data accesses. The persisting data has priority in the L2 cache and for example, the streaming data can only enter if the L2 cache is not full of persisting data. 

When the GPU is configured in Multi-instance GPU (MIG) mode, this feature is turned off. 

#### L2 Policy for Persisting Accesses 

An accesses policy window specifies a contiguous region of global memory and persistence property in the L2 cache for accesses within that region. 

#### L2 Access Properties

There are three types of different global memory data access:
1. `cudaAccessPropertyStreaming` - mem accesses that occur with the streaming property are less likely to be L2 cached because they are often evicted.
2. `cudaAccessPropertyPersisting` - mem access that occur with the persisting property are more likely to persist in the L2 because of the set-aside portion of the L2  
3. `cudaAccessPropertyNormal` - This access property resets the previously applied property. Mem accesses with persisting property from the previous cuda kernels may be retained in the L2 long after intended use. This works as a pallet cleanser. 
##### Reset L2 Access to Normal

A persisting L2 cache from a previous kernel may persist in the L2. Hence a reset to normal is important for streaming or normal mem accesses to utilize the cache with normal priority. Two ways this can be done:
1. Reset previous mem region with the access property `cudaAccessPropertyNormal`
2. Reset all persisting L2 cache lines to normal using `cudaCtxResetPersistingL2Cache()`
3. Eventually untouched lines reset to normal. This is strongly discouraged 

#### Manage Utilization of L2 Set-Aside Cache

Multiple kernels executing concurrently in different cuda streams may have different access policies. However, the L2 set-aside portion is shared among all the kernels. A result is the sum of all concurrent individual use. The benefit is that persisting diminish as the volume of persisting accesses exceeds the L2 cache capacity.

To manage the utilization of the L2 set-aside, one must consider the following:
- size of the L2 set-aside cache
- kernels that are concurrently executing
- Access policy of the other kernels 
- When and how the L2 reset is required to allow normal streaming access. 

### Shared Memory 

shared mem is allocated using the `__shared__` attribute. This mem is much fast than global memory as mentioned in [[CUDA Programming Model]]. It can be used as scratchpad memory (software managed cache) to minimize global mem access. 

The following code is an example 


```c
__global__
void reduce_naive(const int *d_in, int *d_out, int n) {

	size_t index = (blockDim.x * blockIdx.x + threadIdx.x);
		if (index < n) {
		__shared__ int buffer[1024];
		buffer[index] = d_in[index];
		__syncthreads();
		
		for (int stride = 1; stride < blockDim.x; stride*=2) {
			if (index % (2*stride) == 0) {
				buffer[index] = buffer[index] + buffer[index+stride];
			
			__syncthreads();
			d_out[index] = buffer[index];
	    }
	}

}
```

### Distributed Shared Memory 

[[CUDA Programming Model#Thread Block Clusters|Thread block clusters]] allow threads in a block cluster to access shared mem of all participating threads. This partition is called `dsitributed memory access` and the corresponding address space is called `distributed shared memoery address space`.  Threads that belong to block cluster can read/write or perform atomics regardless whether the data belongs to the local thread block. Whether the kernel uses static or dynamic me is up to the thread block. The size of the distributed mem is 
$\text{thread blocks} \times \text{size of shared mem per block}$. Accessing this data requires all thread blocks to exist. A use can guarantee that by using `cluster.sync()`. The user also needs to ensure that the mem operations happen before the exit of  block. 

### Memory Sync Domains 

Some cuda kernels me see degraded performance due to fence/flush operations which are needed to enforce the memory model. 

#### Isolating Traffic with Domains 

Beginning with the Hopper architecture, we have ways to alleviate such inter fences. In exchange for explicit assitance from code, the GPU can reduce the net case by a fence operations. Eacg kernel launch has a domain ID. Writes and fences are tagged with the ID and a fence will only order writes matching the fence's domain. 

When using domains, code must abide by the rule that `ordering or sync between distinct domains on the same GPU requires system-scope fencing`. Within a domain, device-scope fencing remains sufficient. This is necessary for cumulative as one kernel's writes will not be encompassed by a fence in a diff domain. 

#### Using Domains in Cuda 

Domains are accessible via the launch attribute `cudaLaunchAttributeMemSyncDomain` and `cudaLaunchAttributeMemSynchDomainMap`. The latter provides a mapping from logical to physical domains. 

The remote domain is intended for kernels performing remote memory access. 

### Async Concurrent Execution 

Cuda exposes the following operations that can work concurrently with one another:
- Computation on the host
- Computation on the device
- Memory transfers from host to device and vise versa 
- Memory transfers within the memory of a given device and among devices

#### Concurrent Operations Between Host & Device 

Concurrent host execution are done through an sync lib that returns control to the host thread before the device completes the task. Using async calls, many device operations can be queued together to be executed by the cuda driver. This relieves the host thread of managing the device. The following are async with w.r.t host:
- Kernel launches
- Mem copies withing the device
- Mem copies from host to device of block less than 64KB 
- Mem copies by functions with the Async prefix
- Mem set function calls. 

#### Concurrent Kernel Launches

Some devices can execute multiple concurrent kernels. The max number of kernel launches depends on the compute capability of the device. A kernel from one cuda context cannot execute concurrently with a kernel from a diff context. If the user wants to run multiple kernels from multiple procs. simultaneously on one SM, one must enable MPS. 

#### Explicit Sync

One way is to use `cudaDeviceSynchronize()` which waits until the preceding commands in all streams of all host threads have completed. 

`cudaStreamSynchronize()` takes stream as an input and waits until that stream has synchronized. 

`cudaStreamWaitEvent()` takes and [[Cuda Event]] as an input and waits fir the event to be done. 

### Unified Virtual Address Space 

When the application runs as a 64-but proc. a single address soace is used for the host and devices. All host mem allocations made via the cuda API and all device mem allocations are within the virtual space. As a result:
- The location of any me  on the host allocated through cuda or on the device use the unified address space, can be determined by the value of the pointer using `cudaPointerGetAttributes()`
- When copying to/from mem of a any device using this address space, the `cudaMemcpyKind` param of `cudaMemcpy*()` is set to default to determine locations from the pointers. This also works for host pointers not allocated using cuda as long as the current device uses the unified mem.
- Allocations via `cudaHostAlloc()` are automatically portable to all devices and pointers returned by this func can be directly used with the kernel running on these devices. 

### Inter-process Communication 

Any device pointer or event handle created by a host thread can directly be referenced by other threads within the same proc. It is not valid outside of the proc. and cannot be directly referenced by threads of other procs. 

To combat this and shared pointers and event across procs one must use the IPC API. The IPC API is only supported for 64-bit systems on Linux. Note: the IPC API is not supported for the `cudaMallocManaged` allocations. 

Using this API, a kernel can get the IPC handle for a given device pointer using `cudaIpcGetMemHandle()`, pass it to another proc. using standard IPC mechanism (e.g. teleprocess shared memory or files) and use `CudaIpcOpenMemHandle()` to retrieve a device point from the IPC handle. 

Noe that `cudaMalloc()` may return a pointer of a block while the IPC handle will return the entire block and lead to information disclosure between procs. To prevent this, it is important to share allocations with a 2MB alligned size. 

