#GPU #Memory-Model #CUDA

[[CUDA_C_Programming_Guide]]

### Kernels

Kernels are the CUDA functions that get executed $N$ times by $N$ different cuda threads. These functions must be void, have a `__global__` or `__device__` and when invoked must be called in one of two ways:
1. using the triple shevron method `<<<>>>` 
2. Idk the second method. 

Each thread has a unique ID and is usually how we chose to access and synchronize. Here is a code snippet to sum two vectors 

```c
__global__ void VecAdd(float *A, float *B, float *C) {
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}

int main(void) {
	VecAdd<<<1,N>>>(A,B,C);
}
```

### Thread Hierarchy 

ThreadIdx is a 3-component vector, so that threads can be identified using a one-dimensional, two-dimensional, or three-dimensional thread index which can create a 1- 2- or 3-D threads called a thread block. 

The index of a thread and its thread ID relate to each other in a straightforward way. For a 1D block, they are the same, for two 2D block of size (Dx, Dy), the thread ID of thread of index (x,y) is (x+y *Dx). For a 3D block of size (Dx,Dy,Dz) the thread ID of a thread of index (x,y,z) is (x+y* * Dx + z *Dx * Dy).

As an example, the following code adds two matrices A and B of size $N\times N$.

```c
__global__
void matAdd(float A[N][N], float B[N][N], float C[N][N]) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	C[i][j] = A[i][j] + B[i][j];
}

int main(void) {
	int numBlocks = 1;
	dim3 threadsPerBlock(N, N);
	matAdd<<<numBlocks, threadsPerBlock>>>(A,B,C);
}
```

There is a limit for the threads per block (usually 1024). However, a kernel can be executed on multiple equally shaped blocks making the max threads be threads per block * number of blocks 

```c
∕∕ Kernel definition
__global__ void MatAdd(float A[N][N], 
					   float B[N][N],
                       float C[N][N]) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i < N && j < N)
		C[i][j] = A[i][j] + B[i][j];
}


int main()
{
	...
	∕∕ Kernel invocation
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(N ∕ threadsPerBlock.x, N ∕ threadsPerBlock.y);
	MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
...
}
```

A thread block of 16x16 (256) threads is arbitrary but common. The grid is created with enough blocks to have on thread per matrix element as before. Thread blocks are required to execute independently.

Threads within a block can cooperate by sharing data through some shared memory and by sync their execution to coordinate memory accesses. 

You can use `__syncthreads()` as one tool. For efficient cooperation, shared memory is near each process's L1 cache and `__syncthreads()` is lightweight. 

### Thread Block Clusters 

Now there is an optional level of hierarchy called Thread Block Clusters, that are made up of thread blocks. Similar to how threads in a block are guaranteed to be co-scheduled on an SM, thread-blocks in a cluster are guaranteed to be co-scheduled on a GPU Processing Cluster (GPC).  

These cluster can also be up to 3D. 

![[Screenshot From 2026-10-25 17-06-21 2.png]]

The max number of thread-blocks per cluster is `8` or the max size based on hardware limitations. A thread-block cluster can be enabled in a kernel using a compile time attribute using `__cluster_dims__(X,Y,Z)` or using the launch API `cudaLaunchKernelEx.`

```c
__global__ void 
__cluster_dims__(2,1,1) cluster_kernel(...);

int main(void) {
	dim3 threadPerBlock(16, 16);
	dim3 numBlock(N / threadPerBlock.x, N / threadPerBlock.y);
	cluster_kernel<<<numBlock,threadPerBlock>>>(...);
}
```

this can also be set at runtime but its annoying to show. 

### Memory Hierarchy 

Threads may access different levels of memory spaces. Each thread has its own thread-private memory. Threads in a block have shaved memory and thread blocks in a cluster can perform read-write atomics on each other's share mem. All threads have access to the global memory.

There also exists the texture and constant memory space which is accessible to all. The global, texture and constant spaces persist across kernel launches by the same applications.

![[Screenshot From 2026-10-26 13-51-17.png]]

### Heterogeneous Programming 

The cuda model assumes that cuda threads execute on physically separate device that operate as co-processor to the host running the C++ program. This is the case when the kernel execute on a GPU and the rest on the CPU. 

Cuda assumes that the device and host maintain their own mem spaces in DRAM. Therefore a program manages the global, const and texture me space visible to the kernel via calls to the cuda runtime (including allocation and deallocation as well as data transfers). 

Unified mem provides managed memory to bridge the host and device mem spaces. This mem is accessible to both the host and device as a single coherent mem image with a common address space. This allows to simplify the task of porting application by eliminating the need to explicitly mirror the data. 

### Async SIMT Programming Model 

In cuda, a thread is the lowest level of abstraction for doing a computation. Starting with devices based on the `ampere GPU architecture`, the cuda model provides acceleration via async programming model.

### Async Operations

