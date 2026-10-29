#patching 

[[NVBit]]

## Getting Started

The way NVBit works is that you create shared object .so that then gets called at runtime. To make any tool that uses NVBit you your folder should look like this 

```
my_tool:
	- inject_func.cu
	- tool_name.cu
	- Makefile (optional) 
```

The `inject_func` file contains the actual function you want to patch while the tool_name.cu usually has all of host code and is usually where you reference the NVBit APIs. 

The injected function should have a function notation as such:

```c
extern "C" __device__ __noinline__ void instr_name(...);
```

the `extern "C"` that the function name is exported as it is called otherwise, the injection may fail. 

## NVBit API 

the API is split in three main categories:
1. Inspections functions you call to obtain information about the context / kernel
2. A function callback you create that gets triggered when something happens in the CUContext (e.g start, end, event). 
3. Injection functions 

All of the callbacks and other functions can be found in the `nvbit.h` file.

### Callbacks 

`void nvbit_at_init()` - this function gets called once as soon as the program starts. In this part of the program you can allocate memory on the GPU. 

`void nvbit_at_term()` - this function gets called at the end of the program. 

In both cases, no GPU calls should be made. 

`void nvbit_at_ctx_init(CUcontext ctx)` - this function is called the moment that the cuda context has been launched. Note: that you should not make any memory allocations as it will result in a **deadlock**. 

`void nvbit_at_ctx_term(CUcontext ctx)` 

`void nvbit_tool_init(CUcontext ctx)` - this callback is active at the launch of the first kernel and can be used to init any data 

```c
void nvbit_at_cuda_event(
	CUcontext ctx, 
	int is_exit, 
	nvbit_api_cuda_t cbid,
	const char* event_name, 
	void* params,CUresult* pStatus);
```

this function is called at the beginning (and end) of a cuda driver call. If is_exit = 0 then at the beginning, if is_exit = 1  then also at the end. The cbid variable can tell you what kind of driver it is by referencing the `nvbit_api_cuda_t` enum. 

```c
void nvbit_at_graph_node_launch(
	CUcontext ctx, 
	CUfunction func,
	CUstream stream,
	uint64_t launch_handle);
```

similar as `nvbit_tool_init` but for graph nodes 


### Inspection APIs 

```c
std::vector<CUfunction> nvbit_get_related_functions(
	CUcontext ctx,
	CUfunction func);
```

This will return all related functions to a given kernel, including the kernel itself (I think). 

```c
const std::vector<Instr*>& nvbit_get_instrs(
	CUcontext ctx, 
	CUfunction func);
```

this will return all the instructions that a function has. 

`uint32_t nvbit_get_sm_family(CUcontext cuctx);` - get the ID of the [[iGuard#Background|SM]] family of where the kernel is located. 

`bool nvbit_is_func_kernel(CUcontext ctx, CUfunction func);` - returns true if the functions is a kernel (i.e. has the `__global__` or `__device__` prefix)

- [ ] Add the rest of the inspection APIs

### Injection APIs 

the `ipoint_t` lets you specify if you want to insert your function before or after the current function. 

