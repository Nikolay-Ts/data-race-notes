#data-race #GPU #instrumentation #state-machine

[[MSc/M1/data-races/sources/md/HiRace|HiRace]]
## Intro

Data-races are very important and likely in the GPU due to the [[CUDA Programming Model#Memory Hierarchy|memory hierarchy]]. Most of the data-race checkers such as [[iGuard]] and [[Compute Sanitizer]] are NVIDIA specific and, compute sanitizer for example, does not detect global memory races. iGuard and its evolution heavily depend on [[NVBit]]  and does not seem to be actively maintained nor function on on later NVIDIA architectures from 2024.

This paper presents HiRace, which aims to serve the needs of today's HPC and ML practitioners. HiRace works at the source level by instrumenting code via Clang's source rewrite API. Source level instrumentation allows us to avoid reliance on compiler contracts and optimizations.  

### Design Novelties

Single Access Record: the most significant one and it should be sufficient to encode all necessary info about `all` prior accesses to identify races.  This design ensures that memory overhead is constant and small. In contrast, other designs maintain multiple accessor records significantly more mem. 

Explicit State-Machine: HiRace achieves a modest size (const mem space per location) access history through use of an explicit finite-state-machine that encodes access history compactly. 

Correctness: Having a state-machine allows the development of a Murphi formal reference model 

## Methodology 

The design is based around the built-in sync primitives in GPUs (e.g `__syncthreads`) as opposed to focusing on the full generality of GPU parallel programming patterns. The methods do not preclude the handling of fine-grained thread sync, but focus on optimizing the most common and impact forms of thread communication in GPUs. 

All of the relating possible thread access histories to a single mem location. For example, if a thread reads the same mem location multiple times, we do not need to record each access separately. A single thread ID and a flag indicating a read by that ID is enough. Even more, in a program with only two threads, if both reads concurrently read the same me location, it suffices to only track the ID of either accessor along with the flag identifying the multiple threads have read it. 

In the general case, this would require $2^{T}$ flags to track all possible thread relations between $T$ accessing thread. However, due to the [[CUDA Programming Model#Memory Hierarchy|mem hierarchy]] and asymmetric nature of shared mem scopes, instead of tracking individual threads, only the hierarchical scope shared by accessing threads needs to be tracked. 

HiRace is based on a Finite State machine (FSM) that encodes the [[Happens-Before Relation|happens before]] logic introduced before. Whereas FSM is tailored to the current [[CUDA Programming Model]] it can easily be extended to other models with deeper hierarchies. This FSM has 25 states and 1200 transitions. They are sufficient to describe interactions of all atomic and no-atomic reads and writes so a single mem loc as well as needed syn history from warp-scoped and block-scoped barrier primitives. 

Rather than dealing with each state of the SM, this sections develops similar state machine for simple concurrency models and two scenarios to show the key idea to this approach:
1. How the FSM performs with atomics 
2. How the source of instrumentation works and is expresses as FSM transitions.

```c
__global__ void noSync(int *data) {
	int i = threadIdx.x + blockIdx.x * TPB;
	int val = data[0];
	data[0] = i + val; // race
}
```

Take this program with no atomics and a race as one thread can read data[0] as the other writes to it. Here is how it would be described as an FSM. 

![[Pasted image 20261026201613.png]]

They claim that this FSM will a singleo prior accessor ID is enough to track all races. To monitor *data* we create a `shadow[k]` entry for each array element. Each shadow entry maintains the following info:
- Block-scalar clock
- Warp-scalar clock
- ID of acting thread
- Most recent access (TID)
- State of the FSM

![[Pasted image 20261026201836.png]]


Look at the following example:

```c
__global__ void racy(int *data, int n) {
	int i = threadIdx.x
	int val = data[0];
	__syncthreads();
	if (i > 0) {
		data[i-1] = i; // race 
	}
}
```

- [ ] finish reading this paper as well 