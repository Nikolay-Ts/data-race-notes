#data-race  #GPU #hardware

[[isca20_ScoRD]]

## Intro 

More and more applications across different industries start using GPUs. This also means that you need a more sophisticated syncing methods. To fulfill this sync need, GPUs have enabled rich synching primitives such as various flavors of atomics, fence, barrier and acquire/release ops. Programming models such as cuda have a [[CUDA Programming Model#Thread Hierarchy|hierarchies]] of threads. As a result, cuda has sync ops with non-global scope qualifiers that can be used with the sync op. Cuda has three scopes, `block`, `device` & `system`. For example a read/write atomic with the `block` scope will only affect threads from the same block. This is also true for mem. Threads in the same [[GPU Memory Consistency#GPU Architecture|SM]]/block share the same L1 cache and scratchpad mem. This however brings a new dimension that programmers have to consider; is the software synced to the correct scope. This paper declares that if a scope does not encompass both the producer & consumer that the program intends to sync, then it is not correctly synced. 

For example: consider two threads perform `atomicExch` op on the same variable within a `block` scope. If these threads belong to diff block, then they may not observe the effect of eahc other's op. Consequently, a scoped race will arise.

This work present Scoped Race Detector (ScoRD), an efficient hardware-based detector for scoped races & global mem races. This design is also not limited to cuda. Scored also extends the [[Happens-Before Relation]] data-race detection tool by introducing scopes to detect scoped race due to atomics & fences. ScoRD thus dynamically infers lock & unlock ops & extends lockset-based mechanisms. ScoRD starts with a small hardware state (less than 3KB) to hold info on active sync operations along with the logic for detecting races on the GPU. ScoRD keeps MD at the granularity of 4 bytes to track the identity of the last accessor. The MD is kept in global mem. They observe that mem races happen in close time to each other & only a small part of allocated me participates in a race; thus they use keep the MD for recently accesses mem addr using a direct-mapped software cache. This helps us reduce metadata overhead by 16x  to a reasonable 12.5% without greatly sacrificing the accuracy. 



ScoRD has low performance overhead (35% on average) & no false positives. Moreover, it can be turned on only during software testing and debugging and can be turned of during production. 

In summary, here are the following contributions:
- ScoRD, hardware-based scoped race-detector for reporting races in GPU 
- ScoR benchmark suite containing several applications & micro-benches that make use of scoped syn primitives  

## Methodology 

ScoRD reports the instruction pointer & data race of the me instruction associate with the resultant race either due to insufficient scope or due to absence of sync. It further reports whether conflicting accesses were form the same [[CUDA Programming Model#Thread Hierarchy|thread-block]] or different blocks & the type of race due to insufficient scope in the lock/unlock. It does not stop the program but just accumulates the races in a mem buffer. 

From a high-level POV, ScoRD, on a mem access (load/store/atomics), performs a preliminary checks to find out if the access is *trivially* race-free. This captures simple yet common circumstances where a race cannot happen (e.g. access to a location in program order) & acts as a filter to a more involved race detection. If the preliminary fails, two types of check are deployed:
1. Happens-before relations to check if a race is due to insufficient sync
2. infers lock/unlock ops by monitoring atomics, fences and lockset based algs extended to scopes to detect lock-based races 


While lockset-based detection is restricted to detecting errors only in locking, happens-before-based detection applies a broader range of sync ops. However a race needs to manifest during exec for the happens-before to detect it. In contrast, lockset-based detection can even detect potential races that have not yet manifested. 

### Hardware Modifications & MD 

Since in a typical GPU, the smallest scheduling entity is a warp, we maintain state & MD at the warp granularity. 

![[Pasted image 20261028114847.png]]

this depicts the hardware modification. First, each SM is extended to keep track of the ID. 