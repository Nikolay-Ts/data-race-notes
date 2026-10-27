#memory-model #CPU 

[[primer2020_2nd_edition]]

A relaxed memory consistency model is one where we allow the CPU to modify the order of the store and load operations between and inside of threads. The benefit of this is that, it can be optimized and increase CPU throughput. The downside is that we need to formalize tools to use in order to enforce that some operations cannot happen before others (e.g memory fences).

## Example Relaxed Consistency Model (XC)

We will refer to this model as XC from now on. XC assumes that a global memory order exists as is true for strong models of SC and TSO, as well as the largely defunt relaxed models for Alpha and SPARC relaxed memory order (RMO).

XC provides a FENCE instruction that programmers can indicate when order is needed, otherwise, load and stores are unordered. Other relaxed consistency models call a FENCE a barrier, a memory barrier or a sync. Let core $C_{i}$ execute some loads/store and $X_i$ then a FENCE and then some loads/stores by $Y_i$. The FENCE ensure thats all the loads/stores of $C_i$ will happen before any $Y_i$ operations.  A FENCE instruction does not specify an address and two FENCEs  by the same core stay ordered. Note that a FENCE does not affect the order of memory operations on the other cores. 

XC's memory is guaranteed to respect (preserve) program order for:
- load -> FENCE
- store -> FENCE
- FENCE -> FENCE
- FENCE -> load
- FENCE -> store 
And also implements sequential ordering whens loading from the same address twice or loading and then saving or storing to the same address twice. 

This allows us to enforce sequential instructions within a core when needed. 

### Formalising XC 

### Implementing XC 