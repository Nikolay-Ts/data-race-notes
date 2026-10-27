#cache-coherence #CPU 

[[primer2020_2nd_edition]]

## Baseline System Model

We consider a system with multiple processors and a shared memory. The cores can perform loads and stores to all (physical) addresses. Each core has its own private cache and a last-level cache (LLC) which is shared by all the cores. Cache will refer to as the private cache and not the LLC unless said otherwise. 

## How Incoherence Could Possibly Occur 

The rise of this issue comes from one fundamental problem: there exists multiple access with access to caches and memory. In modern systems, the actors are the cores, DMA engines and external devices that can read/write to caches and memory (though we primarily focus on cores). 

Incoherence would occur when two or more cores have a copy of the same variable and one or more of the cores have a wrong copy of it. To solve this issue, we introduce `cache coherent protocols` that make the store from core A visible to core B.

## The Cache Coherence Interface

Informally, the goal is for a write to be made visible to all other cores. The cores interact with the coherence protocol via a coherence interface that provides two methods:
1. Read-request - takes in a memory location as a param and returns a value
2. Write-request - accept a memory location and value and returns an Ack.

There are many different protocols in nature and we tend to put them into two categories based on the nature of their coherence interfaces - specifically based on whether there is a clean separation of coherence from the consistency model or whether they are indivisible. 

#### Consistency-Agnostic Coherence

In the first category, a write is made visible to other cores before returning.  Because write are propagated synchronously, the interface is identical to that of an atomic memory system with no cache. Thus any system interacting with it can treat it as atomic memory with no cache present. This interface provides a nice separation of concerns. The cache coherence protocol abstracts away the caches completely and only the memory is contained within the coherence box. 

#### Consistency-directed Coherence 

This is a more recent category where write are propagated asynchronously. A write can thus return before it has been made visible to all cores, allowing for stale values to be observed. However, in order to correctly enforce consistency, the coherence protocols in this class must ensure that the order in which writes are eventually made visible adheres to the ordering rules of the consistency model. 

This category emerged to support throughput-based GPGPU.

## Consistency-Agnostic Coherence Invariants

What invariant must coherence protocols satisfy to make the caches invisible and present abstraction of an atomic system. 

A coherence through the single-write-multiple-reader invariant is that for any given memory location at any given moment, there is either a single core that may write to it or some number of cores that may read it. Thus, there is never a time when a given memory location may be written by one core and simultaneously read or written to by any other cores. 

In addition to SWMR, the data must be propagated correctly. This is important because if a write happens, all the other cores that have a, now stale, copy in their own cache, must be notified and invalidate and fetch the fresh copy of the new value. 