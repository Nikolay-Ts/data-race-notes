#cache-coherence #CPU 
## Intro 

When having a multi-core architecture you have several cores that shared a variable but still have their own cache. This synchronization between caches can be tricky because two cores may view the same variable differently due to the different caches. The difficulty is generally referred to as the `cache coherence problem`. 

informally, a system is coherent if ant read of a data item returns the most recently written value of that item. This can be broken into two aspect:
1. Coherence - defines what values can be returned by a read
2. Consistency - determines when a written value will be returned by a read

## Coherency 

A memory system is coherent if 

(1) A read by core $P$ to location $X$ that follows a write by $P$ to $X$ with no writes of $X$ by other cores occurring between the write and read, always returns the value written by $P$ to $X$. 

(2) A read by a core to location $X$ that follows a write by another core to $X$ returns the written value, if the read and write are sufficiently separated in time and no other writes to $X$ occur between the two accesses. 

(3) Writes to the same location are `serialised`. That means, two writes to the same location by any core are seen in the same order by all cores. For example, if CPU B writes 2 to $X$ after time step 3, cores can never read the value at $X$ as 2 and then later read it as 1

### Basic Schemes For Enforcing Coherence 

The caches usually provide both `migration` and `replication` of shared data items. 

Migration - A data item can be moved to a local cache and used there in a transparent manner. Migration reduces both the latency and access a shared data item that is allocated remotely and the bandwidth demand n the shared memory. 

Replication - When shared data are being concurrently read, the caches make a copy of the data into the location cache. Replication reduces both the latency of access and contention of a read for a shared item 

These two tools are so important that many processors implement these in hardware to maintain coherent caches. The protocols are called `cache-coherent protocols`. A key element to their implementation is tracking the state of any sharing of a data block. 

The most popular protocol is `snooping`. Every cache that has a copy of the data from a block of physical memory also has a copy of the sharing status of the block, but no centralized state is kept. The caches are all accessible via some broadcast (data bus or network) and all cache controllers monitor or `snoop` on the medium to determine whether they have a copy of a data block that is requested on the bus/switch. 

### Snooping Protocols 

One method to ensure coherence is to have a core have `exclusive` rights to a data block. This style of protocol is called a `write invalidate protocol` because it invalidates copies in other caches on a write.

Exclusive access ensures that no other readable/writable copies of an item exist when the write occurs. Thus, a read after this exclusive right would cause a cache miss and would have core B request a copy from Core A (assuming core A did the write and B had a local copy) which would then be sent over the data bus/network. 

If two cores do try to write at the same time, because this ex is atomic, one of them will win and the other core will have to way and also obtain a new copy of the data first. Therefore this protocol also enforces `write serialisation`.