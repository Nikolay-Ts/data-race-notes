#thoughts #data-race 

## Recap 

Eraser is the founder of them all and was the first to use the [[Happens-Before Relation]]. It used the set of all locks and all locks held and with enough runs it will know if a shared variable is exposed if the set $C(v)$ for some shared variable $v$ was empty ([[Eraser]], [[eraser]]). However it is only on the CPU, it uses vector clocks so it is also very expensive. 

FastTrack builds on happens-before but notices that most memory accesses are writes that are totally ordered in practice. So instead of maintaining a heavy vector clock for everything, it uses an `epoch` ($O(1)$ clock + thread ID) for writes and single-reader reads, only falling back to full vector clocks when there are concurrent reads ([[FastTrack]], [[fast-track]]). It gives constant-time fast paths on the CPU, but it is still CPU-focused and vector clocks do not scale to $10^5$ GPU threads.

ScoRD is an efficient hardware-based scoped race detector for GPUs ([[ScoRD]], [[isca20_ScoRD]]). It extends each SM with a small hardware state (~3 KB) and combines happens-before with lockset analysis (inferring locks from atomic instructions & fences). The cool thing about ScoRD is that it uses a software cache in global mem to track recently accessed metadata, cutting the metadata footprint down by 16x (to ~12.5%) with only ~35% runtime overhead. The main drawback is that it requires custom hardware modifications, so we cannot run it on existing silicon.

PreDataR brings predictive race detection to GPUs using vector clocks ([[PreDataR]], [[asplos21_predatar]]). Unlike pure happens-before which only catches races if they physically collide during that exact run, PreDataR uses a partial-order relation called GWCP (GPU Weakest Causality Partial order) to predict races that *could* happen under alternative valid thread schedules. It is sound (no false positives), but tracking full vector clocks across thousands of GPU threads causes prohibitive memory and runtime overheads.

SuperCollider is a very interesting tool as it is both a compiler tool-chain and a run-time. The good thing is that it is easy to understand and uses a simple technique of sleep for random times and then checking again if the value is valid. It is also fully on-device which is very good for memory performance and is also stateless, (meaning low metadata overhead). It can also work with async data copying and tensor memory acceleration. The main drawback is that existing binaries would need to recompile with the sanitizer flag of NVCC ([[Super Collider]], [[super-collider]]). 

iGUARD is the main paper that this research is based on since the main goal is to have a inter-process data-race detector that does not require any recompilation. The main tool iGUARD uses is [[NVBit]] ([[MICRO_2019_NVBit]]), which is a dynamic instrumentation tool which allows for the injection of arbitrary code after a given [[Cuda Events]]. It tackles race conditions also induced due to a different scope or due to ITS and tries to use memory primitives such as [[Relaxed Memory Consistency|fences]] to infer locking and unlock. 

The way it works is that you have MD for each memory location (4 bytes of MD per byte of shared mem), which has the sync operations & tracks which threads use them and with what scope. All the stores, loads and atomics are accounted for in the MD. It uses a similar technique to [[ScoRD]] to detect data races.

The main novelty is how it infers the locking protocols and ITS detection. For ITS it assumes that the locking is per-warp until two or more same-warp threads try to lock/unlock to which after it becomes sub-warp locking. Because of the unified virtual mem and some optimization, the transfer of the MD is not that bad but still has around 17x mem overhead ([[iGuard]], [[SOSP21_iGUARD]]).

HiRace was created because of data-races due to the scope/hierarchy ([[HiRace]], [[sc24_hirace]]). The main novelties it brought is that it showed that with a flag set for (multiple access), you only need to record the last accessor of a mem location. It also brought a 25-state finite state machine which can be proven using Murphi. The drawback is that it relies on Clang source rewriting, requiring source access and recompilation.

Compute Sanitizer ([[Compute Sanitizer]], [[Computer Sanitizer]]) is the official NVIDIA tool suite. Its `racecheck` tool ([[RaceCheck]]) dynamically catches hazards (RAW, WAR, WAW) without recompilation, while `synccheck` ([[Synccheck]]) checks barrier usage. The main limitation is that RaceCheck only checks shared memory (scratchpad), missing global memory races, has very high runtime overhead (often 20x-50x+), and only runs within a single process context.

---

## How to Categorize Data-Race Detectors

Looking at all the literature, we can categorize race detectors across 4 main dimensions:

### 1. Instrumentation Level (Where is the code patched?)
- **Dynamic Binary Instrumentation (DBI):** Instruments machine instructions (SASS / binary) at load-time or runtime without source code. Examples: [[iGuard]] (via [[NVBit]]), [[Compute Sanitizer]], [[Eraser]] (via ATOM). 
  - *Pros:* No recompilation required; works on existing commercial / closed-source binaries.
  - *Cons:* Architecture-specific, higher runtime overhead, harder to map back to high-level source lines.
- **Compiler IR Pass:** Instruments intermediate representation (LLVM IR / PTX) during compilation. Examples: [[Super Collider]] (via NVCC/LLVM pass).
  - *Pros:* Deep compiler optimizations, access to register allocation and new PTX instructions (like `cp.async`).
  - *Cons:* Requires recompiling all kernels with custom flags (`--sanitize=supercollider`).
- **Source-Level Rewriting:** Rewrites source AST before compilation. Examples: [[HiRace]] (via Clang LibTooling).
  - *Pros:* Vendor-agnostic (can target NVIDIA, AMD, Intel); preserves full source abstractions.
  - *Cons:* Breaks if code uses non-standard extensions; requires full source access and recompilation.
- **Hardware-Assisted:** Modifies processor architecture / SM pipeline. Examples: [[ScoRD]].
  - *Pros:* Near-zero overhead (~35%); transparent to software.
  - *Cons:* Requires new silicon; impossible to deploy on existing GPUs.

### 2. Core Detection Logic (How are races identified?)
- **Lockset Analysis:** Enforces a consistent locking discipline ($C(v) \leftarrow C(v) \cap L$). Catches latent races even if thread interleavings do not directly collide at runtime. Examples: [[Eraser]], locking subsystem of [[iGuard]] and [[ScoRD]].
- **Happens-Before (HB) / Vector Clocks:** Tracks Lamport causal ordering between memory accesses and sync operations. Precise with zero false positives on observed executions. Examples: [[FastTrack]], [[ScoRD]], [[iGuard]].
- **Finite State Machine (FSM):** Encodes the state of each memory location into an explicit transition diagram based on access types and barriers. Formally verifiable. Examples: [[HiRace]] (25 states).
- **Stateless Value-Sampling & Delays:** Injects random delays (`nanosleep`) between weak accesses and redundant strong reads. Sync-agnostic; detects races if values change unexpectedly. Examples: [[Super Collider]] (and [[DataCollider]] on CPU).
- **Predictive Partial Orders:** Formulates alternative valid executions from a single execution trace (e.g. GWCP). Examples: [[PreDataR]].

### 3. Metadata (MD) Footprint & Management (How heavy is the state?)
- **Heavyweight Shadow Memory:** Dedicated shadow memory per word or byte (e.g. iGUARD allocates 16B MD per 4B mem $\to$ up to 17x overhead), managed via [[Unified Memory Programming|UVM]] page migration.
- **Single Access Record:** Only stores the last accessor ID + multiple-access flags (HiRace), maintaining constant-size shadow memory per location.
- **Cache-Filtered Shadow Memory:** Keeps MD only for recently accessed addresses in a direct-mapped software cache (ScoRD), cutting overhead to ~12.5%.
- **Fully Stateless (Zero MD):** No persistent shadow memory tables at all (SuperCollider). Uses only short-lived registers.

### 4. GPU Architecture & Concurrency Support
- **Memory Scopes:** Can the tool differentiate between `block`, `device`, and `system` memory scopes for atomics and fences?
- **[[Independent Thread Scheduling|Independent Thread Scheduling (ITS)]]:** Does the detector handle sub-warp divergence and warp-yield correctly, or does it assume lockstep warps?
- **Modern Hardware Features:** Support for Cooperative Groups, Tensor Memory Accelerator (TMA), and asynchronous bulk copies (`cp.async`).
- **Inter-Process Communication (IPC):** Can the tool track races across independent processes sharing GPU memory?

---

## Comparison Table

| Tool | Target Arch | Instrumentation Level | Recompile? | Detection Paradigm | Metadata Overhead | Runtime Overhead | Memory Checked | Scope & ITS Support | Predictive? | False Positives? | Source Link |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[[Eraser]]** | CPU | Dynamic Binary (ATOM) | **No** | Lockset ($C(v) \cap L$) | Shadow Mem (locks) | ~10x-30x | CPU Shared | N/A (CPU locks) | **Yes** (for locks) | **Yes** (imprecise) | [[eraser]] |
| **[[FastTrack]]** | CPU | Bytecode / VM (RoadRunner) | **Yes** (Bytecode) | Epoch-based Happens-Before | Epochs ($O(1)$) + VC fallback | ~2x-8x | CPU Heap / Shared | N/A (CPU threads) | **No** (manifest only) | **No** (sound) | [[fast-track]] |
| **[[ScoRD]]** | GPU | Hardware Extension (SM) | **No** | Hybrid HB + Lockset | Cache-Filtered (12.5%) | ~35% (very low) | Global & Shared | Scopes: Block, Dev, Sys; No ITS | **Partial** (Lockset part) | **No** | [[isca20_ScoRD]] |
| **[[PreDataR]]** | GPU | Dynamic Trace Analysis | **No** | Predictive Partial Order (GWCP) | Heavy (full Vector Clocks) | >10x-50x | Global & Shared | Scopes & atomics; No ITS | **Yes** (GWCP predictions) | **No** (sound) | [[asplos21_predatar]] |
| **[[iGuard]]** | GPU (NVIDIA) | Dynamic Binary ([[NVBit]]) | **No** | Hybrid HB + Lockset (infer locks) | Heavy UVM Shadow (4x-17x) | ~1.5x-3x | Global & Shared | Scopes: Block & Dev; Full ITS | **Partial** (inferred lockset) | **No** | [[SOSP21_iGUARD]] |
| **[[Compute Sanitizer]]** | GPU (NVIDIA) | Dynamic Binary (Sanitizer API) | **No** | Hazard tracking (RAW, WAR, WAW) | Internal driver state | >10x-50x | **Shared only** (scratchpad) | Warp & Block barriers (`synccheck`) | **No** (manifest only) | **No** | [[Computer Sanitizer]] |
| **[[HiRace]]** | GPU (Vendor Agnostic) | Source AST (Clang LibTooling) | **Yes** | 25-State FSM (Murphi-verified) | Single Access Record (constant) | ~1.3x-2x | Global & Shared | Hierarchical scopes; barriers | **No** (manifest only) | **No** | [[sc24_hirace]] |
| **[[Super Collider]]** | GPU (NVIDIA) | Compiler Pass (NVCC/LLVM) + Runtime | **Yes** | Stateless Redundant Reads & Delays | **0% (Stateless)** | ~2x-5x | Global, Shared, Generic | Warp/block shuffling; TMA/async | **Probabilistic** (via delays) | **No** (probabilistic sound) | [[super-collider]] |

---

## GPU Concurrency & Feature Matrix

To clearly see how modern GPU mechanisms are supported across the GPU tools:

| Feature                           | [[ScoRD]]                     | [[PreDataR]]         | [[iGuard]]           | [[Compute Sanitizer]]       | [[HiRace]]               | [[Super Collider]]              |                           |
| :-------------------------------- | :---------------------------- | :------------------- | :------------------- | :-------------------------- | :----------------------- | :------------------------------ | ------------------------- |
| **Shared Memory (`__shared__`)**  | ✅ Yes                         | ✅ Yes                | ✅ Yes                | ✅ Yes (primary)             | ✅ Yes                    | ✅ Yes                           |                           |
| **Global Memory (`__global__`)**  | ✅ Yes                         | ✅ Yes                | ✅ Yes                | ❌ No                        | ✅ Yes                    | ✅ Yes                           |                           |
| **Scoped Atomics & Fences**       | ✅ Block, Dev, Sys             | ✅ Scoped atomics     | ✅ Block & Dev        | ❌ No                        | ✅ Scope hierarchy        | ❌ Sync-agnostic                 |                           |
| [[Independent Thread Scheduling]] | ITS]] (Sub-warp Divergence)** | ❌ No                 | ❌ No                 | ✅ Yes (warp vs thread lock) | ⚠️ Partial (`synccheck`) | ❌ Warp-scalar only              | ✅ Yes (`nanosleep` yield) |
| **Cooperative Groups**            | ❌ No                          | ❌ No                 | ✅ Yes                | ⚠️ Basic barrier checks     | ❌ No                     | ✅ Yes                           |                           |
| **Async Bulk Copy / TMA**         | ❌ No                          | ❌ No                 | ❌ No                 | ❌ No                        | ❌ No                     | ✅ Yes (`cp.async` checking)     |                           |
| **No Source Recompile Required**  | ✅ Yes (Hardware)              | ✅ Yes (Binary trace) | ✅ **Yes (NVBit)**    | ✅ **Yes (Sanitizer)**       | ❌ No (Clang rewrite)     | ❌ No (NVCC flag)                |                           |
| **Inter-Process (IPC) Ready**     | ❌ Hardware only               | ❌ Trace per proc     | ⚠️ Single-process MD | ❌ Single context            | ❌ Single-process AST     | ⚠️ Logic ready, needs recompile |                           |

---

## Key Insights for Inter-Process Race Detection (My Research)

My research project focus is building an **inter-process GPU data-race detector that does not require recompilation** (see [[inter-process_communication_in_linux]], [[Meeting-notes/26.10.14]], and [[Documentation.md]]). 

Looking at the table above, here is how the existing categories impact our design:

### 1. The Recompilation Bottleneck
- Tools like [[Super Collider]] and [[HiRace]] have huge advantages in memory footprint (0% for SuperCollider, compact FSM for HiRace).
- However, both require recompiling the application from source. If our requirement is to inspect arbitrary pre-compiled binaries or vendor libraries sharing GPU memory, **Dynamic Binary Instrumentation (DBI)** via [[NVBit]] ([[MICRO_2019_NVBit]]) or the NVIDIA Sanitizer API is the only viable path.

### 2. The Metadata Problem Across Process Boundaries
- [[iGuard]] is our closest baseline because it runs dynamically on SASS via NVBit and understands scopes and ITS.
- However, iGUARD allocates its shadow metadata in device memory using `cudaMallocManaged()` within the address space of a **single process**.
- In an inter-process setting (where Process A and Process B share memory via `cudaIpcGetMemHandle()` or POSIX shared memory mapped to CUDA virtual memory):
  1. Process A and Process B have separate CPU and GPU virtual address spaces.
  2. If Process A writes to shared memory, Process B's iGUARD instrumentation cannot see Process A's metadata unless the metadata itself is located in an IPC-shared memory pool.
  3. Therefore, to adapt iGUARD for IPC, we must allocate the shadow metadata in an IPC-shared region (`cudaIpcMemHandle` or POSIX shm) so both processes can read and update the same lock/accessor records.

### 3. Can We Borrow Stateless Value-Sampling Dynamically?
- [[Super Collider]] works seamlessly across kernels, GPUs, and processes *because* it is completely stateless. It does not need to coordinate shadow metadata between processes; it simply reads the shared memory address twice with a delay.
- *Open Idea for the Project:* Could we implement SuperCollider's redundant-read and nanosleep perturbation *dynamically* via NVBit instead of through a compiler pass? 
  - If we inject weak read duplication and `nanosleep` directly at SASS level, we could get cross-process race detection without recompilation and with 0% metadata overhead.
  - The trade-off would be instrumentation overhead and register pressure, but it avoids the entire IPC metadata sharing nightmare.
