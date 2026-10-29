#data-race #GPU #cuda #sanitizer

[[Computer Sanitizer]]

NVIDIA Compute Sanitizer is a functional correctness tool suite for CUDA applications that runs dynamically without recompilation. It replaces the older `cuda-memcheck` tool suite.

It contains four sub-tools:
1. `memcheck` - out of bounds, misaligned mem access
2. `racecheck` - [[RaceCheck|data-race hazard detection]] in shared memory (RAW, WAR, WAW)
3. `initcheck` - uninitialized device global memory
4. `synccheck` - [[Synccheck|synchronization primitives correctness]] (`__syncthreads`, `__syncwarp`)
