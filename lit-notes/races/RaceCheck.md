#data-race #GPU 

[[Computer Sanitizer|Computer Sanitizer]]

## What is RaceCheck 

This is a run-time shared mem data access hazard detector. The primary use is to identify mem race conditions in [[CUDA Runtime|cuda]] applications that use shared memory.

A shared on chip memory variable has the `__shared__` qualifier. All threads in a [[CUDA Programming Model#Thread Hierarchy|thread-block]] can access it simultaneously.  Shared mem goes out of scope when the thread block completes its execution. This mem is on chip and is frequently used for intra-thread communication (wrongful usage may lead to a data-race). RaceCheck is built to identify these hazard.

## What are Hazards

A data hazard is when two or more threads try to access the same mem location simultaneously and at least one of the threads must perform a write. RaceCheck identifies three types of hazards:
- Write-After-Write (WAW) - This occurs when two threads attempt to write to the same mem location; The resulting value depends on the relative order of the two accesses.
- Writer-Afer-Read (WAR) - this occurs when one thread performs a read and the other a write. In this case, the writing thread is ordered before the reading & the value returned to the reading one is not the original value in memory.
- Read-After-Write (RAW) - This happens when one is a read, the other a write. In this case the read happens before the writing threads commits the operation. 

## Racecheck Report Modes 

Hazard reports - these reports contain detailed info about one particular hazard. Each hazard report is byte accurate and represent info about the conflicting access that affect this by of shared mem.

Analysis reports - these contain a post analysis set of reports. These reports are produced by analyzing multiple hazard reports && examining active device state. 

