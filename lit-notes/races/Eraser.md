
#data-race #CPU

[[eraser]]

## Introduction

This software is to create a tool that can dynamically check for race conditions. It is based on Lamport's *happens-before relation* which checks that conflicting memory accesses from diff. threads are separated by a synchronisation event. Eraser checks that threads follow a consistent *locking discipline*. A discipline is a programming policy to ensure no races (the simplest one being a mutex). 

data race - when two or more threads attempt to access shared memory where at least one of the threads tries to write to that object. 

### Related work 

Another way to do this was to use *monitors*, which are a group of shared objects together with procedures that are allowed to access them all grouped in a single anonymous lock that is automatically locked and unlocked at the entrance and exit of scope (similar to C++ lock guards). 

Monitors thus provide a static, compile time guarantee that access to the critical section is serialised $\therefore$ free of any data races. The issue with monitors is that they cannot help with dynamically allocated shared variables. 

The issue with Lamport's lock is that it cannot detect if a variable has been modified outside of a lock unless another one comes, causing potential races

![[Pasted image 20261013165142.png]]

even though y is shared, the *happens-before* would not spot the fault in this case. 

## The Lockset Algorithm 

At runtime, a shared variables $v$ will have a corresponding set $C(v)$ which contain all the locks. Eraser monitors each that $v$ has been accessed and which locks were held, if a lock was not held while accessing $v$, it is removed from the set $C(v)$. The more the program runs the more accurate this set becomes.   

upon access: 
$$C(v)= C(v)\cap L$$where $L$ is the set of all the locks that are currently held. if $C(v)$ becomes empty, then no lock is consistently protecting $v$ which may lead to a data race. 

```python
L = all locks
locks_held = [all the locks held by thread t]
for v in shared_variables:
	C[v] = l
	
# upon access of v by t
C[v] = C[v].intersect(locks_held)
if (C[v].isEmpty()):
	raise Unprotected_shared_variable 
```
### Improving the Discipline

you can have a shared variable that is inited once and then only read, or have read-write locks that do not need a mutex but can have multiple reads. That is why we want to loosen the discipline to support such cases. 