#data-race  #CPU 

## Intro 

Multi-threaded programs lead to race conditions. In general, dynamic race detectors fall into two categories, depending on whether they report false alarms. *Precise* race detectors never produce false alarms, instead they compute the [[Happens-Before Relation]] for the observed trace and sensation and report iff the observed relation is represented using `vector clocks` (which are very expensive). Thus if the program has $bn$ threads, the CC needs $O(n)$ storage space and each op requires $O(n)$ time. 

Motivated in part by performance limitations, *imprecise* race detectors have been developed which may provide better coverage but can report false positive. For example [[Eraser|erasor's]] Lockset algorithm enforces lock-based sync discipline & reports an  error if no lock is consistently held on each access to particular mem lock. It may report an error however, on programs with alternative sync. 

A limitation of static race detectors & imprecise dynamic ones is the potential for a large number of false alarms. It becomes difficult and time consuming to figure out which alarms are not false. 

FastTrack uses adaptive representation of the happens-before relation to provide constant-time fast paths for the common case, without loss of precision or correctness in the general case.  

A VC alg would record the most recent write to each var $x$ by each thread $t$. FastTrack on the other hand exploits the observation that all writes to $x$ are totally ordered by the happens-before relation and so it records info only about the *last* write to $x$, specifically to the clock and thread identifier of the write. The pair of a clock and thread id will be called an `epoch`. 