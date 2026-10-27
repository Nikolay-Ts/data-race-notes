[[inter-process_communication_in_linux]]

#IPC #CPU

## Shared Storage

Shared files might be the most basic method of IPC. A producer can write to a file and consumer can read from it. The main issue is preventing race conditions as both procs could access the same data. To prevent this, we can lock the file as follows:
- The producer must gain an exclusive lock to write to a file
- The consumer can gain a shared lock which would allow other readers to also read but exclude the writer from writing. 

Usually the locking is done using `fcntl` which accepts a file descriptor. Note that different programs may have different file descriptors that still point to the same physical file. Linux also provides a function `flock` which is a thin wrapper of `fcntl`. This also allows you to either lock the whole file or certain bytes of the file.

## Using Pipes and Message Queues

## Sockets and Signals 