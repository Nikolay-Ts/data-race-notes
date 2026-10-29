#GPU #cuda #sync

[[Compute Sanitizer]]

This is a tool that can identify whether a cuda application is correctly using the `__syncthreads()` & `__syncwarp()` sync APIs.

## Using Synccheck 

this tool can be enabled by using the compute-sanitizer tool with the `--tool synccheck` flag turned on.
