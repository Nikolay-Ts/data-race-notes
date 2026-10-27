#GPU #unified-memory

[[CUDA_C_Programming_Guide]]

#### Example of unified memory 

```c
__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - first %d characters: '", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
	  printf("'\n");
}
```

this simply prints the first 8 characters of an input arracy to stdout 

