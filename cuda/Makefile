NVCC ?= nvcc
ARCH ?= native            # e.g. make ARCH=sm_86 if your nvcc predates -arch=native (CUDA < 11.5)
NVFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo

all: matmul reduce

matmul: matmul.cu common.cuh
	$(NVCC) $(NVFLAGS) -o $@ matmul.cu

reduce: reduce.cu common.cuh
	$(NVCC) $(NVFLAGS) -o $@ reduce.cu

run: all
	./matmul && ./reduce

clean:
	rm -f matmul reduce
.PHONY: all run clean
