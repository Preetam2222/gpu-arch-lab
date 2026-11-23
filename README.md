# gpu-arch-lab

A trace-driven cache simulator (C++), a Python test harness, and two CUDA kernels. The cache
experiments and the GPU kernels are about the same idea: reuse data close to the compute.

```
cachesim/   C++17 simulator: set-associative, LRU/FIFO/random, write-back/write-allocate,
            multi-level (L1-L3), 3C miss classification (compulsory/capacity/conflict), JSON output
harness/    Python: independent reference model, trace generators, pytest suite, experiments
cuda/       matmul.cu  shared-memory tiled GEMM   (+ naive baseline)
            reduce.cu  warp-shuffle reduction     (+ divergent shared-mem baseline)
```

## Run

```bash
make test            # build simulator, run 78 tests
make experiment      # markdown tables: tiling, stride, associativity, replacement policy
make run-cuda        # needs nvcc + NVIDIA GPU; prints PASS/FAIL and timings
                     # old nvcc: make -C cuda ARCH=sm_75   (T4)  /  sm_86 (RTX 30xx)
```

Simulator CLI: `cachesim --l1 32K:8:64 --l2 256K:8:64 --policy lru < trace.txt`
(trace lines: `R 0x1000` / `W 4096`).

## What the tests establish

- Hand-computed cases: sequential miss rate = elem/line, direct-mapped ping-pong is all conflict
  misses, LRU thrashes on a cyclic sweep one line over capacity, LRU vs FIFO divergence, writebacks.
- Differential fuzz: 36 random (config x policy x seed) runs and 2 two-level runs match an
  independently written Python model on every counter.
- 3C invariant: compulsory + capacity + conflict == misses.
- `test_kernel_logic.py` emulates both CUDA kernels' index math on the CPU (ragged sizes, barriers
  as phase boundaries). It does not replace running them on a GPU.

## Cache result that motivates the GPU kernel (n=48 doubles, 8KB 4-way L1)

| workload | miss rate |
|---|---|
| naive ijk | 10.6% |
| tiled, tile=8 | 1.1% |
| tiled, tile=16 | 0.9% |

## Status

- C++ and Python: built and tested (78 pass).
- CUDA: all correctness checks pass on a Tesla T4 (Google Colab, sm_75); also compiles clean for
  sm_80 and sm_90.

### GPU results (Tesla T4, sm_75, Colab)

| GPU | matmul naive | matmul tiled | reduce baseline | reduce shuffle |
|---|---|---|---|---|
| Tesla T4 | 5.543 ms, 387.5 GFLOP/s | 2.893 ms, 742.3 GFLOP/s (1.9x) | 1.339 ms, 50.1 GB/s | 0.249 ms, 269.4 GB/s (5.4x) |

Matmul: fp32, 1024x1024x1024. Reduce: 2^24 floats (67 MB). Single run each, so treat the
numbers as approximate.

- Shuffle reduction reaches about 84% of the T4's ~320 GB/s peak DRAM bandwidth, so it is
  close to memory-bound.
- Tiled matmul is about 9% of the T4's fp32 peak (~8 TFLOP/s). Register tiling and vectorized
  loads would be the next steps; it is not competitive with cuBLAS.

## Talking points (only claim what you can explain)

- Tiling: each global load reused TILE times; the CPU-cache experiment shows the same effect.
- `Bs[k][tx]` reads are stride-1 across a warp (no bank conflict); `As[ty][k]` is a broadcast.
- Baseline reduction: `tid % (2*s)` leaves scattered active lanes (divergence) and needs many passes;
  shuffle reduces in registers with no barriers, so it is bandwidth-bound.
- Float sums are order-dependent, so tests use an error scaled by sum|a*b| (matmul) or a relative
  tolerance (reduce) against a double-precision CPU reference.
