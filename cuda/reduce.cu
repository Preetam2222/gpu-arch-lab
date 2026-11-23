// Kernel 2: warp-shuffle parallel sum reduction.
//
// Baseline (reduce_baseline): classic shared-memory tree with interleaved addressing.
//   `tid % (2*s) == 0` leaves a shrinking, scattered subset of threads active in every warp
//   (warp divergence), needs log2(blockDim) __syncthreads, and produces one partial per
//   256 inputs, so the host must launch many passes.
// Optimized (reduce_warp_shuffle):
//   1. grid-stride loop: each thread privately accumulates many elements (sequential,
//      coalesced reads; only ~1024 partials survive).
//   2. __shfl_down_sync reduces within a warp via registers: no shared memory, no barriers.
//   3. one shared-memory exchange across the block's warps, then one more shuffle.
//   Two launches total (blocks -> partials -> 1 block).
//
// Build: nvcc -O3 -arch=native -o reduce reduce.cu      Run: ./reduce [log2_n]
#include <cmath>
#include <vector>

#include "common.cuh"

constexpr int BLOCK = 256;  // multiple of 32, <= 1024
constexpr int MAX_BLOCKS = 1024;

__global__ void reduce_baseline(const float* in, float* out, int n) {
    extern __shared__ float s[];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + tid;
    s[tid] = i < (unsigned)n ? in[i] : 0.f;
    __syncthreads();
    for (unsigned st = 1; st < blockDim.x; st *= 2) {
        if (tid % (2 * st) == 0) s[tid] += s[tid + st];  // divergent
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;  // valid in lane 0
}

// Requires blockDim.x to be a multiple of 32 (all lanes participate in every shuffle).
__global__ void reduce_warp_shuffle(const float* in, float* out, int n) {
    __shared__ float warp_sums[32];
    float sum = 0.f;
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) sum += in[i];

    sum = warp_reduce_sum(sum);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) warp_sums[wid] = sum;
    __syncthreads();

    const int nwarps = blockDim.x >> 5;
    sum = threadIdx.x < nwarps ? warp_sums[threadIdx.x] : 0.f;  // first warp gathers the partials
    if (wid == 0) sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

// ---- host drivers: both return the scalar sum of d_in[0..n) ----

static float sum_baseline(const float* d_in, int n, float* buf0, float* buf1) {
    const float* cur = d_in;
    int cnt = n, pass = 0;
    do {
        int blocks = (cnt + BLOCK - 1) / BLOCK;
        float* dst = (pass++ & 1) ? buf1 : buf0;
        reduce_baseline<<<blocks, BLOCK, BLOCK * sizeof(float)>>>(cur, dst, cnt);
        cur = dst;
        cnt = blocks;
    } while (cnt > 1);
    float r;
    CUDA_CHECK(cudaMemcpy(&r, cur, 4, cudaMemcpyDeviceToHost));
    return r;
}

static void launch_shuffle(const float* d_in, int n, float* partials, float* d_out) {
    int blocks = (n + BLOCK - 1) / BLOCK;
    if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
    reduce_warp_shuffle<<<blocks, BLOCK>>>(d_in, partials, n);
    reduce_warp_shuffle<<<1, BLOCK>>>(partials, d_out, blocks);
}

static bool close_enough(double got, double ref) { return std::fabs(got - ref) <= 1e-5 * std::fabs(ref) + 1e-6; }

int main(int argc, char** argv) {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    std::printf("device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    bool ok = true;
    // Sizes chosen to hit: 1 element, partial warp/block, exact block, ragged, multi-pass.
    std::vector<int> sizes = {1, 31, 32, 255, 256, 257, 1000, 65536, 100003, 1 << 20, (1 << 22) + 17};
    int big = 1 << 24;
    if (argc == 2) big = 1 << atoi(argv[1]);

    int maxn = big;
    for (int s : sizes) maxn = s > maxn ? s : maxn;
    std::vector<float> h(maxn);
    srand(42);
    for (auto& x : h) x = rand() / (float)RAND_MAX;  // [0,1): sum grows ~n/2, stresses fp32

    float *d_in, *buf0, *buf1, *partials, *d_out;
    int nbuf = (maxn + BLOCK - 1) / BLOCK + 1;
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)maxn * 4));
    CUDA_CHECK(cudaMalloc(&buf0, (size_t)nbuf * 4));
    CUDA_CHECK(cudaMalloc(&buf1, (size_t)nbuf * 4));
    CUDA_CHECK(cudaMalloc(&partials, MAX_BLOCKS * 4));
    CUDA_CHECK(cudaMalloc(&d_out, 4));
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), (size_t)maxn * 4, cudaMemcpyHostToDevice));

    std::printf("correctness vs double-precision CPU sum:\n");
    sizes.push_back(big);
    for (int n : sizes) {
        double ref = 0;
        for (int i = 0; i < n; ++i) ref += h[i];
        float a = sum_baseline(d_in, n, buf0, buf1);
        launch_shuffle(d_in, n, partials, d_out);
        CUDA_CHECK(cudaGetLastError());
        float b;
        CUDA_CHECK(cudaMemcpy(&b, d_out, 4, cudaMemcpyDeviceToHost));
        bool pa = close_enough(a, ref), pb = close_enough(b, ref);
        ok &= pa && pb;
        std::printf("  n=%-9d ref=%-14.1f baseline %s  shuffle %s\n", n, ref, pa ? "PASS" : "FAIL",
                    pb ? "PASS" : "FAIL");
    }

    std::printf("performance (n=%d, %.0f MB read):\n", big, big * 4.0 / 1e6);
    float t_base = time_ms([&] { sum_baseline(d_in, big, buf0, buf1); });
    float t_shfl = time_ms([&] { launch_shuffle(d_in, big, partials, d_out); });
    std::printf("  baseline %.3f ms  %.1f GB/s\n", t_base, big * 4.0 / (t_base * 1e6));
    std::printf("  shuffle  %.3f ms  %.1f GB/s   (%.2fx)\n", t_shfl, big * 4.0 / (t_shfl * 1e6),
                t_base / t_shfl);
    int clk_khz = 0, bus_bits = 0;  // cudaDeviceProp::memoryClockRate was removed in CUDA 13
    CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    std::printf("  (peak DRAM bandwidth on this GPU ~ %.0f GB/s)\n",
                2.0 * clk_khz * (bus_bits / 8) / 1e6);

    std::printf(ok ? "ALL PASS\n" : "FAILED\n");
    return ok ? 0 : 1;
}
