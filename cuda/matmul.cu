// Kernel 1: shared-memory tiled matrix multiply, C[MxN] = A[MxK] * B[KxN], row-major fp32.
//
// Why tiling: in the naive kernel every thread streams a row of A and a column of B from
// global memory, so each element is fetched K times in total. Here a 32x32 block stages
// 32x32 tiles of A and B in shared memory; each global load is reused 32 times.
// (harness/experiment.py shows the same effect on a CPU cache: ~10x fewer misses.)
//
// Build: nvcc -O3 -arch=native -o matmul matmul.cu      Run: ./matmul [M N K]
#include <cmath>
#include <vector>

#include "common.cuh"

constexpr int TILE = 32;

// Baseline: one thread per output element, no reuse. Adjacent threads (tx) read adjacent
// B columns (coalesced) and the same A element (broadcast), but nothing is cached by hand.
__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// Tiled kernel. Launch with dim3(TILE, TILE) threads and dim3(ceil(N/TILE), ceil(M/TILE)) blocks.
__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        const int a_col = t * TILE + tx;  // this thread's element of the A tile
        const int b_row = t * TILE + ty;  // this thread's element of the B tile
        // Coalesced loads (consecutive tx -> consecutive addresses); zero-pad out of range
        // so ragged M/N/K need no special-casing in the inner loop.
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
        __syncthreads();  // tile fully loaded before anyone reads it

#pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];  // As: broadcast within a warp; Bs: stride-1, conflict-free
        __syncthreads();  // everyone done reading before the next tile overwrites it
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ---- host ----

// Max error over a sample of output rows against a double-precision CPU reference,
// scaled by sum|a*b| (the standard dot-product error scale). Dividing by |ref| alone is wrong
// for zero-mean data: ref can be ~0 while fp32 rounding error is not.
static double check(const std::vector<float>& A, const std::vector<float>& B,
                    const std::vector<float>& C, int M, int N, int K) {
    double worst = 0;
    int step = M > 64 ? M / 64 : 1;  // sample ~64 rows (all rows when small)
    for (int i = 0; i < M; i += step) {
        for (int j = 0; j < N; ++j) {
            double ref = 0, mag = 0;
            for (int k = 0; k < K; ++k) {
                double p = (double)A[(size_t)i * K + k] * B[(size_t)k * N + j];
                ref += p;
                mag += std::fabs(p);
            }
            double err = std::fabs(C[(size_t)i * N + j] - ref) / (mag + 1e-12);
            worst = err > worst ? err : worst;
        }
    }
    return worst;
}

static bool run_case(int M, int N, int K, bool report_perf) {
    std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N);
    srand(1234);
    for (auto& x : hA) x = rand() / (float)RAND_MAX - 0.5f;
    for (auto& x : hB) x = rand() / (float)RAND_MAX - 0.5f;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * 4));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * 4));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    bool ok = true;
    struct Variant { const char* name; void (*kernel)(const float*, const float*, float*, int, int, int); };
    for (Variant v : {Variant{"naive", matmul_naive}, Variant{"tiled", matmul_tiled}}) {
        CUDA_CHECK(cudaMemset(dC, 0xFF, hC.size() * 4));  // poison so unwritten outputs are caught
        v.kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * 4, cudaMemcpyDeviceToHost));
        double err = check(hA, hB, hC, M, N, K);
        bool pass = err < 1e-4;  // fp32 bound ~ K*eps = 6e-5 at K=1024
        ok &= pass;
        std::printf("  %-5s %dx%dx%d  scaled err %.2e  %s", v.name, M, N, K, err, pass ? "PASS" : "FAIL");
        if (report_perf) {
            float ms = time_ms([&] { v.kernel<<<grid, block>>>(dA, dB, dC, M, N, K); });
            std::printf("   %.3f ms  %.1f GFLOP/s", ms, 2.0 * M * N * K / (ms * 1e6));
        }
        std::printf("\n");
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return ok;
}

int main(int argc, char** argv) {
    int dev = 0;
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    std::printf("device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    bool ok = true;
    std::printf("correctness (ragged sizes exercise the bounds checks):\n");
    ok &= run_case(33, 65, 31, false);
    ok &= run_case(100, 77, 129, false);
    ok &= run_case(257, 255, 513, false);

    int M = 1024, N = 1024, K = 1024;
    if (argc == 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    std::printf("performance:\n");
    ok &= run_case(M, N, K, true);

    std::printf(ok ? "ALL PASS\n" : "FAILED\n");
    return ok ? 0 : 1;
}
