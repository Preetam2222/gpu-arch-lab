"""CPU emulation of the CUDA kernels' index math / data flow (no GPU needed).

These transcribe cuda/matmul.cu and cuda/reduce.cu thread-by-thread (barriers become phase
boundaries) and check them against numpy. They catch indexing, padding and reduction-tree bugs;
they cannot catch real-hardware issues (races, bank conflicts, perf) -- run ./matmul and
./reduce on a GPU for those.
"""
import numpy as np
import pytest

TILE = 32


def emulate_matmul_tiled(A, B):
    M, K = A.shape
    _, N = B.shape
    C = np.full((M, N), np.nan)  # poison: any unwritten output fails the comparison
    for by in range((M + TILE - 1) // TILE):
        for bx in range((N + TILE - 1) // TILE):
            acc = np.zeros((TILE, TILE))  # acc[ty][tx]
            for t in range((K + TILE - 1) // TILE):
                As = np.zeros((TILE, TILE))
                Bs = np.zeros((TILE, TILE))
                for ty in range(TILE):          # phase 1: every thread loads
                    for tx in range(TILE):
                        row, col = by * TILE + ty, bx * TILE + tx
                        a_col, b_row = t * TILE + tx, t * TILE + ty
                        As[ty, tx] = A[row, a_col] if (row < M and a_col < K) else 0.0
                        Bs[ty, tx] = B[b_row, col] if (b_row < K and col < N) else 0.0
                # --- __syncthreads ---
                for ty in range(TILE):          # phase 2: every thread accumulates
                    for tx in range(TILE):
                        acc[ty, tx] += sum(As[ty, k] * Bs[k, tx] for k in range(TILE))
                # --- __syncthreads ---
            for ty in range(TILE):
                for tx in range(TILE):
                    row, col = by * TILE + ty, bx * TILE + tx
                    if row < M and col < N:
                        C[row, col] = acc[ty, tx]
    return C


@pytest.mark.parametrize("M,N,K", [(1, 1, 1), (33, 65, 31), (32, 32, 32), (40, 17, 70)])
def test_tiled_matmul_logic(M, N, K):
    rng = np.random.default_rng(M * 1000 + N)
    A, B = rng.standard_normal((M, K)), rng.standard_normal((K, N))
    np.testing.assert_allclose(emulate_matmul_tiled(A, B), A @ B, rtol=1e-9, atol=1e-9)


def shfl_down(v, off):
    """__shfl_down_sync over one warp: lane i reads lane i+off; out of range returns own value."""
    return np.array([v[i] + 0 if i + off >= 32 else v[i + off] for i in range(32)])


def warp_reduce_sum(v):
    for off in (16, 8, 4, 2, 1):
        v = v + shfl_down(v, off)
    return v[0]  # valid in lane 0


def emulate_shuffle_kernel(x, grid, block=256):
    out = np.zeros(grid)
    n = len(x)
    for b in range(grid):
        sums = np.zeros(block)
        for tid in range(block):
            i = b * block + tid
            while i < n:                       # grid-stride loop
                sums[tid] += x[i]
                i += grid * block
        nwarps = block // 32
        warp_sums = np.array([warp_reduce_sum(sums[w * 32:(w + 1) * 32]) for w in range(nwarps)])
        first = np.zeros(32)                   # first warp gathers partials (threadIdx.x < nwarps)
        first[:nwarps] = warp_sums
        out[b] = warp_reduce_sum(first)
    return out


def emulate_baseline_kernel(x, block=256):
    n = len(x)
    blocks = (n + block - 1) // block
    out = np.zeros(blocks)
    for b in range(blocks):
        s = np.array([x[b * block + t] if b * block + t < n else 0.0 for t in range(block)])
        st = 1
        while st < block:
            for tid in range(block):           # all threads evaluate the (divergent) predicate
                if tid % (2 * st) == 0:
                    s[tid] += s[tid + st]
            st *= 2
        out[b] = s[0]
    return out


@pytest.mark.parametrize("n", [1, 31, 32, 255, 256, 257, 1000, 5000, 70001])
def test_shuffle_reduction_two_pass(n):
    x = np.random.default_rng(n).random(n)
    grid = min((n + 255) // 256, 1024)
    partials = emulate_shuffle_kernel(x, grid)
    final = emulate_shuffle_kernel(partials, 1)  # second launch: <<<1, BLOCK>>> over partials
    assert final[0] == pytest.approx(x.sum(), rel=1e-12)


@pytest.mark.parametrize("n", [1, 31, 256, 257, 1000, 70001])
def test_baseline_reduction_multi_pass(n):
    x = np.random.default_rng(n).random(n)
    cur = x
    while True:                                # mirrors the do/while in sum_baseline
        cur = emulate_baseline_kernel(cur)
        if len(cur) == 1:
            break
    assert cur[0] == pytest.approx(x.sum(), rel=1e-12)
