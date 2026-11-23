"""Memory-access trace generators. A trace is a list of (is_write, byte_addr)."""
import random

R, W = False, True


def sequential(n, elem=4, base=0):
    return [(R, base + i * elem) for i in range(n)]


def strided(n, stride, base=0):
    return [(R, base + i * stride) for i in range(n)]


def sweep(nbytes, passes, step=4, base=0):
    return [(R, base + a) for _ in range(passes) for a in range(0, nbytes, step)]


def random_trace(n, addr_space, seed, write_frac=0.3):
    rng = random.Random(seed)
    return [(rng.random() < write_frac, rng.randrange(addr_space)) for _ in range(n)]


def matmul_naive(n, elem=8):
    """C = A*B, ijk order. A, B, C are n*n row-major arrays laid out back to back."""
    A, B, C = 0, n * n * elem, 2 * n * n * elem
    t = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                t.append((R, A + (i * n + k) * elem))
                t.append((R, B + (k * n + j) * elem))  # column walk: stride n*elem
            t.append((W, C + (i * n + j) * elem))
    return t


def matmul_tiled(n, tile, elem=8):
    """Same product, blocked into tile x tile sub-matrices (the CUDA shared-memory idea)."""
    A, B, C = 0, n * n * elem, 2 * n * n * elem
    t = []
    for ii in range(0, n, tile):
        for jj in range(0, n, tile):
            for kk in range(0, n, tile):
                for i in range(ii, min(ii + tile, n)):
                    for j in range(jj, min(jj + tile, n)):
                        for k in range(kk, min(kk + tile, n)):
                            t.append((R, A + (i * n + k) * elem))
                            t.append((R, B + (k * n + j) * elem))
                        t.append((R, C + (i * n + j) * elem))
                        t.append((W, C + (i * n + j) * elem))
    return t


def to_text(trace):
    return "".join(f"{'W' if w else 'R'} {a}\n" for w, a in trace)
