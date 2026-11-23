"""Experiments that motivate the CUDA kernels. Run: python3 harness/experiment.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import sim
import traces as T

L1 = "8K:4:64"


def row(name, tr, spec=L1):
    s = sim.l1(tr, spec)
    return (f"| {name} | {s['accesses']:,} | {s['miss_rate']*100:5.1f}% | "
            f"{s['compulsory']:,} | {s['capacity']:,} | {s['conflict']:,} |")


def main():
    hdr = "| workload | accesses | miss rate | compulsory | capacity | conflict |\n|---|---|---|---|---|---|"

    print(f"## Matmul memory behavior (n=48, doubles, L1 = {L1}, LRU)\n\n{hdr}")
    n = 48
    print(row("naive ijk", T.matmul_naive(n)))
    for tile in (4, 8, 16):
        print(row(f"tiled, tile={tile}", T.matmul_tiled(n, tile)))

    print(f"\n## Stride sweep (4096 reads, L1 = {L1})\n\n{hdr}")
    for stride in (4, 16, 64, 256, 2048):
        print(row(f"stride {stride}B", T.strided(4096, stride)))

    print("\n## Associativity vs conflict misses (8 lines, 1KB-apart, cyclic; 4KB cache)\n\n" + hdr)
    tr = [(T.R, (i % 8) * 1024) for i in range(800)]
    for ways in (1, 2, 4, 8):
        print(row(f"{ways}-way", tr, f"4K:{ways}:64"))

    print("\n## Replacement policy on cyclic sweep of 1 line more than capacity (8-way, 512B)\n\n" + hdr)
    tr = [(T.R, (i % 9) * 64) for i in range(900)]
    for p in ("lru", "fifo", "random"):
        s = sim.l1(tr, "512:8:64", policy=p)
        print(f"| {p} | {s['accesses']:,} | {s['miss_rate']*100:5.1f}% | {s['compulsory']:,} | {s['capacity']:,} | {s['conflict']:,} |")


if __name__ == "__main__":
    main()
