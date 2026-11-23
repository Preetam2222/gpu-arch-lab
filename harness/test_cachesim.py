"""Tests: hand-computed analytic cases, 3C invariants, and differential fuzzing vs ref_cache.py.

Run:  pytest -q harness
"""
import random
import subprocess

import pytest

import sim
import traces as T
from ref_cache import RefCache, RefHierarchy

LINE = 64


def kb(n):
    return n * 1024


# ---------- analytic cases (answers worked out by hand) ----------

def test_sequential_miss_rate_is_elem_over_line():
    n = 4096  # 4-byte elements, 16 per 64B line -> 1 miss per 16 accesses
    s = sim.l1(T.sequential(n), "32K:8:64")
    assert s["misses"] == n * 4 // LINE
    assert s["compulsory"] == s["misses"]


def test_stride_of_one_line_never_hits():
    s = sim.l1(T.strided(1000, LINE), "32K:8:64")
    assert s["hits"] == 0 and s["compulsory"] == 1000


def test_working_set_that_fits_hits_on_second_pass():
    s = sim.l1(T.sweep(kb(16), passes=2), "32K:8:64")
    assert s["misses"] == kb(16) // LINE  # only the cold pass misses


def test_direct_mapped_ping_pong_is_all_conflict():
    a, b = 0, kb(1)  # same set in a 1KB direct-mapped cache
    tr = [(T.R, a if i % 2 == 0 else b) for i in range(100)]
    s = sim.l1(tr, "1K:1:64")
    assert s["misses"] == 100
    assert (s["compulsory"], s["capacity"], s["conflict"]) == (2, 0, 98)
    s2 = sim.l1(tr, "1K:2:64")  # two ways absorb the conflict
    assert s2["misses"] == 2


def test_cyclic_sweep_one_line_too_big_thrashes_lru_but_not_random():
    ways = 8
    nlines = ways + 1
    tr = [(T.R, (i % nlines) * LINE) for i in range(nlines * 5)]
    cfg = f"{ways * LINE}:{ways}:64"  # fully associative, 8 lines
    s = sim.l1(tr, cfg)
    assert s["hits"] == 0  # classic LRU pathology
    assert (s["compulsory"], s["capacity"], s["conflict"]) == (nlines, nlines * 4, 0)
    assert sim.l1(tr, cfg, policy="random")["hits"] > 0


def test_lru_vs_fifo_on_one_set():
    # one 2-way set: A B A C A.  LRU keeps A (2 hits); FIFO evicts A when C arrives (1 hit).
    A, B, C = 0, 64, 128
    tr = [(T.R, x) for x in (A, B, A, C, A)]
    assert sim.l1(tr, "128:2:64", policy="lru")["hits"] == 2
    assert sim.l1(tr, "128:2:64", policy="fifo")["hits"] == 1


def test_writeback_only_for_dirty_victims():
    tr = [(T.W, 0), (T.R, 1024)]  # direct-mapped 1KB: second access evicts dirty line 0
    s = sim.l1(tr, "1K:1:64")
    assert (s["evictions"], s["writebacks"]) == (1, 1)
    tr = [(T.R, 0), (T.R, 1024)]
    assert sim.l1(tr, "1K:1:64")["writebacks"] == 0


def test_write_hit_marks_dirty():
    tr = [(T.R, 0), (T.W, 0), (T.R, 1024)]
    s = sim.l1(tr, "1K:1:64")
    assert s["write_hits"] == 1 and s["writebacks"] == 1


def test_l2_filters_l1_misses():
    tr = T.sweep(kb(16), passes=3)  # exceeds 8KB L1 (thrash), fits 64KB L2
    lv = sim.run(tr, "8K:4:64", l2="64K:8:64")["levels"]
    assert lv[1]["accesses"] == lv[0]["misses"]  # no writes here, so L2 sees exactly L1's misses
    assert lv[1]["misses"] == kb(16) // LINE  # L2 only misses cold


# ---------- invariants + differential fuzz ----------

CONFIGS = [(1024, 1, 64), (1024, 2, 32), (2048, 4, 64), (4096, 8, 64), (512, 8, 64), (4096, 2, 16)]


@pytest.mark.parametrize("policy", ["lru", "fifo"])
@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("seed", range(3))
def test_matches_reference_model(policy, cfg, seed):
    size, assoc, line = cfg
    tr = T.random_trace(3000, addr_space=size * 6, seed=seed * 101 + assoc)
    got = sim.l1(tr, f"{size}:{assoc}:{line}", policy=policy)
    ref = RefCache(size, assoc, line, policy, classify=True)
    for w, a in tr:
        ref.access(a, w)
    for k, v in ref.s.items():
        assert got[k] == v, (k, cfg, policy)
    assert got["hits"] == ref.hits


@pytest.mark.parametrize("policy", ["lru", "fifo"])
def test_two_level_matches_reference(policy):
    levels = [(1024, 2, 64), (8192, 4, 64)]
    tr = T.random_trace(5000, addr_space=40000, seed=7, write_frac=0.4)
    got = sim.run(tr, "1K:2:64", l2="8K:4:64", policy=policy)["levels"]
    ref = RefHierarchy(levels, policy)
    for w, a in tr:
        ref.access(a, w)
    for i, lv in enumerate(ref.levels):
        for k, v in lv.s.items():
            if i > 0 and k in ("compulsory", "capacity", "conflict"):
                continue
            assert got[i][k] == v, (i, k)


@pytest.mark.parametrize("seed", range(5))
def test_3c_sums_to_misses_and_random_is_deterministic(seed):
    tr = T.random_trace(4000, addr_space=20000, seed=seed)
    s = sim.l1(tr, "2K:4:64")
    assert s["compulsory"] + s["capacity"] + s["conflict"] == s["misses"]
    assert s["hits"] + s["misses"] == s["accesses"] == len(tr)
    r1 = sim.l1(tr, "2K:4:64", policy="random")
    assert r1 == sim.l1(tr, "2K:4:64", policy="random")


def test_more_associativity_never_adds_conflict_misses_on_cyclic_pattern():
    tr = [(T.R, (i % 6) * 1024) for i in range(600)]  # 6 lines all mapping to set 0 of a 1KB DM cache
    assert sim.l1(tr, "4K:1:64")["conflict"] > 0
    assert sim.l1(tr, "4K:8:64")["conflict"] == 0


# ---------- the architecture payoff: tiling (what CUDA shared memory does) ----------

def test_tiled_matmul_beats_naive_in_cache():
    n = 32
    naive = sim.l1(T.matmul_naive(n), "4K:4:64")
    tiled = sim.l1(T.matmul_tiled(n, 8), "4K:4:64")
    assert tiled["misses"] < 0.5 * naive["misses"]


# ---------- CLI error handling ----------

@pytest.mark.parametrize("args", [["--l1", "1000:2:64"], ["--l1", "1K:2"], ["--policy", "lru"], ["--bogus"]])
def test_bad_args_fail_cleanly(args):
    p = subprocess.run([sim.BIN, *args], input="R 0\n", capture_output=True, text=True)
    assert p.returncode == 1 and "error" in p.stderr


def test_bad_trace_line_fails_cleanly():
    p = subprocess.run([sim.BIN, "--l1", "1K:2:64"], input="X 0\n", capture_output=True, text=True)
    assert p.returncode == 1
