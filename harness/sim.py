"""Thin subprocess wrapper around the C++ cachesim binary."""
import json
import os
import subprocess

import traces

BIN = os.environ.get(
    "CACHESIM", os.path.join(os.path.dirname(__file__), "..", "cachesim", "cachesim"))


def run(trace, l1, l2=None, l3=None, policy="lru", classify=True):
    """l1/l2/l3 are 'SIZE:ASSOC:LINE' strings, e.g. '32K:8:64'. Returns parsed JSON dict."""
    cmd = [BIN, "--l1", l1, "--policy", policy]
    if l2:
        cmd += ["--l2", l2]
    if l3:
        cmd += ["--l3", l3]
    if not classify:
        cmd.append("--no-classify")
    p = subprocess.run(cmd, input=traces.to_text(trace), capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip())
    return json.loads(p.stdout)


def l1(trace, spec, **kw):
    return run(trace, spec, **kw)["levels"][0]
