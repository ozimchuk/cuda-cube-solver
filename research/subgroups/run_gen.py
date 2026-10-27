"""Build and measure the generic survey tables (groups.GenKind) one after another.

Usage: run_gen.py validate | run_gen.py all [min_factor=2048] [max_factor=12500]
validate: the generic code must reproduce 208, 612 and 304 exactly.
Results go to results.jsonl (kind names g<mode>-<group id>, ids index candidates.json).
"""
import json
import sys
from pathlib import Path

import groups
import survey
from cuda_cube_solver import device

HERE = Path(__file__).parent


def kinds(min_f, max_f):
    out = []
    for gid, g in enumerate(groups.stable_candidates()):
        for mode in (1, 2, 3):
            k = groups.GenKind(mode, g, gid)
            if min_f <= k.n_edge // 495 <= max_f:
                out.append(k)
    return out


def done():
    path = HERE / "results.jsonl"
    return {json.loads(l)["kind"] for l in path.read_text().splitlines()} if path.exists() else set()


def measure(k):
    return survey.run(k.name, False, kind=k, dt=device.DeviceTables(k.gen),
                      extra={"mode": k.mode, "factor": k.n_edge // 495, **k.describe()})


def validate():
    cands = groups.stable_candidates()
    flips = frozenset(groups.EID[((0, 1, 2, 3), f)] for f in groups.product((0, 1), repeat=4))
    full = frozenset(range(384))
    s4 = frozenset(groups.EID[(p, (0, 0, 0, 0))] for p in groups.PERMS)
    d4 = groups.sym_image()
    for g, mode, ref in ((flips, 1, "208"), (d4, 1, "612"), (s4, 2, "304")):
        k = groups.GenKind(mode, g, cands.index(g))
        r = measure(k)
        print(f"{k.name} should equal {ref}: factor {k.n_edge // 495}", flush=True)
    del full


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        validate()
    else:
        lo = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
        hi = int(sys.argv[3]) if len(sys.argv) > 3 else 12500
        todo = [k for k in kinds(lo, hi) if k.name not in done()]
        print(f"{len(todo)} tables to build", flush=True)
        for k in sorted(todo, key=lambda k: k.n_edge):
            measure(k)
