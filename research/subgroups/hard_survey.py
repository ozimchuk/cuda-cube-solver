"""Second survey pass: real search cost of each table on hard and on random positions.

    PYTHONPATH=src:research/subgroups python research/subgroups/hard_survey.py [min_factor] [max_factor] [modes=1]

The first pass ranked tables by their bounds on random positions (avg6).  The superflip ranks
them differently, so this pass builds each kind's block table (factor <= 6,144: building needs
the mod-3 and block tables on a 16 GB card together) and solves three sets:

- the superflip;
- 10 known distance-20 positions (data/distance20/random10.moves.txt);
- the first 100 positions of data/random1000.txt.

The metric is nodes in completed rounds (bounds below the solution length): it does not depend
on timing or on where the last round happens to find its first solution.  Results go to
hard.jsonl (one line per kind, written with fsync; finished kinds are skipped).  Largest kinds
first.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cupy as cp

import groups
import run_gen
from cuda_cube_solver import device, table
from cuda_cube_solver.cubie import MOVES_HTM, CubieState, apply_moves
from cuda_cube_solver.solver import GpuSolver, parse_moves

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
SCRATCH = ROOT / "tables" / "hard-survey-tmp"
OUT = HERE / "hard.jsonl"


def states(path, n):
    lines = [l for l in (ROOT / path).read_text().splitlines() if l.strip()][:n]
    return [apply_moves(CubieState(), [MOVES_HTM[m] for m in parse_moves(l)]) for l in lines]


SETS = {
    "superflip": states("data/distance20/superflip.txt", 1),
    "d20": states("data/distance20/random10.moves.txt", 10),
    "random": states("data/random1000.txt", 100),
}


def complete_nodes(sol):
    return sum(n for b, n in sol.iteration_nodes if b < len(sol.moves))


def measure(k):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    dt = device.DeviceTables(k.gen)
    t = time.perf_counter()
    mod3, blk, counts = table.generate(dt, k, log=lambda m: None)
    del mod3
    table.save_device(blk, SCRATCH / f"prune-ud-{k.name}-blk.bin")
    del blk
    cp.get_default_memory_pool().free_all_blocks()
    res = {"kind": k.name, "factor": k.n_edge // 495, "mode": k.mode, **k.describe(),
           "build_s": round(time.perf_counter() - t, 1)}
    s = GpuSolver(SCRATCH, kind=k)
    for name, sts in SETS.items():
        t = time.perf_counter()
        sols = s.solve(sts)
        res[f"{name}_nodes"] = sum(complete_nodes(x) for x in sols) / len(sols)
        res[f"{name}_s"] = round(time.perf_counter() - t, 2)
        res[f"{name}_len"] = [len(x.moves) for x in sols]
    del s
    cp.get_default_memory_pool().free_all_blocks()
    shutil.rmtree(SCRATCH)
    return res


def main():
    min_f = int(sys.argv[1]) if len(sys.argv) > 1 else 1120
    max_f = int(sys.argv[2]) if len(sys.argv) > 2 else 6144
    done = {json.loads(l)["kind"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    # Modes 2 and 3 keep no flips of the non-E edges: on the superflip they search ~100x more
    # (870-980 G nodes at 3,360) and they already lost on random positions, so mode 1 only.
    modes = {int(m) for m in sys.argv[3].split(",")} if len(sys.argv) > 3 else {1}
    todo = sorted((k for k in run_gen.kinds(min_f, max_f) if k.name not in done and k.mode in modes),
                  key=lambda k: -k.n_edge)
    print(f"{len(todo)} kinds to measure", flush=True)
    for k in todo:
        res = measure(k)
        print(f"{k.name:6} factor {res['factor']:5}  superflip {res['superflip_nodes'] / 1e9:7.2f} G  "
              f"d20 {res['d20_nodes'] / 1e9:6.3f} G  random {res['random_nodes'] / 1e6:6.1f} M", flush=True)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


if __name__ == "__main__":
    main()
