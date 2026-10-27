"""Time one IDA* round at a fixed bound for a set of cubes (kernel diagnostics)."""
import argparse
import time
from pathlib import Path

import cupy as cp
import numpy as np

from cuda_cube_solver.device import module
from cuda_cube_solver.solver import JOB_DTYPE, GpuSolver, parse_moves
from cuda_cube_solver.cubie import MOVES_HTM, CubieState, apply_moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--bound", type=int, default=17)
    ap.add_argument("--kernel", default="search6")
    ap.add_argument("--define", action="append", default=[])
    args = ap.parse_args()
    lines = Path("data/random1000.txt").read_text().splitlines()[: args.count]
    cubes = [apply_moves(CubieState(), [MOVES_HTM[m] for m in parse_moves(l)]) for l in lines]
    s = GpuSolver(Path("tables"), kind="d4t")
    kern = module(*args.define).get_function(args.kernel)
    h, words = s.root_h(cubes)
    jobs = np.zeros(len(cubes), dtype=JOB_DTYPE)
    jobs["frame"] = words
    jobs["h"] = h
    jobs["bound"] = args.bound
    jobs["found"] = 0
    d_jobs = cp.asarray(jobs.view(np.uint8))
    counter = cp.zeros(1, dtype=cp.uint64)
    pre = s.prefixes[4]
    kargs = (s.dt.tables, s.sconst, s.table, s.blk, s.dt.classidx, d_jobs,
             cp.asarray(np.arange(len(cubes), dtype=np.int32)), np.int32(len(cubes)),
             pre, np.int32(len(pre)), np.int32(4), counter, np.uint64(0))
    cp.cuda.Device().synchronize()
    t = time.perf_counter()
    kern((s.grid,), (s.block,), kargs)
    cp.cuda.Device().synchronize()
    dt = time.perf_counter() - t
    out = d_jobs.get().view(JOB_DTYPE)
    nodes = int(out["nodes"].sum())
    print(f"{args.kernel} {args.define} bound {args.bound}: {nodes / 1e9:.3f} G children, {dt:.2f}s, "
          f"{nodes / dt / 1e9:.3f} G children/s, solved {int(out['found'].sum())}/{len(cubes)}")


if __name__ == "__main__":
    main()
