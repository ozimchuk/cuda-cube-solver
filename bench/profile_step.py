"""One search_step launch on a fixed workload, for Nsight Compute.

    ncu --kernel-name search_step --launch-count 1 --set full -o report \
        python bench/profile_step.py
"""
import sys
from pathlib import Path

import cupy as cp
import numpy as np

from cuda_cube_solver.cubie import MOVES_HTM, CubieState, apply_moves
from cuda_cube_solver.solver import JOB_DTYPE, GpuSolver, parse_moves

COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BOUND = int(sys.argv[2]) if len(sys.argv) > 2 else 16


def main():
    lines = Path("data/random1000.txt").read_text().splitlines()[:COUNT]
    cubes = [apply_moves(CubieState(), [MOVES_HTM[m] for m in parse_moves(l)]) for l in lines]
    s = GpuSolver(Path("tables"))
    h, words = s.root_h(cubes)
    jobs = np.zeros(len(cubes), dtype=JOB_DTYPE)
    jobs["frame"] = words
    jobs["h"] = h
    jobs["bound"] = BOUND
    d_jobs = cp.asarray(jobs.view(np.uint8))
    counter = cp.zeros(1, dtype=cp.uint64)
    pre = s.prefixes[4]
    args = (s.dt.tables, s.sconst, s.blk, s.blk, s.blk, s.blk, s.dt.classidx, d_jobs,
            cp.asarray(np.arange(len(cubes), dtype=np.int32)), np.int32(len(cubes)),
            pre, np.int32(len(pre)), np.int32(4), counter, np.uint64(0))
    s.k_search((s.grid,), (s.block,), args)
    cp.cuda.Device().synchronize()
    print("children:", int(d_jobs.get().view(JOB_DTYPE)["nodes"].sum()))


if __name__ == "__main__":
    main()
