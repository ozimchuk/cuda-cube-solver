"""Collect training and evaluation data for the learned-table study.

train.npz: uniformly random entries of the 208 table and their exact values.
eval.npz:  20,000 random positions x 6 frames: 208 index, exact 208 value,
           and the 304 table's lower bound (the fallback of the 1-bit scheme).
"""
import random
import sys
from pathlib import Path

import cupy as cp
import numpy as np

from gpucube import coords, device, table
from gpucube.cubie import MOVES_HTM, CubieState, apply_moves
from gpucube.solver import frames

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("research/learned_table/data")
N_TRAIN = 60_000_000
N_EVAL = 20_000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dt = device.DeviceTables()
    k208 = coords.K208
    n_entries = dt.n_classes * k208.n_edge
    mod3 = table.load_device(Path("tables/prune-ud-208.bin"), (n_entries + 15) // 16)
    blk = table.load_device(Path("tables/prune-ud-208-blk.bin"), table.block_words(n_entries))
    exact = device.module().get_function("exact_at")

    def exact_values(idx):
        out = cp.zeros(len(idx), dtype=cp.uint8)
        exact(((len(idx) + 255) // 256,), (256,), (mod3, blk, idx, np.int32(len(idx)), out))
        return out

    rng = np.random.default_rng(7)
    idx = rng.integers(0, n_entries, size=N_TRAIN, dtype=np.uint64)
    vals = np.concatenate([exact_values(cp.asarray(idx[s:s + 10_000_000])).get()
                           for s in range(0, N_TRAIN, 10_000_000)])
    np.savez(OUT / "train.npz", idx=idx, val=vals)
    print("train", N_TRAIN, "value histogram", np.bincount(vals), flush=True)

    r = random.Random(21)
    states = [apply_moves(CubieState(), [MOVES_HTM[r.randrange(18)] for _ in range(40)]) for _ in range(N_EVAL)]
    allf = [f for s in states for f in frames(s)]
    packed = [coords.pack(f) for f in allf]
    words = cp.asarray(device.pack_states(np.stack([p[0] for p in packed]), np.stack([p[1] for p in packed])))
    n = len(allf)
    eidx = cp.zeros(n, dtype=cp.uint64)
    device.module().get_function("probe_index_only")(((n + 127) // 128,), (128,),
                                                      (dt.tables, dt.classidx, words, np.int32(n), eidx))
    evals = exact_values(eidx)
    del mod3, blk
    cp.get_default_memory_pool().free_all_blocks()
    k304 = coords.EdgeKind(3, 4)
    blk304 = table.load_device(Path("tables/prune-ud-304-blk.bin"), table.block_words(dt.n_classes * k304.n_edge))
    fb = cp.zeros(n, dtype=cp.uint32)
    device.module(*k304.defines).get_function("probe_block")(((n + 127) // 128,), (128,),
                                                              (dt.tables, blk304, dt.classidx, words, np.int32(n), fb))
    np.savez(OUT / "eval.npz", idx=eidx.get(), val=evals.get(), fallback=fb.get().astype(np.uint8))
    print("eval", n, "frames; mean exact", float(evals.mean()), "mean fallback", float(fb.mean()))


if __name__ == "__main__":
    main()
