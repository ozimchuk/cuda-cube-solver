"""Subgroup survey: coset tables of other subgroups that contain every U and D turn.

For each edge variant: build the table on the GPU (mod-3 always; the block table when both
fit on the card), then measure exact coset distances on 20,000 random positions in all six
frames (three axes, each also inverted): avg1 = first frame, avg6 = maximum of the six with
the axis rule (all three axis frames, or all three inverse frames, equal -> +1).
Same positions as research/learned_table (random.Random(21), 40 random moves).

Usage: survey.py KIND... [--save]   (KIND e.g. 208 512 612 808 712; --save writes the block table)
Results are appended to research/subgroups/results.jsonl.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

from cuda_cube_solver import coords, device, table
from cuda_cube_solver.cubie import MOVES_HTM, CubieState, apply_moves
from cuda_cube_solver.solver import frames

HERE = Path(__file__).parent
TABLES = Path("tables")
N_EVAL = 20_000
VRAM = 15.4e9  # bytes usable for the two tables on this card


def eval_words():
    r = random.Random(21)
    states = [apply_moves(CubieState(), [MOVES_HTM[r.randrange(18)] for _ in range(40)]) for _ in range(N_EVAL)]
    allf = [f for s in states for f in frames(s)]
    packed = [coords.pack(f) for f in allf]
    return cp.asarray(device.pack_states(np.stack([p[0] for p in packed]), np.stack([p[1] for p in packed])))


def scores(v):
    v = v.reshape(-1, 6).astype(np.int64)
    m = v.max(1)
    plus = ((v[:, :3] == m[:, None]).all(1) | (v[:, 3:] == m[:, None]).all(1)) & (m > 0)
    return float(v[:, 0].mean()), float((m + plus).mean())


def run(name, save, kind=None, dt=None, extra=None):
    kind = kind or coords.EdgeKind.parse(name)
    dt = dt or device.DeviceTables()
    n_entries = dt.n_classes * kind.n_edge
    mod3_bytes = (n_entries + 15) // 16 * 4
    blk_bytes = table.block_words(n_entries) * 4
    with_block = mod3_bytes + blk_bytes <= VRAM
    t = time.perf_counter()
    path = TABLES / f"prune-ud-{name}.bin"
    if path.exists():
        mod3 = table.load_device(path, (n_entries + 15) // 16)
        counts = json.loads(path.with_suffix(".json").read_text())["depth_counts"]
        blk = None
    else:
        mod3, blk, counts = table.generate(dt, kind, log=lambda m: print(f"  {name} {m}", flush=True),
                                           with_block=with_block)
    gen_s = time.perf_counter() - t
    words = eval_words()
    n = len(words)
    exact = cp.zeros(n, dtype=cp.uint32)
    mod = device.module(*kind.defines)
    mod.get_function("root_distance")(((n + 127) // 128,), (128,),
                                      (dt.tables, mod3, dt.classidx, words, np.int32(n), exact))
    a1, a6 = scores(exact.get())
    res = {"kind": name, "entries": n_entries, "block_gb": round(blk_bytes / 1e9, 2),
           "entry_mean": sum(d * c for d, c in enumerate(counts)) / n_entries,
           "max_depth": len(counts) - 1, "avg1": a1, "avg6": a6, "depth_counts": counts,
           "generate_s": round(gen_s, 1), **(extra or {})}
    if blk is not None:
        bound = cp.zeros(n, dtype=cp.uint32)
        mod.get_function("probe_block")(((n + 127) // 128,), (128,),
                                        (dt.tables, blk, dt.classidx, words, np.int32(n), bound))
        b = bound.get()
        res["block_avg1"], res["block_avg6"] = scores(b)
        res["block_exact"] = float((b == exact.get()).mean())
        if save:
            table.save_device(blk, TABLES / f"prune-ud-{name}-blk.bin")
            (TABLES / f"prune-ud-{name}.json").write_text(json.dumps(
                {"coordinate": f"survey EP{kind.ep} x EO{kind.eo}", "entries": n_entries,
                 "depth_counts": counts}, indent=2) + "\n")
    print(f"{name}: {n_entries:.3e} entries, block {res['block_gb']} GB, entry mean {res['entry_mean']:.4f}, "
          f"avg1 {a1:.4f}, avg6 {a6:.4f}"
          + (f", block avg6 {res['block_avg6']:.4f}" if "block_avg6" in res else ""), flush=True)
    with open(HERE / "results.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")
        fh.flush()
        os.fsync(fh.fileno())  # a crash mid-survey once left a line of NUL bytes
    del mod3, blk
    cp.get_default_memory_pool().free_all_blocks()
    return res


if __name__ == "__main__":
    save = "--save" in sys.argv
    for name in [a for a in sys.argv[1:] if not a.startswith("--")]:
        run(name, save)
