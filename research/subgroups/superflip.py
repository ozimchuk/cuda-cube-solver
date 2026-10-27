"""Why is the superflip slow, and for which tables?

    PYTHONPATH=src python research/subgroups/superflip.py rounds d4t s4-b3 ...
    PYTHONPATH=src python research/subgroups/superflip.py near d4t s4-b3 ...

rounds: solve the superflip and print nodes, time and node rate for each IDA* bound.
lookups: table lookups per node and which frame prunes (kernel LOOKUP_STATS), on the
        superflip and on 100 random positions.
near:   for positions g random moves from the superflip (and, for contrast, from a random
        distance-20 position), the fraction the search would expand in its last round
        (bound 20): table bound h <= 20 - g.  The node count is roughly
        sum over g of 13.35^g times that fraction.
"""
import random
import sys
from pathlib import Path

import numpy as np

from cuda_cube_solver import kinds
from cuda_cube_solver.cubie import MOVES_HTM, CubieState, apply_moves
from cuda_cube_solver.solver import GpuSolver, parse_moves

ROOT = Path(__file__).resolve().parents[2]
SUPERFLIP = (ROOT / "data/distance20/superflip.txt").read_text().split("\n")[0]
D20 = (ROOT / "data/distance20/random10.moves.txt").read_text().split("\n")[0]


def state(text):
    return apply_moves(CubieState(), [MOVES_HTM[m] for m in parse_moves(text)])


def solver(name, tables=Path("tables"), **kw):
    p = kinds.PRESET_BY_NAME.get(name)
    kind, fmt = (p.kind, p.format) if p else (name, "block")
    return GpuSolver(tables, kind=kind, table_format=fmt, **kw)


def lookups(name):
    s = solver(name, collect_stats=True, step_defines=("LOOKUP_STATS",))
    randoms = [state(l) for l in (ROOT / "data/random1000.txt").read_text().splitlines()[:100]]
    for label, states in (("superflip", [state(SUPERFLIP)]), ("random100", randoms)):
        s.stats.fill(0)
        nodes = sum(sol.nodes for sol in s.solve(states))
        st = s.stats.get()
        look, prune = st[:6], st[8:14]
        print(f"{name:7} {label:10} lookups/node {look.sum() / nodes:.2f}  pruned/node {prune.sum() / nodes:.2f}  "
              f"lookups by frame {[round(x / nodes, 2) for x in look]}  prunes by frame {[round(x / nodes, 2) for x in prune]}")


def rounds(name):
    s = solver(name)
    times = []
    sol = s.solve([state(SUPERFLIP)], log=lambda m: times.append(float(m.split("s")[0])))[0]
    prev = 0.0
    print(f"{name}: {sol.nodes / 1e9:.1f} G nodes, {sol.seconds:.1f} s")
    for (bound, nodes), t in zip(sol.iteration_nodes, times):
        dt = t - prev
        prev = t
        if nodes > 1e8:
            print(f"  bound {bound}: {nodes / 1e9:7.2f} G nodes {dt:6.1f} s {nodes / dt / 1e9 if dt else 0:5.2f} G/s")


def walks(start, g, n, rng):
    """n random move sequences of length g from start (no two turns of the same face in a row)."""
    out = []
    for _ in range(n):
        seq, last = [], -1
        while len(seq) < g:
            m = rng.randrange(18)
            if m // 3 != last:
                seq.append(m)
                last = m // 3
        out.append(apply_moves(start, [MOVES_HTM[m] for m in seq]))
    return out


def near(name, n=3000):
    s = solver(name)
    for label, start in (("superflip", state(SUPERFLIP)), ("distance-20", state(D20))):
        rng = random.Random(1)
        row = []
        for g in range(0, 11):
            h = s.root_h(walks(start, g, n if g else 1, rng))[0].max(axis=1)
            row.append(f"{g}:{np.mean(h <= 20 - g):.3f}/{np.mean(h):.1f}")
        print(f"{name:7} {label:11} " + " ".join(row))


if __name__ == "__main__":
    cmd, names = sys.argv[1], sys.argv[2:]
    if cmd == "near":
        print("g: fraction expanded at bound 20 / mean h")
    for name in names:
        {"rounds": rounds, "near": near, "lookups": lookups}[cmd](name)
