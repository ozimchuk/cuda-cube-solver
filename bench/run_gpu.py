"""Solve scrambles from a file with the GPU solver and report cubes/s."""
import argparse
import time
from pathlib import Path

from gpucube.solver import GpuSolver, parse_moves
from gpucube.cubie import MOVES_HTM, CubieState, apply_moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scrambles", type=Path)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--table", type=Path, default=Path("tables/prune-ud-208.bin"))
    ap.add_argument("--prefix", type=int, default=4)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-inverse", action="store_true")
    ap.add_argument("--mode", default="forward", choices=["forward", "bidir"])
    args = ap.parse_args()
    lines = args.scrambles.read_text().splitlines()[: args.count]
    states = [apply_moves(CubieState(), [MOVES_HTM[m] for m in parse_moves(l)]) for l in lines]
    solver = GpuSolver(args.table, prefix_len=args.prefix, inverse=not args.no_inverse, mode=args.mode)
    out = []
    t = time.perf_counter()
    nodes = 0
    for start in range(0, len(states), args.batch):
        sols = solver.solve(states[start:start + args.batch])
        for k, s in enumerate(sols):
            out.append(f"{start + k} {s.seconds:.3f} {len(s.moves)} {s.text()}")
            nodes += s.nodes
    dt = time.perf_counter() - t
    if args.out:
        args.out.write_text("\n".join(out) + "\n")
    print(f"{len(states)} cubes in {dt:.2f}s = {len(states) / dt:.3f} cubes/s, {nodes / dt / 1e9:.3f} G nodes/s, "
          f"{nodes / len(states) / 1e6:.1f} M nodes/cube")


if __name__ == "__main__":
    main()
