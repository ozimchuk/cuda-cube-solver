"""Command line: build the pruning tables, or solve scrambles read from stdin."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="gpucube")
    sub = ap.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-table", help="generate the pruning tables on the GPU (~2 min, 15 GB)")
    build.add_argument("--out", type=Path, default=Path("tables"))
    build.add_argument("--kind", default="208", help="edge variant: 208 (default), 112, 304, ...")
    solve = sub.add_parser("solve", help="solve move-sequence scrambles, one per line on stdin")
    solve.add_argument("--table", type=Path, default=Path("tables/prune-ud-208.bin"))
    solve.add_argument("--batch", type=int, default=100)
    solve.add_argument("--mode", default="forward", choices=["forward", "bidir"])
    args = ap.parse_args(argv)

    if args.command == "build-table":
        from gpucube import coords, table

        print(table.build(args.out, coords.EdgeKind.parse(args.kind)))
        return

    from gpucube.cubie import MOVES_HTM, CubieState, apply_moves
    from gpucube.solver import GpuSolver, parse_moves

    lines = [line.strip() for line in sys.stdin if line.strip()]
    states = [apply_moves(CubieState(), [MOVES_HTM[m] for m in parse_moves(line)]) for line in lines]
    solver = GpuSolver(args.table, mode=args.mode)
    for start in range(0, len(states), args.batch):
        for k, sol in enumerate(solver.solve(states[start:start + args.batch])):
            print(f"{start + k} {len(sol.moves)} {sol.text()}", flush=True)


if __name__ == "__main__":
    main()
