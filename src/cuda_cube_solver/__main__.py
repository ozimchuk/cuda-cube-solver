"""Command line: build the pruning tables, or solve cubes read from stdin."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOLVE_EPILOG = """\
input (one cube per line, from stdin):
  a scramble in face-turn notation, e.g.  R U R' U2 F'   (the cube that the
      scramble produces from solved), or
  a 54-sticker facelet string in U R F D L B face order, e.g.
      DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD
      (any six symbols; the centres define the faces, so colour letters work)
  blank lines and lines starting with # are skipped.

output (one line per cube, in input order):
  <line number> <length> <moves>     an optimal solution in the half-turn
                                     metric; applying <moves> to the cube
                                     solves it
  <line number> error <message>      invalid input (the other lines still run)

Loading the table takes 1-5 s per invocation, so pipe many cubes into one run.
"""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cuda-cube-solver",
                                 description="Optimal Rubik's cube solver (half-turn metric) running on an NVIDIA GPU.")
    sub = ap.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-table", help="generate the pruning tables on the GPU (~2 min, 16 GB VRAM)",
                           description="Generate the pruning tables on the GPU. The default 208 table writes "
                                       "prune-ud-208.bin (7.0 GiB, generation only) and prune-ud-208-blk.bin "
                                       "(7.3 GiB, used for solving).")
    build.add_argument("--out", type=Path, default=Path("tables"), help="output directory (default: tables)")
    build.add_argument("--kind", default="208",
                       help="table variant: 208 (default, needed for solving); 112 and 304 only for the "
                            "experimental --mode multi / --step-extra")
    solve = sub.add_parser("solve", help="solve cubes read from stdin (scrambles or facelet strings)",
                           description="Solve cubes optimally.", epilog=SOLVE_EPILOG,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
    solve.add_argument("--table", type=Path, default=Path("tables/prune-ud-208.bin"),
                       help="path of prune-ud-208.bin; its -blk.bin companion is what is loaded "
                            "(default: tables/prune-ud-208.bin)")
    solve.add_argument("--format", choices=["auto", "moves", "facelets"], default="auto",
                       help="input format (default: auto-detect per line)")
    solve.add_argument("--batch", type=int, default=100, help="cubes searched together on the GPU (default: 100)")
    solve.add_argument("--mode", default="step", choices=["step", "forward", "multi", "bidir"],
                       help="search kernel (default: step; the others are slower, kept for experiments)")
    args = ap.parse_args(argv)

    if args.command == "build-table":
        from cuda_cube_solver import coords, table

        print(table.build(args.out, coords.EdgeKind.parse(args.kind)))
        return

    from cuda_cube_solver.cubie import MOVES_HTM, CubieState, apply_moves
    from cuda_cube_solver.facelets import FaceletError, is_facelet_string, parse_facelets
    from cuda_cube_solver.solver import GpuSolver, parse_moves

    cubes: list[tuple[int, CubieState]] = []
    errors: list[tuple[int, str]] = []  # printed in input order, between solutions
    for number, raw in enumerate(sys.stdin, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            facelets = args.format == "facelets" or (args.format == "auto" and is_facelet_string(line))
            if facelets:
                state = parse_facelets(line)
            else:
                state = apply_moves(CubieState(), [MOVES_HTM[m] for m in parse_moves(line)])
        except (FaceletError, KeyError) as exc:
            reason = exc.args[0] if isinstance(exc, FaceletError) else f"unknown move {exc.args[0]!r}"
            errors.append((number, f"{number} error {reason}"))
            continue
        cubes.append((number, state))

    def flush_errors(before: float) -> None:
        while errors and errors[0][0] < before:
            print(errors.pop(0)[1], flush=True)

    if cubes:
        solver = GpuSolver(args.table, mode=args.mode)
        for start in range(0, len(cubes), args.batch):
            chunk = cubes[start:start + args.batch]
            for (number, _), sol in zip(chunk, solver.solve([state for _, state in chunk])):
                flush_errors(number)
                print(f"{number} {len(sol.moves)} {sol.text()}".rstrip(), flush=True)
    flush_errors(float("inf"))


if __name__ == "__main__":
    main()
