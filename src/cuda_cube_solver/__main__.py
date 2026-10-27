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
The first run builds the table (1-3 min); later runs load it from --tables.
"""

_NO_TABLE = "cuda-cube-solver: no table fits in that GPU memory (the smallest, 304, needs 1.8 GB)"

MEMORY_HELP = ("GPU memory for the table: 'auto' (default: the free VRAM) or a size in GB, e.g. 6; "
               "picks the fastest preset that fits (see `cuda-cube-solver tables`)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cuda-cube-solver",
                                 description="Optimal Rubik's cube solver (half-turn metric) running on an NVIDIA GPU.")
    sub = ap.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-table", help="build a table now instead of on first use",
                           description="Build a pruning table on the GPU. By default the preset that `solve` "
                                       "would pick; only its solving file is written.")
    build.add_argument("--out", type=Path, default=Path("tables"), help="output directory (default: tables)")
    build.add_argument("--memory", default="auto", help=MEMORY_HELP)
    build.add_argument("--table", help="a preset name (see `tables`), or 208, 112, 304 for the experimental "
                                       "--mode options (writes the mod-3 and block tables)")
    tables = sub.add_parser("tables", help="list the table presets and the one `solve` would use",
                            description="List the table presets: the memory ladder from 1.4 GB to 7.8 GB.")
    tables.add_argument("--tables", type=Path, default=Path("tables"), help="table directory (default: tables)")
    tables.add_argument("--memory", default="auto", help=MEMORY_HELP)
    solve = sub.add_parser("solve", help="solve cubes read from stdin (scrambles or facelet strings)",
                           description="Solve cubes optimally.", epilog=SOLVE_EPILOG,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
    solve.add_argument("--tables", type=Path, default=Path("tables"),
                       help="table directory; missing tables are built there (default: tables)")
    solve.add_argument("--memory", default="auto", help=MEMORY_HELP)
    solve.add_argument("--table", help="use this preset instead of choosing by --memory")
    solve.add_argument("--no-build", action="store_true", help="fail instead of building a missing table")
    solve.add_argument("--format", choices=["auto", "moves", "facelets"], default="auto",
                       help="input format (default: auto-detect per line)")
    solve.add_argument("--batch", type=int, default=100, help="cubes searched together on the GPU (default: 100)")
    solve.add_argument("--mode", default="step", choices=["step", "forward", "multi", "bidir"],
                       help="search kernel (default: step; the others are slower experiments that need the "
                            "208 tables from `build-table --table 208`)")
    args = ap.parse_args(argv)

    if getattr(args, "table", None) and args.command == "solve":
        from cuda_cube_solver import kinds

        if args.table not in kinds.PRESET_BY_NAME:
            sys.exit(f"cuda-cube-solver: unknown table {args.table!r}; choose from {', '.join(kinds.PRESET_BY_NAME)}")

    if args.command == "tables":
        from cuda_cube_solver import kinds

        best, better = _pick(args.tables, args.memory, None)
        print("\n".join(kinds.describe(args.tables)))
        print(f"\nsolve would use: {best.name if best else 'nothing (not enough GPU memory)'}"
              + (f"; {better.name} fits too but can't be built here ({better.build_bytes / 1e9:.1f} GB VRAM)"
                 if better else ""))
        return

    if args.command == "build-table":
        from cuda_cube_solver import coords, kinds, table

        if args.table and args.table not in kinds.PRESET_BY_NAME:
            print(table.build(args.out, coords.EdgeKind.parse(args.table)))
            return
        preset = _pick(args.out, args.memory, args.table, to_build=True)[0]
        if preset is None:
            sys.exit(_NO_TABLE)
        _log(f"building table {preset.name} in {args.out}/")
        print(table.build_preset(args.out, preset, log=lambda m: _log(f"  {m}")))
        return

    from cuda_cube_solver.cubie import MOVES_HTM, CubieState, apply_moves
    from cuda_cube_solver.facelets import FaceletError, is_facelet_string, parse_facelets
    from cuda_cube_solver.solver import GpuSolver, NoTableFits, TableNotFound, open_solver, parse_moves

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
        try:
            if args.mode != "step":
                solver = GpuSolver(args.tables, mode=args.mode)
            else:
                solver = open_solver(args.tables, args.memory, args.table, build=not args.no_build, log=_log)
        except (TableNotFound, NoTableFits, ValueError) as exc:
            sys.exit(f"cuda-cube-solver: {exc}")
        for start in range(0, len(cubes), args.batch):
            chunk = cubes[start:start + args.batch]
            for (number, _), sol in zip(chunk, solver.solve([state for _, state in chunk])):
                flush_errors(number)
                print(f"{number} {len(sol.moves)} {sol.text()}".rstrip(), flush=True)
    flush_errors(float("inf"))


def _pick(tables: Path, memory: str, name: str | None, to_build: bool = False):
    from cuda_cube_solver import kinds

    try:
        return kinds.pick(tables, memory, name, to_build)
    except ValueError as exc:
        sys.exit(f"cuda-cube-solver: {exc}")


def _log(message: str) -> None:
    print(f"cuda-cube-solver: {message}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
