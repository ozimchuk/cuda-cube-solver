# cuda-cube-solver — optimal 3×3 solving on the GPU

Finds the **shortest possible solution** of a Rubik's cube on an NVIDIA GPU.
Solutions are optimal in the half-turn metric, where a half turn such as `U2`
counts as one move; random positions need 16–19 moves. The solver keeps a
pattern database in GPU memory and runs an IDA* search on the GPU.

**Speed or memory:** by default it uses the fastest table that fits in the
GPU's free memory (7.8 GB). `--memory GB` picks the fastest table within a
smaller budget, down to 1.4 GB; `--table NAME` picks one directly. The first
run builds the table it needs. Robust variants (`--table d4`, `d4-b3`, `a4`,
`a4-b3`) trade 6–11% of random-cube speed for 3–15× faster flip-heavy
positions such as the superflip. See
[Tables](#tables-trading-memory-for-speed).

## How it compares

All solvers below were run on the same desktop, a Ryzen 5 9600X with an RTX
5060 Ti 16 GB. Times are per cube, solved one at a time. cuda-cube-solver
uses its default, fastest table (`d4t`, 7.8 GB).

| | nxopt (CPU) | vcube (CPU) | copy-optsolver (GPU) | Hayakawa (GPU + CPU) | **cuda-cube-solver** |
|---|---|---|---|---|---|
| Random position, median | 0.976 s | 0.763 s | 4.13 s | 0.570 s | **0.020 s** |
| Random position, worst of 100 | 17.4 s | 7.92 s | 58.5 s | 4.65 s | **0.204 s** |
| 20-move position, median of 10 | 98.8 s | 73.1 s | ~590 s | 29.4 s | **1.02 s** |
| Superflip | 702 s | 633 s | 1,148 s | 48.7 s | **16.0 s**¹ |
| 1,000 random positions, whole machine | 6.0 cubes/s | 7.6 cubes/s | ≈0.19 cubes/s² | 1.20 cubes/s | **37.8 cubes/s** |
| Table | 22.7 GB RAM | 21.2 GiB RAM | 0.9 GB | 3.5 GB | 7.8 GB VRAM (1.4–6.2 GB: slower) |

- **Projects compared:**
  - [vcube](https://github.com/Voltara/vcube) and
    [nxopt](https://github.com/rokicki/cube20src) are the fastest open CPU
    optimal solvers.
  - [copy-optsolver](https://github.com/cmsxbc/copy-optsolver) and
    [Hayakawa's solver](https://github.com/h-hayakawa/rubiks_cube) are the
    other public GPU optimal solvers found.
- **Same answers:** every solver returns the same optimal length for every
  position.
- **CPU threads:** vcube and nxopt search a single cube on one thread, so the
  rows above use one thread for them. In bulk they solve 12 cubes in
  parallel, and the gap shrinks to 5.0–6.3×.
- **No parallel bulk mode in the GPU competitors:** they solve one cube after
  another, each using the whole GPU, so their bulk rate is 1,000 divided by
  the sum of single-cube times.
  - Hayakawa's 1.20 cubes/s was measured on all 1,000 positions.
  - ² copy-optsolver's rate is an estimate, not a measurement: its mean
    time per optimal length on the first 100 positions, weighted by the
    length distribution of the 1,000 (about 90 minutes in total).
- ¹ **The superflip** (every edge flipped in place) is the slowest known
  position for the default table, 16× a typical 20-move position: the table
  does not keep the flips of the four middle-layer edges exactly. With
  `--table d4` (same size, keeps them) it takes **3.2 s**, while random
  positions get 11% slower (33.3 cubes/s); 20-move positions are the same.
  Hayakawa's solver has no such outlier (its table keeps every flip) but
  is about 30× slower on typical positions.

Full tables, methods, builds and raw outputs are in
[`docs/benchmarks.md`](docs/benchmarks.md).

## Requirements

| | To solve | To build a table (once, automatic) |
|---|---|---|
| NVIDIA GPU | Turing (RTX 20-series) or newer, ≥ 2 GB free VRAM; 8.2 GB for the fastest table | 2.8–15.3 GB VRAM, depending on the table |
| Disk | 1.4–7.8 GB | the same |
| Host RAM | ~2 GB | ~2 GB |

You also need:

- Linux;
- Python ≥ 3.11;
- an NVIDIA driver supporting CUDA 13 (R580 or newer);
- CuPy 14.

It has been developed and measured only on an RTX 5060 Ti under Ubuntu.

## Quick start

```bash
git clone https://gitlab.com/ozimchuk/cuda-cube-solver.git
cd cuda-cube-solver
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pip install 'cupy-cuda13x[ctk]==14.2.0'   # CUDA headers, if no CUDA toolkit is installed

echo "R U R' U' F2 D L2 B' R2 U F' L D2 R' B U2 F L'" | .venv/bin/cuda-cube-solver solve
# 1 17 F L F2 D2 L2 B R' U' F U B' L' B2 R D' R2 D2
```

The first run builds the table into `tables/` (1–3 minutes); later runs load it.

### Tables: trading memory for speed

The solver picks the fastest table that fits in the GPU's free memory. `--memory GB` sets a
smaller budget, for example to leave room for other programs. `--table NAME` picks a table
directly. `cuda-cube-solver tables` lists them and shows which one `solve` would use:

| Table | VRAM | Cubes/s | Superflip | VRAM to build | Robust alternative (`--table`) |
|---|---|---|---|---|---|
| `304` | 1.42 GB | 6.0 | 44.6 s | 2.8 GB | |
| `s4-b3` | 2.08 GB | 13.1 | 6.2 s | 5.1 GB | (already robust) |
| `d4f-b3` | 3.12 GB | 16.9 | 77.7 s | 7.7 GB | |
| `a4t-b3` | 4.16 GB | 21.8 | 27.1 s | 10.2 GB | `a4-b3`: 20.7 cubes/s, superflip 4.0 s |
| `a4t` | 5.20 GB | 24.1 | 11.5 s | 10.2 GB | `a4`: 22.2 cubes/s, superflip 4.2 s |
| `d4t-b3` | 6.23 GB | 35.2 | 47.5 s | 15.3 GB | `d4-b3`: 31.6 cubes/s, superflip 3.2 s |
| `d4t` | 7.79 GB | 37.5 | 16.0 s | 15.3 GB | `d4`: 33.3 cubes/s, superflip 3.2 s |

The robust alternatives have the same size as the table in their row. They keep
the flips of the four middle-layer edges exactly, so flip-heavy positions such
as the superflip are 3–15× faster, at 6–11% less speed on random cubes; known
distance-20 positions take the same time with both. `--memory` picks only
from the first column; a robust table is used only when named with `--table`.

Cubes/s is for 1,000 random cubes on an RTX 5060 Ti. All tables give optimal solutions;
smaller tables search more positions per cube. The tables are coset tables for subgroups
that contain the U and D turns, found by a survey of every such subgroup of a family:
[research/subgroups](research/subgroups/README.md). The `-b3` tables store the same
bounds in base 3, about 20% smaller and about 5% slower.

Tables from earlier versions (`prune-ud-208-*`) are no longer used and can be
deleted. Building needs more memory than solving, because it holds the table twice. A card that can
solve with a table but not build it gets a smaller one. `tables` says so, and you can copy
the bigger table's file from a machine with a 16 GB card into `tables/`. For example, an 8 GB
card builds `s4-b3`, but solves at 35 cubes/s with a copied `d4t-b3`
(or at 31.6 cubes/s and a 3.2 s superflip with a copied `d4-b3`). No prebuilt download is
published yet.

`build-table` builds a table without solving anything (`--table NAME`, `--out DIR`). Each
build checks itself and stops with an error if the check fails. Only the solving file is
written, plus a `.json` with the per-depth counts.

### Solving

```bash
echo "DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD" | .venv/bin/cuda-cube-solver solve
# 1 18 R2 D2 R' F U2 R' B' D' R U2 F R' D' R' U2 L2 F B2

.venv/bin/cuda-cube-solver solve < cubes.txt > solutions.txt
```

- **Input:** one cube per line, either as a scramble (the moves that produce
  it from solved) or as a 54-sticker facelet string.
  - Facelet strings use the U R F D L B face order of the `kociemba` package
    and Cube Explorer.
  - Any six symbols work, because the centres define the faces.
- **Output:** one line per cube, in input order: `<line> <length> <moves>`.
  Applying `<moves>` to the cube solves it.
  - Invalid lines print `<line> error <reason>`, and the rest still run.
  - Several optimal solutions usually exist, and repeated runs may return a
    different one.
- **Startup:** each run loads the table first (1–5 s), so pipe many cubes
  into one run rather than starting it once per cube.
- **Options:** run `solve --help`. `--tables DIR` points to a table folder
  other than `tables/`; `--no-build` fails instead of building a missing table.

From Python:

```python
from pathlib import Path
from cuda_cube_solver.facelets import parse_facelets
from cuda_cube_solver.solver import open_solver

solver = open_solver(Path("tables"))   # or memory=6, or table="d4t-b3"; builds a missing table
solution = solver.solve([parse_facelets("DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD")])[0]
print(len(solution.moves), solution.text())
```

## How it works

- **Why a GPU:** an optimal solver is limited by random lookups into a large
  table. This GPU does 3.85 G random lookups/s, against 0.71–0.75 G/s on 12
  CPU threads of this machine (`bench/gpu_gather.py`, `bench/cpu_gather.c`).
- **Table:** a coset table, like nxopt's and vcube's, for a subgroup that
  contains the U and D turns. The default `d4t` keeps all edge flips and the
  E-slice edges' order up to a twisted D4 relabelling; it has 30.2 billion
  entries, the size of nxopt's `EO2,EP2` and vcube's `208`, reduced by the
  16 symmetries that keep the U–D axis. It came out of a
  [survey of such subgroups](research/subgroups/README.md) and searches 26%
  fewer positions than `208`.
  - It is stored as blocks: each 32-byte memory sector holds an 8-bit minimum
    and 124 two-bit offsets.
  - For `d4t` they are exact for 98.1% of lookups from random positions.
- **Search:** IDA* with six table lookups per position: the position turned
  so each of its three axes becomes the U–D axis, plus the inverse of each.
  - A coset rule from nxopt's documentation uses these lookups to exclude
    whole axes of first moves.
  - Each cube is split into 43,254 work items, so a single cube already fills
    the GPU.
  - Every GPU lane does exactly one lookup per step.
- **Limit:** profiling shows one DRAM sector read per lookup, at the card's
  random-read ceiling. Speed therefore scales with memory, not computation.

Design history and measured dead ends are in
[`docs/development-notes.md`](docs/development-notes.md). Postponed options,
such as 8 GB cards, a CPU table generator and larger cards, are in
[`docs/future-work.md`](docs/future-work.md).

## Portability

- **Supported GPUs:** Turing (sm_75) or newer, the oldest generation CUDA 13
  supports.
  - Kernels are compiled at run time for the installed GPU (NVRTC via CuPy),
    and the launch grid scales with its SM count.
  - Older cards might work with `cupy-cuda12x`; this is untested.
- **Measured hardware:** only an RTX 5060 Ti. Speed should scale roughly with
  a card's random-memory-read rate.
- **Larger cards:** cards with 24–32 GB could hold nxopt's next table size
  (22.7 GB); this is not implemented.

## Development

```bash
.venv/bin/pytest                                                        # unit tests (GPU, no tables needed)
.venv/bin/python bench/run_gpu.py data/random1000.txt --count 1000 --batch 100
```

## License

MIT; see [`LICENSE`](LICENSE).
