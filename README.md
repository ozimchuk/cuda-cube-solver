# cuda-cube-solver — optimal 3×3 solving on the GPU

Finds the **shortest possible solution** of a Rubik's cube (optimal in
the half-turn metric, where a half turn such as `U2` counts as one move;
random positions need 16–19 moves) on an NVIDIA GPU. It keeps a 7.3 GiB
pattern database in GPU memory and runs IDA* search on the GPU.

On a desktop with an RTX 5060 Ti 16 GB, against the two fastest open CPU
optimal solvers on the same machine:

- **one cube at a time: ~30× faster than vcube, ~40× faster than nxopt**
  (median 23 ms per random position versus 0.76 s and 0.98 s);
- **1,000 cubes in bulk: 3.6× faster than vcube, 4.6× faster than nxopt**
  (27.7 versus 7.6 and 6.0 cubes/s),

with a table a third the size of theirs (7.8 GB versus 21–23 GB).

## Requirements

| | To solve | To build the table (once) |
|---|---|---|
| NVIDIA GPU | Turing (RTX 20-series) or newer, ≥ 9 GB VRAM | ≥ 16 GB VRAM |
| Disk | 7.3 GiB (`prune-ud-208-blk.bin`) | 14.3 GiB for both table files |
| Host RAM | ~2 GB | ~2 GB |

- Linux, Python ≥ 3.11, an NVIDIA driver supporting CUDA 13 (R580 or
  newer), CuPy 14. Developed and measured only on an RTX 5060 Ti (16 GB)
  under Ubuntu.
- Building the table takes about 2 minutes on that card. Cards with less
  than 16 GB can solve with a table built elsewhere (copy
  `prune-ud-208-blk.bin`); see [8 GB cards](docs/future-work.md#1-supporting-8-gb-cards)
  for options under consideration.

## Quick start

```bash
git clone https://gitlab.com/ozimchuk/cuda-cube-solver.git
cd cuda-cube-solver
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
# CuPy also needs CUDA headers if no CUDA toolkit is installed system-wide:
.venv/bin/pip install 'cupy-cuda13x[ctk]==14.2.0'

.venv/bin/cuda-cube-solver build-table      # writes tables/ (~2 min, 16 GB VRAM)
```

Solve a scramble (the cube it produces from solved), a 54-sticker
facelet string, or a whole file — one cube per line on stdin:

```bash
echo "R U R' U' F2 D L2 B' R2 U F' L D2 R' B U2 F L'" | .venv/bin/cuda-cube-solver solve
# 1 17 F L F2 D2 L2 B R' U' F U B' L' B2 R D' R2 D2

echo "DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD" | .venv/bin/cuda-cube-solver solve
# 1 18 R2 D2 R' F U2 R' B' D' R U2 F R' D' R' U2 L2 F B2

.venv/bin/cuda-cube-solver solve < cubes.txt > solutions.txt
```

- **Output:** `<line number> <length> <moves>` per cube, in input order;
  applying `<moves>` to the cube solves it. Invalid lines print
  `<line number> error <reason>` (e.g. a twisted corner) and the rest
  still run. Several optimal solutions usually exist, and repeated runs
  may return a different one of the same length.
- **Facelet strings** use the common U R F D L B face order (as in the
  `kociemba` package and Cube Explorer); any six symbols work, since the
  centres define the faces, so colour letters such as `W G R …` are fine.
- **Startup:** each invocation loads the table (1–5 s) before solving,
  so pipe many cubes into one run rather than calling it per cube.
- `cuda-cube-solver solve --help` lists the options; the table location
  defaults to `tables/` and can be changed with `--table`.

From Python:

```python
from pathlib import Path
from cuda_cube_solver.facelets import parse_facelets
from cuda_cube_solver.solver import GpuSolver

solver = GpuSolver(Path("tables/prune-ud-208.bin"))   # loads the -blk.bin table
solution = solver.solve([parse_facelets("DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD")])[0]
print(len(solution.moves), solution.text())
```

## Comparison with vcube and nxopt

Machine: Ryzen 5 9600X (6 cores, 12 threads, 30 GB RAM), RTX 5060 Ti
16 GB. Other programs paused; transparent huge pages `always` for the
CPU solvers; solve time only (table loading excluded for every solver).
All three solvers return the same optimal length for every position.
Baseline builds and commands are listed under [Baselines](#baselines).

### Bulk: 1,000 random positions

`data/random1000.txt` (40 random moves each). Each solver uses the whole
machine: vcube and nxopt solve 12 cubes at a time, one per thread;
cuda-cube-solver searches batches of 100 cubes on the GPU.

| Solver | Table | Time | Cubes/s | cuda-cube-solver speedup |
|---|---|---|---|---|
| nxopt `EO2,EP3` (`nxopt23`), 12 threads | 22.7 GB RAM | 166.3 s | 6.01 | 4.6× |
| vcube `308`, 12 threads | 21.2 GiB RAM | 131.1 s | 7.63 | 3.6× |
| **cuda-cube-solver** (default) | 7.8 GB VRAM | **36.2 s** | **27.7** | — |

Optimal lengths: 3× 15, 27× 16, 278× 17, 659× 18, 33× 19 moves.
Raw outputs: `data/cuda-cube-solver_1000.txt`, `data/vcube308_quiet_1000.txt`,
`data/nxopt23_quiet_1000.txt`.

### One cube at a time: first 100 of those positions

The first 100 lines of `data/random1000.txt` — a subset because the
CPU solvers need 1–1.5 s per position on one thread. Their optimal
lengths (2× 16, 32× 17, 65× 18, 1× 19) are close to the full set's
distribution, with slightly fewer 19-movers, so absolute times are a
little optimistic for all three solvers alike.

Time to solve a single cube with nothing else queued. vcube and nxopt
search one cube on one thread (they parallelise only across cubes), so
they run with one worker here; cuda-cube-solver spreads one cube over the whole
GPU (43,254 work items per cube).

| Solver | Median | 90th percentile | Worst | Mean |
|---|---|---|---|---|
| nxopt `EO2,EP3`, 1 thread | 0.976 s | 3.09 s | 17.4 s | 1.53 s |
| vcube `308`, 1 thread | 0.763 s | 2.32 s | 7.92 s | 1.08 s |
| **cuda-cube-solver** | **0.023 s** | **0.072 s** | **0.263 s** | **0.034 s** |

By optimal solution length (median time per cube):

| Optimal length | Cubes | nxopt | vcube | cuda-cube-solver |
|---|---|---|---|---|
| 16 | 2 | 0.03 s | 0.02 s | 0.002 s |
| 17 | 32 | 0.17 s | 0.13 s | 0.009 s |
| 18 | 65 | 1.35 s | 0.95 s | 0.036 s |
| 19 | 1 | 10.1 s | 7.92 s | 0.263 s |

Raw outputs: `data/single_cube/`.

### Reading these numbers

- The single-cube gap (~30–40×) is larger than the bulk gap (~4×)
  because the CPU solvers use one of 12 threads for a single cube.
- These are single-machine numbers. On large servers vcube reports 55
  cubes/s (32 cores, 170 GB table) and nxopt 33 cubes/s (16 cores,
  256 GB); cuda-cube-solver has not been run on comparable GPU hardware.
- [GPU-Rubiks](https://github.com/Nico-Adamo/GPU-Rubiks), the only other
  public GPU solver found, solves in the quarter-turn metric with a small
  per-cubie heuristic and is meant for scrambles up to ~15 moves; on this
  GPU its solver did not finish cubes needing 8 or more quarter turns
  (see `docs/future-work.md`). cuda-cube-solver solves its scramble distribution
  (`data/gpurubiks/`) in 0.3 ms per cube at 13 quarter turns, but in the
  half-turn metric, so this is not a like-for-like comparison.

## How it works

**Why a GPU.** An optimal solver is limited by random lookups into a
large table. Measured on this machine (`bench/cpu_gather.c`,
`bench/gpu_gather.py`): 0.71–0.75 G random lookups/s on 12 CPU threads
versus 3.85 G/s on the GPU.

**Table.** An independent implementation of the coset coordinate used by
nxopt (`EO2,EP2`) and vcube (`208`): corner twist × which positions hold
U-layer corners, reduced by the 16 symmetries that keep the U–D axis
(9,930 classes); E-slice edge positions × their order × flips of the
other eight edges. 30,199,910,400 entries. One GPU breadth-first search
(≈ 2 minutes) writes `prune-ud-208-blk.bin` (7.3 GiB, used for solving:
per 32-byte memory sector an 8-bit minimum and 124 two-bit offsets,
exact for 99.6% of entries) and `prune-ud-208.bin` (7.0 GiB, distances
mod 3, used during generation and by experimental modes). Per-depth entry
counts match vcube's generator exactly.

**Search.** IDA* with six lookups per position: the position turned so
each of its three axes becomes the U–D axis, and the inverse of each. A
coset rule from nxopt's documentation, re-derived for this table, uses
those lookups to exclude whole axes of first moves and to add one to the
bound when all three axes agree. Each cube is split into 43,254 work
items (one per canonical 4-move start), so a single cube already fills
the GPU; many cubes can be searched together. In the kernel every lane
does exactly one table lookup per step, and moves and symmetry
conjugation are byte-permute (`prmt`) instructions on a packed state.

**Limit.** Profiling a representative search round with Nsight Compute
shows one DRAM sector per table lookup at this card's random-read
ceiling (122 GB/s): the solver is bound by memory reads, not computation.
Faster memory or a card large enough for the next table size are the
paths to more speed. The history of the design and the experiments that
did not pay off are in [`docs/development-notes.md`](docs/development-notes.md).

## Portability

- NVIDIA GPUs from Turing (sm_75) on, the oldest generation CUDA 13
  supports; kernels are compiled at run time for the installed GPU
  (NVRTC via CuPy), and the launch grid scales with its SM count. The
  kernels only need Pascal-era instructions, so older cards should work
  with a CUDA 12 build of CuPy (`cupy-cuda12x`) — untested.
- Only measured on an RTX 5060 Ti. Throughput should scale roughly with
  a card's random-memory-read rate. Cards with 24–32 GB could hold
  nxopt's next table size (`EO2,EP3`, 22.7 GB), which needs ~2.3× fewer
  nodes on CPUs; not implemented or tested.

## Development

```bash
.venv/bin/pytest                                     # unit tests (GPU, no tables needed)
.venv/bin/python bench/run_gpu.py data/random1000.txt --count 1000 --batch 100
```

`docs/development-notes.md` records the design history and measured dead
ends; `docs/future-work.md` the postponed options (8 GB cards, a CPU
table generator, larger cards); `research/learned_table/` a negative
result on neural-network table compression.

## Baselines

Built outside this repository and used only as benchmarks (both GPL-3.0;
no code from them is included here):

- vcube, <https://github.com/Voltara/vcube>, commit `9f5bc2cc`, clang 22
  `-O3 -mlzcnt -mbmi -mbmi2 -mavx2`; `./vc-optimal -c 308 --ordered`.
- nxopt from cube20src, <https://github.com/rokicki/cube20src>, commit
  `f430921a`, `g++ -DPS=16 -O3 -march=native -DHALF -DTHREADS -DEO2 -DEP3`;
  `nxopt23 -t 12 -` with moves converted to `U1R3…` (`data/random1000.nxopt.txt`).

Single-cube runs: `vc-optimal -c 308 --ordered --workers=1` and
`nxopt23 -t 1 -` on the first 100 positions; cuda-cube-solver with
`bench/run_gpu.py --count 100 --batch 1`.

## License

No license has been chosen yet, so all rights are reserved for now.
