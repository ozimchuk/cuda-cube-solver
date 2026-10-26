# cuda-cube-solver — optimal 3×3 solving on the GPU

Finds the **shortest possible solution** of a Rubik's cube on an NVIDIA GPU.
Solutions are optimal in the half-turn metric, where a half turn such as `U2`
counts as one move; random positions need 16–19 moves. The solver keeps a
7.3 GiB pattern database in GPU memory and runs an IDA* search on the GPU.

## How it compares

All solvers below were run on the same desktop, a Ryzen 5 9600X with an RTX
5060 Ti 16 GB. Times are per cube, solved one at a time.

| | nxopt (CPU) | vcube (CPU) | copy-optsolver (GPU) | Hayakawa (GPU + CPU) | **cuda-cube-solver** |
|---|---|---|---|---|---|
| Random position, median | 0.976 s | 0.763 s | 4.13 s | 0.570 s | **0.023 s** |
| Random position, worst of 100 | 17.4 s | 7.92 s | 58.5 s | 4.65 s | **0.263 s** |
| 20-move position, median of 10 | 99.5 s | 74.8 s | ~590 s | 29.4 s | **1.53 s** |
| Superflip | 720 s | 808 s | 1,148 s | **48.7 s**¹ | 51.5 s |
| 1,000 random positions, whole machine | 6.0 cubes/s | 7.6 cubes/s | ≈0.19 cubes/s² | 1.20 cubes/s | **27.7 cubes/s** |
| Table | 22.7 GB RAM | 21.2 GiB RAM | 0.9 GB | 3.5 GB | 7.8 GB VRAM |

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
  parallel, and the gap shrinks to 3.6–4.6×.
- **No parallel bulk mode in the GPU competitors:** they solve one cube after
  another, each using the whole GPU, so their bulk rate is 1,000 divided by
  the sum of single-cube times.
  - Hayakawa's 1.20 cubes/s was measured on all 1,000 positions.
  - ² copy-optsolver's rate is an estimate, not a measurement: its mean
    time per optimal length on the first 100 positions, weighted by the
    length distribution of the 1,000 (about 90 minutes in total).
- ¹ **The superflip** is the one case where another solver is faster than
  the default, and only with help from the CPU:
  - Hayakawa's 48.7 s uses the GPU plus 11 CPU threads. On the GPU alone it
    takes 53.1 s, slower than cuda-cube-solver's 51.5 s, which uses the GPU
    only.
  - With the optional `112` table (`--step-extra 1`, +2.6 GB VRAM),
    cuda-cube-solver solves it in 10.3 s, but that makes everything else
    slower.

Full tables, methods, builds and raw outputs are in
[`docs/benchmarks.md`](docs/benchmarks.md).

## Requirements

| | To solve | To build the table (once) |
|---|---|---|
| NVIDIA GPU | Turing (RTX 20-series) or newer, ≥ 9 GB VRAM | ≥ 16 GB VRAM |
| Disk | 7.3 GiB | 14.3 GiB |
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

.venv/bin/cuda-cube-solver build-table              # once: ~2 min, writes tables/
echo "R U R' U' F2 D L2 B' R2 U F' L D2 R' B U2 F L'" | .venv/bin/cuda-cube-solver solve
# 1 17 F L F2 D2 L2 B R' U' F U B' L' B2 R D' R2 D2
```

### Building the table

`build-table` runs a breadth-first search on the GPU, in about 2 minutes on
an RTX 5060 Ti. Use `--out DIR` to write somewhere other than `tables/`.

It writes three files:

- `prune-ud-208-blk.bin` (7.3 GiB): the only file solving reads.
- `prune-ud-208.bin` (7.0 GiB): needed only to rebuild table variants.
- `prune-ud-208.json`: entry counts and timing for each depth.

The build checks itself before writing and stops with an error if the
check fails. Its per-depth counts match vcube's generator; the
[expected values](docs/development-notes.md#expected-table-contents) are
listed in the development notes.

**Cards with 9–15 GB** can solve but not build. Copy
`prune-ud-208-blk.bin` from a machine with a 16 GB card; no prebuilt
download is published yet. Options for 8 GB cards are in
[future-work.md](docs/future-work.md#1-supporting-8-gb-cards).

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
  other than `tables/`.

From Python:

```python
from pathlib import Path
from cuda_cube_solver.facelets import parse_facelets
from cuda_cube_solver.solver import GpuSolver

solver = GpuSolver(Path("tables"))
solution = solver.solve([parse_facelets("DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD")])[0]
print(len(solution.moves), solution.text())
```

## How it works

- **Why a GPU:** an optimal solver is limited by random lookups into a large
  table. This GPU does 3.85 G random lookups/s, against 0.71–0.75 G/s on 12
  CPU threads of this machine (`bench/gpu_gather.py`, `bench/cpu_gather.c`).
- **Table:** an independent implementation of the coset coordinate used by
  nxopt (`EO2,EP2`) and vcube (`208`). It has 30.2 billion entries, reduced
  by the 16 symmetries that keep the U–D axis.
  - It is stored as blocks: each 32-byte memory sector holds an 8-bit minimum
    and 124 two-bit offsets.
  - These are exact for 99.6% of entries.
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
