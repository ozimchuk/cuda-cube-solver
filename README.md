# gpucube — optimal 3×3 solving on the GPU

An optimal half-turn-metric Rubik's cube solver that keeps its pattern
database in GPU memory and runs IDA* on the GPU. On a desktop with an
RTX 5060 Ti 16 GB it solves random positions 2.5× faster than vcube and
3.1× faster than nxopt (the two fastest open CPU optimal solvers) on the
same machine, with a smaller table.

## Results

1,000 random positions (`data/random1000.txt`), Ryzen 5 9600X (6 cores,
12 threads, 30 GB RAM), RTX 5060 Ti 16 GB. Quiet machine; transparent
huge pages `always` for the CPU solvers; solve time only (table loading
excluded for every solver).

| Solver | Table | Time | Cubes/s |
|---|---|---|---|
| nxopt `EO2,EP3` (`nxopt23`), g++ `-O3 -march=native`, 12 threads | 22.7 GB RAM | 166.3 s | 6.01 |
| vcube `308`, clang 22 `-O3`, 12 threads | 21.2 GiB RAM | 131.1 s | 7.63 |
| **gpucube** (forward mode) | 15.3 GB VRAM | 52.9 s | **18.9** |
| gpucube (bidirectional mode) | 7.8 GB VRAM | 66.8 s | 15.0 |

All solvers return identical optimal lengths on all 1,000 positions
(3× 15, 27× 16, 278× 17, 659× 18, 33× 19). Raw outputs are in `data/`.

These are single-machine numbers. On large servers vcube reports 55
cubes/s (32 cores, 170 GB table) and nxopt 33 cubes/s (16 cores, 256 GB);
this project has not been run on comparable GPU hardware.

## How it works

**Why a GPU.** Optimal IDA* is bound by random lookups into a large
table. Measured on this host (`bench/cpu_gather.c`, `bench/gpu_gather.py`):
0.71–0.75 G random byte lookups/s on 12 CPU threads versus 3.85 G/s on
the GPU.

**Table.** An independent implementation of the coset coordinate used by
nxopt (`EO2,EP2`) and vcube (`208`): corner twist × which positions hold
U-layer corners, reduced by the 16 UD-preserving symmetries (9,930
classes); E-slice positions × order of the E edges × flips of the eight
non-E edges. 30,199,910,400 entries. One GPU BFS (≈2 minutes) writes two
encodings:

- `prune-ud-208.bin` (7.03 GiB): distance mod 3. Along a path of right
  multiplications the coset distance changes by at most one, so exact
  values are tracked from the root.
- `prune-ud-208-blk.bin` (7.26 GiB): per 32-byte sector an 8-bit minimum
  and 124 two-bit offsets. Absolute coset lower bounds in one memory
  transaction; exact for 99.6% of positions.

Per-depth entry counts match vcube's generator exactly at every depth.

**Search.**

- Six lookups per node: the position conjugated onto the UD, RL and FB
  axes, and the inverse of each (the inverse evolves by left
  multiplication, so it uses the block table).
- Coset axis rule (from nxopt's documentation, re-derived here): the
  table's subgroup contains every U and D turn, so a shortest coset
  representative never ends in U or D. An inverse frame whose bound
  equals the remaining budget excludes first moves on its axis; three
  tight forward (or inverse) frames prune the node. The bound must be the
  coset distance itself — mixing in the true-distance bound `parent − 1`
  breaks the rule.
- Packed state: corners in two 32-bit words, edges in three. Moves,
  inverse (left) moves and symmetry conjugation are `prmt` byte permutes;
  the edge coordinate is read with bit tricks and small shared-memory
  tables, no loops.
- Work items are (cube, canonical 4-move prefix); a persistent kernel
  pulls items with an atomic counter; many cubes are searched at once.
- Bidirectional mode (`--mode bidir`) also extends the solution from
  whichever end has fewer allowed moves and inherits one lookup from the
  parent, as nxopt does. It needs only the block table (half the VRAM)
  but is slower here: nodes fall only 5%, while every frame then needs a
  block lookup and back moves need left multiplication.

**Things that did not pay off (measured on the 1,000 positions):**

| Variant | Nodes/cube | Cubes/s |
|---|---|---|
| default (forward, inverse-frame inheritance) | 55.7 M | 19.0 |
| forward frames from the block table instead of mod 3 | 56.6 M | 18.9 |
| `--mode multi`: + `112` table (2.6 GB) | 42.1 M | 15.9 |
| `--mode multi`: + `112` + `304` tables (4.0 GB) | 39.2 M | 12.1 |
| prefix length 3 / 5 (default 4) | — | 17.3 / 16.1 |

The extra tables (`gpucube build-table --kind 112|304`, both
cross-checked against vcube's generator) raise the six-lookup average by
0.11 and cut nodes by 31%, but only ~8% of children need them and a warp
executes the extra index computations whenever any lane does, so the
node rate falls faster than the node count.

A larger table kept in host RAM and read by the GPU over PCIe is not
viable on this machine: random reads from GPU kernels into pinned host
memory reach ~60 M/s (`bench/gpu_host_gather.py`), versus 3,850 M/s
from VRAM.

**Where the time goes.** With every lookup forced into L2 the kernel is
only ~13% faster than with real DRAM lookups, so it is balanced between
memory and instruction issue. 99.6% of nodes have three or more moves of
budget left: the search lives in the top ~8 levels and is bounded by
table information, which 16 GB of VRAM caps at this table size (the next
size, nxopt `EO2,EP3`, needs ~23 GB).

## Usage

Requires an NVIDIA GPU with ≥16 GB, CUDA 13 runtime, CuPy 14.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[test]'
# CuPy also needs CUDA headers if no system toolkit is installed:
.venv/bin/pip install 'cupy-cuda13x[ctk]==14.2.0'

.venv/bin/gpucube build-table               # writes tables/ (15 GB, ~2 min)
.venv/bin/gpucube build-table --kind 112      # optional extra tables for --mode multi
.venv/bin/gpucube build-table --kind 304
echo "R U R' U' F2 D L2 B' R2 U F' L D2 R' B U2 F L'" | .venv/bin/gpucube solve
.venv/bin/python bench/run_gpu.py data/random1000.txt --count 1000 --batch 100
.venv/bin/pytest
```

## Baselines

Built outside this repository and used only as benchmarks (both GPL-3.0;
no code from them is included here):

- vcube, <https://github.com/Voltara/vcube>, commit `9f5bc2cc`, clang 22
  `-O3 -mlzcnt -mbmi -mbmi2 -mavx2`; `./vc-optimal -c 308 --ordered`.
- nxopt from cube20src, <https://github.com/rokicki/cube20src>, commit
  `f430921a`, `g++ -DPS=16 -O3 -march=native -DHALF -DTHREADS -DEO2 -DEP3`;
  `nxopt23 -t 12 -` with moves converted to `U1R3…` (`data/random1000.nxopt.txt`).
