# cuda-cube-solver — optimal 3×3 solving on the GPU

An optimal half-turn-metric Rubik's cube solver that keeps its pattern
database in GPU memory and runs IDA* on the GPU. On a desktop with an
RTX 5060 Ti 16 GB, against the two fastest open CPU optimal solvers on
the same machine:

- **one cube at a time: ~30× faster than vcube, ~40× faster than nxopt**
  (median 23 ms per random position versus 0.76 s and 0.98 s);
- **1,000 cubes in bulk: 3.6× faster than vcube, 4.6× faster than nxopt**
  (27.7 versus 7.6 and 6.0 cubes/s),

with a table a third the size of theirs (7.8 GB versus 21–23 GB).

The Python package and command are named `gpucube`.

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

**Development history (same 1,000 positions, cubes/s):** three axis
frames 7.8 → + inverse frames and block table 9.3 → branch-free edge
conjugation 9.6 → coset axis rule 18.9 → inverse-frame inheritance 19.0
→ per-node move masks (lookup-lane efficiency 22% → 45%) 23.7 →
step-machine kernel with per-node frame order, block table only (lane
efficiency 99%, half the VRAM) 28.0.

**Step machine.** Each warp iteration, every lane performs exactly one
table lookup for its current child; choosing moves, push, pop and
fetching work run between steps. The child is evaluated on a temporary
copy of the frame, so a pruned child needs no undo. Frames are taken in
a per-node order (frames at the budget edge first), encoded as three
6-bit masks. The search is then bound by DRAM random-access throughput
(~2.5 lookups per child at ~1.5 G children/s ≈ 3.7 G lookups/s against
a measured ceiling of 3.85 G/s).

Extra tables in the step machine (`--step-extra 1|2`) cut nodes by 30%
but nearly double lookup steps per child (2.5 → 4.8): with the axis rule
removing most easily pruned moves up front, ~46% of the remaining
children survive all six main lookups and then pay five more per extra
table. Net loss (18.4 / 13.8 cubes/s).

**Where the time goes (forward kernel, before the step machine).** With every lookup forced into L2 the kernel is
only ~13% faster than with real DRAM lookups, so it is balanced between
memory and instruction issue. 99.6% of nodes have three or more moves of
budget left: the search lives in the top ~8 levels and is bounded by
table information, which 16 GB of VRAM caps at this table size (the next
size, nxopt `EO2,EP3`, needs ~23 GB).

## Portability

- Any NVIDIA GPU of the Pascal generation (sm_61) or newer; kernels are
  compiled at run time for the installed device (NVRTC via CuPy). The
  launch grid scales with the SM count.
- Solving needs ≈ 9 GB of VRAM (7.8 GB block table). Generating the
  tables needs 16 GB (the BFS holds both encodings); on smaller cards,
  copy prebuilt tables.
- Only measured on an RTX 5060 Ti. The search is balanced between random
  memory access and instruction issue, so throughput should scale with
  both; cards with 24–32 GB could hold nxopt's next table size
  (`EO2,EP3`, 22.7 GB), which needs ~2.3× fewer nodes on CPUs.

## Future work

Postponed options — 8 GB-card support (base-3 table encoding, prebuilt
table download, a single-source CPU table generator), further speed on
16 GB, larger cards, publishing — are written up with measurements in
[`docs/future-work.md`](docs/future-work.md).

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

Single-cube runs: `vc-optimal -c 308 --ordered --workers=1` and
`nxopt23 -t 1 -` on the first 100 positions; cuda-cube-solver with
`bench/run_gpu.py --count 100 --batch 1`.
