# Benchmarks

All measurements are from one machine: a Ryzen 5 9600X (6 cores, 12 threads,
30 GB RAM) and an RTX 5060 Ti 16 GB. Other programs were paused. The CPU
solvers ran with transparent huge pages set to `always`. Times are solve
time only; table loading is excluded for every solver. Every solver returns
the same optimal length for every position it solved. Builds and commands
are listed [at the end](#builds-and-commands).

## Bulk: 1,000 random positions

The input is `data/random1000.txt` (40 random moves each). Each solver uses
the whole machine: vcube and nxopt solve 12 cubes at a time, one per
thread, and cuda-cube-solver searches batches of 100 cubes on the GPU.

| Solver | Table | Time | Cubes/s | cuda-cube-solver speedup |
|---|---|---|---|---|
| nxopt `EO2,EP3` (`nxopt23`), 12 threads | 22.7 GB RAM | 166.3 s | 6.01 | 4.6× |
| vcube `308`, 12 threads | 21.2 GiB RAM | 131.1 s | 7.63 | 3.6× |
| **cuda-cube-solver** | 7.8 GB VRAM | **36.2 s** | **27.7** | — |

Optimal lengths: 3× 15, 27× 16, 278× 17, 659× 18, 33× 19 moves.

Raw outputs: `data/cuda-cube-solver_1000.txt`, `data/vcube308_quiet_1000.txt`,
`data/nxopt23_quiet_1000.txt`.

## One cube at a time: the first 100 of those positions

This is the time to solve a single cube with nothing else queued:

- vcube and nxopt search one cube on one thread, because they parallelise
  only across cubes.
- cuda-cube-solver spreads one cube over the whole GPU (43,254 work items).

The set is the first 100 lines of `data/random1000.txt`. A subset is used
because the CPU solvers need 1–1.5 s per position on one thread. Its
lengths (2× 16, 32× 17, 65× 18, 1× 19) have slightly fewer 19-movers than
the full set, so the absolute times are a little optimistic for every
solver alike.

| Solver | Median | 90th percentile | Worst | Mean |
|---|---|---|---|---|
| nxopt `EO2,EP3`, 1 thread | 0.976 s | 3.09 s | 17.4 s | 1.53 s |
| vcube `308`, 1 thread | 0.763 s | 2.32 s | 7.92 s | 1.08 s |
| **cuda-cube-solver** | **0.023 s** | **0.072 s** | **0.263 s** | **0.034 s** |

Median time per cube by optimal solution length:

| Optimal length | Cubes | nxopt | vcube | cuda-cube-solver |
|---|---|---|---|---|
| 16 | 2 | 0.03 s | 0.02 s | 0.002 s |
| 17 | 32 | 0.17 s | 0.13 s | 0.009 s |
| 18 | 65 | 1.35 s | 0.95 s | 0.036 s |
| 19 | 1 | 10.1 s | 7.92 s | 0.263 s |

Raw outputs: `data/single_cube/`.

## Worst case: positions that need 20 moves

No position needs more than 20 moves, and random scrambles almost never
need 20, so the tables above do not show the worst case. These runs use
known distance-20 positions from <https://cube20.org/distance20s/>, one
cube at a time.

| 10 positions (`random10.txt`) | Median | Worst | Total |
|---|---|---|---|
| nxopt `EO2,EP3`, 1 thread | 99.5 s | 104.6 s | 974 s |
| vcube `308`, 1 thread | 74.8 s | 124.0 s | 770 s |
| **cuda-cube-solver** | **1.53 s** | **1.69 s** | **15.3 s** |

On the 100 positions in `random100.txt`, cuda-cube-solver takes a median of
1.54 s and at most 1.97 s per cube. The CPU solvers were not run on those.

The superflip (every edge flipped in place) is much harder than a typical
20-mover for all three solvers. Their shared table design sees little of
its distance, so the search starts from a bound of 6 instead of about 12.

| Superflip | Time |
|---|---|
| vcube `308`, 1 thread | 808 s |
| nxopt `EO2,EP3`, 1 thread | 720 s |
| **cuda-cube-solver** | **51.5 s** |
| cuda-cube-solver + `112` table (`bench/run_gpu.py --step-extra 1`) | 10.3 s |

The extra `112` table (`cuda-cube-solver build-table --kind 112`, +2.6 GB
VRAM) makes positions like the superflip about 5× faster, but it makes
everything else slower:

- 2.00 s instead of 1.53 s median on the 10 positions above;
- about 1.5× slower on random scrambles.

It is therefore not the default.

Raw outputs and inputs: `data/distance20/`.

## Other GPU solvers

These are the two other public GPU optimal solvers found. Both were built
and run on the same card, one cube at a time, and their optimal lengths
match ours on all 100 positions.

- [copy-optsolver](https://github.com/cmsxbc/copy-optsolver) is a C++/CUDA
  port of Kociemba's optimal solver, with about 0.9 GB of tables.
- [h-hayakawa/rubiks_cube](https://github.com/h-hayakawa/rubiks_cube) is
  the solver behind Hayakawa et al., "GPU-acceleration of optimal
  permutation-puzzle solving", PASCO 2015. It uses a 3.5 GB table, and the
  GPU and 11 CPU threads share the search.

| | copy-optsolver | Hayakawa | **cuda-cube-solver** |
|---|---|---|---|
| First 100 random positions, median | 4.13 s | 0.570 s | **0.023 s** |
| First 100 random positions, mean | 3.99 s | 0.740 s | **0.034 s** |
| First 100 random positions, worst | 58.5 s | 4.65 s | **0.263 s** |
| copy-optsolver's hard 19-mover (`19f.txt`) | 39.5 s | 10.0 s | **0.277 s** |
| 10 distance-20 positions, median | ~590 s (6 of 10 finished in 1 hour) | 29.4 s | **1.53 s** |
| Superflip | 1,148 s | **48.7 s** | 51.5 s (10.3 s with the `112` table) |

**Hayakawa's solver** is the closer competitor:

- It is about 25× slower on random positions and 19× slower on distance-20
  positions.
- It is slightly faster than cuda-cube-solver's default on the superflip.
- Its CPU threads handle 7–9% of the search. With one CPU thread its times
  are about the same on random and distance-20 positions, and 8% slower on
  the superflip (53.1 s). Its superflip result therefore comes from its
  table, not the CPU.

**copy-optsolver** reports 1.30 s for the 19-mover and 38.2 s for the
superflip on an RTX 4090. Its search depends on the hot part of its table
staying in L2 cache, which is 72 MB on the 4090 and 32 MB here. That is why
it is about 30× slower on this card.

**[GPU-Rubiks](https://github.com/Nico-Adamo/GPU-Rubiks)** is not a direct
competitor:

- It solves in the quarter-turn metric with a small per-cubie heuristic and
  is meant for scrambles up to about 15 moves.
- On this GPU its solver did not finish cubes needing 8 or more quarter
  turns (see [future-work.md](future-work.md)).
- cuda-cube-solver solves its scramble distribution (`data/gpurubiks/`) in
  0.3 ms per cube at 13 quarter turns, but in the half-turn metric, so the
  comparison is not like-for-like.

Raw outputs: `data/other_gpu/`.

## Reading these numbers

- **Single cube versus bulk:** the single-cube gap over the CPU solvers
  (about 30–40×) is larger than the bulk gap (about 4×). For a single cube
  the CPU solvers use only one of the 12 threads.
- **Single machine only:** on large servers, vcube reports 55 cubes/s
  (32 cores, 170 GB table) and nxopt 33 cubes/s (16 cores, 256 GB).
  cuda-cube-solver has not been run on comparable GPU hardware.

## Builds and commands

None of the code from these solvers is included in this repository. They
were built outside it and used only as benchmarks.

### CPU solvers (both GPL-3.0)

- **vcube**, <https://github.com/Voltara/vcube>, commit `9f5bc2cc`:
  - built with clang 22, `-O3 -mlzcnt -mbmi -mbmi2 -mavx2`;
  - run as `./vc-optimal -c 308 --ordered`.
- **nxopt** from cube20src, <https://github.com/rokicki/cube20src>, commit
  `f430921a`:
  - built with `g++ -DPS=16 -O3 -march=native -DHALF -DTHREADS -DEO2 -DEP3`;
  - run as `nxopt23 -t 12 -`, with moves converted to `U1R3…` notation
    (`data/random1000.nxopt.txt`).
- **Single-cube runs:** `vc-optimal -c 308 --ordered --workers=1` and
  `nxopt23 -t 1 -`, on the first 100 positions, the distance-20 positions
  and the superflip.

### cuda-cube-solver

`bench/run_gpu.py --count N --batch 1` for the single-cube runs and
`--batch 100` for the bulk run.

### Other GPU solvers (neither has a license)

Both were built with CUDA 13.4 `nvcc` for `sm_120` and g++ 13.

- **copy-optsolver**, commit `2278a4d`:
  - its CPU build generates the tables, then its `c++/cuda` build is used
    to solve;
  - run as `gpu_solve -b`, with one scramble per line;
  - its `parity_test 50000` passes on this card.
- **Hayakawa's rubiks_cube**, commit `4d9e834`:
  - `solver_settings.h` set to `N_GPU 1` and `N_THREADS 12` (6 made no
    difference);
  - its `main.c` only solves random cubes it generates itself, so a small
    local driver feeds it the same scrambles through its `solve()` and
    checks each solution;
  - table generation took 82 minutes.
