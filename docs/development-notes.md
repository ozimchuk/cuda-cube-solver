# Development notes

How the solver got to its current design, and the measured experiments
that did not pay off. All numbers are on `data/random1000.txt` (1,000
random positions) on an RTX 5060 Ti 16 GB unless stated; every variant
returned the same optimal lengths. Postponed ideas and open questions
are in [`future-work.md`](future-work.md).

## History (cubes/s on the 1,000 positions)

| Step | Cubes/s |
|---|---|
| Three axis frames (UD, RL, FB), mod-3 table | 7.8 |
| + inverse frames, block table for them | 9.3 |
| Branch-free edge conjugation (`prmt` byte permutes, no loops) | 9.6 |
| Coset axis rule (from nxopt's documentation) | 18.9 |
| Inverse-frame value inherited along the move's axis | 19.0 |
| Per-node move masks (lookup-lane efficiency 22% → 45%) | 23.7 |
| Step-machine kernel, per-node frame order, block table only (lane efficiency 99%, half the VRAM) | 28.0 |

Later re-measurements of the final kernel give 27.6–27.8 cubes/s; the
README and `docs/benchmarks.md` use the most recent run (27.7).

## Expected table contents

`cuda-cube-solver build-table` should report these entry counts per
depth (also stored in `tables/prune-ud-208.json`). Depths 1–11 equal the
counts printed by vcube's generator for its `208` table; vcube does not
print 12 and 13.

| Depth | Entries |
|---|---|
| 0 | 1 |
| 1 | 2 |
| 2 | 22 |
| 3 | 200 |
| 4 | 2,137 |
| 5 | 23,903 |
| 6 | 287,698 |
| 7 | 3,560,276 |
| 8 | 44,164,674 |
| 9 | 530,379,944 |
| 10 | 5,233,704,011 |
| 11 | 19,675,963,236 |
| 12 | 4,711,576,986 |
| 13 | 247,310 |
| total | 30,199,910,400 |

## The forward kernel (before the step machine)

Each thread runs its own depth-first search and checks a child's frames
in a fixed order, stopping at the first that prunes. A warp executes a
frame's index computation whenever any of its 32 lanes needs it, so most
of that work ran with few active lanes. Measured with this kernel:

| Variant | Nodes/cube | Cubes/s |
|---|---|---|
| forward kernel, inverse-frame inheritance | 55.7 M | 19.0 |
| forward frames from the block table instead of mod 3 | 56.6 M | 18.9 |
| `--mode multi`: + `112` table (2.6 GB) | 42.1 M | 15.9 |
| `--mode multi`: + `112` + `304` tables (4.0 GB) | 39.2 M | 12.1 |
| prefix length 3 / 5 (default 4) | — | 17.3 / 16.1 |

The extra tables (`cuda-cube-solver build-table --kind 112|304`, both
cross-checked against vcube's generator) raise the six-lookup average by
0.11 and cut nodes by 31%, but every warp pays for their index
computation whenever one lane needs it.

**Bidirectional search** (`--mode bidir`, nxopt's other refinement):
extend whichever end of the solution has fewer allowed moves and inherit
one lookup from the parent. 15.0 cubes/s versus 18.9: nodes fall only
5%, while every frame needs a block lookup and back moves need left
multiplication.

## The step machine (current default)

Each warp iteration, every lane performs exactly one table lookup for its
current child; choosing moves, push, pop and fetching work run between
steps. The child is evaluated on a temporary copy of the frame, so a
pruned child needs no undo. Frames are taken in a per-node order (frames
at the budget edge first), encoded as three 6-bit masks.

Measured with it, all without gain:

| Variant | Cubes/s (baseline 27.6–28.0) | Why |
|---|---|---|
| Extra tables as extra steps (`--step-extra 1` / `2`) | 18.4 / 13.8 | lookup steps per child 2.5 → 4.8: ~46% of children survive the main lookups |
| One bit per entry + `304` fallback (`--table-format bit`, 5.2 GB) | 17.2 | 4.03 vs 2.48 lookups per child |
| Lazy forward frames (`LAZY_FORWARD`, gap 2/3/4) | 28.1 / 27.8 / 27.3 | lookups saved = nodes added |
| Streaming cache hints (`CACHE_HINT`) | 26.9 | nothing was being evicted |
| Reuse child frames at push / saved-frame pop stack (removed) | 27.7 / 26.8 | not instruction-bound |

## Where the time goes

Nsight Compute on a representative round (bound 17,
`bench/profile_step.py 10 17`, `CUDA_CUBE_SOLVER_LINEINFO=1`): DRAM
traffic 122 GB/s, exactly one 32-byte sector per table lookup at the
random-read ceiling of this card (3.8 G sectors/s, `bench/gpu_gather.py`).
Compute 66%; 70% of warp instructions run with fewer than 8 active lanes
(the divergent push/pop/control code) but that is not the bottleneck.
Local memory is 7% of L2 traffic, mostly hits. The search is therefore
limited by lookups per child × the card's random-read rate; every
instruction-side change measured as zero.

A larger table kept in host RAM and read by the GPU over PCIe is not
viable on this machine: random reads from GPU kernels into pinned host
memory reach ~60 M/s (`bench/gpu_host_gather.py`).

Learned table encodings (small neural networks, tensor cores) were tested
and do not work: the networks reach only the majority-class accuracy.
See `research/learned_table/`.
