# Subgroup survey: other coset tables with U and D in the subgroup

Every table here is a coset table for a subgroup H that contains all U and D turns. That keeps
the 16-symmetry reduction, the six frame lookups, the axis rule and the inherited lookup.
nxopt's twelve standard variants (`nxopt.md` in cube20src) jump from 7.8 GB (`208` =
EO2,EP2) to 22.7 GB (EO2,EP3). The variants below come from subgroups nxopt does not use:
instead of forgetting a whole piece of edge information, they forget it only up to a smaller
relabelling group.

## The family

A table index is corner class (9,930) × E-slice positions (495) × an edge factor. The edge
factor is the E content up to a relabelling group L, times a non-E part:

- **E content:** the labels of the four E edges in position order, with their flips. That is an
  element of the signed permutation group B4 (384 elements). L is a subgroup of B4 acting on the
  labels. It must be normalised by the symmetry image (the D4 of label permutations, with flips
  0000 or 1111, that the 16 UD symmetries induce), so that symmetry reduction stays exact. 79 of
  B4's 1,659 subgroups pass.
- **Non-E part (mode):** 1 = the flips of the 8 non-E edges (256); 2 = which non-E positions hold
  U edges (70); 3 = that split up to swapping U and D (35).

`groups.py` enumerates the subgroups and builds the class tables. A survey kind is named
`g<mode>-<id>`, where id indexes `candidates.json`. The kernels handle every kind through one
generic path (`EDGE_EP=9`). `test_groups.py` checks, for all 95 kinds with factors from 2,048 to
12,288, that unrank/index round-trip and that the index is constant on cosets, also after every
move. Symmetry reduction stays exact because only normalised subgroups are candidates. nxopt's
own tables come out as special cases: g1-32 = `208`,
g2-49 = `304`, and g1-26 = `612`.

## Bounds (exact coset distances, 20,000 random positions × 6 frames)

`survey.py` uses the same positions as `research/learned_table`. avg6 is the maximum of the six
frames plus the axis rule. Tables of 8.5 GB and up were built as mod-3 tables only. A mod-3 table
and its block table don't fit together on a 16 GB card, so those rows have no block avg6.

| Block table | Factor | Kinds | Best | avg6 | avg6 from block | Worst avg6 | nxopt reference |
|---|---|---|---|---|---|---|---|
| 1.42 GB | 1,120 | 1 | g2-49 = `304` | 11.133 | 11.122 | | `304` |
| 2.60 GB | 2,048 | 6 | g1-49 | 11.578 | 11.556 | 11.449 | |
| 2.84 GB | 2,240 | 2 | g2-31 | 11.409 | 11.393 | 11.397 | |
| 3.90 GB | 3,072 | 15 | g1-43 | 11.694 | 11.676 | 11.619 | |
| 4.26 GB | 3,360 | 26 | g3-7 | 11.650 | 11.625 | 11.497 | |
| 5.20 GB | 4,096 | 3 | **g1-31** | **11.807** | 11.789 | 11.787 | `512`: 11.787 |
| 7.79 GB | 6,144 | 21 | **g1-27** | **11.953** | 11.939 | 11.854 | `208`: 11.862 |
| 8.52 GB | 6,720 | 14 | g2-7 | 11.899 | | 11.811 | |
| 11.37 GB | 8,960 | 1 | `808` | 11.969 | | | |
| 15.59 GB | 12,288 | 12 | **g1-7** | **12.188** | | 12.056 | EO2,EP3 (22.7 GB): 12.23 |

The winners:
- **g1-27** (7.8 GB) is the same D4 of E-label permutations as `612`. The four permutations
  outside the diagonal-swap subgroup {id, swap FL/BR labels, swap FR/BL labels, both} also flip
  all four E edges. Of the three D4 twists, untwisted g1-26 = `612` scores 11.935, the symmetry
  image itself (g1-28) scores 11.909, and g1-27 scores 11.953.
- **g1-31** (5.2 GB): an A4 of E-label permutations, with flips tied to the permutation. It beats
  `512`, which is the untwisted version of the same A4 (g1-30).
- **g1-7** (15.6 GB): only the diagonal swaps, no flips. It beats `712` (Klein four, 12.124).

What the numbers show:
- **Mode 1 wins.** Full edge orientation is worth more than E order or the U/D layer split. Every
  8.5 GB table (modes 2 and 3) scores below the best 7.8 GB tables.
- **The twist matters as much as the group order.** At 7.8 GB the 21 kinds range from 11.854 to
  11.953. g1-26, g1-27 and g1-28 have the same group order and the same permutation image; they
  differ only in which elements carry flips.

## Solving (1,000 random cubes, `bench/run_gpu.py data/random1000.txt --count 1000 --batch 100 --kind K`)

Measured on an RTX 5060 Ti 16 GB, step mode, block table only. Every kind found the same
solution lengths for all 1,000 cubes (mean 17.692), and every solution was checked to solve its
cube.

| Kind | Table | Cubes/s | M nodes/cube | G nodes/s |
|---|---|---|---|---|
| `208` | 7.79 GB | 27.56 | 52.5 | 1.45 |
| `612` | 7.79 GB | 30.51, 30.65 | 42.7 | 1.30 |
| g1-26 (= `612` group, generic path) | 7.79 GB | 33.47 | 43.5 | 1.46 |
| **g1-27** | 7.79 GB | **37.87, 37.82, 37.80** | **39.1** | **1.48** |
| **g1-27, base 3** (`--table-format b3`) | **6.23 GB** | **34.84, 34.56** | 40.7 | 1.41 |
| `512` | 5.20 GB | 20.15 | 67.4 | 1.36 |
| g1-31 | 5.20 GB | 23.93 | 64.1 | 1.53 |

- **g1-27 is 37% faster than `208`, using the same memory.** It searches 26% fewer nodes, and its
  node rate is slightly higher.
- **`612`'s 10% lower node rate was its code path, not its group.** The same group through the
  generic path (g1-26) runs at `208`'s node rate. The two tables number classes differently, so
  their block minima, and hence their node counts, differ slightly.
- **For 8 GB cards: base-3 g1-27.** At 6.23 GB it is 26% faster than `208` at 7.8 GB. Its bounds
  were checked on the 20,000 positions: no violations, 92.4% exact, avg6 11.927.
- **For 5.2 GB, g1-31 beats `512` by 19%.**

## Hard positions (`hard_survey.py`, `superflip.py`)

The bounds above rank tables on random positions. The superflip ranks them differently, so a
second pass built every kind up to 7.8 GB that keeps all edge flips (mode 1, 55 kinds; plus 4
mode-3 kinds before they were dropped) and counted real search nodes on three sets. The counts
include completed rounds only: they do not depend on timing or on where the last round finds
its first solution.

Pareto front per size (no other table of that size is better on all three):

| Size | Kind | Superflip | 10 distance-20, per cube | 100 random, per cube |
|---|---|---|---|---|
| 7.8 GB | g1-27 = `d4t` | 36.9 G | 1.36 G | **9.2 M** |
| 7.8 GB | g1-26 = `d4` | **6.3 G** | **1.35 G** | 10.2 M |
| 5.2 GB | g1-31 = `a4t` | 20.7 G | 2.13 G | **14.9 M** |
| 5.2 GB | g1-30 (`512`'s group) | **8.4 G** | **2.08 G** | 16.0 M |
| 3.9 GB | g1-43 = `d4f` | 84.7 G | **3.17 G** | **20.4 M** |
| 3.9 GB | g1-38 (and 5 between) | **38.9 G** | 3.62 G | 23.9 M |
| 2.6 GB | g1-50 | 44.1 G | 4.03 G | **27.3 M** |
| 2.6 GB | g1-49 = `s4` | **14.1 G** | **3.77 G** | 28.0 M |
| 1.95 GB | g1-60 … g1-62 (5 kinds) | 62–98 G | 6.6–7.3 G | 41–45 M |

For comparison, `208` (g1-32) costs 27.5 G, 1.84 G and 12.4 M, behind both 7.8 GB tables on
all three sets. The mode-3 tables, which keep no flips of the 8 non-E edges, cost 870–980 G
nodes on the superflip.

What the numbers show:
- **The twist is the trade-off.** At 7.8 GB and 5.2 GB the front is exactly two tables: a group
  and its twisted version, whose elements outside a subgroup also flip all four E edges. The
  twist wins 7–11% on random positions. The untwisted group keeps the E flips exactly and is
  2.5–6× faster on the superflip, and as good or better on distance-20 positions.
- **At 2.6 GB there is no trade-off worth taking:** `s4` keeps the E flips exactly, is best on
  the superflip and on distance-20 positions, and only 2.6% behind on random positions.
- **The 3.9 GB tables are weak on hard positions:** `d4f` (its pure all-flip element hides the
  E flips) costs 84.7 G on the superflip, worse than `s4` at 2.6 GB.

Why the superflip was slow at all, from `superflip.py` (`rounds`, `lookups`):
- **Not the root bound:** `d4t` starts at 9, `208` at 6, yet `d4t` searches more.
- **Not lookups per node:** about 2.0 on the superflip against 2.4 on random positions.
- **Work imbalance:** with 4-move work items the node rate fell to 0.55 G/s, while a few items
  finished their huge subtrees. With 5- and 6-move items it is 2.3 G/s; the solver now picks
  them for cubes whose last round was large.
- **Base 3 and the block format:** a block stores each entry as its minimum plus at most 3, and
  base 3 is lossier still. Around the superflip this matters: `d4t-b3` searches 2.8× the nodes
  of `d4t`, and `s4` gives the superflip a bound of 9 in block format, 10 in base 3.

Robust and fast presets, solving times (1,000 random cubes, the superflip, 10 distance-20
positions; identical solution lengths):

| Size | Fast | Random | Superflip | Distance-20 | Robust | Random | Superflip | Distance-20 |
|---|---|---|---|---|---|---|---|---|
| 7.79 GB | `d4t` | 37.8/s | 16.0 s | 10.0 s | `d4` | 33.3/s | 3.2 s | 10.0 s |
| 6.23 GB | `d4t-b3` | 35.2/s | 47.5 s | 10.5 s | `d4-b3` | 31.6/s | 3.2 s | 10.8 s |
| 5.20 GB | `a4t` | 24.1/s | 11.5 s | 16.0 s | `a4` | 22.2/s | 4.2 s | 15.7 s |
| 4.16 GB | `a4t-b3` | 21.9/s | 27.1 s | 18.1 s | `a4-b3` | 20.7/s | 4.0 s | 17.6 s |

Base 3 hurts only the twisted tables on the superflip: `d4-b3` even searches fewer nodes than
`d4` (5.2 G against 6.3 G), while `d4t-b3` searches 2.8× more than `d4t`.

A second table (`GpuSolver(extra_kind=...)`, kernel `EXTRA_GEN_NEC`, `EXTRA_FRAMES`) does not
help: `d4t` + `s4` (10.4 GB) takes 5.7–19 s on the superflip and costs 8–28% on random cubes,
all dominated by `d4` alone.

## In the solver

The ladder became the solver's table presets (`src/cuda_cube_solver/kinds.py`, `cuda-cube-solver
tables`), under short names: `s4` = g1-49, `d4f` = g1-43, `a4t` = g1-31, `d4t` = g1-27, and
`d4` = g1-26 and `a4` = g1-30 (robust options, only by name; see "Hard positions"). Full
memory ladder (`ladder.sh`, 1,000 cubes, same solution lengths for every table):

| Preset | VRAM | Cubes/s |
|---|---|---|
| `304` | 1.42 GB | 6.0 |
| `s4-b3` (block: 2.60 GB, 13.6) | 2.08 GB | 13.1 |
| `d4f-b3` (block: 3.90 GB, 17.3) | 3.12 GB | 16.9 |
| `a4t-b3` | 4.16 GB | 21.8 |
| `a4t` | 5.20 GB | 24.1 |
| `d4t-b3` | 6.23 GB | 35.2 |
| `d4t` | 7.79 GB | 37.1–37.9 |

## Reproduce

```
PYTHONPATH=src:research/subgroups python research/subgroups/run_gen.py validate
PYTHONPATH=src:research/subgroups python research/subgroups/run_gen.py all   # about 1 h; skips kinds already in results.jsonl
PYTHONPATH=src python research/subgroups/survey.py 208 512 612 --save        # named kinds
```

`results.jsonl` holds one line per kind: depth counts, avg1/avg6 and block avg6. To solve with a
survey kind, build its block table first. `--b3` also builds the base-3 table:

```
PYTHONPATH=src:research/subgroups python research/subgroups/build.py g1-27 --b3   # ~5 min
PYTHONPATH=src python bench/run_gpu.py data/random1000.txt --count 1000 --batch 100 --kind g1-27 [--table-format b3]
```
