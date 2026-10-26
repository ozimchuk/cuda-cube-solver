# Future work and postponed options

Status at commit `e597d5c` (RTX 5060 Ti 16 GB, Ryzen 5 9600X, 30 GB RAM):
28.0 cubes/s on `data/random1000.txt`, default `step` mode, one 7.8 GB
block table (`prune-ud-208-blk.bin`). Table generation needs 15.3 GB of
VRAM (mod-3 table and block table at once).

## 1. Supporting 8 GB cards

**Problem.** Solving needs ≈ 9 GB of VRAM (7.8 GB table plus small
tables); an 8 GB card has ≈ 7.3–7.6 GB free after the display. Building
the tables needs ≥ 16 GB.

### 1a. Base-3 block encoding (recommended first step)

Pack 5 entries per byte (3⁵ = 243 ≤ 256): a 32-byte sector holds one
header byte and 155 entries, 6.2 GB for the `208` table.

A plain "minimum + {0, 1, ≥2}" code loses the 11-vs-12 distinction,
because almost every block contains a depth-9 entry. Instead each block
stores its minimum `m` and a cut `b ∈ [m, m+3]` (6 + 2 bits), and the
three codes mean *≤ b* (bound `m`), *b + 1*, and *≥ b + 2*; the cut is
chosen per block to maximise the sum of encoded bounds. The result is
still an admissible coset lower bound, so the axis rule stays valid.

Measured on 20,000 random positions (six frames each):

| Encoding | Size | avg1 | avg6 + axis rule | entries exact |
|---|---|---|---|---|
| 2-bit blocks (current) | 7.8 GB | 10.905 | 11.848 | 100% |
| base-3, cut = block minimum | 6.2 GB | 10.716 | 11.648 | 81% |
| **base-3, cut chosen per block** | **6.2 GB** | 10.846 | **11.823** | 94% |

Expected cost ≈ 5–6% more nodes. Decoding is one byte load plus a 1.3 KB
shared-memory digit table: still one memory transaction per lookup.

Work (≈ 1 hour, no C++ build):

1. Compile-time switch (`TABLE_B3`) changing `block_lb` and the block
   geometry; all kernels share that function.
2. Converter kernel (GPU, CuPy) that recovers exact depths from the two
   existing tables — the 2-bit block value is exact unless it reads
   "≥ min + 3", where the mod-3 table resolves it uniquely — and writes
   `prune-ud-208-b3.bin` chunk by chunk.
3. Solver option `--table-format b3`; validate decoded bounds against
   exact values and run the 1,000-position benchmark (expect ≈ 26–27
   cubes/s here, identical lengths).

Limitation: building still needs a ≥ 16 GB GPU; 8 GB users need the
prebuilt file (1b) or a CPU generator (1c).

### 1b. Distribute the prebuilt table

Publish the 6.2 GB `prune-ud-208-b3.bin` (e.g. a GitLab release asset)
with a checksum. Zero code; enough if the audience is small.

### 1c. Single-source CPU generator (if others must build tables)

- Mark the cube logic in `kernels.cu` (moves, symmetry conjugation,
  coordinates, table index, unrank, decode) `__host__ __device__`; add a
  header with portable versions of `__byte_perm`, `__dp4a`,
  `__vcmpgeu4`, `__popc`, `__ffs` for g++.
- OpenMP host versions of the BFS passes (forward/backward layer,
  histogram, final encoding), GCC atomics instead of CUDA atomics.
- Python compiles the file with `g++ -O3 -fopenmp` at first use and
  calls it through `ctypes` (the CPU counterpart of CuPy/NVRTC):
  `cuda-cube-solver build-table --device cpu`.
- BFS on exact 4-bit depths (15 GB RAM), then base-3 blocks with the
  per-block cut. Expected 5–15 min on 12 threads (vcube builds the same
  table in 4 min 21 s, nxopt in 2 min 24 s, both with SIMD code).
- Validate: per-depth counts equal vcube's; tables byte-identical to the
  GPU build. Search stays GPU-only.
- Bonus: on a machine with ≥ 48 GB RAM it could build the 22.7 GB-class
  `308` table for 24–32 GB cards (section 3).

Cost: several hours and a C++ toolchain requirement for table building.

### 1d. GPU generation with host RAM assistance (not recommended)

Keep only the random-access BFS state in VRAM — three states (unvisited,
visited at even depth, at odd depth) in base 3, 20 entries per 32-bit
word with compare-and-swap, 6.0 GB — and after each layer stream it to
host RAM, where a 4-bit depth array (15 GB) records newly reached
entries (an unvisited entry can never neighbour an entry two layers
back, so parity suffices). Build the base-3 table from RAM at the end.
≈ 4–6 min, needs an 8 GB GPU and ≈ 16 GB RAM. More complex than 1c for
little gain; superseded by 1c.

### 1e. Threshold bit + small exact fallback (implemented; slower than hoped)

From `research/learned_table/`: one bit per entry, d ≥ 11, and the 304
table (1.4 GB) when the bit is clear. 5.2 GB in total; avg6 11.844 versus
11.862 exact. Implemented as `--table-format bit`
(`cuda_cube_solver.table.build_bit_table`, kernel define `TABLE_BIT`; the fallback
read is deferred to the lane's next step).

Measured on the 1,000 positions (RTX 5060 Ti, register cap
`SEARCH_MIN_BLOCKS=2`): **17.2 cubes/s versus 28.0** for the 2-bit block
table, identical lengths and node counts. The search is memory-bound and
the bit table needs 4.03 lookup steps per child versus 2.48: capping 12
at 11 loses many first-lookup prunes, plus ~0.5 fallback reads per child.
Lesson: average bounds (avg1/avg6) do not predict speed here; lookups per
child do.

Use for cards with ≈ 6 GB; for 8 GB cards base-3 (1a) should be faster,
since it keeps 12 distinct in most blocks (94% exact). Not yet measured.

### Rejected for 8 GB

- **GPU reading a table in host RAM over PCIe.** Measured 60 M random
  reads/s (`bench/gpu_host_gather.py`) versus 3,850 M/s from VRAM; a BFS
  needs 10¹¹–10¹² random accesses.
- **CPU backends for CUDA code** (PyCUDA/CuPy have none; Numba's CUDA
  simulator ≈ 1,000× too slow; HIP-CPU, chipStar/POCL, SYCL emulate
  warps, which the search kernel relies on). 1c gets the single-source
  benefit without them.
- **Smaller tables only** (`112` 2.6 GB, `304` 1.4 GB): avg6 11.55 /
  11.12 versus 11.82 for base-3 `208`; roughly 2× more nodes.
- **Lossy pair compression** (min of neighbouring entries): `208`
  halved loses 0.27 of avg6, far worse than base-3's 0.026.

## 2. Further speed on 16 GB

**Profile (Nsight Compute 2026.2, `bench/profile_step.py 10 17`, a
bound-17 round as in real solves; `CUDA_CUBE_SOLVER_LINEINFO=1` for source
attribution).** DRAM throughput 122 GB/s = one DRAM sector per table
lookup at 3.8 G sectors/s, the random-read ceiling measured on this card
(`bench/gpu_gather.py`). Compute 66%, 9 active threads per warp on
average (70% of warp instructions run with < 8 lanes: the divergent
control code), local memory 7% of L2 sectors and mostly hits. The search
is memory-bound; instruction-side changes do not move it. (An earlier
profile of a bound-16 round, with only ~27 children per work item,
looked compute-bound — prefix replay and control dominated there — and
was not representative.)

Measured with no gain on the 1,000 positions (all correct, off by
default where kept):

| Variant | Cubes/s (baseline 27.6–27.8) | Why |
|---|---|---|
| Lazy forward frames (`LAZY_FORWARD`, gap 2/3/4) | 28.1 / 27.8 / 27.3 | lookups saved = nodes added |
| Streaming cache hints (`CACHE_HINT`) | 26.9 | nothing was being evicted |
| Reuse child inverse frames at push (removed) | 27.7 | not instruction-bound |
| Saved-frame pop stack (removed) | 26.8 | not instruction-bound |
| `h`/`c` arrays out of local memory (kept) | 27.8 | local traffic was cache hits |

**Sibling locality: none.** A U turn keeps every corner orientation
value but moves corners between positions, so the twist coordinate — and
with it the corner class, the top of the table index — changes for most
moves; siblings land billions of entries apart (measured: 0.0% of lookups
share a sector with the lane's previous lookup of the same frame). This
rules out block-parallel IDA* style cooperative expansion for memory
reasons.

Remaining ideas on this card:

- The step kernel is at the DRAM random-access ceiling (≈ 2.5 lookups
  per child, ≈ 3.7 G lookups/s against a measured 3.85 G/s); gains must
  come from fewer lookups or more information per lookup.
- Extra tables (`--step-extra`) cut nodes 30% but nearly double lookup
  steps per child, because ~46% of children survive the main lookups.
  Would need a much cheaper way to decide which survivors to recheck.
- Base-3 encoding (1a) frees 1.6 GB on 16 GB cards; only useful if the
  extra-table problem is solved.

## 2b. Other GPU solvers

[GPU-Rubiks](https://github.com/Nico-Adamo/GPU-Rubiks) (commit
`4889310f`, Dec 2023; coursework project): quarter-turn metric, 12
moves, IDA* with a per-cubie "Manhattan" heuristic, one 24-thread block
per subproblem with a shared-memory stack (after arXiv:1705.02843).
Its README reports 13.4 s per cube for 13-quarter-turn scrambles on an
unnamed GPU and recommends depths up to 15. Built here with CUDA 13
(`sm_120`) and with CUDA 12.8 in Pascal-compatible mode
(`compute_60` → `sm_120`): its GPU solver did not finish 5–7 of 12
depth-8 scrambles within 20 s (only cubes solvable in ≤ 6 moves
finished), while its CPU solver worked — most likely a warp-synchronous
assumption broken on post-Volta GPUs. cuda-cube-solver solves the same scramble
distribution (`data/gpurubiks/`, half-turn metric) in 0.2–0.3 ms per
cube up to 13 quarter turns and 4 ms at 20. Nothing to adopt (see
sibling locality above).

## 3. Larger cards (24–32 GB)

nxopt's next table size, `EO2,EP3` (`308`), is 22.7 GB (≈ 23.4 GB in
block format, ≈ 18 GB in base-3) and needs ~2.3× fewer nodes on CPUs
(avg6 12.23 vs 11.87). The edge-coordinate code already supports the
variant (`EdgeKind(3, 8)`, tested); generation needs either a ≥ 48 GB
GPU or the CPU generator (1c) on a ≥ 48 GB machine. Untested on this
hardware.

Threshold-bit variant (research/learned_table): `308` as one bit
"≥ q" is 11 GB; with the `112` fallback (2.6 GB) it fits 16 GB cards.
Whether it beats `208` depends on the `308` value distribution — needs
the table built first.

## 4. Neural/tensor-core table compression (tested, negative)

Small and large MLPs predicting the 208 table from its coordinates reach
only the majority-class accuracy (65%); learned per-entry cuts are no
better than a constant. Tensor cores have no role in this design. See
`research/learned_table/README.md`.

## 5. Publishing

- No license yet (needed before making the repository public).
- Claims supported so far: on this desktop, 3.7× vcube and 4.7× nxopt
  on 1,000 random positions with identical optimal lengths. Not
  supported: "fastest in the world" (vcube reports 55 cubes/s on a
  32-core, 170 GB server).
