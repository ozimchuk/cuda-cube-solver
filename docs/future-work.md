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

Measured on 20,000 random positions (six frames each, same method as the
README's averages):

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
  `gpucube build-table --device cpu`.
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

- The step kernel is at the DRAM random-access ceiling (≈ 2.5 lookups
  per child, ≈ 3.7 G lookups/s against a measured 3.85 G/s); gains must
  come from fewer lookups or more information per lookup.
- Extra tables (`--step-extra`) cut nodes 30% but nearly double lookup
  steps per child, because ~46% of children survive the main lookups.
  Would need a much cheaper way to decide which survivors to recheck.
- Base-3 encoding (1a) frees 1.6 GB on 16 GB cards; only useful if the
  extra-table problem is solved.

## 3. Larger cards (24–32 GB)

nxopt's next table size, `EO2,EP3` (`308`), is 22.7 GB (≈ 23.4 GB in
block format, ≈ 18 GB in base-3) and needs ~2.3× fewer nodes on CPUs
(avg6 12.23 vs 11.87). The edge-coordinate code already supports the
variant (`EdgeKind(3, 8)`, tested); generation needs either a ≥ 48 GB
GPU or the CPU generator (1c) on a ≥ 48 GB machine. Untested on this
hardware.

## 4. Publishing

- No license yet (needed before making the repository public).
- Claims supported so far: on this desktop, 3.7× vcube and 4.7× nxopt
  on 1,000 random positions with identical optimal lengths. Not
  supported: "fastest in the world" (vcube reports 55 cubes/s on a
  32-core, 170 GB server).
