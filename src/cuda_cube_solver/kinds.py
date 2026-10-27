"""Table kinds and the memory ladder of table presets.

A table entry is a coset of a subgroup H that contains all U and D turns: corner class (9,930) x
E-slice positions (495) x an edge factor.  The ``coords.EdgeKind`` tables (208, 112, 304, ...) keep
or forget whole pieces of edge information.  A ``GroupKind`` keeps the flips of the 8 non-E edges
and the E-edge content (labels in position order with their flips: an element of the signed
permutation group B4) up to a relabelling group L <= B4.  L must be normalised by the relabelling
the 16 UD symmetries induce, so symmetry reduction stays exact.  research/subgroups surveys all
such L; the four below are the winners at their sizes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np

from cuda_cube_solver import coords

PERMS = list(permutations(range(4)))
PRANK = {p: i for i, p in enumerate(PERMS)}
IDENT = ((0, 1, 2, 3), (0, 0, 0, 0))


def _mul(a, b):
    """a after b; (perm, flips) acting on E labels: label x -> p[x], its flip ^= f[x]."""

    (pa, fa), (pb, fb) = a, b
    return tuple(pa[pb[x]] for x in range(4)), tuple(fb[x] ^ fa[pb[x]] for x in range(4))


def _inverse(a):
    p, f = a
    q = tuple(p.index(z) for z in range(4))
    return q, tuple(f[q[z]] for z in range(4))


def _sym_image():
    """The relabellings of the E edges that the 16 UD symmetries induce."""

    return {(tuple(int(es.perm[8 + x]) - 8 for x in range(4)), tuple(int(es.flip[8 + x]) for x in range(4)))
            for es in coords.ud_actions()[1]}


def _closure(gens):
    group, frontier = {IDENT}, [IDENT]
    while frontier:
        frontier = [y for y in {_mul(g, x) for x in frontier for g in gens} if y not in group]
        group.update(frontier)
    return group


class GroupKind:
    """Edge factor: non-E flips (256) x E content (384) up to the group (kernel EDGE_EP=9, EDGE_EO=1)."""

    def __init__(self, name: str, gens):
        self.name = name
        self.group = _closure(gens)
        if any(_mul(_mul(s, x), _inverse(s)) not in self.group for s in _sym_image() for x in self.group):
            raise ValueError(f"{name}: group not normalised by the UD symmetries; symmetry reduction would be wrong")
        # Classes of the 384 contents (index perm_rank * 16 + flips by position) under relabelling.
        cls = -np.ones(384, dtype=np.int64)
        reps: list[int] = []
        for c in range(384):
            if cls[c] >= 0:
                continue
            q, g = PERMS[c // 16], [(c >> i) & 1 for i in range(4)]
            for kp, kf in self.group:
                cls[PRANK[tuple(kp[q[i]] for i in range(4))] * 16 + sum((g[i] ^ kf[q[i]]) << i for i in range(4))] = len(reps)
            reps.append(c)
        parity = [bin(c & 15).count("1") & 1 for c in range(384)]
        gen_class = np.zeros(384, np.uint16)
        gen_rep = np.zeros((2, 384), np.uint16)
        if all(sum(f) % 2 == 0 for _, f in self.group):
            # Total flip parity is even, so the E flip parity follows from the non-E flips:
            # number the classes within each parity.
            count = [0, 0]
            index = []
            for r in reps:
                p = parity[r]
                index.append(count[p])
                gen_rep[p, count[p]] = r
                count[p] += 1
            self.nec = len(reps) // 2
            gen_class[:] = [index[cls[c]] for c in range(384)]
        else:
            self.nec = len(reps)
            gen_class[:] = cls
            for c in range(384):
                gen_rep[parity[c], cls[c]] = c
        self.n_edge = 495 * 256 * self.nec
        self.defines = ("EDGE_EP=9", "EDGE_EO=1", f"GEN_NEC={self.nec}")
        self.gen = (gen_class, gen_rep)


# research/subgroups names: s4 = g1-49, d4f = g1-43, a4t = g1-31, d4t = g1-27, d4 = g1-26, a4 = g1-30.
GROUP_KINDS = {
    "s4": [((0, 1, 3, 2), (0, 0, 0, 0)), ((0, 2, 1, 3), (0, 0, 0, 0)), ((1, 0, 2, 3), (0, 0, 0, 0))],
    "d4f": [((0, 1, 2, 3), (1, 1, 1, 1)), ((0, 3, 2, 1), (0, 0, 0, 0)), ((1, 0, 3, 2), (0, 0, 0, 0))],
    "a4t": [((0, 2, 3, 1), (0, 1, 1, 0)), ((1, 0, 3, 2), (1, 1, 1, 1))],
    "d4t": [((0, 3, 2, 1), (0, 0, 0, 0)), ((1, 0, 3, 2), (1, 1, 1, 1))],
    "d4": [((1, 2, 3, 0), (0, 0, 0, 0)), ((0, 3, 2, 1), (0, 0, 0, 0))],
    "a4": [((0, 2, 3, 1), (0, 0, 0, 0)), ((1, 0, 3, 2), (0, 0, 0, 0))],
}


def get(name: str):
    """A kind by table name: a GroupKind above, or a coords.EdgeKind name like "208"."""

    if name in GROUP_KINDS:
        return GroupKind(name, GROUP_KINDS[name])
    return coords.EdgeKind.parse(name)


N_CLASSES = 9930
B3_ENTRIES = 155
BLOCK_ENTRIES = 124


@dataclass(frozen=True)
class Preset:
    """A solving table: a kind in block format (2-bit offsets) or base 3 (denser, a little slower)."""

    kind: str
    format: str  # "block" or "b3"
    cubes_per_s: float  # 1,000 random cubes, RTX 5060 Ti 16 GB
    superflip_s: float  # the superflip, same card
    auto: bool = True  # chosen by --memory; False: only by name

    @property
    def name(self) -> str:
        return self.kind if self.format == "block" else f"{self.kind}-b3"

    @property
    def file(self) -> str:
        return f"prune-ud-{self.kind}-{'blk' if self.format == 'block' else 'b3'}.bin"

    @property
    def entries(self) -> int:
        return N_CLASSES * get(self.kind).n_edge

    @property
    def table_bytes(self) -> int:
        """Size of the solving table (file and VRAM)."""
        per = BLOCK_ENTRIES if self.format == "block" else B3_ENTRIES
        return -(-self.entries // per) * 32

    @property
    def build_bytes(self) -> int:
        """Peak VRAM to build it: the mod-3 table and the block table together."""
        return (self.entries + 15) // 16 * 4 + -(-self.entries // BLOCK_ENTRIES) * 32


# The memory ladder, smallest first: each step is faster on random cubes and bigger.
PRESETS = [
    Preset("304", "block", 6.0, 44.6),
    Preset("s4", "b3", 13.1, 6.2),
    Preset("d4f", "b3", 16.9, 77.7),
    Preset("a4t", "b3", 21.8, 27.1),
    Preset("a4t", "block", 24.1, 11.5),
    Preset("d4t", "b3", 35.2, 47.5),
    Preset("d4t", "block", 37.5, 16.0),
    # Robust alternatives, only by name: the same groups without the twist keep the E-slice
    # edges' flips exactly.  6-11% slower on random cubes, as fast on distance-20 positions,
    # 3-15x faster on flip-heavy ones such as the superflip.
    Preset("a4", "b3", 20.7, 4.0, auto=False),
    Preset("a4", "block", 22.2, 4.2, auto=False),
    Preset("d4", "b3", 31.6, 3.2, auto=False),
    Preset("d4", "block", 33.3, 3.2, auto=False),
]
PRESET_BY_NAME = {p.name: p for p in PRESETS}

# Solving needs some VRAM besides the table (search buffers, corner tables).
SOLVE_MARGIN = 400 << 20


def built(preset: Preset, tables: Path) -> bool:
    return (tables / preset.file).exists()


def choose(tables: Path, memory: int, build_memory: int | None = None,
           to_build: bool = False) -> tuple[Preset | None, Preset | None]:
    """The fastest preset whose table fits in ``memory`` bytes and is built or can be built
    (in ``build_memory``, default ``memory``), and the fastest preset that would fit but can't
    be built here (None if that is no faster).  ``to_build``: ignore tables already built."""

    build_memory = memory if build_memory is None else build_memory
    fits = [p for p in PRESETS if p.auto and p.table_bytes + SOLVE_MARGIN <= memory]
    usable = [p for p in fits if (built(p, tables) and not to_build) or p.build_bytes + SOLVE_MARGIN <= build_memory]
    best = usable[-1] if usable else None
    better = fits[-1] if fits and fits[-1] is not best else None
    return best, better


def pick(tables: Path, memory="auto", name: str | None = None,
         to_build: bool = False) -> tuple[Preset | None, Preset | None]:
    """``choose`` for a ``memory`` budget in GB or "auto" (the free VRAM; a larger budget is cut
    to it), or the preset ``name``."""

    if name:
        if name not in PRESET_BY_NAME:
            raise ValueError(f"unknown table {name!r}; choose from {', '.join(PRESET_BY_NAME)}")
        return PRESET_BY_NAME[name], None
    import cupy as cp

    free = cp.cuda.runtime.memGetInfo()[0]
    budget = free if memory == "auto" else min(free, int(float(memory) * 1e9))
    return choose(tables, budget, free, to_build)


def describe(tables: Path) -> list[str]:
    rows = [f"{'preset':8} {'VRAM':>8} {'cubes/s':>8} {'superflip':>10} {'build VRAM':>11}  file"]
    for p in PRESETS:
        rows.append(f"{p.name:8} {p.table_bytes / 1e9:6.2f}GB {p.cubes_per_s:8.1f} {p.superflip_s:9.1f}s "
                    f"{p.build_bytes / 1e9:9.1f}GB  {p.file}{'' if built(p, tables) else '  (not built)'}"
                    f"{'' if p.auto else '  (only with --table)'}")
    return rows


def write_meta(path: Path, preset: Preset, counts: list[int], seconds: float) -> None:
    path.write_text(json.dumps({"preset": preset.name, "entries": preset.entries, "depth_counts": counts,
                                "seconds": round(seconds, 1)}, indent=2) + "\n")
