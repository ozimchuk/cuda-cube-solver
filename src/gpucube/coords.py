"""Coordinates of the pruning-table quotient, host side (numpy).

The quotient keeps, relative to the UD axis:

* corners: twist (3^7) and which positions hold U-layer corners (8C4 = 70);
* edges: which positions hold E-slice edges (12C4 = 495), the order of the
  four E-slice edges in those positions (4! = 24), and the orientation of
  the eight non-E edges read in position order (2^8).

The corner part is reduced by the 16 UD-preserving symmetries.  A table
index is ``corner_class * N_EDGE + edge_coord``.

Packed byte formats match the GPU kernels: corner byte ``cp | co << 3``,
edge byte ``ep | eo << 4``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations

import numpy as np

from gpucube.geometry import Symmetry, ud_symmetries
from gpucube.cubie import MOVES_HTM, CubieState

N_TWIST = 3**7
N_CCOMB = 70
N_CORNER_RAW = N_TWIST * N_CCOMB
N_SLICE = 495
N_EORDER = 24
N_EO8 = 256
N_EDGE = N_SLICE * N_EORDER * N_EO8


@dataclass(frozen=True)
class EdgeKind:
    """Edge coordinate variant, named like vcube/nxopt tables ("208", "112", "304").

    ``ep``: 1 = E-slice positions only, 2 = plus the order of the E edges,
    3 = plus which non-E positions hold U-layer edges.  ``eo``: 4 = flips of
    the E edges, 8 = flips of the non-E edges, 12 = all flips.
    """

    ep: int
    eo: int

    @property
    def n_ep(self) -> int:
        return {1: 1, 2: 24, 3: 70}[self.ep]

    @property
    def n_eo(self) -> int:
        return {4: 16, 8: 256, 12: 2048}[self.eo]

    @property
    def n_edge(self) -> int:
        return N_SLICE * self.n_ep * self.n_eo

    @property
    def name(self) -> str:
        return f"{self.ep}{self.eo:02d}"

    @property
    def defines(self) -> tuple[str, ...]:
        return (f"EDGE_EP={self.ep}", f"EDGE_EO={self.eo}")

    @classmethod
    def parse(cls, name: str) -> "EdgeKind":
        return cls(int(name[0]), int(name[1:]))


K208 = EdgeKind(2, 8)


def _combination_tables(bits: int, choose: int):
    rank = np.full(1 << bits, 0xFFFF, dtype=np.uint16)
    unrank = []
    for r, combo in enumerate(combinations(range(bits), choose)):
        mask = sum(1 << i for i in combo)
        rank[mask] = r
        unrank.append(mask)
    return rank, np.array(unrank, dtype=np.uint16)


CCOMB_RANK, CCOMB_UNRANK = _combination_tables(8, 4)
SLICE_RANK, SLICE_UNRANK = _combination_tables(12, 4)

# E-order: the 2-bit labels of the four E edges in ascending position order,
# packed little-endian (first E position in the low bits).
EORDER_RANK = np.full(256, 0xFF, dtype=np.uint8)
EORDER_UNRANK = np.zeros(24, dtype=np.uint8)
for _r, _perm in enumerate(permutations(range(4))):
    _key = sum(v << (2 * j) for j, v in enumerate(_perm))
    EORDER_RANK[_key] = _r
    EORDER_UNRANK[_r] = _key
TWIST_WEIGHTS = 3 ** np.arange(7)


def pack(state: CubieState) -> tuple[np.ndarray, np.ndarray]:
    corners = np.array([p | (o << 3) for p, o in zip(state.cp, state.co)], dtype=np.uint8)
    edges = np.array([p | (o << 4) for p, o in zip(state.ep, state.eo)], dtype=np.uint8)
    return corners, edges


def corner_raw(corners: np.ndarray) -> np.ndarray:
    """``twist * 70 + ccomb`` for packed corners of shape (..., 8)."""

    co = (corners[..., :7] >> 3).astype(np.int64)
    twist = co @ TWIST_WEIGHTS
    dmask = (((corners >> 2) & 1).astype(np.int64) << np.arange(8)).sum(axis=-1)
    return twist * N_CCOMB + CCOMB_RANK[dmask].astype(np.int64)


def edge_coord(edges: np.ndarray, kind: EdgeKind = K208) -> np.ndarray:
    """Edge coordinate for packed edges of shape (..., 12)."""

    edges = np.asarray(edges)
    flat = edges.reshape(-1, 12)
    out = np.empty(len(flat), dtype=np.int64)
    for row, e in enumerate(flat):
        emask = eorder = umask = eo_e = eo_n = 0
        ecount = ocount = 0
        for i in range(12):
            b = int(e[i])
            if b & 8:
                emask |= 1 << i
                eorder |= (b & 3) << (2 * ecount)
                eo_e |= (b >> 4) << ecount
                ecount += 1
            else:
                umask |= int((b & 15) < 4) << ocount
                eo_n |= (b >> 4) << ocount
                ocount += 1
        ep = {1: 0, 2: int(EORDER_RANK[eorder]), 3: int(CCOMB_RANK[umask])}[kind.ep]
        if kind.eo == 4:
            eo = eo_e
        elif kind.eo == 8:
            eo = eo_n
        else:
            eo = sum(((int(e[i]) >> 4) & 1) << i for i in range(11))
        out[row] = (int(SLICE_RANK[emask]) * kind.n_ep + ep) * kind.n_eo + eo
    return out.reshape(edges.shape[:-1])


def corners_from_raw(raw: np.ndarray) -> np.ndarray:
    """Representative packed corners for raw coordinates (vectorized)."""

    raw = np.asarray(raw, dtype=np.int64)
    twist, ccomb = np.divmod(raw, N_CCOMB)
    digits = (twist[..., None] // TWIST_WEIGHTS) % 3
    co = np.concatenate([digits, (-digits.sum(axis=-1, keepdims=True)) % 3], axis=-1)
    dmask = CCOMB_UNRANK[ccomb].astype(np.int64)
    is_d = (dmask[..., None] >> np.arange(8)) & 1
    # U cubies 0..3 fill non-D positions in order, D cubies 4..7 fill D positions.
    cp = np.where(is_d, 4 + np.cumsum(is_d, axis=-1) - 1, np.cumsum(1 - is_d, axis=-1) - 1)
    return (cp | (co << 3)).astype(np.uint8)


def edges_from_coord(coord: int, kind: EdgeKind = K208) -> np.ndarray:
    """Representative packed edges; matches the GPU ``unrank``."""

    rest, eo = divmod(int(coord), kind.n_eo)
    srank, ep = divmod(rest, kind.n_ep)
    emask = int(SLICE_UNRANK[srank])
    eorder = int(EORDER_UNRANK[ep]) if kind.ep == 2 else 0xE4
    umask = int(CCOMB_UNRANK[ep]) if kind.ep == 3 else 0x0F
    edges = np.zeros(12, dtype=np.uint8)
    ecount = ocount = n_u = n_d = 0
    for i in range(12):
        if emask >> i & 1:
            b = 8 | ((eorder >> (2 * ecount)) & 3)
            flip = (eo >> ecount) & 1 if kind.eo == 4 else 0
            ecount += 1
        else:
            if umask >> ocount & 1:
                b, n_u = n_u, n_u + 1
            else:
                b, n_d = 4 + n_d, n_d + 1
            flip = (eo >> ocount) & 1 if kind.eo == 8 else 0
            ocount += 1
        if kind.eo == 12:
            flip = (eo >> i) & 1 if i < 11 else bin(eo).count("1") & 1
        edges[i] = b | (flip << 4)
    return edges


# --- symmetry action on packed states ------------------------------------


@dataclass(frozen=True)
class EdgeSym:
    """Cubie-level edge action of a symmetry: ``M g M^-1``."""

    perm: np.ndarray      # position p -> position P(p); also relabels cubies
    perm_inv: np.ndarray
    flip: np.ndarray      # a(p): 1 if M swaps the two sticker slots of p


@dataclass(frozen=True)
class CornerSym:
    perm: np.ndarray
    perm_inv: np.ndarray
    shift: np.ndarray     # slot offset added at position p
    reflect: bool         # reflections reverse the sticker cycle


def _edge_sym(sym: Symmetry) -> EdgeSym:
    perm = sym.edge[0::2] // 2
    flip = sym.edge[0::2] % 2
    assert np.array_equal(sym.edge[1::2] // 2, perm)
    return EdgeSym(perm, np.argsort(perm), flip)


def _corner_sym(sym: Symmetry) -> CornerSym:
    perm = sym.corner[0::3] // 3
    shift = sym.corner[0::3] % 3
    second = sym.corner[1::3] % 3
    reflect = bool(((second - shift) % 3 == 2).all())
    assert reflect == (sym.det < 0)
    return CornerSym(perm, np.argsort(perm), shift, reflect)


def conj_edges(edges: np.ndarray, es: EdgeSym) -> np.ndarray:
    """Packed-edge ``M g M^-1`` (vectorized over leading axes)."""

    src = edges[..., es.perm_inv]
    cubie = src & 15
    o = src >> 4
    o_new = o ^ es.flip[es.perm_inv] ^ es.flip[cubie]
    return (es.perm[cubie] | (o_new << 4)).astype(np.uint8)


def conj_corners(corners: np.ndarray, cs: CornerSym) -> np.ndarray:
    """Packed-corner ``M g M^-1`` (vectorized over leading axes).

    Sticker k of cubie c sits in slot (k + o) of position p.  The symmetry
    maps slot j of p to slot ``s(p) +/- j`` of P(p), and relabels the
    sticker the same way using the cubie's home position.
    """

    src = corners[..., cs.perm_inv]
    cubie = (src & 7).astype(np.int64)
    o = (src >> 3).astype(np.int64)
    pos_shift = cs.shift[cs.perm_inv]
    cub_shift = cs.shift[cubie]
    sign = -1 if cs.reflect else 1
    # new slot of sticker k: pos_shift + sign*(k+o); new sticker index: cub_shift + sign*k
    o_new = (pos_shift + sign * o - cub_shift) % 3
    return (cs.perm[cubie] | (o_new << 3)).astype(np.uint8)


@lru_cache(maxsize=1)
def ud_actions() -> tuple[tuple[CornerSym, ...], tuple[EdgeSym, ...]]:
    syms = ud_symmetries()
    return tuple(_corner_sym(s) for s in syms), tuple(_edge_sym(s) for s in syms)


@dataclass(frozen=True)
class CornerClasses:
    classidx: np.ndarray   # raw -> class << 4 | sym, with conj(sym, raw) = rep(class)
    rep: np.ndarray        # class -> representative raw
    stab: np.ndarray       # class -> 16-bit mask of syms fixing the representative raw


@lru_cache(maxsize=1)
def corner_classes() -> CornerClasses:
    corner_syms, _ = ud_actions()
    raws = np.arange(N_CORNER_RAW)
    states = corners_from_raw(raws)
    images = np.stack([corner_raw(conj_corners(states, cs)) for cs in corner_syms])  # (16, N)
    orbit_min = images.min(axis=0)
    reps = np.unique(orbit_min)
    class_of_rep = np.full(N_CORNER_RAW, -1, dtype=np.int64)
    class_of_rep[reps] = np.arange(len(reps))
    sym = np.argmax(images == orbit_min[None, :], axis=0)
    classidx = (class_of_rep[orbit_min] << 4 | sym).astype(np.uint32)
    stab = ((images[:, reps] == reps[None, :]).astype(np.uint32) << np.arange(16)[:, None]).sum(axis=0)
    return CornerClasses(classidx, reps.astype(np.uint32), stab.astype(np.uint16))


def table_index(corners: np.ndarray, edges: np.ndarray, kind: EdgeKind = K208) -> int:
    cc = corner_classes()
    _, edge_syms = ud_actions()
    packed = int(cc.classidx[int(corner_raw(corners))])
    cls, s = packed >> 4, packed & 15
    return cls * kind.n_edge + int(edge_coord(conj_edges(edges, edge_syms[s]), kind))


def apply_move(corners: np.ndarray, edges: np.ndarray, move: int):
    m = MOVES_HTM[move]
    cp = np.array(m.cp)
    co = np.array(m.co)
    c = corners[..., cp]
    o = ((c >> 3) + co) % 3
    c = (c & 7) | (o << 3)
    e = edges[..., list(m.ep)] ^ (np.array(m.eo, dtype=np.uint8) << 4)
    return c.astype(np.uint8), e.astype(np.uint8)
