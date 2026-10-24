"""Device-side constant tables and kernel loading."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from gpucube import coords
from gpucube.geometry import ud_symmetries
from gpucube.cubie import MOVES_HTM

KERNEL_PATH = Path(__file__).with_name("kernels.cu")


def _selector(sources: list[int]) -> int:
    return sum((src & 7) << (4 * i) for i, src in enumerate(sources))


def build_tables() -> bytes:
    c_sel = np.zeros((18, 2), np.uint32)
    c_add = np.zeros((18, 2), np.uint32)
    e_sel01 = np.zeros((18, 3), np.uint32)
    e_sel2 = np.zeros((18, 3), np.uint32)
    e_mask = np.zeros((18, 3), np.uint32)
    e_flip = np.zeros((18, 3), np.uint32)
    lc = np.zeros((18, 2), np.uint32)
    le = np.zeros((18, 3), np.uint32)
    for m, move in enumerate(MOVES_HTM):
        for w in range(2):
            c_sel[m, w] = _selector(list(move.cp[4 * w:4 * w + 4]))
            c_add[m, w] = sum(move.co[4 * w + j] << (3 + 8 * j) for j in range(4))
        for w in range(3):
            src = move.ep[4 * w:4 * w + 4]
            e_sel01[m, w] = _selector([s if s < 8 else 0 for s in src])
            e_sel2[m, w] = _selector([s - 8 if s >= 8 else 0 for s in src])
            e_mask[m, w] = sum((0xFF if s >= 8 else 0) << (8 * j) for j, s in enumerate(src))
            e_flip[m, w] = sum(move.eo[4 * w + j] << (4 + 8 * j) for j in range(4))
        for w in range(2):
            lc[m, w] = sum((move.cp[4 * w + j] | move.co[4 * w + j] << 3) << (8 * j) for j in range(4))
        for w in range(3):
            le[m, w] = sum((move.ep[4 * w + j] | move.eo[4 * w + j] << 4) << (8 * j) for j in range(4))

    _, edge_syms = coords.ud_actions()
    syms = ud_symmetries()
    cs_sel01 = np.zeros((16, 3), np.uint32)
    cs_sel2 = np.zeros((16, 3), np.uint32)
    cs_mask = np.zeros((16, 3), np.uint32)
    cs_flip = np.zeros((16, 3), np.uint32)
    cs_rel = np.zeros((16, 3), np.uint32)
    for k, es in enumerate(edge_syms):
        for w in range(3):
            src = [int(x) for x in es.perm_inv[4 * w:4 * w + 4]]
            cs_sel01[k, w] = _selector([x if x < 8 else 0 for x in src])
            cs_sel2[k, w] = _selector([x - 8 if x >= 8 else 0 for x in src])
            cs_mask[k, w] = sum((0xFF if x >= 8 else 0) << (8 * j) for j, x in enumerate(src))
            cs_flip[k, w] = sum(int(es.flip[x]) << (4 + 8 * j) for j, x in enumerate(src))
            cs_rel[k, w] = sum((int(es.perm[4 * w + j]) | int(es.flip[4 * w + j]) << 4) << (8 * j) for j in range(4))
    pext6 = np.zeros((64, 64), np.uint8)
    for mask in range(64):
        bits = [i for i in range(6) if mask >> i & 1]
        for value in range(64):
            pext6[mask, value] = sum(((value >> b) & 1) << j for j, b in enumerate(bits))
    compact4 = np.zeros((16, 256), np.uint8)
    for mask in range(16):
        fields = [i for i in range(4) if mask >> i & 1]
        for value in range(256):
            compact4[mask, value] = sum(((value >> (2 * f)) & 3) << (2 * j) for j, f in enumerate(fields))
    s_src = np.zeros((16, 12), np.uint8)
    s_rel = np.zeros((16, 16), np.uint8)
    s_fsrc = np.zeros((16, 12), np.uint8)
    s_fcub = np.zeros((16, 16), np.uint8)
    s_mul = np.zeros((16, 16), np.uint8)
    for k, es in enumerate(edge_syms):
        s_src[k] = es.perm_inv
        s_rel[k, :12] = es.perm
        s_fsrc[k] = es.flip[es.perm_inv]
        s_fcub[k, :12] = es.flip
    for t in range(16):
        for s in range(16):
            product = syms[t].matrix @ syms[s].matrix
            s_mul[t, s] = next(i for i, u in enumerate(syms) if np.array_equal(u.matrix, product))

    parts = [
        c_sel, c_add, e_sel01, e_sel2, e_mask, e_flip, lc, le,
        cs_sel01, cs_sel2, cs_mask, cs_flip, cs_rel,
        s_src, s_rel, s_fsrc, s_fcub, s_mul,
        coords.CCOMB_RANK.astype(np.uint16),
        coords.CCOMB_UNRANK.astype(np.uint8),
        coords.EORDER_RANK.astype(np.uint8),
        coords.EORDER_UNRANK.astype(np.uint8),
        coords.SLICE_UNRANK.astype(np.uint16),
        coords.SLICE_RANK.astype(np.uint16),
        pext6,
        compact4,
    ]
    blob = bytearray()
    for part in parts:
        align = part.dtype.itemsize
        blob += b"\0" * (-len(blob) % align)
        blob += part.tobytes()
    blob += b"\0" * (-len(blob) % 4)
    return bytes(blob)


@lru_cache(maxsize=None)
def module(*defines: str):
    import cupy as cp

    options = ("--std=c++17",) + tuple(f"-D{d}" for d in defines)
    if os.environ.get("GPUCUBE_LINEINFO"):
        options += ("-lineinfo",)  # source attribution in Nsight Compute
    return cp.RawModule(code=KERNEL_PATH.read_text(), options=options)


class DeviceTables:
    """Constant tables resident on the GPU."""

    def __init__(self):
        import cupy as cp

        blob = build_tables()
        size = cp.zeros(1, dtype=cp.uint32)
        module().get_function("tables_size")((1,), (1,), (size,))
        if int(size[0]) != len(blob):
            raise AssertionError(f"Tables layout mismatch: device {int(size[0])}, host {len(blob)}")
        self.tables = cp.asarray(np.frombuffer(blob, dtype=np.uint8))
        cc = coords.corner_classes()
        self.n_classes = len(cc.rep)
        self.classidx = cp.asarray(cc.classidx)
        self.class_rep = cp.asarray(cc.rep)
        self.class_stab = cp.asarray(cc.stab)


def pack_states(corners: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """(n, 8) and (n, 12) packed bytes -> (n, 5) uint32 words."""

    words = np.concatenate([corners, edges], axis=-1).astype(np.uint8)
    return np.ascontiguousarray(words).view(np.uint32).reshape(-1, 5)
