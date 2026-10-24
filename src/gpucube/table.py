"""Breadth-first generation of the mod-3 pruning table on the GPU."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from gpucube import coords
from gpucube.device import DeviceTables, module

GRID = 36 * 32
BLOCK = 256
BLOCK_ENTRIES = 124
CHUNK_WORDS = 1 << 26


def save_device(array, path: Path) -> None:
    """Write a device array to disk in chunks (keeps host memory small)."""

    with open(path, "wb") as fh:
        for start in range(0, len(array), CHUNK_WORDS):
            array[start:start + CHUNK_WORDS].get().tofile(fh)


def load_device(path: Path, n_words: int):
    import cupy as cp

    out = cp.empty(n_words, dtype=cp.uint32)
    with open(path, "rb") as fh:
        for start in range(0, n_words, CHUNK_WORDS):
            chunk = np.fromfile(fh, dtype=np.uint32, count=min(CHUNK_WORDS, n_words - start))
            out[start:start + len(chunk)].set(chunk)
    if path.stat().st_size != n_words * 4:
        raise ValueError(f"{path}: size {path.stat().st_size}, expected {n_words * 4}")
    return out


def block_words(n_entries: int) -> int:
    return -(-n_entries // BLOCK_ENTRIES) * 8


def generate(dt: DeviceTables, kind: coords.EdgeKind = coords.K208, log=print):
    """Return (mod-3 table, block table, per-depth counts).

    Counts are table entries.  The block table stores, per 32-byte sector,
    a minimum depth and 124 two-bit offsets from it (3 = at least +3).
    """

    import cupy as cp

    n_entries = dt.n_classes * kind.n_edge
    n_words = (n_entries + 15) // 16
    table = cp.full(n_words, 0xFFFFFFFF, dtype=cp.uint32)
    blk = cp.zeros(block_words(n_entries), dtype=cp.uint32)
    blk[::8] = 0xFF
    mod = module(*kind.defines)
    counter32 = cp.zeros(1, dtype=cp.uint32)
    mod.get_function("seed_solved")((1,), (32,), (dt.tables, table, blk, dt.classidx, dt.class_stab, counter32))
    counts = [int(counter32[0])]
    remaining = n_entries - counts[0]
    forward = mod.get_function("expand_forward")
    backward = mod.get_function("expand_backward")
    counter = cp.zeros(1, dtype=cp.uint64)
    depth = 0
    while remaining:
        counter.fill(0)
        use_forward = counts[-1] < remaining // 4
        t = time.perf_counter()
        if use_forward:
            forward((GRID,), (BLOCK,), (dt.tables, table, blk, dt.classidx, dt.class_rep, dt.class_stab,
                                        np.uint64(n_words), np.uint64(n_entries), np.uint32(depth), counter))
        else:
            backward((GRID,), (BLOCK,), (dt.tables, table, blk, dt.classidx, dt.class_rep,
                                         np.uint64(n_words), np.uint64(n_entries), np.uint32(depth), counter))
        cp.cuda.Device().synchronize()
        found = int(counter[0])
        log(f"depth {depth + 1:2d} {'fwd' if use_forward else 'bwd'} found {found:>14,d} "
            f"({time.perf_counter() - t:7.1f}s)")
        if found == 0:
            raise RuntimeError(f"{remaining} entries unreachable")
        counts.append(found)
        remaining -= found
        depth += 1
    return table, blk, counts


def histogram(table, n_entries: int) -> list[int]:
    import cupy as cp

    counts = cp.zeros(4, dtype=cp.uint64)
    module().get_function("count_values")((GRID,), (BLOCK,), (table, np.uint64(len(table)), np.uint64(n_entries), counts))
    return [int(v) for v in counts.get()]


def build(out_dir: Path, kind: coords.EdgeKind = coords.K208, log=print) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = DeviceTables()
    t = time.perf_counter()
    table, blk, counts = generate(dt, kind, log)
    elapsed = time.perf_counter() - t
    n_entries = dt.n_classes * kind.n_edge
    hist = histogram(table, n_entries)
    expected = [sum(counts[d] for d in range(r, len(counts), 3)) for r in range(3)] + [0]
    if hist != expected:
        raise AssertionError(f"histogram {hist} != depth counts {expected}")
    max_min = max(int((blk[start:start + CHUNK_WORDS:8] & 0xFF).max()) for start in range(0, len(blk), CHUNK_WORDS))
    if max_min >= len(counts):
        raise AssertionError("block table has an unset block minimum")
    path = out_dir / f"prune-ud-{kind.name}.bin"
    save_device(table, path)
    save_device(blk, out_dir / f"prune-ud-{kind.name}-blk.bin")
    meta = {
        "coordinate": f"UD corner class (twist x 8C4, 16 syms) x slice 12C4 x EP{kind.ep} x EO{kind.eo}",
        "entries": n_entries,
        "classes": dt.n_classes,
        "encoding": "2-bit distance mod 3, 16 entries per little-endian uint32",
        "block_table": f"prune-ud-{kind.name}-blk.bin: per 32-byte sector, u8 minimum depth then 124 two-bit offsets (3 = at least +3)",
        "depth_counts": counts,
        "seconds": round(elapsed, 1),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return path
