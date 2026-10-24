"""Batched optimal HTM solver: IDA* on the GPU with a resident pruning table."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from gpucube import coords
from gpucube.device import DeviceTables, module, pack_states
from gpucube.geometry import axis_rotation, move_map
from gpucube.table import block_words, load_device
from gpucube.cubie import MOVE_NAMES_HTM, MOVES_HTM, CubieState, apply_moves

JOB_DTYPE = np.dtype(
    [
        ("frame", np.uint32, (30,)),
        ("h", np.uint32, (6,)),
        ("bound", np.uint32),
        ("found", np.uint32),
        ("solution_length", np.uint32),
        ("solution", np.uint8, (24,)),
        ("nodes", np.uint64),
    ],
    align=True,
)


def allowed(last: int, move: int) -> bool:
    if last < 0:
        return True
    lf, mf = last // 3, move // 3
    return mf != lf and not (lf >= 3 and mf == lf - 3)


@lru_cache(maxsize=None)
def canonical_prefixes(length: int) -> np.ndarray:
    seqs = [()]
    for _ in range(length):
        seqs = [s + (m,) for s in seqs for m in range(18) if allowed(s[-1] if s else -1, m)]
    return np.array(seqs, dtype=np.uint8).reshape(len(seqs), length)


def search_const() -> bytes:
    rot = move_map(axis_rotation())
    frame_move = np.array([list(range(18)), rot, [rot[rot[m]] for m in range(18)]], dtype=np.uint8)
    allow = np.array([[allowed(last, m) for m in range(18)] for last in range(-1, 18)], dtype=np.uint8)
    return frame_move.tobytes() + allow.tobytes()


def frames(state: CubieState) -> list[CubieState]:
    """The three axis frames of ``state`` followed by their inverses."""

    rot = axis_rotation()
    once = rot.conjugate(state)
    axes = [state, once, rot.conjugate(once)]
    return axes + [f.inverse() for f in axes]


@dataclass
class Solution:
    moves: list[int]
    nodes: int
    seconds: float
    iteration_nodes: list[tuple[int, int]] = field(default_factory=list)  # (bound, nodes)

    def text(self) -> str:
        return " ".join(MOVE_NAMES_HTM[m] for m in self.moves)


class GpuSolver:
    def __init__(self, table_path: Path, prefix_len: int = 4, grid: int = 36 * 8, block: int = 256,
                 inverse: bool = True, collect_stats: bool = False, mode: str = "forward",
                 extra_tables: int = 2):
        """``mode``: "forward" (mod-3 table for the axis frames, block table for
        inverses), "multi" (block tables only: 208 plus ``extra_tables`` of
        112 and 304), or "bidir" (208 block table, bidirectional search)."""

        import cupy as cp

        self.cp = cp
        self.mode = mode
        self.dt = DeviceTables()
        n_entries = self.dt.n_classes * coords.N_EDGE
        n_words = (n_entries + 15) // 16
        self.inverse = inverse or mode in ("bidir", "multi")
        if mode in ("bidir", "multi"):
            self.table = None
        else:
            self.table = load_device(table_path, n_words)
        if self.inverse:
            self.blk = load_device(table_path.with_name(f"{table_path.stem}-blk.bin"), block_words(n_entries))
        else:
            self.blk = self.table
        self.extra = []
        if mode == "multi":
            for name in ("112", "304")[:extra_tables]:
                kind = coords.EdgeKind.parse(name)
                path = table_path.with_name(f"prune-ud-{name}-blk.bin")
                self.extra.append((kind, load_device(path, block_words(self.dt.n_classes * kind.n_edge))))
        self.sconst = cp.asarray(np.frombuffer(search_const(), dtype=np.uint8))
        self.prefix_len = prefix_len
        self.prefixes = {k: cp.asarray(canonical_prefixes(k)) for k in range(prefix_len + 1)}
        self.grid, self.block = grid, block
        mod = module()
        self.n_frames = 6 if self.inverse else 3
        # per-frame lookup counts [0:6], prune counts [8:14], children by budget 0,1,2,3+ [16:20]; slows the kernel
        self.stats = cp.zeros(24, dtype=cp.uint64) if collect_stats else None
        if mode == "multi":
            self.k_search = mod.get_function(f"search_multi{len(self.extra)}")
        else:
            self.k_search = mod.get_function("search_bidir" if mode == "bidir" else f"search{self.n_frames}")
        self.k_root = mod.get_function("root_distance")
        self.k_block = mod.get_function("probe_block")

    def root_h(self, states: list[CubieState]) -> np.ndarray:
        cp = self.cp
        all_frames = [f for s in states for f in frames(s)]
        packed = [coords.pack(f) for f in all_frames]
        words = pack_states(np.stack([p[0] for p in packed]), np.stack([p[1] for p in packed]))
        n = len(all_frames)
        out = cp.zeros(n, dtype=cp.uint32)
        if self.mode in ("bidir", "multi"):
            d_words = cp.asarray(words)
            self.k_block(((n + 127) // 128,), (128,),
                         (self.dt.tables, self.blk, self.dt.classidx, d_words, np.int32(n), out))
            for kind, blk in self.extra:
                extra = cp.zeros(n, dtype=cp.uint32)
                module(*kind.defines).get_function("probe_block")(
                    ((n + 127) // 128,), (128,), (self.dt.tables, blk, self.dt.classidx, d_words, np.int32(n), extra))
                out = cp.maximum(out, extra)
        else:
            self.k_root(((n + 127) // 128,), (128,),
                        (self.dt.tables, self.table, self.dt.classidx, cp.asarray(words), np.int32(n), out))
        return out.get().reshape(len(states), 6), words.reshape(len(states), 30)

    def solve(self, states: list[CubieState], log=None) -> list[Solution]:
        cp = self.cp
        t0 = time.perf_counter()
        h, words = self.root_h(states)
        jobs = np.zeros(len(states), dtype=JOB_DTYPE)
        jobs["frame"] = words
        jobs["h"] = h
        jobs["bound"] = h[:, : self.n_frames].max(axis=1)
        results: list[Solution | None] = [None] * len(states)
        for i, s in enumerate(states):
            if s == CubieState():
                results[i] = Solution([], 0, 0.0)
        d_jobs = cp.asarray(jobs.view(np.uint8)).view(cp.uint8)
        counter = cp.zeros(1, dtype=cp.uint64)
        history: list[list[tuple[int, int]]] = [[] for _ in states]
        prev_nodes = np.zeros(len(states), dtype=np.uint64)
        while True:
            active = [i for i, r in enumerate(results) if r is None]
            if not active:
                break
            host_jobs = d_jobs.get().view(JOB_DTYPE)
            groups: dict[int, list[int]] = {}
            for i in active:
                groups.setdefault(min(int(host_jobs["bound"][i]), self.prefix_len), []).append(i)
            for plen, ids in groups.items():
                counter.fill(0)
                pre = self.prefixes[plen]
                if self.mode == "multi":
                    xs = [blk for _, blk in self.extra] + [self.blk] * (2 - len(self.extra))
                    tables = (self.blk, xs[0], xs[1])
                elif self.mode == "bidir":
                    tables = (self.blk,)
                else:
                    tables = (self.table, self.blk)
                args = (self.dt.tables, self.sconst, *tables, self.dt.classidx, d_jobs,
                        cp.asarray(np.array(ids, dtype=np.int32)), np.int32(len(ids)),
                        pre, np.int32(len(pre)), np.int32(plen), counter)
                if self.mode != "multi":
                    args += (self.stats if self.stats is not None else np.uint64(0),)
                self.k_search((self.grid,), (self.block,), args)
            cp.cuda.Device().synchronize()
            host_jobs = d_jobs.get().view(JOB_DTYPE)
            now = time.perf_counter() - t0
            for i in active:
                history[i].append((int(host_jobs["bound"][i]), int(host_jobs["nodes"][i] - prev_nodes[i])))
                prev_nodes[i] = host_jobs["nodes"][i]
                if host_jobs["found"][i]:
                    n = int(host_jobs["solution_length"][i])
                    results[i] = Solution(list(host_jobs["solution"][i][:n]), int(host_jobs["nodes"][i]), now, history[i])
                else:
                    host_jobs["bound"][i] += 1
            d_jobs[:] = cp.asarray(host_jobs.view(np.uint8))
            if log:
                log(f"{now:8.2f}s solved {sum(r is not None for r in results)}/{len(states)}")
        for state, sol in zip(states, results):
            if apply_moves(state, [MOVES_HTM[m] for m in sol.moves]) != CubieState():
                raise AssertionError("solution does not solve the cube")
        return results


def parse_moves(text: str) -> list[int]:
    index = {name: i for i, name in enumerate(MOVE_NAMES_HTM)}
    return [index[tok] for tok in text.split()]
