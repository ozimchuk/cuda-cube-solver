"""Batched optimal HTM solver: IDA* on the GPU with a resident pruning table."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from cuda_cube_solver import coords, kinds
from cuda_cube_solver.device import DeviceTables, module, pack_states
from cuda_cube_solver.geometry import axis_rotation, move_map
from cuda_cube_solver.table import b3_words, block_words, load_device
from cuda_cube_solver.cubie import MOVE_NAMES_HTM, MOVES_HTM, CubieState, apply_moves

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
    allow = np.array([[allowed(last, m) for m in range(18)] for last in range(-1, 18)], dtype=bool)
    seqs, last = np.zeros((1, 0), dtype=np.uint8), np.zeros(1, dtype=np.int64)
    for _ in range(length):
        i, m = np.nonzero(allow[last])  # row-major: each sequence's children in move order
        seqs = np.concatenate([seqs[i], m[:, None].astype(np.uint8)], axis=1)
        last = m + 1
    return seqs


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


class TableNotFound(FileNotFoundError):
    """A pruning table file is missing."""


def _table_file(tables: Path, name: str) -> Path:
    path = tables / name
    if not path.exists():
        m = re.fullmatch(r"prune-ud-(\d+|[a-z0-9]+)(?:-blk)?(-b3)?\.bin", name)
        hint = f"; build it with `cuda-cube-solver build-table --table {m[1]}{m[2] or ''} --out {tables}`" if m else ""
        raise TableNotFound(f"{path} not found{hint}")
    return path


class NoTableFits(RuntimeError):
    """No table preset fits in the GPU memory budget."""


def open_solver(tables: Path = Path("tables"), memory="auto", table: str | None = None, build: bool = True,
                log=print) -> "GpuSolver":
    """A solver with the fastest table preset that fits ``memory`` (GB, or "auto": the free VRAM),
    or the preset named ``table`` (see ``kinds.PRESETS``).  A missing table is built first
    (1-3 min) unless ``build`` is False."""

    from cuda_cube_solver import table as tables_mod

    tables = Path(tables)
    if tables.suffix == ".bin":  # a table file inside the directory, as GpuSolver accepts
        tables = tables.parent
    preset = kinds.pick(tables, memory, table)[0]
    if preset is None:
        raise NoTableFits("no table fits in that GPU memory (the smallest, 304, needs 1.8 GB)")
    if not (tables / preset.file).exists():
        if not build:
            raise TableNotFound(f"{tables / preset.file} not found")
        log(f"building table {preset.name} ({preset.table_bytes / 1e9:.2f} GB, needs "
            f"{preset.build_bytes / 1e9:.1f} GB VRAM, 1-3 min) in {tables}/")
        log(f"wrote {tables_mod.build_preset(tables, preset, log=lambda m: log(f'  {m}'))}")
    return GpuSolver(tables, kind=preset.kind, table_format=preset.format)


class GpuSolver:
    def __init__(self, tables: Path = Path("tables"), prefix_len: int = 4, grid: int | None = None, block: int = 256,
                 inverse: bool = True, collect_stats: bool = False, mode: str = "step",
                 extra_tables: int = 2, step_extra: int = 0, table_format: str = "block",
                 step_defines: tuple[str, ...] = (), kind="208",
                 deeper_prefix_at: tuple[float, ...] = (5e7, 3e9), extra_kind=None):
        """``mode``: "step" (default: step-machine kernel, block table only),
        "forward" (mod-3 table for the axis frames, block table for inverses),
        "multi" (block tables only: 208 plus ``extra_tables`` of 112 and 304),
        or "bidir" (208 block table, bidirectional search).  ``kind`` picks another
        main table variant (step mode, no extra tables): a ``kinds.get`` name such as "d4t", or an
        object with ``name``, ``n_edge``, ``defines`` and optionally ``gen`` (research/subgroups).
        ``extra_kind``: a second table of a generic kind (e.g. "s4", same format as the main
        one), looked up only when the main table's frames do not prune (step mode)."""

        import cupy as cp

        # A tables directory; a path to a table file inside it is accepted too.
        tables = Path(tables)
        if tables.suffix == ".bin":
            tables = tables.parent
        self.cp = cp
        self.mode = mode
        self.kind = kinds.get(kind) if isinstance(kind, str) else kind
        kind = self.kind.name
        self.dt = DeviceTables(getattr(self.kind, "gen", None))
        if (kind != "208" or table_format == "b3") and (mode != "step" or table_format == "bit" or step_extra):
            raise ValueError("another table kind or b3 needs mode='step' and no extra tables")
        if extra_kind is not None and (mode != "step" or table_format == "bit" or step_extra):
            raise ValueError("extra_kind needs mode='step' and no step_extra")
        n_entries = self.dt.n_classes * self.kind.n_edge
        n_words = (n_entries + 15) // 16
        self.inverse = inverse or mode in ("bidir", "multi", "step")
        if mode in ("bidir", "multi", "step"):
            self.table = None
        else:
            self.table = load_device(_table_file(tables, "prune-ud-208.bin"), n_words)
        self.table_format = table_format
        if table_format == "bit":
            # "blk" holds the threshold-bit table; the 304 block table is the fallback.
            if mode != "step" or step_extra:
                raise ValueError("table_format='bit' needs mode='step' without extra tables")
            self.blk = load_device(_table_file(tables, "prune-ud-208-bit11.bin"), (n_entries + 31) // 32)
            k304 = coords.EdgeKind(3, 4)
            self.fallback = load_device(_table_file(tables, "prune-ud-304-blk.bin"),
                                        block_words(self.dt.n_classes * k304.n_edge))
        elif table_format == "b3":
            self.blk = load_device(_table_file(tables, f"prune-ud-{kind}-b3.bin"), b3_words(n_entries))
        elif self.inverse:
            self.blk = load_device(_table_file(tables, f"prune-ud-{kind}-blk.bin"), block_words(n_entries))
        else:
            self.blk = self.table
        self.extra = []
        if mode in ("multi", "step"):
            n_extra = extra_tables if mode == "multi" else step_extra
            for name in ("112", "304")[:n_extra]:
                ek = coords.EdgeKind.parse(name)
                path = _table_file(tables, f"prune-ud-{name}-blk.bin")
                self.extra.append((ek, load_device(path, block_words(self.dt.n_classes * ek.n_edge))))
        if extra_kind is not None:
            ek = kinds.get(extra_kind) if isinstance(extra_kind, str) else extra_kind
            if getattr(ek, "gen", None) is None or "EDGE_EO=1" not in ek.defines:
                raise ValueError("extra_kind must be a generic kind with non-E flips (kinds.GroupKind)")
            n_extra = self.dt.n_classes * ek.n_edge
            if table_format == "b3":
                arr = load_device(_table_file(tables, f"prune-ud-{ek.name}-b3.bin"), b3_words(n_extra))
            else:
                arr = load_device(_table_file(tables, f"prune-ud-{ek.name}-blk.bin"), block_words(n_extra))
            self.extra.append((ek, arr))
        self.extra_kind = self.extra[0][0] if extra_kind is not None else None
        self.sconst = cp.asarray(np.frombuffer(search_const(), dtype=np.uint8))
        self.prefix_len = prefix_len
        self.prefixes = {k: cp.asarray(canonical_prefixes(k)) for k in range(prefix_len + 1)}
        # Work items are the canonical move sequences of one length.  A round of billions of
        # nodes needs finer items, or a few large subtrees keep most of the GPU idle (superflip:
        # 0.5 G nodes/s with 4-move items, 2.4 G/s with 6); a small round pays for every item.
        # So a cube whose last round reached these node counts gets longer prefixes.
        self.deeper_prefix_at = deeper_prefix_at
        # Persistent kernel: 8 blocks per SM is enough to keep every SM fed on any device.
        sms = cp.cuda.Device().attributes["MultiProcessorCount"]
        self.grid, self.block = grid or 8 * sms, block
        mod = module()
        self.n_frames = 6 if self.inverse else 3
        # per-frame lookup counts [0:6], prune counts [8:14], children by budget 0,1,2,3+ [16:20]; slows the kernel
        self.stats = cp.zeros(64, dtype=cp.uint64) if collect_stats else None
        if mode == "multi":
            self.k_search = mod.get_function(f"search_multi{len(self.extra)}")
        elif mode == "step":
            defines = ("STEP_ALL_BLOCK", f"STEP_EXTRA={len(self.extra)}") + tuple(step_defines)
            if kind != "208":
                defines += self.kind.defines
            if table_format == "b3":
                defines += ("TABLE_B3",)
            if table_format == "bit":
                defines += ("TABLE_BIT",)
            if self.extra_kind is not None:
                defines += ("EXTRA1_KIND=3", f"EXTRA_GEN_NEC={self.extra_kind.nec}")
            step_mod = module(*defines)
            if self.extra_kind is not None:
                gen_class = cp.ndarray((384,), cp.uint16, step_mod.get_global("EXTRA_GEN_CLASS"))
                gen_class[...] = cp.asarray(self.extra_kind.gen[0])
            self.k_search = step_mod.get_function("search_step")
        else:
            self.k_search = mod.get_function("search_bidir" if mode == "bidir" else f"search{self.n_frames}")
        self.k_root = mod.get_function("root_distance")
        probe_defines = (self.kind.defines if kind != "208" else ()) + (("TABLE_B3",) if table_format == "b3" else ())
        self.k_block = module(*probe_defines).get_function("probe_block")

    def root_h(self, states: list[CubieState]) -> np.ndarray:
        cp = self.cp
        all_frames = [f for s in states for f in frames(s)]
        packed = [coords.pack(f) for f in all_frames]
        words = pack_states(np.stack([p[0] for p in packed]), np.stack([p[1] for p in packed]))
        n = len(all_frames)
        out = cp.zeros(n, dtype=cp.uint32)
        if self.table_format == "bit":
            module().get_function("probe_bit")(((n + 127) // 128,), (128,),
                                               (self.dt.tables, self.dt.classidx, self.blk, self.fallback,
                                                cp.asarray(words), np.int32(n), out))
        elif self.mode in ("bidir", "multi", "step"):
            d_words = cp.asarray(words)
            self.k_block(((n + 127) // 128,), (128,),
                         (self.dt.tables, self.blk, self.dt.classidx, d_words, np.int32(n), out))
            for kind, blk in self.extra:
                extra = cp.zeros(n, dtype=cp.uint32)
                gen = getattr(kind, "gen", None)
                tables = DeviceTables(gen).tables if gen is not None else self.dt.tables  # its own classes
                fmt = ("TABLE_B3",) if self.table_format == "b3" else ()
                module(*kind.defines, *fmt).get_function("probe_block")(
                    ((n + 127) // 128,), (128,), (tables, blk, self.dt.classidx, d_words, np.int32(n), extra))
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
                last = history[i][-1][1] if history[i] else 0
                plen = self.prefix_len + sum(last >= t for t in self.deeper_prefix_at)
                groups.setdefault(min(int(host_jobs["bound"][i]), plen), []).append(i)
            for plen, ids in groups.items():
                counter.fill(0)
                if plen not in self.prefixes:
                    self.prefixes[plen] = cp.asarray(canonical_prefixes(plen))
                pre = self.prefixes[plen]
                if self.mode == "multi":
                    xs = [blk for _, blk in self.extra] + [self.blk] * (2 - len(self.extra))
                    tables = (self.blk, xs[0], xs[1])
                elif self.mode == "bidir":
                    tables = (self.blk,)
                elif self.mode == "step" and self.table_format == "bit":
                    tables = (self.blk, self.blk, self.blk, self.fallback)
                elif self.mode == "step":
                    xs = [blk for _, blk in self.extra] + [self.blk] * (2 - len(self.extra))
                    tables = (self.blk, self.blk, xs[0], xs[1])
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
