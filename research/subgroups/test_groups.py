"""Checks for the generic survey tables (run: PYTHONPATH=src:research/subgroups pytest research/subgroups).

For every candidate: unrank and index are inverse, and the index is constant on cosets: relabelling
the E edges by an element of the group (and the non-E edges by the mode's group) keeps the index,
also after any move.
"""
import random

import cupy as cp
import numpy as np
import pytest

import groups
import run_gen
from cuda_cube_solver import coords, device
from cuda_cube_solver.cubie import MOVES_HTM, CubieState, apply_moves

KINDS = run_gen.kinds(2048, 12500)


def relabel(state, k, rng, mode):
    """Left-multiply: E label x -> p[x] with flip f[x]; non-E labels by the mode's group."""
    kp, kf = groups.ELEMS[k]
    perm = list(range(8))
    if mode == 1:
        rng.shuffle(perm)
    else:
        u, d = list(range(4)), list(range(4, 8))
        rng.shuffle(u)
        rng.shuffle(d)
        perm = (d + u) if mode == 3 and rng.random() < 0.5 else (u + d)
    ep, eo = [], []
    for c, o in zip(state.ep, state.eo):
        if c >= 8:
            ep.append(8 + kp[c - 8])
            eo.append(o ^ kf[c - 8])
        else:
            ep.append(perm[c])
            eo.append(o)
    return CubieState(state.cp, state.co, tuple(ep), tuple(eo))


def indices(dt, kind, states, move=-1):
    packed = [coords.pack(s) for s in states]
    words = cp.asarray(device.pack_states(np.stack([p[0] for p in packed]), np.stack([p[1] for p in packed])))
    n = len(states)
    out = cp.zeros(n, dtype=cp.uint64)
    tmp = cp.zeros((n, 5), dtype=cp.uint32)
    device.module(*kind.defines).get_function("probe_index")(
        ((n + 127) // 128,), (128,), (dt.tables, dt.classidx, words, np.int32(n), np.int32(move), out, tmp))
    return out.get()


@pytest.mark.parametrize("kind", KINDS, ids=lambda k: k.name)
def test_unrank_and_cosets(kind):
    dt = device.DeviceTables(kind.gen)
    rng = random.Random(kind.gid * 7 + kind.mode)
    total = dt.n_classes * kind.n_edge
    idx = np.array([0, total - 1] + [rng.randrange(total) for _ in range(500)], dtype=np.uint64)
    back = cp.zeros(len(idx), dtype=cp.uint64)
    device.module(*kind.defines).get_function("probe_unrank")(
        ((len(idx) + 127) // 128,), (128,), (dt.tables, dt.classidx, dt.class_rep, cp.asarray(idx),
                                            np.int32(len(idx)), back))
    assert np.array_equal(back.get(), idx)
    states = [apply_moves(CubieState(), [MOVES_HTM[rng.randrange(18)] for _ in range(30)]) for _ in range(40)]
    group = sorted(kind.group)
    others = [relabel(s, rng.choice(group), rng, kind.mode) for s in states]
    for move in [-1] + list(range(18)):
        assert np.array_equal(indices(dt, kind, states, move), indices(dt, kind, others, move))
