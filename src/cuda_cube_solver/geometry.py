"""Sticker-level geometry for the 3x3 cube: the 48 symmetries and their cubie actions.

Symmetries are signed permutation matrices acting on 3D sticker slots
(x = R, y = U, z = F).  Conjugating a state by a symmetry ``M`` is
``g' = M g M^-1`` on sticker slots, which is valid for reflections as
well as rotations.  Everything is validated against ``model.MOVES_HTM``.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations, product

import numpy as np

from cuda_cube_solver.cubie import CORNER_NAMES, EDGE_NAMES, MOVES_HTM, CubieState

FACE_NORMAL = {
    "U": (0, 1, 0), "R": (1, 0, 0), "F": (0, 0, 1),
    "D": (0, -1, 0), "L": (-1, 0, 0), "B": (0, 0, -1),
}


def _center(name: str) -> tuple[int, ...]:
    return tuple(sum(axis) for axis in zip(*(FACE_NORMAL[f] for f in name)))


# Slot (position, k) holds sticker k of the home ordering; corners first.
CORNER_SLOTS = [(_center(n), FACE_NORMAL[f]) for n in CORNER_NAMES for f in n]
EDGE_SLOTS = [(_center(n), FACE_NORMAL[f]) for n in EDGE_NAMES for f in n]
_CORNER_INDEX = {slot: i for i, slot in enumerate(CORNER_SLOTS)}
_EDGE_INDEX = {slot: i for i, slot in enumerate(EDGE_SLOTS)}


def cubie_to_slots(state: CubieState) -> tuple[np.ndarray, np.ndarray]:
    """Return slot -> home-slot maps (24 corner slots, 24 edge slots)."""

    corners = np.empty(24, dtype=np.int64)
    edges = np.empty(24, dtype=np.int64)
    for i in range(8):
        for k in range(3):
            corners[3 * i + (k + state.co[i]) % 3] = 3 * state.cp[i] + k
    for i in range(12):
        for k in range(2):
            edges[2 * i + (k + state.eo[i]) % 2] = 2 * state.ep[i] + k
    return corners, edges


def slots_to_cubie(corners: np.ndarray, edges: np.ndarray) -> CubieState:
    cp, co, ep, eo = [], [], [], []
    for i in range(8):
        content = int(corners[3 * i])
        cp.append(content // 3)
        co.append((-(content % 3)) % 3)
    for i in range(12):
        content = int(edges[2 * i])
        ep.append(content // 2)
        eo.append(content % 2)
    return CubieState(tuple(cp), tuple(co), tuple(ep), tuple(eo))


def _apply_matrix(matrix: np.ndarray, vector: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(v) for v in matrix @ np.array(vector))


class Symmetry:
    """A cube symmetry as permutations of corner and edge sticker slots."""

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        self.det = int(round(np.linalg.det(matrix)))
        self.corner = np.array(
            [_CORNER_INDEX[(_apply_matrix(matrix, c), _apply_matrix(matrix, n))] for c, n in CORNER_SLOTS]
        )
        self.edge = np.array(
            [_EDGE_INDEX[(_apply_matrix(matrix, c), _apply_matrix(matrix, n))] for c, n in EDGE_SLOTS]
        )
        self.corner_inv = np.argsort(self.corner)
        self.edge_inv = np.argsort(self.edge)

    def preserves_ud(self) -> bool:
        return abs(int(self.matrix[1, 1])) == 1

    def conjugate_slots(self, corners: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``M g M^-1``; accepts a trailing axis of 24 slots (vectorized)."""

        return (
            self.corner[corners[..., self.corner_inv]],
            self.edge[edges[..., self.edge_inv]],
        )

    def conjugate(self, state: CubieState) -> CubieState:
        return slots_to_cubie(*self.conjugate_slots(*cubie_to_slots(state)))


@lru_cache(maxsize=1)
def all_symmetries() -> tuple[Symmetry, ...]:
    syms = []
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.int64)
            for row, col in enumerate(perm):
                matrix[row, col] = signs[row]
            syms.append(Symmetry(matrix))
    identity = [s for s in syms if np.array_equal(s.matrix, np.eye(3, dtype=np.int64))]
    syms.remove(identity[0])
    return (identity[0], *syms)


@lru_cache(maxsize=1)
def ud_symmetries() -> tuple[Symmetry, ...]:
    """The 16 symmetries preserving the UD axis, identity first."""

    syms = tuple(s for s in all_symmetries() if s.preserves_ud())
    assert len(syms) == 16
    return syms


@lru_cache(maxsize=1)
def axis_rotation() -> Symmetry:
    """120-degree rotation about the URF diagonal: x->y->z->x (R->U->F->R)."""

    matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.int64)
    return Symmetry(matrix)


def compose_slots(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]):
    """Slot-level ``compose(a, b)``: apply ``b`` after ``a``."""

    return a[0][b[0]], a[1][b[1]]


def move_map(sym: Symmetry) -> list[int]:
    """Index of ``M m M^-1`` for every HTM move ``m``; raises if not closed."""

    lookup = {MOVES_HTM[i]: i for i in range(18)}
    return [lookup[sym.conjugate(MOVES_HTM[m])] for m in range(18)]


def validate() -> None:
    for state in MOVES_HTM:
        assert slots_to_cubie(*cubie_to_slots(state)) == state
    for sym in all_symmetries():
        mapping = move_map(sym)
        assert sorted(mapping) == list(range(18))
    rot = axis_rotation()
    # Frame 1 should move the RL axis onto UD (or FB): check the face of U.
    assert move_map(rot)[0] // 3 in (1, 2, 4, 5)
