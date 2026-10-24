"""Authoritative cubie model.

Permutation arrays use the Kociemba convention: ``cp[position]`` is the
corner cubie occupying that position.  Applying transform ``b`` after state
``a`` therefore gives ``new.cp[i] = a.cp[b.cp[i]]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

CORNER_NAMES = ("URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB")
EDGE_NAMES = ("UR", "UF", "UL", "UB", "DR", "DF", "DL", "DB", "FR", "FL", "BL", "BR")
FACE_NAMES = ("U", "R", "F", "D", "L", "B")


@dataclass(frozen=True, slots=True)
class CubieState:
    cp: tuple[int, ...] = tuple(range(8))
    co: tuple[int, ...] = (0,) * 8
    ep: tuple[int, ...] = tuple(range(12))
    eo: tuple[int, ...] = (0,) * 12

    def then(self, transform: "CubieState") -> "CubieState":
        return compose(self, transform)

    def inverse(self) -> "CubieState":
        cp_inv = [0] * 8
        ep_inv = [0] * 12
        for position, cubie in enumerate(self.cp):
            cp_inv[cubie] = position
        for position, cubie in enumerate(self.ep):
            ep_inv[cubie] = position
        return CubieState(
            tuple(cp_inv),
            tuple((-self.co[cp_inv[i]]) % 3 for i in range(8)),
            tuple(ep_inv),
            tuple((-self.eo[ep_inv[i]]) % 2 for i in range(12)),
        )

    def is_legal(self) -> bool:
        return (
            sorted(self.cp) == list(range(8))
            and sorted(self.ep) == list(range(12))
            and sum(self.co) % 3 == 0
            and sum(self.eo) % 2 == 0
            and permutation_parity(self.cp) == permutation_parity(self.ep)
            and all(0 <= value < 3 for value in self.co)
            and all(0 <= value < 2 for value in self.eo)
        )


IDENTITY = CubieState()

# Quarter-turn transforms in U, R, F, D, L, B order.
_FACE_QUARTERS = (
    CubieState(
        (3, 0, 1, 2, 4, 5, 6, 7),
        (0,) * 8,
        (3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11),
        (0,) * 12,
    ),
    CubieState(
        (4, 1, 2, 0, 7, 5, 6, 3),
        (2, 0, 0, 1, 1, 0, 0, 2),
        (8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0),
        (0,) * 12,
    ),
    CubieState(
        (1, 5, 2, 3, 0, 4, 6, 7),
        (1, 2, 0, 0, 2, 1, 0, 0),
        (0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11),
        (0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0),
    ),
    CubieState(
        (0, 1, 2, 3, 5, 6, 7, 4),
        (0,) * 8,
        (0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11),
        (0,) * 12,
    ),
    CubieState(
        (0, 2, 6, 3, 4, 1, 5, 7),
        (0, 1, 2, 0, 0, 2, 1, 0),
        (0, 1, 10, 3, 4, 5, 9, 7, 8, 2, 6, 11),
        (0,) * 12,
    ),
    CubieState(
        (0, 1, 3, 7, 4, 5, 2, 6),
        (0, 0, 1, 2, 0, 0, 2, 1),
        (0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7),
        (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1),
    ),
)


def compose(a: CubieState, b: CubieState) -> CubieState:
    """Return the result of applying transform ``b`` after state ``a``."""

    return CubieState(
        tuple(a.cp[b.cp[i]] for i in range(8)),
        tuple((a.co[b.cp[i]] + b.co[i]) % 3 for i in range(8)),
        tuple(a.ep[b.ep[i]] for i in range(12)),
        tuple((a.eo[b.ep[i]] + b.eo[i]) % 2 for i in range(12)),
    )


def power(transform: CubieState, exponent: int) -> CubieState:
    result = IDENTITY
    for _ in range(exponent % 4):
        result = compose(result, transform)
    return result


MOVE_NAMES_HTM: tuple[str, ...] = tuple(
    suffix
    for face in FACE_NAMES
    for suffix in (face, f"{face}2", f"{face}'")
)
MOVES_HTM: tuple[CubieState, ...] = tuple(
    power(_FACE_QUARTERS[face], turn)
    for face in range(6)
    for turn in (1, 2, 3)
)
MOVE_NAMES_QTM: tuple[str, ...] = tuple(
    suffix for face in FACE_NAMES for suffix in (face, f"{face}'")
)
MOVES_QTM: tuple[CubieState, ...] = tuple(
    power(_FACE_QUARTERS[face], turn)
    for face in range(6)
    for turn in (1, 3)
)

FIXED_CORNER_FACES = (0, 1, 2)  # U, R, F leave DBL fixed.
MOVE_NAMES_2_HTM: tuple[str, ...] = tuple(
    suffix
    for face in ("U", "R", "F")
    for suffix in (face, f"{face}2", f"{face}'")
)
MOVES_2_HTM: tuple[CubieState, ...] = tuple(
    power(_FACE_QUARTERS[face], turn)
    for face in FIXED_CORNER_FACES
    for turn in (1, 2, 3)
)
MOVE_NAMES_2_QTM: tuple[str, ...] = tuple(
    suffix for face in ("U", "R", "F") for suffix in (face, f"{face}'")
)
MOVES_2_QTM: tuple[CubieState, ...] = tuple(
    power(_FACE_QUARTERS[face], turn)
    for face in FIXED_CORNER_FACES
    for turn in (1, 3)
)


def apply_moves(state: CubieState, moves: Iterable[CubieState]) -> CubieState:
    for move in moves:
        state = compose(state, move)
    return state


def permutation_parity(permutation: tuple[int, ...] | list[int]) -> int:
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return inversions & 1


def pack_state(state: CubieState) -> tuple[int, int]:
    """Pack a legal cubie state into two unsigned 64-bit words.

    The representation is deliberately redundant but cheap to transform:
    ``lo`` contains corner permutation, corner orientation, and edge
    orientation; ``hi`` contains edge permutation.
    """

    lo = 0
    hi = 0
    for i, value in enumerate(state.cp):
        lo |= value << (4 * i)
    for i, value in enumerate(state.co):
        lo |= value << (32 + 2 * i)
    for i, value in enumerate(state.eo):
        lo |= value << (48 + i)
    for i, value in enumerate(state.ep):
        hi |= value << (4 * i)
    return lo, hi


def unpack_state(lo: int, hi: int) -> CubieState:
    return CubieState(
        tuple((lo >> (4 * i)) & 0xF for i in range(8)),
        tuple((lo >> (32 + 2 * i)) & 0x3 for i in range(8)),
        tuple((hi >> (4 * i)) & 0xF for i in range(12)),
        tuple((lo >> (48 + i)) & 1 for i in range(12)),
    )


def apply_packed(lo: int, hi: int, move: CubieState) -> tuple[int, int]:
    return pack_state(compose(unpack_state(lo, hi), move))
