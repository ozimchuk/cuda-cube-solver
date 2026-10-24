import random

import pytest

from cuda_cube_solver.cubie import MOVES_HTM, CubieState, apply_moves
from cuda_cube_solver.facelets import FaceletError, is_facelet_string, parse_facelets, to_facelets

SOLVED = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"


def _random_states(n, seed):
    rng = random.Random(seed)
    return [apply_moves(CubieState(), [MOVES_HTM[rng.randrange(18)] for _ in range(rng.randrange(1, 40))])
            for _ in range(n)]


def test_solved_round_trip():
    assert parse_facelets(SOLVED) == CubieState()
    assert to_facelets(CubieState()) == SOLVED


def test_random_round_trip():
    for state in _random_states(200, 1):
        assert parse_facelets(to_facelets(state)) == state


def test_matches_reference_implementation():
    cc = pytest.importorskip("kociemba.pykociemba.cubiecube")
    fc = pytest.importorskip("kociemba.pykociemba.facecube")
    for state in _random_states(200, 2):
        ref = cc.CubieCube(list(state.cp), list(state.co), list(state.ep), list(state.eo)).toFaceCube().to_String()
        assert to_facelets(state) == ref
        back = fc.FaceCube(ref).toCubieCube()
        assert parse_facelets(ref) == CubieState(tuple(back.cp), tuple(back.co), tuple(back.ep), tuple(back.eo))
    example = "DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD"  # the kociemba package's README example
    back = fc.FaceCube(example).toCubieCube()
    assert parse_facelets(example) == CubieState(tuple(back.cp), tuple(back.co), tuple(back.ep), tuple(back.eo))


def test_colour_letters():
    state = _random_states(1, 3)[0]
    s = to_facelets(state)
    colours = s.translate(str.maketrans("URFDLB", "WRGYOB"))
    assert is_facelet_string(colours)
    assert parse_facelets(colours) == state


@pytest.mark.parametrize("mutate, message", [
    (lambda s: s[:53], "54 stickers"),
    (lambda s: s.replace("U", "R", 1), "has 8 stickers"),
    (lambda s: _swap(s, 8, 9, 20, 9, 20, 8), "twisted"),       # rotate the URF corner in place
    (lambda s: _swap(s, 5, 10, 10, 5), "flipped"),             # flip the UR edge in place
    (lambda s: _swap(s, 5, 7, 10, 19, 7, 5, 19, 10), "swapped"),  # exchange the UR and UF edges
])
def test_illegal_cubes(mutate, message):
    with pytest.raises(FaceletError, match=message):
        parse_facelets(mutate(SOLVED))


def _swap(s, *idx):
    """Move stickers: the first half of idx are targets, the second half sources."""
    n = len(idx) // 2
    out = list(s)
    for t, src in zip(idx[:n], idx[n:]):
        out[t] = s[src]
    return "".join(out)
