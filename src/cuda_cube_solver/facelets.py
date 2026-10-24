"""54-sticker facelet strings <-> cubie states.

The common "Kociemba" layout (also used by the kociemba package and Cube
Explorer): faces in the order U, R, F, D, L, B, nine stickers each, every
face read row by row from its top-left sticker as seen when looking at
that face, with U viewed with B at the top and D viewed with F at the
top (side faces with U at the top).  Index 4 + 9 * face is the centre.

Any six distinct symbols may be used: the centre stickers define which
symbol belongs to which face, so both "UUUUUUUUURRR..." and colour
strings such as "WWWWWWWWWRRR..." are accepted.
"""

from __future__ import annotations

from cuda_cube_solver.cubie import CubieState, permutation_parity

FACES = "URFDLB"

# Sticker indices of each corner position, U/D sticker first, then clockwise.
CORNER_FACELETS = (
    (8, 9, 20), (6, 18, 38), (0, 36, 47), (2, 45, 11),       # URF UFL ULB UBR
    (29, 26, 15), (27, 44, 24), (33, 53, 42), (35, 17, 51),  # DFR DLF DBL DRB
)
# Sticker indices of each edge position (U/D or F/B sticker first).
EDGE_FACELETS = (
    (5, 10), (7, 19), (3, 37), (1, 46),       # UR UF UL UB
    (32, 16), (28, 25), (30, 43), (34, 52),   # DR DF DL DB
    (23, 12), (21, 41), (50, 39), (48, 14),   # FR FL BL BR
)
CORNER_FACES = ("URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB")
EDGE_FACES = ("UR", "UF", "UL", "UB", "DR", "DF", "DL", "DB", "FR", "FL", "BL", "BR")


class FaceletError(ValueError):
    """The string does not describe a legal cube."""


def is_facelet_string(text: str) -> bool:
    s = "".join(text.split())
    return len(s) == 54 and len(set(s)) == 6


def parse_facelets(text: str) -> CubieState:
    """Cubie state of a 54-sticker string; raises FaceletError if illegal."""

    s = "".join(text.split())
    if len(s) != 54:
        raise FaceletError(f"expected 54 stickers, got {len(s)}")
    centres = [s[4 + 9 * f] for f in range(6)]
    if len(set(centres)) != 6:
        raise FaceletError("the six centre stickers must all differ")
    face_of = {c: FACES[f] for f, c in enumerate(centres)}
    try:
        faces = [face_of[c] for c in s]
    except KeyError as exc:
        raise FaceletError(f"sticker {exc.args[0]!r} matches no centre") from None
    for f in FACES:
        if faces.count(f) != 9:
            raise FaceletError(f"face {f} has {faces.count(f)} stickers, expected 9")

    cp, co = [], []
    for pos, slots in enumerate(CORNER_FACELETS):
        stickers = [faces[i] for i in slots]
        twist = next((k for k in range(3) if stickers[k] in "UD"), None)
        if twist is None:
            raise FaceletError(f"corner at {CORNER_FACES[pos]} has no U or D sticker")
        rest = (stickers[(twist + 1) % 3], stickers[(twist + 2) % 3])
        cubie = next((j for j, name in enumerate(CORNER_FACES)
                      if name[0] == stickers[twist] and (name[1], name[2]) == rest), None)
        if cubie is None:
            raise FaceletError(f"corner at {CORNER_FACES[pos]} has impossible colours {''.join(stickers)}")
        cp.append(cubie)
        co.append(twist)

    ep, eo = [], []
    for pos, (a, b) in enumerate(EDGE_FACELETS):
        pair = faces[a] + faces[b]
        if pair in EDGE_FACES:
            ep.append(EDGE_FACES.index(pair))
            eo.append(0)
        elif pair[::-1] in EDGE_FACES:
            ep.append(EDGE_FACES.index(pair[::-1]))
            eo.append(1)
        else:
            raise FaceletError(f"edge at {EDGE_FACES[pos]} has impossible colours {pair}")

    if sorted(cp) != list(range(8)):
        raise FaceletError("some corner appears twice")
    if sorted(ep) != list(range(12)):
        raise FaceletError("some edge appears twice")
    if sum(co) % 3:
        raise FaceletError("a corner is twisted (corner orientations do not sum to 0 mod 3)")
    if sum(eo) % 2:
        raise FaceletError("an edge is flipped (edge orientations do not sum to 0 mod 2)")
    if permutation_parity(cp) != permutation_parity(ep):
        raise FaceletError("two pieces are swapped (corner and edge permutation parities differ)")
    return CubieState(tuple(cp), tuple(co), tuple(ep), tuple(eo))


def to_facelets(state: CubieState) -> str:
    """54-sticker string (face letters) of a cubie state."""

    out = list("".join(f * 9 for f in FACES))
    for pos, slots in enumerate(CORNER_FACELETS):
        name = CORNER_FACES[state.cp[pos]]
        for k in range(3):
            out[slots[(k + state.co[pos]) % 3]] = name[k]
    for pos, slots in enumerate(EDGE_FACELETS):
        name = EDGE_FACES[state.ep[pos]]
        for k in range(2):
            out[slots[(k + state.eo[pos]) % 2]] = name[k]
    return "".join(out)
