import random

import numpy as np
import pytest

from cuda_cube_solver import coords, geometry
from cuda_cube_solver.cubie import MOVES_HTM, CubieState, apply_moves, compose


def _random_states(n: int, seed: int) -> list[CubieState]:
    rng = random.Random(seed)
    return [apply_moves(CubieState(), [MOVES_HTM[rng.randrange(18)] for _ in range(rng.randrange(40))]) for _ in range(n)]


def test_all_48_symmetries_preserve_the_move_set() -> None:
    geometry.validate()
    assert len(geometry.all_symmetries()) == 48
    assert len(geometry.ud_symmetries()) == 16


def test_packed_conjugation_matches_sticker_conjugation() -> None:
    for sym in geometry.all_symmetries():
        cs, es = coords._corner_sym(sym), coords._edge_sym(sym)
        for state in _random_states(5, 1):
            want = coords.pack(sym.conjugate(state))
            corners, edges = coords.pack(state)
            assert np.array_equal(coords.conj_corners(corners, cs), want[0])
            assert np.array_equal(coords.conj_edges(edges, es), want[1])


def test_corner_classes_match_nxopt_count() -> None:
    classes = coords.corner_classes()
    assert len(classes.rep) == 9930
    assert classes.stab[0] & 1


KINDS = [coords.EdgeKind(2, 8), coords.EdgeKind(1, 12), coords.EdgeKind(3, 4), coords.EdgeKind(3, 8),
         coords.EdgeKind(5, 12), coords.EdgeKind(6, 12), coords.EdgeKind(7, 12), coords.EdgeKind(8, 8)]


@pytest.mark.parametrize("kind", KINDS, ids=lambda k: k.name)
def test_edge_coord_roundtrip(kind) -> None:
    rng = random.Random(7)
    for coord in [0, kind.n_edge - 1] + [rng.randrange(kind.n_edge) for _ in range(300)]:
        assert coords.edge_coord(coords.edges_from_coord(coord, kind), kind) == coord


@pytest.mark.parametrize("kind", KINDS, ids=lambda k: k.name)
def test_quotient_action_is_well_defined(kind) -> None:
    for state in _random_states(30, 3):
        corners, edges = coords.pack(state)
        rep_c = coords.corners_from_raw(coords.corner_raw(corners))
        rep_e = coords.edges_from_coord(coords.edge_coord(edges, kind), kind)
        for m in range(18):
            a = coords.apply_move(corners, edges, m)
            b = coords.apply_move(rep_c, rep_e, m)
            assert coords.corner_raw(a[0]) == coords.corner_raw(b[0])
            assert coords.edge_coord(a[1], kind) == coords.edge_coord(b[1], kind)


@pytest.mark.parametrize("kind", KINDS, ids=lambda k: k.name)
def test_symmetry_action_is_well_defined(kind) -> None:
    _, edge_syms = coords.ud_actions()
    for state in _random_states(30, 13):
        _, edges = coords.pack(state)
        rep_e = coords.edges_from_coord(coords.edge_coord(edges, kind), kind)
        for es in edge_syms:
            assert coords.edge_coord(coords.conj_edges(edges, es), kind) == \
                coords.edge_coord(coords.conj_edges(rep_e, es), kind)


@pytest.mark.parametrize("kind", KINDS, ids=lambda k: k.name)
def test_gpu_index_and_unrank_match_host(kind) -> None:
    cp = pytest.importorskip("cupy")
    from cuda_cube_solver import device

    dt = device.DeviceTables()
    mod = device.module(*kind.defines)
    states = _random_states(300, 5)
    packed = [coords.pack(s) for s in states]
    words = device.pack_states(np.stack([p[0] for p in packed]), np.stack([p[1] for p in packed]))
    n = len(states)
    out_idx = cp.zeros(n, dtype=cp.uint64)
    out_state = cp.zeros((n, 5), dtype=cp.uint32)
    mod.get_function("probe_index")(((n + 127) // 128,), (128,),
                                    (dt.tables, dt.classidx, cp.asarray(words), np.int32(n), np.int32(-1),
                                     out_idx, out_state))
    got = out_idx.get()
    for k in range(n):
        assert got[k] == coords.table_index(*packed[k], kind)
    rng = random.Random(11)
    total = dt.n_classes * kind.n_edge
    idx = np.array([0, total - 1] + [rng.randrange(total) for _ in range(2000)], dtype=np.uint64)
    back = cp.zeros(len(idx), dtype=cp.uint64)
    mod.get_function("probe_unrank")(((len(idx) + 127) // 128,), (128,),
                                     (dt.tables, dt.classidx, dt.class_rep, cp.asarray(idx), np.int32(len(idx)), back))
    assert np.array_equal(back.get(), idx)


def test_gpu_moves_left_moves_and_index_match_host() -> None:
    cp = pytest.importorskip("cupy")
    from cuda_cube_solver import device

    dt = device.DeviceTables()
    states = _random_states(200, 2)
    packed = [coords.pack(s) for s in states]
    words = device.pack_states(np.stack([p[0] for p in packed]), np.stack([p[1] for p in packed]))
    probe = device.module().get_function("probe_index")
    n = len(states)
    for move in [-1] + list(range(18)) + [-2 - m for m in range(18)]:
        out_idx = cp.zeros(n, dtype=cp.uint64)
        out_state = cp.zeros((n, 5), dtype=cp.uint32)
        probe(((n + 127) // 128,), (128,),
              (dt.tables, dt.classidx, cp.asarray(words), np.int32(n), np.int32(move), out_idx, out_state))
        got_idx, got_state = out_idx.get(), out_state.get()
        for k in range(0, n, 5):
            if move >= -1:
                corners, edges = packed[k]
                if move >= 0:
                    corners, edges = coords.apply_move(corners, edges, move)
            else:
                corners, edges = coords.pack(compose(MOVES_HTM[-2 - move], states[k]))
            assert np.array_equal(got_state[k], device.pack_states(corners[None], edges[None])[0])
            assert got_idx[k] == coords.table_index(corners, edges)


@pytest.mark.parametrize("name", ["s4", "d4f", "a4t", "d4t", "d4", "a4"])
def test_group_kind_unrank_and_cosets(name) -> None:
    """Unrank inverts the index, and the index is constant on cosets: relabelling the E edges by a
    group element and the non-E edges arbitrarily keeps it, also after any move."""

    cp = pytest.importorskip("cupy")
    from cuda_cube_solver import device, kinds

    kind = kinds.get(name)
    dt = device.DeviceTables(kind.gen)
    mod = device.module(*kind.defines)
    rng = random.Random(name)
    total = dt.n_classes * kind.n_edge
    idx = np.array([0, total - 1] + [rng.randrange(total) for _ in range(500)], dtype=np.uint64)
    back = cp.zeros(len(idx), dtype=cp.uint64)
    mod.get_function("probe_unrank")(((len(idx) + 127) // 128,), (128,),
                                     (dt.tables, dt.classidx, dt.class_rep, cp.asarray(idx), np.int32(len(idx)), back))
    assert np.array_equal(back.get(), idx)

    def relabel(state: CubieState) -> CubieState:
        kp, kf = rng.choice(sorted(kind.group))
        others = list(range(8))
        rng.shuffle(others)
        ep = tuple(8 + kp[c - 8] if c >= 8 else others[c] for c in state.ep)
        eo = tuple(o ^ kf[c - 8] if c >= 8 else o for c, o in zip(state.ep, state.eo))
        return CubieState(state.cp, state.co, ep, eo)

    def indices(states, move):
        packed = [coords.pack(s) for s in states]
        words = cp.asarray(device.pack_states(np.stack([p[0] for p in packed]), np.stack([p[1] for p in packed])))
        out = cp.zeros(len(states), dtype=cp.uint64)
        tmp = cp.zeros((len(states), 5), dtype=cp.uint32)
        mod.get_function("probe_index")(((len(states) + 127) // 128,), (128,),
                                        (dt.tables, dt.classidx, words, np.int32(len(states)), np.int32(move), out, tmp))
        return out.get()

    states = _random_states(40, 17)
    relabelled = [relabel(s) for s in states]
    for move in [-1] + list(range(18)):
        assert np.array_equal(indices(states, move), indices(relabelled, move))


def test_preset_choice(tmp_path) -> None:
    from cuda_cube_solver import kinds

    names = lambda pair: tuple(p.name if p else None for p in pair)
    assert names(kinds.choose(tmp_path, int(16e9))) == ("d4t", None)
    assert names(kinds.choose(tmp_path, int(8e9), int(16e9))) == ("d4t-b3", None)
    # An 8 GB card can't build d4t-b3 or d4f-b3 ...
    assert names(kinds.choose(tmp_path, int(8e9))) == ("s4-b3", "d4t-b3")
    # ... but uses them once they are copied in.
    (tmp_path / "prune-ud-d4t-b3.bin").touch()
    assert names(kinds.choose(tmp_path, int(8e9))) == ("d4t-b3", None)
    assert names(kinds.choose(tmp_path, int(1e9))) == (None, None)
    # build-table ignores copied-in tables: it picks what this card can build.
    assert names(kinds.choose(tmp_path, int(8e9), to_build=True)) == ("s4-b3", "d4t-b3")


def test_group_kind_must_be_normalised_by_the_symmetries() -> None:
    from cuda_cube_solver import kinds

    with pytest.raises(ValueError, match="not normalised"):
        kinds.GroupKind("swap01", [((1, 0, 2, 3), (0, 0, 0, 0))])
    with pytest.raises(ValueError, match="not normalised"):
        kinds.GroupKind("flip01", [((0, 1, 2, 3), (1, 1, 0, 0))])
