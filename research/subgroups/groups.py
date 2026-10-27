"""Subgroups L of the signed permutations of the four E edges (B4, order 384) for the survey.

An E-edge content is the labels of the E edges in position order with their flips:
element (perm, flips) of B4, indexed perm_rank * 16 + flips (flip bit i = i-th E position).
L relabels from the left: label x -> p[x], flip of that edge ^= f[x].  Only subgroups
normalised by the relabelling of the 16 UD symmetries keep symmetry reduction exact.
"""
from functools import lru_cache
from itertools import permutations, product

import numpy as np

from cuda_cube_solver import coords

PERMS = list(permutations(range(4)))
PRANK = {p: i for i, p in enumerate(PERMS)}
ELEMS = [(p, f) for p in PERMS for f in product((0, 1), repeat=4)]  # f indexed by label
EID = {e: i for i, e in enumerate(ELEMS)}
IDENT = EID[((0, 1, 2, 3), (0, 0, 0, 0))]


def mul(a, b):
    """a after b (both act on labels)."""
    (pa, fa), (pb, fb) = ELEMS[a], ELEMS[b]
    return EID[(tuple(pa[pb[x]] for x in range(4)), tuple(fb[x] ^ fa[pb[x]] for x in range(4)))]


MUL = np.array([[mul(a, b) for b in range(384)] for a in range(384)], dtype=np.int16)
INV = np.array([int(np.flatnonzero(MUL[a] == IDENT)[0]) for a in range(384)], dtype=np.int16)


def closure(gens):
    group = {IDENT}
    frontier = [IDENT]
    while frontier:
        nxt = []
        for x in frontier:
            for g in gens:
                y = int(MUL[g, x])
                if y not in group:
                    group.add(y)
                    nxt.append(y)
        frontier = nxt
    return frozenset(group)


@lru_cache(maxsize=1)
def sym_image():
    _, edge_syms = coords.ud_actions()
    out = set()
    for es in edge_syms:
        p = tuple(int(es.perm[8 + x]) - 8 for x in range(4))
        f = tuple(int(es.flip[8 + x]) for x in range(4))
        out.add(EID[(p, f)])
    return frozenset(out)


@lru_cache(maxsize=1)
def all_subgroups():
    found = {frozenset([IDENT])}
    frontier = list(found)
    while frontier:
        nxt = []
        for g in frontier:
            for e in range(384):
                if e not in g:
                    h = closure(list(g) + [e]) if len(g) < 48 else closure(_gens(g) + [e])
                    if h not in found:
                        found.add(h)
                        nxt.append(h)
        frontier = nxt
    return found


def _gens(g):
    gens, cur = [], frozenset([IDENT])
    for e in sorted(g):
        if e not in cur:
            gens.append(e)
            cur = closure(gens)
    return gens


def normalised(g):
    return all(int(MUL[MUL[s, x], INV[s]]) in g for s in sym_image() for x in g)


def flip_parity(e):
    return sum(ELEMS[e][1]) & 1


def content_index(e_label):
    """Element as stored by position: labels in position order and flips by position."""
    p, f = ELEMS[e_label]
    # element e maps "identity content" to: position i holds label p[i] with flip f[p[i]]
    return PRANK[p] * 16 + sum(f[p[i]] << i for i in range(4))


def classes(group):
    """class_of[384 content index], number of classes, reps[class] (content index), even_only."""
    # content c = (perm in position order q, flips by position g); relabel by k: label q[i] -> kp[q[i]],
    # flip at i ^= kf[q[i]]
    cls = -np.ones(384, dtype=np.int64)
    reps = []
    for c in range(384):
        if cls[c] >= 0:
            continue
        q, g = PERMS[c // 16], [(c >> i) & 1 for i in range(4)]
        for k in group:
            kp, kf = ELEMS[k]
            q2 = tuple(kp[q[i]] for i in range(4))
            g2 = sum((g[i] ^ kf[q[i]]) << i for i in range(4))
            cls[PRANK[q2] * 16 + g2] = len(reps)
        reps.append(c)
    even_only = all(flip_parity(k) == 0 for k in group)
    return cls, len(reps), reps, even_only


def candidates():
    """Normalised subgroups with their effective E factor (given tracked non-E flips)."""
    out = []
    for g in all_subgroups():
        if normalised(g):
            _, n, _, even = classes(g)
            out.append((g, n, even))
    return out


if __name__ == "__main__":
    print("sym image size", len(sym_image()))
    subs = all_subgroups()
    print("subgroups of B4:", len(subs))
    c = candidates()
    print("normalised:", len(c))
    from collections import Counter
    print(sorted(Counter((len(g), n, even) for g, n, even in c).items()))


class GenKind:
    """Table variant EP 9: E content up to the group, non-E part by mode (1 flips, 2 split70, 3 split35)."""

    NNE = {1: 256, 2: 70, 3: 35}

    def __init__(self, mode, group, gid):
        self.mode, self.group, self.gid = mode, frozenset(group), gid
        cls, n, reps, even = classes(self.group)
        gen_class = np.zeros(384, np.uint16)
        gen_rep = np.zeros((2, 384), np.uint16)
        parity = [bin(c & 15).count("1") & 1 for c in range(384)]
        if mode == 1 and even:
            nec = n // 2
            index, count = {}, [0, 0]
            for k, r in enumerate(reps):
                p = parity[r]
                index[k] = count[p]
                gen_rep[p, count[p]] = r
                count[p] += 1
            assert count == [nec, nec]
            gen_class[:] = [index[int(cls[c])] for c in range(384)]
        else:
            nec = n
            gen_class[:] = cls
            for c in range(384):
                gen_rep[parity[c] if mode == 1 else 0, cls[c]] = c
                if mode != 1:
                    gen_rep[1, cls[c]] = c
        self.nec = nec
        self.n_edge = 495 * self.NNE[mode] * nec
        self.defines = ("EDGE_EP=9", f"EDGE_EO={mode}", f"GEN_NEC={nec}")
        self.name = f"g{mode}-{gid}"
        self.gen = (gen_class, gen_rep)

    def describe(self):
        perms = {ELEMS[e][0] for e in self.group}
        flips = [e for e in self.group if ELEMS[e][0] == (0, 1, 2, 3)]
        return {"order": len(self.group), "perm_image": len(perms), "pure_flips": len(flips),
                "odd_flips": any(flip_parity(e) for e in self.group)}


def stable_candidates():
    """Normalised subgroups in a fixed order (cached in candidates.json)."""
    import json
    from pathlib import Path
    path = Path(__file__).with_name("candidates.json")
    if path.exists():
        return [frozenset(g) for g in json.loads(path.read_text())]
    groups = sorted((sorted(g) for g, _, _ in candidates()), key=lambda g: (len(g), g))
    path.write_text(json.dumps(groups) + "\n")
    return [frozenset(g) for g in groups]


def by_name(name):
    """The GenKind for a survey name like "g1-26"."""
    mode, gid = name[1:].split("-")
    return GenKind(int(mode), stable_candidates()[int(gid)], int(gid))
