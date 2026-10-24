"""Feasibility study: learned 1-bit encoding of the 208 pruning table.

Scheme (admissible by construction): a predictor gives a cut q(x) from the
table coordinates; the table stores one bit b(x) = [d(x) >= q(x)]; the bound
is q(x) if b(x) else the fallback table's bound (304, 1.4 GB).  Memory:
3.8 GB of bits + 1.4 GB fallback + the predictor, versus 7.8 GB for 208.

Predictors of q compared: constant; per corner class; per (corner class,
slice); small and large neural networks (classifiers over d, q chosen to
maximise the expected bound with the fallback's mean as the alternative).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("research/learned_table/data")
N_EDGE = 495 * 24 * 256
N_CLASSES = 9930
N_VALUES = 14
dev = torch.device("cuda")


def features(idx: np.ndarray):
    idx = idx.astype(np.int64)
    cls, e = np.divmod(idx, N_EDGE)
    rest, eo8 = np.divmod(e, 256)
    slice_, eorder = np.divmod(rest, 24)
    return cls, slice_, eorder, eo8


class Net(nn.Module):
    def __init__(self, emb_cls, emb_slice, emb_order, hidden):
        super().__init__()
        self.cls = nn.Embedding(N_CLASSES, emb_cls)
        self.slice = nn.Embedding(495, emb_slice)
        self.order = nn.Embedding(24, emb_order)
        width = emb_cls + emb_slice + emb_order + 8
        layers = []
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, N_VALUES))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cls, slice_, order, eo8):
        bits = ((eo8[:, None] >> torch.arange(8, device=eo8.device)) & 1).float()
        x = torch.cat([self.cls(cls), self.slice(slice_), self.order(order), bits], dim=1)
        return self.mlp(x)

    def macs(self):
        return sum(m.in_features * m.out_features for m in self.mlp if isinstance(m, nn.Linear))


def best_cut(probs: torch.Tensor, alt: float) -> torch.Tensor:
    """argmax_q  q * P(d >= q) + alt * P(d < q)."""
    tail = probs.flip(1).cumsum(1).flip(1)  # P(d >= q)
    q = torch.arange(N_VALUES, device=probs.device, dtype=probs.dtype)
    return (q * tail + alt * (1 - tail)).argmax(1)


def train(net, data, epochs, batch=1 << 16, lr=3e-3):
    cls, sl, od, eo, val = data
    n = len(val)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    steps = epochs * (n // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    lossf = nn.CrossEntropyLoss()
    t = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        total = 0.0
        for s in range(0, n - batch + 1, batch):
            b = perm[s:s + batch]
            loss = lossf(net(cls[b], sl[b], od[b], eo[b]), val[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        print(f"  epoch {ep + 1}: loss {total / (n // batch):.4f} ({time.time() - t:.0f}s)", flush=True)


def scores(bound: np.ndarray):
    v = bound.reshape(-1, 6)
    m = v.max(1)
    plus = ((v[:, :3] == m[:, None]).all(1) | (v[:, 3:] == m[:, None]).all(1)) & (m > 0)
    return v[:, 0].mean(), (m + plus).mean()


def main():
    tr = np.load(DATA / "train.npz")
    ev = np.load(DATA / "eval.npz")
    t_cls, t_sl, t_od, t_eo = (torch.as_tensor(a, device=dev) for a in features(tr["idx"]))
    t_val = torch.as_tensor(tr["val"].astype(np.int64), device=dev)
    e_cls, e_sl, e_od, e_eo = (torch.as_tensor(a, device=dev) for a in features(ev["idx"]))
    e_val = ev["val"].astype(np.int64)
    fallback = ev["fallback"].astype(np.int64)
    alt = float(fallback.mean())

    def report(name, q, extra_gb):
        q = np.asarray(q, dtype=np.int64)
        bound = np.where(e_val >= q, q, fallback)
        a1, a6 = scores(bound)
        hit = (e_val >= q).mean()
        print(f"{name:34s} avg1 {a1:.4f}  avg6+axis {a6:.4f}  bit set {hit:.3f}  "
              f"memory {3.78 + 1.42 + extra_gb:.2f} GB", flush=True)

    a1, a6 = scores(e_val)
    print(f"{'208 exact (7.8 GB)':34s} avg1 {a1:.4f}  avg6+axis {a6:.4f}")
    a1, a6 = scores(fallback)
    print(f"{'304 fallback alone (1.4 GB)':34s} avg1 {a1:.4f}  avg6+axis {a6:.4f}")

    # constant cut
    hist = np.bincount(tr["val"], minlength=N_VALUES).astype(np.float64)
    probs = torch.as_tensor(hist / hist.sum())[None, :]
    q_const = int(best_cut(probs, alt)[0])
    report(f"constant cut q={q_const}", np.full_like(e_val, q_const), 0.0)

    # per-class and per-(class, slice) empirical distributions
    for name, keys_tr, keys_ev, n_keys in [
        ("cut per corner class", features(tr["idx"])[0], features(ev["idx"])[0], N_CLASSES),
        ("cut per (class, slice)", features(tr["idx"])[0] * 495 + features(tr["idx"])[1],
         features(ev["idx"])[0] * 495 + features(ev["idx"])[1], N_CLASSES * 495),
    ]:
        counts = np.zeros((n_keys, N_VALUES), dtype=np.float32)
        np.add.at(counts, (keys_tr, tr["val"]), 1)
        counts += hist / hist.sum()  # prior for sparse cells
        p = torch.as_tensor(counts / counts.sum(1, keepdims=True))
        q_tab = best_cut(p, alt).numpy()
        report(name, q_tab[keys_ev], n_keys / 2**30)

    data = (t_cls, t_sl, t_od, t_eo, t_val)
    for name, net, epochs in [
        ("NN small", Net(16, 8, 4, [64, 64]), 3),
        ("NN large", Net(64, 32, 16, [512, 512, 256]), 4),
    ]:
        net = net.to(dev)
        print(f"{name}: {net.macs():,} MACs per lookup (MLP part)", flush=True)
        train(net, data, epochs)
        net.eval()
        with torch.no_grad():
            logits = torch.cat([net(e_cls[s:s + 65536], e_sl[s:s + 65536], e_od[s:s + 65536], e_eo[s:s + 65536])
                                for s in range(0, len(e_val), 65536)])
            probs = logits.softmax(1).double()
            pred = probs.argmax(1).cpu().numpy()
            q = best_cut(probs, alt).cpu().numpy()
        print(f"  {name}: argmax accuracy {(pred == e_val).mean():.3f}, "
              f"within 1: {(np.abs(pred - e_val) <= 1).mean():.3f}")
        report(name, q, 0.0)


if __name__ == "__main__":
    main()
