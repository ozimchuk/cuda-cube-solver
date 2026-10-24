# Learned / threshold encodings of the pruning table

Question: can a neural network (run on tensor cores) predict the 208
table well enough that a tiny per-entry correction suffices, so larger
tables fit in VRAM?

## Method

- `collect.py`: 60 M uniformly random entries of the 208 table with exact
  values (block table, "≥ min+3" resolved with the mod-3 table), and an
  evaluation set of 20,000 random positions × 6 frames with exact 208
  values and the 304 table's bound.
- `train.py`: admissible 1-bit scheme. A predictor gives a cut q(x); the
  table stores b(x) = [d(x) ≥ q(x)]; the bound is q(x) if b(x), else the
  304 table's bound (1.4 GB). Memory 3.8 GB of bits + 1.4 GB. Predictors:
  constant, per corner class, per (class, slice), small and large MLPs
  (classifiers; q maximises the expected bound).

## Results (evaluation frames; avg6 includes the coset axis rule)

| Encoding | Memory | avg1 | avg6 |
|---|---|---|---|
| 208 exact | 7.8 GB | 10.921 | 11.862 |
| 304 alone | 1.4 GB | 10.289 | 11.122 |
| 1 bit "≥ 11" + 304 | 5.2 GB | 10.815 | 11.844 |
| cut per corner class | 5.2 GB | 10.815 | 11.844 |
| cut per (class, slice) | 5.2 GB | 10.826 | 11.833 |
| small MLP (7.3 k MACs/lookup) | 5.2 GB | 10.816 | 11.843 |
| large MLP (458 k MACs/lookup) | 5.2 GB | 10.835 | 11.842 |

Value distribution of the table: 11 = 65.1%, 10 = 17.3%, 12 = 15.6%,
≤ 9 = 2.0%.

## Conclusions

- **Networks do not learn the table.** Argmax accuracy 0.652 (small) and
  0.660 (large) against 0.651 for always predicting 11; cross-entropy
  falls only from the prior's ≈ 0.94 to 0.84 nats. A coset distance is a
  highly non-local function of these coordinates. Learned cuts are no
  better than a constant, so tensor cores have no role.
- **A constant-threshold bit plus a small exact fallback is a strong
  encoding**: 99.85% of the 208 table's six-lookup average at 67% of the
  memory. It beats the base-3 block encoding considered for 8 GB cards
  (11.823 at 6.2 GB), at the cost of a second lookup when the bit is
  clear (20% of entries; often skippable, since then d ≤ 10 and the
  fallback cannot prune when the remaining budget is ≥ 10).
- **Lead for larger tables:** nxopt's `EO2,EP3` table (22.7 GB at 2 bits)
  as a threshold bit is 11 GB; with the 112 fallback (2.6 GB) it fits 16
  GB. Its value depends on that table's distribution (nxopt reports avg1
  11.36 vs 10.94 for 208), which needs the table built (CPU generator on
  ≥ 48 GB RAM, or a ≥ 24 GB GPU).

## Follow-up: implemented and measured

`--table-format bit` in the solver (5.2 GB of VRAM). Identical optimal
lengths and node counts on the 1,000 positions, but 17.2 cubes/s versus
28.0 for the 2-bit block table: lookup steps per child rise from 2.48 to
4.03 because capping 12 at 11 loses first-lookup prunes (plus fallback
reads). In the memory-bound solver, lookups per child — not the average
bound — decide speed.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python research/learned_table/collect.py research/learned_table/data
python -m venv /tmp/torchenv && /tmp/torchenv/bin/pip install torch numpy --index-url https://download.pytorch.org/whl/cu128
/tmp/torchenv/bin/python research/learned_table/train.py research/learned_table/data
```

(`collect.py` needs the 208 mod-3, 208 block and 304 block tables; data
is ≈ 0.6 GB and not committed.)
