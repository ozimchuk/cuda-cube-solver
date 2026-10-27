#!/bin/bash
# Memory ladder: build each table preset if missing, then solve the same 1,000 cubes with each.
# Run from the repo root.  Output: research/subgroups/ladder.txt
set -e
PY=${PY:-python}
OUT=research/subgroups/ladder.txt
export PYTHONPATH=src
for t in 304 s4-b3 d4f-b3 a4t-b3 a4t d4t-b3 d4t; do
  kind=${t%-b3}
  if [ "$kind" = "$t" ]; then fmt=block; file=tables/prune-ud-$kind-blk.bin; else fmt=b3; file=tables/prune-ud-$kind-b3.bin; fi
  [ -f $file ] || $PY -m cuda_cube_solver build-table --table $t
  r=$($PY bench/run_gpu.py data/random1000.txt --count 1000 --batch 100 --kind $kind --table-format $fmt | tail -1)
  echo "$t $r" | tee -a $OUT
done
