"""Build survey kinds' block tables into tables/ for solving (no results.jsonl line).

    PYTHONPATH=src:research/subgroups python research/subgroups/build.py g1-27 [--b3]

--b3 also saves the mod-3 table and builds the base-3 table from it (--table-format b3).
"""
import json
import sys
from pathlib import Path

import cupy as cp

import groups
from cuda_cube_solver import device, table

TABLES = Path("tables")

if __name__ == "__main__":
    b3 = "--b3" in sys.argv
    for name in [a for a in sys.argv[1:] if not a.startswith("--")]:
        k = groups.by_name(name)
        dt = device.DeviceTables(k.gen)
        mod3, blk, counts = table.generate(dt, k, log=lambda m: print(f"  {name} {m}", flush=True))
        table.save_device(blk, TABLES / f"prune-ud-{name}-blk.bin")
        (TABLES / f"prune-ud-{name}.json").write_text(json.dumps(
            {"coordinate": f"survey {name} {k.describe()}", "entries": dt.n_classes * k.n_edge,
             "depth_counts": counts}, indent=2) + "\n")
        if b3:
            table.save_device(mod3, TABLES / f"prune-ud-{name}.bin")
        del mod3, blk
        cp.get_default_memory_pool().free_all_blocks()
        if b3:
            table.build_b3_table(TABLES, k, dt)
        print(name, "saved", flush=True)
