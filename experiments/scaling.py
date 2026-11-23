"""Computational cost and scaling (Section 6.8, Table 19): hills maps of 80 x 80 to 500 x 500 cells, three
instances per size (start and goal at least 0.6 N apart), 8, 16 and 32 headings, the same instances for all
heading sets. Every configuration runs in its own process so that its peak resident memory can be measured.

Usage: python experiments/scaling.py [--sizes 80 160 320 500] [--headings 8 16 32] [--out results] [--merge]
       (--merge adds the new sizes to an existing results/scaling.csv instead of replacing it)
The 500 x 500 rows take about 1.5 h on one core: python experiments/scaling.py --sizes 500 --merge
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
import resource
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


def one(size, H, idx):
    from heastar import instances as I
    from heastar.energy import NOMINAL
    from heastar.lattice import Lattice
    inst = I.synthetic("hills", idx, size=size, seed=51)
    lat = Lattice(inst.tm, NOMINAL, H)
    acts, info = lat.plan(inst.start, lat.heading_index(inst.h8), inst.goal)
    return dict(size=size, headings=H, idx=idx, states=H * size * size, precompute_s=lat.precompute_s,
                table_mb=lat.table_mb, expanded=info["expanded"], plan_s=info["plan_ms"] / 1e3,
                found=acts is not None, dist=float(np.hypot(*np.subtract(inst.goal, inst.start))),
                peak_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        print(json.dumps(one(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))))
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[80, 160, 320, 500])
    ap.add_argument("--headings", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="results")
    ap.add_argument("--merge", action="store_true",
                    help="keep the rows of other sizes / heading sets already in <out>/scaling.csv")
    a = ap.parse_args()
    rows = []
    for size in a.sizes:
        for H in a.headings:
            for i in range(a.n):
                r = subprocess.run([sys.executable, __file__, "--one", str(size), str(H), str(i)],
                                   capture_output=True, text=True, check=True)
                rows.append(json.loads(r.stdout.strip().splitlines()[-1]))
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    old = os.path.join(a.out, "scaling.csv")
    if a.merge and os.path.exists(old):
        prev = pd.read_csv(old)
        keep = ~(prev["size"].isin(df["size"].unique()) & prev["headings"].isin(df["headings"].unique()))
        df = pd.concat([prev[keep], df]).sort_values(["size", "headings", "idx"]).reset_index(drop=True)
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    df.to_csv(os.path.join(a.out, "scaling.csv"), index=False)
    t = df.groupby(["size", "headings"]).agg(states=("states", "first"), precompute_s=("precompute_s", "mean"),
                                              table_mb=("table_mb", "mean"), peak_mb=("peak_mb", "mean"),
                                              expanded=("expanded", "mean"), plan_min_s=("plan_s", "min"),
                                              plan_max_s=("plan_s", "max")).round(1)
    t.to_csv(os.path.join(a.out, "tables", "table19_scaling.csv"))
    print(t.to_string())
