"""32 headings (Section 6.2): the synthetic experiment repeated with a 32-heading lattice (all lattice directions
with integer components up to 3, 7-18 deg apart), same instances, primitives and cost model.

Usage: python experiments/headings32.py [--n 50] [--out results]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import sys
import time
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from heastar import instances as I          # noqa: E402
from heastar import maps as M               # noqa: E402
from heastar import planners as PL          # noqa: E402
from heastar import stats as ST             # noqa: E402
from heastar.energy import Model            # noqa: E402

PLAN = ("TA*+turn+S", "TA*+turn-16", "HEA*", "TA*+turn-32", "HEA*-32")


def job(args):
    kind, idx = args
    inst = I.synthetic(kind, idx)
    rows = []
    for kappa in (1.0, 5.0):
        model = Model(kappa=kappa)
        lats = PL.build_lattices(inst, model, (8, 16, 32))
        planned = PL.plan_all(inst, model, lats, PLAN)
        res = PL.score(inst, planned, model)
        for p in PLAN:
            H = 32 if p.endswith("32") else 16 if (p == "HEA*" or p.endswith("16")) else 8
            rows.append(dict(group=kind, idx=idx, kappa=kappa, planner=p, energy=res[p]["energy"],
                             plan_ms=res[p]["plan_ms"], precompute_s=lats[H].precompute_s, table_mb=lats[H].table_mb))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    jobs = [(k, i) for k in M.FAMILIES for i in range(a.n)]
    t0 = time.time()
    with Pool() as pool:
        df = pd.DataFrame([r for rs in pool.imap_unordered(job, jobs) for r in rs])
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    df.to_csv(os.path.join(a.out, "headings32.csv"), index=False)
    out = []
    for kappa in (1.0, 5.0):
        w = df[df.kappa == kappa].pivot_table(index=["group", "idx"], columns="planner", values="energy")
        for a_, b in (("TA*+turn-32", "TA*+turn-16"), ("HEA*-32", "HEA*"), ("HEA*-32", "TA*+turn-32"),
                      ("HEA*", "TA*+turn-16"), ("HEA*-32", "TA*+turn+S"), ("HEA*", "TA*+turn+S")):
            m, lo, ti, hi = ST.median_change(w[a_], w[b])
            out.append(dict(kappa=kappa, comparison=f"{a_} vs {b}", median=f"{m:+.1f}%", lower=lo, tied=ti, higher=hi))
    t = pd.DataFrame(out)
    cost = df[df.kappa == 1.0].groupby("planner")[["plan_ms", "precompute_s", "table_mb"]].mean().round(1)
    t.to_csv(os.path.join(a.out, "tables", "headings32.csv"), index=False)
    cost.to_csv(os.path.join(a.out, "tables", "headings32_cost.csv"))
    print(t.to_string(index=False))
    print(cost.to_string())
