"""One-at-a-time sensitivity (Section 6.6, Table 16): HEA*-8 against A*, TA* and TA*+turn on 60 synthetic
instances (20 per family), when the planner knows the varied parameter.

Usage: python experiments/sensitivity.py [--n 20] [--out results]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import sys
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from heastar import instances as I          # noqa: E402
from heastar import planners as PL          # noqa: E402
from heastar import stats as ST             # noqa: E402
from heastar.energy import NOMINAL          # noqa: E402

PLAN = ("A*", "TA*", "TA*+turn", "HEA*-8")


def settings():
    base = NOMINAL
    mr, ml = base.mu_r_arr, base.mu_l0_arr
    return {
        "L = 0.4 m": base.with_(L=0.4),
        "L = 0.6 m (default)": base,
        "L = 0.9 m": base.with_(L=0.9),
        "L = 1.2 m": base.with_(L=1.2),
        "mu x 0.6": base.with_(mu_r=tuple(mr * 0.6), mu_l0=tuple(ml * 0.6)),
        "mu x 1.4": base.with_(mu_r=tuple(mr * 1.4), mu_l0=tuple(ml * 1.4)),
        "turning moment x 5": base.with_(kappa=5.0),
    }


def job(args):
    kind, idx = args
    inst = I.synthetic(kind, idx)
    rows = []
    for name, model in settings().items():
        lats = PL.build_lattices(inst, model, (8,))
        res = PL.score(inst, PL.plan_all(inst, model, lats, PLAN), model)
        for p in PLAN:
            rows.append(dict(setting=name, group=kind, idx=idx, planner=p, energy=res[p]["energy"]))
    return rows


def summarize(df):
    out = []
    for name in settings():
        w = df[df.setting == name].pivot_table(index=["group", "idx"], columns="planner", values="energy")
        r = dict(setting=name, n=len(w))
        for b in ("A*", "TA*", "TA*+turn"):
            r[f"vs {b}"] = f"{ST.median_change(w['HEA*-8'], w[b])[0]:+.1f}%"
        out.append(r)
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    jobs = [(k, i) for k in ("flat", "hills", "rocky") for i in range(a.n)]
    with Pool() as pool:
        rows = [r for rs in pool.imap_unordered(job, jobs) for r in rs]
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "sensitivity_raw.csv"), index=False)
    t = summarize(df)
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    t.to_csv(os.path.join(a.out, "tables", "table16_sensitivity.csv"), index=False)
    print(t.to_string(index=False))
