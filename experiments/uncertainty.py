"""Joint parameter uncertainty (Section 6.6, Table 17).

Each of the 150 synthetic instances draws one "true" parameter set:
  kappa               log-uniform in [0.5, 6]
  global scales       of mu_r and of mu_l0, each uniform in [0.6, 1.4]
  per-surface factors for every class and coefficient, uniform in [0.7, 1.3]
  radius-law slope    (0.15 in Eq. 4) uniform in [0.075, 0.30]
Every path is scored with the true model. Cases:
  consistent    the planners plan with the true model
  mismatch      the planners plan with the nominal model (uncalibrated robot)
  conservative  the planners plan with kappa = 5 and nominal friction
Drivetrain efficiency is not varied (it does not affect relative savings, Section 3.7).

Usage: python experiments/uncertainty.py [--n 50] [--out results]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from heastar import instances as I          # noqa: E402
from heastar import maps as M               # noqa: E402
from heastar import planners as PL          # noqa: E402
from heastar import stats as ST             # noqa: E402
from heastar.energy import NOMINAL, SURFACES  # noqa: E402

PLAN = ("TA*", "TA*+turn", "HEA*-8", "TA*+turn-16", "HEA*")
SEED = 31


def draw(rng):
    kappa = float(np.exp(rng.uniform(np.log(0.5), np.log(6.0))))
    sr, sl = rng.uniform(0.6, 1.4, 2)
    fr = rng.uniform(0.7, 1.3, len(SURFACES))
    fl = rng.uniform(0.7, 1.3, len(SURFACES))
    wb = float(rng.uniform(0.075, 0.30))
    true = NOMINAL.with_(kappa=kappa, wong_b=wb, mu_r=tuple(NOMINAL.mu_r_arr * sr * fr),
                         mu_l0=tuple(NOMINAL.mu_l0_arr * sl * fl))
    return true, dict(kappa=kappa, scale_r=sr, scale_l=sl, wong_b=wb)


def job(args):
    kind, idx = args
    inst = I.synthetic(kind, idx)
    true, prm = draw(np.random.default_rng([SEED, M.FAMILIES.index(kind), idx]))
    rows = []
    for case, model in (("consistent", true), ("mismatch", NOMINAL), ("conservative", NOMINAL.with_(kappa=5.0))):
        res = PL.score(inst, PL.plan_all(inst, model, PL.build_lattices(inst, model), PLAN), true)
        r = dict(case=case, group=kind, idx=idx, **prm)
        r.update({f"E_{p}": res[p]["energy"] for p in PLAN})
        rows.append(r)
    return rows


BINS = [(0.5, 1), (1, 2), (2, 4), (4, 6), (0.5, 6)]
COMPS = [("TA*+turn", "TA*"), ("HEA*-8", "TA*+turn"), ("HEA*", "TA*+turn-16")]


def summarize(df):
    rows = []
    for case in ("consistent", "mismatch"):
        d = df[df.case == case]
        for a, b in COMPS:
            r = dict(case=case, comparison=f"{a} vs {b}")
            for lo, hi in BINS:
                q = d[(d.kappa > lo if lo > 0.5 else d.kappa >= lo) & (d.kappa <= hi)]
                m, _, _, higher = ST.median_change(q[f"E_{a}"], q[f"E_{b}"])
                r["all" if (lo, hi) == (0.5, 6) else f"kappa in ({lo}, {hi}]"] = f"{m:+.1f}% ({higher})"
            rows.append(r)
    t = pd.DataFrame(rows)
    n = {f"kappa in ({lo}, {hi}]": int(((df.case == 'consistent') & (df.kappa > lo if lo > 0.5 else df.kappa >= lo)
                                       & (df.kappa <= hi)).sum()) for lo, hi in BINS[:-1]}
    # excess over the true optimum of each lattice (the consistent arc lattice is optimal for the true model)
    c = df[df.case == "consistent"].set_index(["group", "idx"])
    mm = df[df.case == "mismatch"].set_index(["group", "idx"])
    cs = df[df.case == "conservative"].set_index(["group", "idx"])
    ex = []
    for lab, sel in (("kappa > 2", c.kappa > 2), ("kappa <= 1", c.kappa <= 1), ("all", c.kappa > 0)):
        for H, opt, pt in ((8, "HEA*-8", "TA*+turn"), (16, "HEA*", "TA*+turn-16")):
            o = c.loc[sel, f"E_{opt}"]
            for case, d in (("mismatch", mm), ("conservative", cs)):
                for p in (opt, pt):
                    e = (d.loc[sel, f"E_{p}"] / o - 1) * 100
                    ex.append(dict(kappa=lab, lattice=H, case=case, planner=p, n=int(sel.sum()),
                                   median_excess_pct=round(float(e.median()), 1),
                                   p90_excess_pct=round(float(e.quantile(0.9)), 1),
                                   worse_than_point_turns=int((d.loc[sel, f"E_{opt}"] > d.loc[sel, f"E_{pt}"] + 1e-6).sum())
                                   if p == opt else ""))
    return t, n, pd.DataFrame(ex)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    jobs = [(k, i) for k in M.FAMILIES for i in range(a.n)]
    with Pool() as pool:
        rows = [r for rs in pool.imap_unordered(job, jobs) for r in rs]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    df.to_csv(os.path.join(a.out, "uncertainty_raw.csv"), index=False)
    t, n, ex = summarize(df)
    t.to_csv(os.path.join(a.out, "tables", "table17_uncertainty.csv"), index=False)
    ex.to_csv(os.path.join(a.out, "tables", "uncertainty_excess.csv"), index=False)
    print("instances per bin:", n)
    print(t.to_string(index=False))
    print(ex.to_string(index=False))
