"""Discretisation study (Section 6.7, Table 18) on 60 synthetic instances (20 per family): median energy change
of HEA*-8 relative to TA*+turn on the default lattice, with the 10th / 90th percentiles and the number of
instances in which HEA*-8 used more energy.

  sampling interval  plan with 0.1-1.0 cell sampling, re-score with a fine reference interval of 0.05 cells
  arc sizes          A1-A2, A2-A4-A8, A1-A16, A1-A2-A3-A4-A6-A8 instead of A1-A2-A4-A8
  cell size          maps refined to 0.5 m cells with the same metric radii (arcs A2-A16 in cells)

Usage: python experiments/discretisation.py [--n 20] [--out results]
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
from heastar import paths as P              # noqa: E402
from heastar.energy import NOMINAL          # noqa: E402
from heastar.lattice import Lattice, make_primitives   # noqa: E402

REF_DS = 0.05
ARC_SETS = {"A1, A2, A4, A8 (default)": (1, 2, 4, 8), "A1, A2 only": (1, 2), "A2, A4, A8": (2, 4, 8),
            "A1-A16": (1, 2, 4, 8, 16), "A1, A2, A3, A4, A6, A8": (1, 2, 3, 4, 6, 8)}


def ref_energy(sc, acts, lat, ref_prims):
    """Energy of an action sequence with the geometry re-sampled at the reference interval."""
    el = []
    for h, x, y, pi in acts:
        p = ref_prims[h][pi]
        el.append(P.Rot(x, y, p.dtheta) if p.kind == "R" else P.Curve(p.samples + (x, y), p.radius, p.kind, p.dtheta))
    return sc.evaluate(el)["energy"]


def job(args):
    kind, idx = args
    inst = I.synthetic(kind, idx)
    tm, s, g = inst.tm, inst.start, inst.goal
    sc = P.Scorer(tm, NOMINAL)
    rows = []
    base = Lattice(tm, NOMINAL, 8)
    h0 = base.heading_index(inst.h8)
    ref8 = make_primitives(8, ds=REF_DS)[2]
    a_pt, _ = base.plan(s, h0, g, arcs=False)
    e_pt = ref_energy(sc, a_pt, base, ref8)
    for ds in (0.1, 0.25, 0.5, 1.0):
        lat = base if ds == 0.1 else Lattice(tm, NOMINAL, 8, ds=ds)
        a, info = lat.plan(s, h0, g)
        e = ref_energy(sc, a, lat, ref8)
        rows.append(dict(study="sampling", setting=f"{ds} cell", group=kind, idx=idx, E_ref_pt=e_pt, E=e,
                         plan_vs_ref=(info["cost"] - e) / e * 100))
    for name, scales in ARC_SETS.items():
        if name.endswith("(default)"):
            continue
        lat = Lattice(tm, NOMINAL, 8, arc_scales=scales)
        a, _ = lat.plan(s, h0, g)
        e = ref_energy(sc, a, lat, make_primitives(8, scales, ds=REF_DS)[2])
        rows.append(dict(study="arcs", setting=name, group=kind, idx=idx, E_ref_pt=e_pt, E=e))
    # 0.5 m cells, same metric radii
    tm2 = M.upsample(tm, 2)
    s2, g2 = (2 * s[0], 2 * s[1]), (2 * g[0], 2 * g[1])
    if tm2.free[s2] and tm2.free[g2]:
        sc2 = P.Scorer(tm2, NOMINAL)
        lat2 = Lattice(tm2, NOMINAL, 8, arc_scales=(2, 4, 8, 16))
        a, _ = lat2.plan(s2, h0, g2)
        b, _ = lat2.plan(s2, h0, g2, arcs=False)
        if a and b:
            rows.append(dict(study="cell", setting="cell size 0.5 m", group=kind, idx=idx,
                             E_ref_pt=sc2.evaluate(lat2.elements(b))["energy"], E=sc2.evaluate(lat2.elements(a))["energy"]))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    jobs = [(k, i) for k in M.FAMILIES for i in range(a.n)]
    with Pool() as pool:
        df = pd.DataFrame([r for rs in pool.imap_unordered(job, jobs) for r in rs])
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    df.to_csv(os.path.join(a.out, "discretisation.csv"), index=False)
    out = []
    for (study, setting), d in df.groupby(["study", "setting"], sort=False):
        ch = (d.E - d.E_ref_pt) / d.E_ref_pt * 100
        r = dict(setting=setting, n=len(d), median=f"{ch.median():+.1f}%",
                 p10_p90=f"{ch.quantile(0.1):+.1f} / {ch.quantile(0.9):+.1f}%", hea8_higher=int((ch > 1e-6).sum()))
        if study == "sampling":
            r["plan_vs_ref_median_abs"] = f"{d.plan_vs_ref.abs().median():.2f}%"
            r["plan_vs_ref_max_abs"] = f"{d.plan_vs_ref.abs().max():.2f}%"
        out.append(r)
    t = pd.DataFrame(out)
    t.to_csv(os.path.join(a.out, "tables", "table18_discretisation.csv"), index=False)
    print(t.to_string(index=False))
