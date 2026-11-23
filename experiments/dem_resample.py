"""Resampling of the DEM (Section 6.7): the DEM experiment (synthetic surfaces, kappa = 1, 8-heading lattice)
repeated with nearest-neighbour, bilinear and cubic upsampling of the ~3.5 m Terrarium tiles to 1 m cells.

The crops, synthetic surfaces, starts and goals are those of the bilinear experiment; the traversability is
recomputed from the resampled elevation, and instances whose endpoints become blocked or disconnected are skipped.

Usage: python experiments/dem_resample.py [--n 25] [--out results]
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
from heastar.energy import NOMINAL          # noqa: E402

ORDERS = {"nearest": 0, "bilinear": 1, "cubic": 3}
PLAN = ("TA*", "TA*+turn", "HEA*-8")


def job(args):
    region, idx = args
    base, _ = I.dem(region, idx)
    x, y = base.origin
    n = base.tm.shape[0]
    rows, slopes = [], []
    for name, order in ORDERS.items():
        src = I.dem_source(region, order=order)
        z = np.array(src.elev[x:x + n, y:y + n], float)
        free = ~M.steep_mask(z, 1.0, NOMINAL.robot.max_grade_deg)
        free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
        tm = M.TerrainMap(z, base.tm.surface, free, 1.0, region)
        g = tm.grade_deg()[1:-1, 1:-1].ravel()
        slopes.append(dict(resampling=name, region=region, idx=idx, mean=g.mean(), p95=np.percentile(g, 95),
                           steep_share=(g > 20).mean()))
        if not (free[base.start] and free[base.goal] and M.connected(free, base.start, base.goal)):
            continue
        inst = PL.Instance(tm, base.start, base.goal, base.h8, "dem_syn", region, idx)
        res = PL.score(inst, PL.plan_all(inst, NOMINAL, PL.build_lattices(inst, NOMINAL, (8,)), PLAN), NOMINAL)
        for p in PLAN:
            rows.append(dict(resampling=name, region=region, idx=idx, planner=p, energy=res[p]["energy"]))
    return rows, slopes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    for r in M.REGIONS:
        for o in ORDERS.values():
            I.dem_source(r, order=o)
    jobs = [(r, i) for r in M.REGIONS for i in range(a.n)]
    rows, slopes = [], []
    with Pool() as pool:
        for r, s in pool.imap_unordered(job, jobs):
            rows += r
            slopes += s
    df, sl = pd.DataFrame(rows), pd.DataFrame(slopes)
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    df.to_csv(os.path.join(a.out, "dem_resample_raw.csv"), index=False)
    sl.to_csv(os.path.join(a.out, "dem_resample_slopes.csv"), index=False)
    out = []
    for name in ORDERS:
        w = df[df.resampling == name].pivot_table(index=["region", "idx"], columns="planner", values="energy").dropna()
        s = sl[sl.resampling == name]
        out.append(dict(resampling=name, instances=len(w), mean_slope_deg=round(s["mean"].mean(), 1),
                        p95_slope_deg=round(s.p95.mean(), 1), steep_share_pct=round(100 * s.steep_share.mean(), 1),
                        hea8_vs_ta=f"{ST.median_change(w['HEA*-8'], w['TA*'])[0]:+.1f}%",
                        turn_vs_ta=f"{ST.median_change(w['TA*+turn'], w['TA*'])[0]:+.1f}%",
                        arcs_hea8_vs_turn=f"{ST.median_change(w['HEA*-8'], w['TA*+turn'])[0]:+.1f}%"))
    t = pd.DataFrame(out)
    t.to_csv(os.path.join(a.out, "tables", "dem_resample.csv"), index=False)
    print(t.to_string(index=False))
