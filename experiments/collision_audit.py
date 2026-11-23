"""Collision audit (Section 6.7).

All instances of the synthetic and DEM experiments are re-planned (kappa = 1) and every path is checked with a
sampling of 0.01 cells: a path *penetrates* if it enters the interior of a blocked cell and *touches* if it
touches a blocked cell's boundary or corner (including a diagonal squeeze between two blocked cells).
With --naive, HEA*-8 is also planned with a non-conservative check that only tests the cells containing the
samples, taken every 0.25 cells (straight moves at 5 points), to show why the conservative check is needed.

Usage: python experiments/collision_audit.py [--data synthetic dem] [--out results]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import math
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from heastar import instances as I          # noqa: E402
from heastar import maps as M               # noqa: E402
from heastar import paths as P              # noqa: E402
from heastar import planners as PL          # noqa: E402
from heastar.energy import NOMINAL          # noqa: E402
from heastar.lattice import Lattice, make_primitives   # noqa: E402

FINE = 0.01


def fine_points(pl, fine_prims):
    pts = []
    if pl.acts is not None:
        for h, x, y, pi in pl.acts:
            p = fine_prims[pl.lattice.H][h][pi]
            if p.kind != "R":
                pts.append(p.samples + (x, y))
    else:
        for a, b in zip(pl.vertices[:-1], pl.vertices[1:]):
            pts.append(P.line_samples(a, b, FINE))
    return np.vstack(pts)


def depth(free, Q):
    """Largest penetration depth into a blocked cell (cells); >= 0 means touching, > 0 entering."""
    n, m = free.shape
    best = -np.inf
    base = np.floor(Q + 0.5).astype(int)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cx, cy = base[:, 0] + dx, base[:, 1] + dy
            ok = (cx >= 0) & (cy >= 0) & (cx < n) & (cy < m)
            blk = np.ones(len(Q), bool)
            blk[ok] = ~free[cx[ok], cy[ok]]
            if blk.any():
                d = 0.5 - np.maximum(np.abs(Q[:, 0] - cx), np.abs(Q[:, 1] - cy))
                best = max(best, float(d[blk].max()))
    return best


class NaiveLattice(Lattice):
    """Lattice whose collision check only tests the cells containing samples every 0.25 cells."""

    def __init__(self, tm, model, H):
        import heastar.paths as PP
        orig = PP.swept_offsets
        PP.swept_offsets = lambda S, margin=None: sorted({tuple(c) for c in P.cell_of(S).tolist()})
        try:
            super().__init__(tm, model, H, ds=0.25)
        finally:
            PP.swept_offsets = orig


def audit(inst, naive):
    fine = {H: make_primitives(H, ds=FINE)[2] for H in (8, 16)}
    lats = PL.build_lattices(inst, NOMINAL)
    planned = PL.plan_all(inst, NOMINAL, lats)
    if naive:
        nl = NaiveLattice(inst.tm, NOMINAL, 8)
        acts, info = nl.plan(inst.start, nl.heading_index(inst.h8), inst.goal)
        planned["HEA*-8 (naive check)"] = PL.Planned("HEA*-8 (naive check)", nl.elements(acts), 0, 0, acts, nl) \
            if acts else None
    rows = []
    for name, pl in planned.items():
        r = dict(data=inst.data, group=inst.group, idx=inst.idx, planner=name, found=pl is not None)
        if pl is not None:
            if name.endswith("(naive check)"):
                pts = np.vstack([make_primitives(8, ds=FINE)[2][h][pi].samples + (x, y) for h, x, y, pi in pl.acts
                                 if pl.lattice.prims[h][pi].kind != "R"])
            else:
                pts = fine_points(pl, fine)
            d = depth(inst.tm.free, pts)
            r.update(max_depth=d, penetrating=d > 1e-6, touching=d >= -1e-9)
        rows.append(r)
    return rows


def job(args):
    data, group, idx, naive = args
    if data == "synthetic":
        insts = [I.synthetic(group, idx)]
    else:
        insts = list(I.dem(group, idx))
    return [r for inst in insts for r in audit(inst, naive and data == "synthetic")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["synthetic", "dem"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--no-naive", dest="naive", action="store_false", help="skip the naive-check HEA*-8 variant")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    jobs = []
    if "synthetic" in a.data:
        jobs += [("synthetic", k, i, a.naive) for k in M.FAMILIES for i in range(a.n or 50)]
    if "dem" in a.data:
        for r in M.REGIONS:
            I.dem_source(r)
        jobs += [("dem", r, i, False) for r in M.REGIONS for i in range(a.n or 25)]
    with Pool() as pool:
        df = pd.DataFrame([r for rs in pool.imap_unordered(job, jobs) for r in rs])
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    df.to_csv(os.path.join(a.out, "collision_audit.csv"), index=False)
    s = df.groupby(["data", "planner"]).agg(paths=("found", "sum"), penetrating=("penetrating", "sum"),
                                             touching=("touching", "sum"), max_depth=("max_depth", "max"))
    s.to_csv(os.path.join(a.out, "tables", "collision_audit_summary.csv"))
    print(s.to_string())
