"""Optimality check (Section 6.7): on small maps (40 x 40, 10 per family) HEA* is compared with Dijkstra's
algorithm on the same lattice (h = 0), and the heuristic (9) is checked against the exact cost-to-go of every
reachable state, computed by a backward Dijkstra search from the goal.

Usage: python experiments/optimality.py [--n 10] [--size 40] [--headings 8 16] [--out results]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import heapq
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from heastar import instances as I          # noqa: E402
from heastar import maps as M               # noqa: E402
from heastar import paths as P              # noqa: E402
from heastar.energy import NOMINAL          # noqa: E402
from heastar.lattice import Lattice         # noqa: E402

SEED = 41


def cost_to_go(lat, goal):
    """Exact minimum energy from every state (h, x, y) to the goal cell with any heading."""
    H = lat.H
    rev = {}
    for h in range(H):
        for pi, p in enumerate(lat.prims[h]):
            c = lat.cost[h, :, :, pi]
            xs, ys = np.nonzero(np.isfinite(c))
            for x, y in zip(xs.tolist(), ys.tolist()):
                v = ((h + p.dh) % H, x + p.off[0], y + p.off[1])
                rev.setdefault(v, []).append(((h, x, y), float(c[x, y])))
    dist = {}
    pq = [(0.0, (h, goal[0], goal[1])) for h in range(H)]
    heapq.heapify(pq)
    while pq:
        d, u = heapq.heappop(pq)
        if u in dist:
            continue
        dist[u] = d
        for w, c in rev.get(u, ()):
            if w not in dist and (w[1], w[2]) != tuple(goal):
                heapq.heappush(pq, (d + c, w))
    return dist


def job(args):
    kind, idx, size, H = args
    inst = I.synthetic(kind, idx, size=size, seed=SEED)
    lat = Lattice(inst.tm, NOMINAL, H)
    h0 = lat.heading_index(inst.h8)
    a1, i1 = lat.plan(inst.start, h0, inst.goal)
    a2, i2 = lat.plan(inst.start, h0, inst.goal, heuristic=False)
    sc = P.Scorer(inst.tm, NOMINAL)
    e1, e2 = sc.evaluate(lat.elements(a1))["energy"], sc.evaluate(lat.elements(a2))["energy"]
    ctg = cost_to_go(lat, inst.goal)
    hm = lat.heuristic_map(inst.goal)
    viol, worst, ratio = 0, 0.0, []
    for (h, x, y), d in ctg.items():
        hv = hm[x, y]
        if hv > d + 1e-6:
            viol += 1
            worst = max(worst, hv - d)
        if d > 0:
            ratio.append(hv / d)
    return dict(headings=H, group=kind, idx=idx, E_hea=e1, E_dijkstra=e2, rel_diff=(e1 - e2) / e2,
                exp_hea=i1["expanded"], exp_dijkstra=i2["expanded"], states=len(ctg), violations=viol,
                worst_violation_J=worst, h_ratio_median=float(np.median(ratio)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--size", type=int, default=40)
    ap.add_argument("--headings", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    jobs = [(k, i, a.size, H) for H in a.headings for k in M.FAMILIES for i in range(a.n)]
    with Pool() as pool:
        df = pd.DataFrame(pool.map(job, jobs))
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    df.to_csv(os.path.join(a.out, "optimality.csv"), index=False)
    s = df.groupby("headings").agg(instances=("idx", "size"), max_rel_diff=("rel_diff", lambda v: v.abs().max()),
                                   states_checked=("states", "sum"), violations=("violations", "sum"),
                                   h_ratio_median=("h_ratio_median", "median"),
                                   exp_hea=("exp_hea", "sum"), exp_dijkstra=("exp_dijkstra", "sum"))
    s["fewer_expansions_pct"] = (1 - s.exp_hea / s.exp_dijkstra) * 100
    s.to_csv(os.path.join(a.out, "tables", "optimality_summary.csv"))
    print(s.to_string())
