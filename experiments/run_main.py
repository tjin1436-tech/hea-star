"""Main experiment (Sections 6.1-6.5): nine planners on 150 synthetic instances and 100 DEM crops with synthetic
and WorldCover surfaces, for the nominal model (kappa = 1) and the high turning loss (kappa = 5).

Every planner plans with the kappa with which its path is scored. Each path is also replayed with the
stop-and-turn and pure-pursuit controllers (Section 6.4) and its grade is checked along its length (Section 6.5).

Usage:
    python experiments/run_main.py --data synthetic             # 150 instances
    python experiments/run_main.py --data dem                   # 100 crops x 2 surface versions (downloads data)
    python experiments/run_main.py --data synthetic --n 2       # quick test
Output (appended, so an interrupted run resumes): results/main_raw.csv, results/primitive_use.csv
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import csv
import math
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from heastar import instances as I          # noqa: E402
from heastar import maps as M               # noqa: E402
from heastar import paths as P              # noqa: E402
from heastar import planners as PL          # noqa: E402
from heastar import tracking as TR          # noqa: E402
from heastar.energy import SURFACES, Model  # noqa: E402

KAPPAS = (1.0, 5.0)
FIELDS = ["data", "group", "idx", "kappa", "planner", "success", "energy", "time_s", "length", "rotation_deg",
          "rot_inplace_deg", "point_turns", "arcs", "plan_ms", "expanded", "precompute_s", "grade_lon_deg",
          "grade_lat_deg", "soft_share", "arc_share", "st_energy", "st_collision", "pp_energy", "pp_collision",
          "pp20_energy", "dist", "relief", "start_x", "start_y", "goal_x", "goal_y", "h0"]
USE_FIELDS = ["data", "group", "idx", "kappa", "planner", "kind", "surface", "count", "length"]
SOFT = (SURFACES.index("sand"), SURFACES.index("grass"))


def path_shares(tm, elems):
    """Share of the path length on soft ground (sand, grass) and on arcs."""
    tot = soft = arc = 0.0
    for el in elems:
        if isinstance(el, P.Curve):
            seg = np.linalg.norm(np.diff(el.pts, axis=0), axis=1)
            c = P.cell_of((el.pts[:-1] + el.pts[1:]) / 2)
            s = tm.surface[c[:, 0], c[:, 1]]
            tot += seg.sum()
            soft += seg[np.isin(s, SOFT)].sum()
            if el.kind.startswith("A"):
                arc += seg.sum()
    return (soft / tot, arc / tot) if tot > 0 else (math.nan, math.nan)


def primitive_use(inst, pl):
    """Counts of point turns and arcs by surface class (surface at the primitive's middle sample)."""
    rows = {}
    for h, x, y, pi in pl.acts:
        p = pl.lattice.prims[h][pi]
        if p.kind == "S":
            continue
        mid = p.samples[len(p.samples) // 2] + (x, y)
        c = P.cell_of(mid)
        s = SURFACES[inst.tm.surface[c[0], c[1]]]
        k = ("point_turn" if p.kind == "R" else p.kind, s)
        cnt, L = rows.get(k, (0, 0.0))
        rows[k] = (cnt + 1, L + p.length * inst.tm.cell)
    return rows


def run_instance(inst, kappas=KAPPAS, planners=PL.PLANNERS, track=True, headings=(8, 16)):
    rows, use = [], []
    tm = inst.tm
    for kappa in kappas:
        model = Model(kappa=kappa)
        lats = PL.build_lattices(inst, model, headings)
        planned = PL.plan_all(inst, model, lats, planners)
        scored = PL.score(inst, planned, model)
        sc = P.Scorer(tm, model)
        for name in planners:
            r = dict(data=inst.data, group=inst.group, idx=inst.idx, kappa=kappa, planner=name)
            r.update(scored[name])
            pl = planned[name]
            if pl is not None:
                r["precompute_s"] = pl.lattice.precompute_s if pl.lattice is not None else 0.0
                r["grade_lon_deg"], r["grade_lat_deg"] = P.grade_along(tm, pl.elems)
                r["soft_share"], r["arc_share"] = path_shares(tm, pl.elems)
                if track:
                    st = TR.stop_and_turn(tm, model, pl.elems, inst.start, inst.th0, sc)
                    pp = TR.pure_pursuit(tm, model, pl.elems, inst.start, inst.th0, 60.0, sc)
                    pp20 = TR.pure_pursuit(tm, model, pl.elems, inst.start, inst.th0, 20.0, sc)
                    r.update(st_energy=st["energy"], st_collision=st["collision"], pp_energy=pp["energy"],
                             pp_collision=pp["collision"], pp20_energy=pp20["energy"])
                if pl.acts is not None and name.startswith("HEA*"):
                    for (kind, s), (cnt, L) in primitive_use(inst, pl).items():
                        use.append(dict(data=inst.data, group=inst.group, idx=inst.idx, kappa=kappa,
                                        planner=name, kind=kind, surface=s, count=cnt, length=L))
            r.update(dist=math.dist(inst.start, inst.goal) * tm.cell, relief=float(np.ptp(tm.elev)),
                     start_x=inst.start[0], start_y=inst.start[1], goal_x=inst.goal[0], goal_y=inst.goal[1],
                     h0=inst.h8)
            rows.append(r)
    return rows, use


def job(args):
    data, group, idx, cache = args
    t0 = time.time()
    if data == "synthetic":
        insts = [I.synthetic(group, idx)]
    else:
        insts = list(I.dem(group, idx, cache=cache))
    rows, use = [], []
    try:
        for inst in insts:
            r, u = run_instance(inst)
            rows += r
            use += u
    except Exception as e:                       # report and continue with the other instances
        import traceback
        traceback.print_exc()
        return (data, group, idx, f"FAILED: {e!r}"), [], [], time.time() - t0
    return (data, group, idx), rows, use, time.time() - t0


def append(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["data"] if r["data"] == "synthetic" else "dem", r["group"], int(r["idx"]))
                for r in csv.DictReader(f)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "dem"], required=True)
    ap.add_argument("--n", type=int, default=None, help="instances per family (50) / crops per region (25)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default="data")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw, usep = os.path.join(a.out, "main_raw.csv"), os.path.join(a.out, "primitive_use.csv")
    if a.data == "synthetic":
        jobs = [("synthetic", k, i, a.cache) for k in M.FAMILIES for i in range(a.n or 50)]
    else:
        for r in M.REGIONS:                       # download once, before forking
            I.dem_source(r, a.cache)
        jobs = [("dem", r, i, a.cache) for r in M.REGIONS for i in range(a.n or 25)]
    done = done_keys(raw)
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already done)", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for k, (key, rows, use, dt) in enumerate(pool.imap_unordered(job, jobs)):
            append(raw, FIELDS, rows)
            if use:
                append(usep, USE_FIELDS, use)
            print(f"[{k + 1}/{len(jobs)}] {key} {dt:.0f}s  elapsed {time.time() - t0:.0f}s", flush=True)
