"""Collision audit of all planned paths (second review, major comment 2).
Re-plans every instance of the main experiments and checks each path geometrically with a fine
sampling (0.01 cell): a point is in collision if it lies in a blocked cell. We report
  penetrating: the path enters the interior of a blocked cell (depth > 1e-6 cell);
  touching:    the path touches a blocked cell's boundary or corner (closed cell), incl. diagonal squeezes.
Usage: python collision_audit.py <code_dir> <out.csv> [--dem]"""
import argparse, csv, math, os, sys
from multiprocessing import Pool
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("code"); ap.add_argument("out"); ap.add_argument("--dem", action="store_true")
ap.add_argument("--n", type=int, default=50); ap.add_argument("--crops", type=int, default=25)
A = ap.parse_args()
sys.path.insert(0, os.path.abspath(A.code))
import energy_model as EM    # noqa
import lattice as LT         # noqa
import run_energy as RE      # noqa
import run_dem as RD         # noqa

FINE = 0.01


def geometry(lat, acts):
    """Fine sample points (cell coords) of an action sequence."""
    pts = []
    for (x, y, h), pi in acts:
        p = lat.prims[h][pi]
        if p["name"] == "R":
            continue
        off = np.array(p["off"], float)
        if p["name"] == "S":
            n = int(math.ceil(np.linalg.norm(off) / FINE)) + 1
            S = np.linspace([0, 0], off, n)
        else:
            th = h * LT.STEP
            n = int(math.ceil(p["length"] / FINE)) * 3 + 1      # bezier speed varies: oversample
            S = LT._bezier(off, th, th + p["dh"] * LT.STEP, n)
        pts.append(S + [x, y])
    return np.vstack(pts) if pts else np.zeros((0, 2))


def depth(free, P):
    """Max penetration depth into blocked cells (negative = clearance to nearest blocked cell touched)."""
    n = free.shape[0]
    best = -np.inf
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cx = np.rint(P[:, 0]).astype(int) + dx
            cy = np.rint(P[:, 1]).astype(int) + dy
            ok = (cx >= 0) & (cy >= 0) & (cx < n) & (cy < n)
            blk = np.zeros(len(P), bool)
            blk[ok] = ~free[cx[ok], cy[ok]]
            blk[~ok] = True
            if blk.any():
                d = 0.5 - np.maximum(np.abs(P[:, 0] - cx), np.abs(P[:, 1] - cy))
                best = max(best, float(d[blk].max()))
    return best


def audit(tm, full, s, g, h0, rb, label, idx):
    rows = []
    lats = {"TA*+turn": LT.Lattice(tm, rb, arcs=False), "HEA* (ours)": full}
    for m in ("A*", "TA*", "TA*+turn", "HEA* (ours)"):
        if m in ("A*", "TA*"):
            p, _ = LT.grid_astar(full, s, g, energy=(m == "TA*"))
            acts, lat = (full.grid_path_to_actions(p, h0) if p else None), full
        else:
            acts, _ = lats[m].plan(s, h0, g)
            lat = lats[m]
        if not acts:
            rows.append(dict(map=label, map_id=idx, method=m, found=False)); continue
        d = depth(tm.free, geometry(lat, acts))
        n_arc = sum(1 for (x, y, h), pi in acts if lat.prims[h][pi]["name"].startswith("A"))
        rows.append(dict(map=label, map_id=idx, method=m, found=True, max_depth=d,
                         penetrating=d > 1e-6, touching=d >= -1e-9, n_arcs=n_arc,
                         energy=lat.evaluate(acts)["energy"]))
    return rows


def job_syn(a):
    kind, idx, size, seed, rb = a
    rng = np.random.default_rng([seed, RE.KINDS.index(kind), idx])
    for _ in range(20):
        tm = RE.make_map(kind, size, rng, rb)
        ends = RE.pick(tm, rng)
        if ends is None:
            continue
        s, g, h0 = ends
        full = LT.Lattice(tm, rb)
        if LT.grid_astar(full, s, g, energy=False)[0] is None:
            continue
        return audit(tm, full, s, g, h0, rb, kind, idx)
    return []


def job_dem(a):
    region, idx, size, seed, rb, dem_path = a
    elev1m = np.load(dem_path, mmap_mode="r")
    rng = np.random.default_rng([seed, list(RD.REGIONS).index(region), idx])
    for _ in range(20):
        tm = RD.crop_map(elev1m, size, rng, rb)
        if tm is None:
            return []
        ends = RE.pick(tm, rng)
        if ends is None:
            continue
        s, g, h0 = ends
        full = LT.Lattice(tm, rb)
        if LT.grid_astar(full, s, g, energy=False)[0] is None:
            continue
        return audit(tm, full, s, g, h0, rb, region, idx)
    return []


if __name__ == "__main__":
    from scipy.ndimage import zoom
    rb = EM.Robot()
    if A.dem:
        here = os.path.dirname(os.path.abspath(__file__))
        tiles = os.path.join(here, "..", "results_dem", "tiles")
        dems = {}
        for reg, (lat, lon, _) in RD.REGIONS.items():
            elev, mpp = RD.fetch_region(lat, lon, cache=tiles)
            p = A.out + f".{reg}.npy"; np.save(p, zoom(elev, mpp, order=1).astype(np.float32)); dems[reg] = p
        jobs = [(r, i, 80, 21, rb, dems[r]) for r in RD.REGIONS for i in range(A.crops)]
        fn = job_dem
    else:
        jobs = [(k, i, 80, 7, rb) for k in RE.KINDS for i in range(A.n)]
        fn = job_syn
    with Pool() as pool:
        rows = [r for rs in pool.map(fn, jobs) for r in rs]
    if A.dem:
        for p in dems.values():
            os.remove(p)
    keys = sorted({k for r in rows for k in r}, key=lambda k: ["map", "map_id", "method", "found"].index(k)
                  if k in ("map", "map_id", "method", "found") else 9)
    with open(A.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("done", len(rows))
