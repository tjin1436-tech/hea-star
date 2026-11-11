"""Additional experiments requested in the review (major comments 2, 3, 5).

Modes (python rev_experiments.py <mode> [--n N] [--out DIR]):
  features     re-run the 150 synthetic instances of the paper (same seeds) for TA*+turn and HEA*,
               record where the arc primitives help (arc count, share of path on sand/grass, ...)
  uncertainty  Monte-Carlo model uncertainty. Each instance draws a 'true' parameter set
               (turning-moment gain, rolling / lateral friction scale, per-surface noise, Wong
               slope coefficient, regeneration).  Two cases are evaluated with the true model:
                 consistent - every planner plans with the true parameters;
                 mismatch   - every planner plans with the nominal parameters (model error).
  optimality   small maps: HEA* vs Dijkstra (h = 0) on the same lattice, and exhaustive check
               h(s) <= exact cost-to-go for every lattice state (backward Dijkstra).
  discret      discretisation: arc sampling interval, arc-radius set and grid resolution.
All planned paths are scored with the stated true model.
"""
import argparse
import csv
import heapq
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import energy_model as EM      # noqa: E402
import lattice as LT           # noqa: E402
import run_energy as RE        # noqa: E402

NOMINAL_TERRAINS = dict(EM.TERRAINS)
KINDS = RE.KINDS


def set_terrains(scale_r=1.0, scale_l=1.0, per=None):
    per = per or {}
    EM.TERRAINS.clear()
    for k, (r, l) in NOMINAL_TERRAINS.items():
        fr, fl = per.get(k, (1.0, 1.0))
        EM.TERRAINS[k] = (r * scale_r * fr, l * scale_l * fl)


def set_wong(b):
    EM.WONG_B = b


def new_case(kind, idx, size, seed, rb):
    """Same map / start / goal / heading generator as run_energy.run_case."""
    rng = np.random.default_rng([seed, KINDS.index(kind), idx])
    for _ in range(20):
        tm = EM.make_map(kind, size, rng, rb)
        ends = RE.pick(tm, rng)
        if ends is None:
            continue
        s, g, h0 = ends
        full = LT.Lattice(tm, rb)
        if LT.grid_astar(full, s, g, energy=False)[0] is None:
            continue
        return tm, full, s, g, h0
    return None


def plan_all(lat_full, lat_pt, s, g, h0):
    """Actions of A*, TA* (both executed with point turns), TA*+turn and HEA*."""
    out = {}
    for m in ("A*", "TA*"):
        p, _ = LT.grid_astar(lat_full, s, g, energy=(m == "TA*"))
        out[m] = ("full", lat_full.grid_path_to_actions(p, h0) if p else None)
    out["TA*+turn"] = ("pt", lat_pt.plan(s, h0, g)[0])
    out["HEA*"] = ("full", lat_full.plan(s, h0, g)[0])
    return out


def score(plans, true_full, true_pt):
    lat = {"full": true_full, "pt": true_pt}
    return {m: (lat[k].evaluate(a)["energy"] if a else math.nan) for m, (k, a) in plans.items()}


# ---------------------------------------------------------------- features
def job_features(args):
    kind, idx, size, seed = args
    rb = EM.Robot()
    c = new_case(kind, idx, size, seed, rb)
    if c is None:
        return None
    tm, full, s, g, h0 = c
    pt = LT.Lattice(tm, rb, arcs=False)
    a_h, _ = full.plan(s, h0, g)
    a_p, _ = pt.plan(s, h0, g)
    e_h, e_p = full.evaluate(a_h)["energy"], pt.evaluate(a_p)["energy"]
    soft = np.isin(tm.terrain, [EM.TERRAIN_NAMES.index("sand"), EM.TERRAIN_NAMES.index("grass")])
    n_arc, arc_len, soft_len, tot_len, arc_soft = 0, 0.0, 0.0, 0.0, 0
    for (x, y, h), pi in a_h:
        p = full.prims[h][pi]
        if p["name"] == "R":
            continue
        S = p["samples"]
        mid = S[len(S) // 2]
        on_soft = bool(soft[int(round(x + mid[0])), int(round(y + mid[1]))])
        tot_len += p["length"]
        soft_len += p["length"] * on_soft
        if p["name"].startswith("A"):
            n_arc += 1
            arc_len += p["length"]
            arc_soft += on_soft
    return dict(map=kind, map_id=idx, E_turn=e_p, E_hea=e_h, gain_pct=(e_h - e_p) / e_p * 100,
                dist=float(np.hypot(g[0] - s[0], g[1] - s[1])), length=tot_len, n_arcs=n_arc,
                arc_share=arc_len / max(tot_len, 1e-9), soft_share_path=soft_len / max(tot_len, 1e-9),
                arcs_on_soft=arc_soft / max(n_arc, 1), relief=float(np.ptp(tm.elev)))


# ---------------------------------------------------------------- uncertainty
def draw_params(rng):
    return dict(turn_gain=float(np.exp(rng.uniform(np.log(0.5), np.log(6.0)))),
                scale_r=float(rng.uniform(0.6, 1.4)), scale_l=float(rng.uniform(0.6, 1.4)),
                per={k: (float(rng.uniform(0.7, 1.3)), float(rng.uniform(0.7, 1.3))) for k in NOMINAL_TERRAINS},
                wong_b=float(rng.uniform(0.075, 0.30)),
                regen=0.0 * float(rng.choice([0.0, 0.0, 0.3, 0.6])))   # admissibility needs regen = 0 (draw kept for RNG stream)


def build(tm, rb, prm=None):
    if prm:
        set_terrains(prm["scale_r"], prm["scale_l"], prm["per"]); set_wong(prm["wong_b"])
    else:
        set_terrains(); set_wong(0.15)
    full, pt = LT.Lattice(tm, rb), LT.Lattice(tm, rb, arcs=False)
    set_terrains(); set_wong(0.15)
    return full, pt


CONSERVATIVE = os.environ.get('CONSERVATIVE') == '1'


def job_uncertainty(args):
    kind, idx, size, seed = args
    rng = np.random.default_rng([seed, 99, KINDS.index(kind), idx])
    prm = draw_params(rng)
    rb0 = EM.Robot()
    rbt = EM.Robot(turn_gain=prm["turn_gain"], regen=prm["regen"])
    c = new_case(kind, idx, size, seed, rb0)
    if c is None:
        return []
    tm, nom_full, s, g, h0 = c
    nom_pt = LT.Lattice(tm, rb0, arcs=False)
    tru_full, tru_pt = build(tm, rbt, prm)
    rows = []
    cases = [("mismatch", (nom_full, nom_pt)), ("consistent", (tru_full, tru_pt))]
    if CONSERVATIVE:   # planner assumes a high turning moment (kappa = 5), nominal friction
        rbc = EM.Robot(turn_gain=5.0)
        cases.append(("conservative", (LT.Lattice(tm, rbc), LT.Lattice(tm, rbc, arcs=False))))
    for case, (pf, pp) in cases:
        e = score(plan_all(pf, pp, s, g, h0), tru_full, tru_pt)
        r = dict(case=case, map=kind, map_id=idx, turn_gain=prm["turn_gain"], scale_r=prm["scale_r"],
                 scale_l=prm["scale_l"], wong_b=prm["wong_b"], regen=prm["regen"])
        r.update({f"E_{m}": v for m, v in e.items()})
        rows.append(r)
    return rows


# ---------------------------------------------------------------- optimality
class ZeroH(LT.Lattice):
    def heuristic(self, x, y, goal):
        return 0.0


def cost_to_go(lat, goal):
    """Exact optimal cost from every (x,y,h) to the goal cell (any heading): backward Dijkstra."""
    n = lat.n
    rev = {}
    for h in range(8):
        for pi, p in enumerate(lat.prims[h]):
            c = lat.cost[h][pi]
            xs, ys = np.nonzero(np.isfinite(c))
            for x, y in zip(xs, ys):
                v = (x + p["off"][0], y + p["off"][1], (h + p["dh"]) % 8)
                rev.setdefault(v, []).append(((x, y, h), c[x, y]))
    dist = {}
    pq = [(0.0, (goal[0], goal[1], h)) for h in range(8)]
    heapq.heapify(pq)
    while pq:
        d, u = heapq.heappop(pq)
        if u in dist:
            continue
        dist[u] = d
        if (u[0], u[1]) == tuple(goal) and d > 0:
            continue
        for w, c in rev.get(u, ()):
            if w not in dist and (w[0], w[1]) != tuple(goal):
                heapq.heappush(pq, (d + c, w))
    return dist


def job_optimality(args):
    kind, idx, size, seed = args
    rb = EM.Robot()
    c = new_case(kind, idx, size, seed, rb)
    if c is None:
        return None
    tm, full, s, g, h0 = c
    a1, i1 = full.plan(s, h0, g)
    dj = ZeroH.__new__(ZeroH); dj.__dict__.update(full.__dict__)
    a2, i2 = dj.plan(s, h0, g)
    e1, e2 = full.evaluate(a1)["energy"], full.evaluate(a2)["energy"]
    ctg = cost_to_go(full, g)
    viol, worst, ratio = 0, 0.0, []
    for (x, y, h), d in ctg.items():
        hv = full.heuristic(x, y, g)
        if hv > d + 1e-6:
            viol += 1
            worst = max(worst, hv - d)
        if d > 0:
            ratio.append(hv / d)
    return dict(map=kind, map_id=idx, E_hea=e1, E_dijkstra=e2, rel_diff=(e1 - e2) / e2,
                exp_hea=i1["expanded"], exp_dijkstra=i2["expanded"], states_checked=len(ctg),
                h_violations=viol, worst_violation_J=worst, h_over_cost_median=float(np.median(ratio)))


# ---------------------------------------------------------------- discretisation
def upsample(tm, f=2):
    z = zoom(tm.elev, f, order=1)
    t = np.kron(tm.terrain, np.ones((f, f), int))
    fr = np.kron(tm.free, np.ones((f, f), bool))
    fr[0, :] = fr[-1, :] = fr[:, 0] = fr[:, -1] = False
    return EM.TerrainMap(z, t, fr, tm.cell / f)


def with_settings(tm, rb, ds=0.1, scales=(1, 2, 4, 8), arcs=True):
    LT.SAMPLE_DS, LT.ARC_SCALES = ds, tuple(scales)
    lat = LT.Lattice(tm, rb, arcs=arcs)
    LT.SAMPLE_DS, LT.ARC_SCALES = 0.1, (1, 2, 4, 8)
    return lat


def job_discret(args):
    kind, idx, size, seed = args
    rb = EM.Robot()
    c = new_case(kind, idx, size, seed, rb)
    if c is None:
        return []
    tm, full, s, g, h0 = c
    rows = []
    ref = with_settings(tm, rb, ds=0.05)                      # fine reference for edge costs
    pt = LT.Lattice(tm, rb, arcs=False)
    e_pt = pt.evaluate(pt.plan(s, h0, g)[0])["energy"]
    base = dict(map=kind, map_id=idx, E_turn=e_pt)
    # (a) sampling interval: plan with ds, score with the fine reference (same primitive indices)
    for ds in (0.1, 0.25, 0.5, 1.0):
        lat = full if ds == 0.1 else with_settings(tm, rb, ds=ds)
        a, _ = lat.plan(s, h0, g)
        rows.append(dict(base, study="sample_ds", setting=ds, E_plan=lat.evaluate(a)["energy"],
                         E_ref=ref.evaluate(a)["energy"]))
    # (b) arc-radius set (radii ~2.9 k m)
    for sc in ((1, 2, 4, 8), (1, 2), (2, 4, 8), (1, 2, 4, 8, 16), (1, 2, 3, 4, 6, 8)):
        lat = full if sc == (1, 2, 4, 8) else with_settings(tm, rb, scales=sc)
        a, _ = lat.plan(s, h0, g)
        rows.append(dict(base, study="arc_set", setting="-".join(map(str, sc)),
                         E_plan=lat.evaluate(a)["energy"], E_ref=ref.evaluate(a)["energy"] if sc == (1, 2, 4, 8) else math.nan))
    # (c) grid resolution 0.5 m (same metric arc radii: scales doubled)
    tm2 = upsample(tm, 2)
    s2, g2 = (2 * s[0], 2 * s[1]), (2 * g[0], 2 * g[1])
    if tm2.free[s2] and tm2.free[g2]:
        f2 = with_settings(tm2, rb, scales=(2, 4, 8, 16))
        p2 = LT.Lattice(tm2, rb, arcs=False)
        ah, _ = f2.plan(s2, h0, g2)
        ap, _ = p2.plan(s2, h0, g2)
        if ah and ap:
            rows.append(dict(base, study="cell_0.5m", setting=0.5, E_turn=p2.evaluate(ap)["energy"],
                             E_plan=f2.evaluate(ah)["energy"], E_ref=math.nan))
    return rows


# ---------------------------------------------------------------- any-angle point-turn baseline
def seg_energy(tm, rb, a, b, ds=0.1):
    """Straight drive a->b (cell coords) with the true model; inf if it touches a blocked cell
    (same conservative swept-cell check as the lattice primitives)."""
    from scipy.ndimage import map_coordinates
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    if L < 1e-9:
        return 0.0
    k = int(math.ceil(L / ds)) + 1
    P = np.linspace(a, b, k)
    n = tm.elev.shape[0]
    a0 = np.rint(a).astype(int)
    for dx, dy in LT.swept_cells(P - a0):
        cx, cy = a0[0] + dx, a0[1] + dy
        if cx < 0 or cy < 0 or cx >= n or cy >= n or not tm.free[cx, cy]:
            return math.inf
    z = map_coordinates(tm.elev, [P[:, 0], P[:, 1]], order=1, mode="nearest")
    seg = L / (k - 1) * tm.cell
    mid = (P[:-1] + P[1:]) / 2
    mx, my = np.rint(mid[:, 0]).astype(int), np.rint(mid[:, 1]).astype(int)
    mu_r, mu_l0 = tm.mu_r[mx, my], tm.mu_l0[mx, my]
    return float(np.sum(EM.track_energy(seg, np.diff(z) / seg, mu_r, mu_l0, math.inf, rb)))


def poly_energy(tm, rb, pts, th0):
    E, th = 0.0, th0
    for a, b in zip(pts[:-1], pts[1:]):
        d = np.asarray(b, float) - np.asarray(a, float)
        if np.linalg.norm(d) < 1e-9:
            continue
        nth = math.atan2(d[1], d[0])
        dth = (nth - th + math.pi) % (2 * math.pi) - math.pi
        x, y = int(round(a[0])), int(round(a[1]))
        E += EM.rotate_energy(abs(dth), tm.mu_r[x, y], tm.mu_l0[x, y], rb)
        e = seg_energy(tm, rb, a, b)
        if not np.isfinite(e):
            return math.inf
        E += e
        th = nth
    return E


def waypoints(lat, acts):
    pts = []
    for (x, y, h), pi in acts:
        if not pts or pts[-1] != (x, y):
            pts.append((x, y))
    (x, y, h), pi = acts[-1]
    p = lat.prims[h][pi]
    pts.append((x + p["off"][0], y + p["off"][1]))
    return pts


def shortcut(tm, rb, pts, th0):
    """Greedy energy-decreasing line-of-sight shortcutting (any-angle point turns)."""
    pts = list(pts)
    best = poly_energy(tm, rb, pts, th0)
    i = 0
    while i < len(pts) - 2:
        for j in range(len(pts) - 1, i + 1, -1):
            cand = pts[:i + 1] + pts[j:]
            e = poly_energy(tm, rb, cand, th0)
            if e < best - 1e-9:
                pts, best = cand, e
                break
        else:
            i += 1
    return pts, best


def job_anyangle(args):
    kind, idx, size, seed = args
    rb = EM.Robot()
    c = new_case(kind, idx, size, seed, rb)
    if c is None:
        return None
    tm, full, s, g, h0 = c
    pt = LT.Lattice(tm, rb, arcs=False)
    a_p, _ = pt.plan(s, h0, g)
    e_lat = pt.evaluate(a_p)["energy"]
    wp = waypoints(pt, a_p)
    th0 = h0 * LT.STEP
    e_poly = poly_energy(tm, rb, wp, th0)
    t0 = time.perf_counter()
    sp, e_sc = shortcut(tm, rb, wp, th0)
    t_sc = (time.perf_counter() - t0) * 1e3
    a_h, _ = full.plan(s, h0, g)
    e_h = full.evaluate(a_h)["energy"]
    return dict(map=kind, map_id=idx, E_turn_lattice=e_lat, E_turn_poly_check=e_poly, E_turn_anyangle=e_sc,
                n_wp_before=len(wp), n_wp_after=len(sp), smooth_ms=t_sc, E_hea=e_h,
                hea_vs_turn_pct=(e_h - e_lat) / e_lat * 100, hea_vs_anyangle_pct=(e_h - e_sc) / e_sc * 100)


# ---------------------------------------------------------------- main
def write(rows, path):
    rows = [r for r in rows if r]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["features", "uncertainty", "optimality", "discret", "anyangle"])
    ap.add_argument("--n", type=int, default=50, help="instances per map family")
    ap.add_argument("--size", type=int, default=80)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seed = {"features": 7, "uncertainty": 31, "optimality": 41, "discret": 7, "anyangle": 7}[a.mode]
    fn = {"features": job_features, "uncertainty": job_uncertainty, "optimality": job_optimality,
          "discret": job_discret, "anyangle": job_anyangle}[a.mode]
    jobs = [(k, i, a.size, seed) for k in KINDS for i in range(a.n)]
    t0 = time.time()
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, jobs)):
            rows.extend(r if isinstance(r, list) else [r])
            if (i + 1) % 10 == 0:
                print(f"{a.mode}: {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    write(rows, os.path.join(a.out, f"{a.mode}.csv"))
    print(f"done {a.mode} in {time.time() - t0:.0f}s")
