"""Benchmark: energy-aware planners for a tracked (skid-steer) robot on terrain maps.

Usage:  python run_energy.py --maps 50 --size 80 --out results_energy [--workers 4]
Writes energy_raw.csv, energy_summary.csv, energy_stats.csv (paired Wilcoxon vs ours) and
example.png (illustration only).
"""
import argparse
import csv
import math
import os
from collections import defaultdict
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from energy_model import TERRAIN_NAMES, Robot, make_map
from lattice import DIRS, Lattice, grid_astar

KINDS = ("flat", "hills", "rocky")
OURS = "HEA* (ours)"
METHODS = ("A*", "TA*", "TA*+turn", "HEA*-constmu", OURS)
METRICS = ("energy", "time_s", "length", "point_turns", "rotation_deg", "plan_ms", "expanded")


def pick(tm, rng, min_frac=0.6):
    n = tm.elev.shape[0]
    free = np.argwhere(tm.free)
    for _ in range(500):
        a, b = free[rng.integers(len(free))], free[rng.integers(len(free))]
        if np.linalg.norm(a - b) >= min_frac * n:
            return tuple(int(v) for v in a), tuple(int(v) for v in b), int(rng.integers(8))
    return None


def run_case(args):
    kind, idx, size, seed, rb = args
    rng = np.random.default_rng([seed, KINDS.index(kind), idx])
    for _ in range(20):
        tm = make_map(kind, size, rng, rb)
        ends = pick(tm, rng)
        if ends is None:
            continue
        s, g, h0 = ends
        full = Lattice(tm, rb)
        path, _ = grid_astar(full, s, g, energy=False)
        if path is None:   # unreachable pair, resample
            continue
        break
    else:
        return []
    return compare(tm, full, s, g, h0, rb, kind, idx)


def compare(tm, full, s, g, h0, rb, kind, idx):
    """Run every method on one map / start / goal / start heading; all scored with the true model."""
    lats = {"TA*+turn": Lattice(tm, rb, arcs=False), "HEA*-constmu": Lattice(tm, rb, const_mu=True), OURS: full}
    rows = []
    for m in METHODS:
        if m in ("A*", "TA*"):
            p, info = grid_astar(full, s, g, energy=(m == "TA*"))
            acts = full.grid_path_to_actions(p, h0) if p else None
            ev = full.evaluate(acts) if acts else None
        else:
            acts, info = lats[m].plan(s, h0, g)
            ev = lats[m].evaluate(acts) if acts else None   # true model in every lattice
        r = dict(map=kind, map_id=idx, method=m, success=ev is not None,
                 plan_ms=info["time_ms"], expanded=info["expanded"])
        r.update(ev or {k: math.nan for k in ("energy", "time_s", "length", "point_turns", "rotation_deg")})
        rows.append(r)
    return rows


def summarize(rows):
    agg = defaultdict(list)
    for r in rows:
        agg[(r["map"], r["method"])].append(r)
        agg[("all", r["method"])].append(r)
    out = []
    for (kind, m), rs in sorted(agg.items(), key=lambda kv: (kv[0][0], METHODS.index(kv[0][1]))):
        ok = [r for r in rs if r["success"]]
        d = dict(map=kind, method=m, cases=len(rs), success=f"{len(ok) / len(rs):.3f}")
        for k in METRICS:
            v = np.array([r[k] for r in ok], float)
            d[k] = f"{v.mean():.1f} ± {v.std():.1f}"
        out.append(d)
    return out


def stats(rows, kinds=KINDS):
    by = defaultdict(dict)
    for r in rows:
        by[(r["map"], r["map_id"])][r["method"]] = r
    out = []
    for kind in tuple(kinds) + ("all",):
        for m in METHODS[:-1]:
            pairs = [(v[m], v[OURS]) for (k, _), v in by.items()
                     if (kind == "all" or k == kind) and v[m]["success"] and v[OURS]["success"]]
            for met in ("energy", "time_s", "length", "point_turns"):
                a = np.array([x[met] for x, _ in pairs], float)
                b = np.array([y[met] for _, y in pairs], float)
                d = b - a
                p = 1.0 if np.allclose(d, 0) else wilcoxon(a, b, zero_method="zsplit").pvalue
                rel = np.median((b - a) / np.maximum(a, 1e-9)) * 100 if met == "energy" else math.nan
                better = int((b < a - 1e-9).sum())
                out.append(dict(map=kind, baseline=m, metric=met, n=len(pairs), mean_baseline=f"{a.mean():.1f}",
                                mean_ours=f"{b.mean():.1f}", median_change_pct=f"{rel:.1f}",
                                ours_lower_in=better, ours_higher_in=int((b > a + 1e-9).sum()),
                                p_value=f"{p:.2e}", verdict="n.s." if p >= 0.05 else
                                ("ours lower" if b.mean() < a.mean() else "ours higher")))
    return out


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_example(size, seed, rb, out):
    """Illustration only: one 'hills' map with the paths of three planners."""
    rng = np.random.default_rng(seed)
    tm = make_map("hills", size, rng, rb)
    s, g, h0 = pick(tm, rng)
    full = Lattice(tm, rb)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    cmap = matplotlib.colors.ListedColormap(["#d9d9d9", "#a1d99b", "#bcbddc", "#fdd49e"])
    axes[0].imshow(np.where(tm.free, tm.terrain, np.nan).T, origin="lower", cmap=cmap, vmin=-0.5, vmax=3.5)
    axes[0].contour(tm.elev.T, levels=10, colors="k", linewidths=0.4, alpha=0.5)
    axes[0].imshow(np.where(tm.free, np.nan, 1).T, origin="lower", cmap="Greys", vmin=0, vmax=1)
    axes[1].imshow(np.where(tm.free, tm.elev, np.nan).T, origin="lower", cmap="terrain")
    styles = {"A*": "tab:blue", "TA*": "tab:orange", OURS: "tab:red"}
    for m, col in styles.items():
        if m == OURS:
            acts, _ = full.plan(s, h0, g)
        else:
            p, _ = grid_astar(full, s, g, energy=(m == "TA*"))
            acts = full.grid_path_to_actions(p, h0)
        e = full.evaluate(acts)["energy"]
        pts = []
        for (x, y, h), pi in acts:
            S = full.prims[h][pi]["samples"]
            pts.extend((x + S[:, 0], y + S[:, 1]) for _ in [0])
        xs = np.concatenate([p[0] for p in pts]); ys = np.concatenate([p[1] for p in pts])
        for ax in axes:
            ax.plot(xs, ys, color=col, lw=1.8, label=f"{m}: {e / 1000:.2f} kJ")
    for ax in axes:
        ax.plot(*s, "go", ms=8); ax.plot(*g, "r*", ms=12)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_title("terrain: " + ", ".join(f"{n}" for n in TERRAIN_NAMES) + " (grey→orange)")
    axes[1].set_title("elevation")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=50, help="maps per map kind")
    ap.add_argument("--size", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results_energy")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rb = Robot()
    jobs = [(k, i, a.size, a.seed, rb) for k in KINDS for i in range(a.maps)]
    rows = []
    with Pool(a.workers) as pool:
        for i, rs in enumerate(pool.imap_unordered(run_case, jobs)):
            rows.extend(rs)
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(jobs)} cases done", flush=True)
    rows.sort(key=lambda r: (r["map"], r["map_id"], METHODS.index(r["method"])))
    write_csv(rows, os.path.join(a.out, "energy_raw.csv"))
    summ = summarize(rows)
    write_csv(summ, os.path.join(a.out, "energy_summary.csv"))
    st = stats(rows)
    write_csv(st, os.path.join(a.out, "energy_stats.csv"))
    for r in summ:
        print(f"{r['map']:6s} {r['method']:13s} succ={r['success']} E={r['energy']:>16s} J  T={r['time_s']:>14s} s  "
              f"L={r['length']:>13s}  turns={r['point_turns']:>10s}  plan={r['plan_ms']:>14s} ms")
    print()
    for r in st:
        if r["metric"] == "energy":
            print(f"{r['map']:6s} vs {r['baseline']:13s} median ΔE={r['median_change_pct']:>6s}%  "
                  f"lower in {r['ours_lower_in']}/{r['n']}  p={r['p_value']} {r['verdict']}")
    plot_example(a.size, a.seed + 100, rb, os.path.join(a.out, "example.png"))
    print(f"Saved to {a.out}/")
