"""Sensitivity of the energy saving to robot geometry (track contact length L / gauge B) and to
the terrain friction coefficients (all mu_r and mu_l0 scaled together), and to a gain on the
turning moment (turn_gain; Morales et al. 2009 measurements suggest the Wong model underestimates it).

Usage:  python sensitivity_energy.py --maps 20 --out results_energy_sens
"""
import argparse
import csv
import os
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

import energy_model as EM
import run_energy as RE

BASE_TERRAINS = dict(EM.TERRAINS)


def job(args):
    kind, i, size, seed, rb, mu_scale = args
    EM.TERRAINS.update({k: (v[0] * mu_scale, v[1] * mu_scale) for k, v in BASE_TERRAINS.items()})
    return RE.run_case((kind, i, size, seed, rb))


def saving(rows, base):
    by = {}
    for r in rows:
        by.setdefault((r["map"], r["map_id"]), {})[r["method"]] = r
    d = [(v[RE.OURS]["energy"] - v[base]["energy"]) / v[base]["energy"] * 100
         for v in by.values() if v[RE.OURS]["success"] and v[base]["success"]]
    return f"{np.median(d):.1f}", f"{np.mean(d):.1f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=20)
    ap.add_argument("--size", type=int, default=80)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="results_energy_sens")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    settings = ([("L", L, 1.0) for L in (0.4, 0.6, 0.9, 1.2)] + [("mu_scale", s, s) for s in (0.6, 1.4)]
                + [("turn_gain", k, 1.0) for k in (2.0, 5.0)])
    out = []
    with Pool() as pool:
        for name, val, mu_scale in settings:
            rb = replace(EM.Robot(), **{name: val}) if name in ("L", "turn_gain") else EM.Robot()
            jobs = [(k, i, a.size, a.seed, rb, mu_scale) for k in RE.KINDS for i in range(a.maps)]
            rows = [r for rs in pool.map(job, jobs) for r in rs]
            row = dict(param=name, value=val)
            for b in ("A*", "TA*", "TA*+turn"):
                med, mean = saving(rows, b)
                row[f"dE_vs_{b}_median_pct"] = med
                row[f"dE_vs_{b}_mean_pct"] = mean
            out.append(row)
            print(row, flush=True)
    with open(os.path.join(a.out, "sensitivity_energy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"Saved to {a.out}/")
