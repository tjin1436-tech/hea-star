"""Paired statistics (Section 5.4).

For a comparison of HEA* (or any planner) with a baseline on paired instances:
  * median relative energy change (E - E_base) / E_base with a 95 % percentile-bootstrap CI (5000 resamples);
    instances whose baseline energy is zero are excluded from relative changes
  * Hodges-Lehmann estimate of the paired difference (J) with a 95 % bootstrap CI (5000 resamples)
  * matched-pairs rank-biserial correlation r, computed on the non-tied pairs
  * two-sided Wilcoxon signed-rank test that keeps ties and splits their ranks (zero_method="zsplit"),
    with Holm correction over the baselines of each data set and model scenario
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, wilcoxon

TIE = 1e-6   # J


def hodges_lehmann(d):
    d = np.asarray(d, float)
    w = (d[:, None] + d[None, :]) / 2
    return float(np.median(w[np.triu_indices(len(d))]))


def rank_biserial(d):
    d = np.asarray(d, float)
    d = d[np.abs(d) > TIE]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def bootstrap_ci(f, x, B, rng):
    n = len(x)
    v = np.array([f(x[rng.integers(n, size=n)]) for _ in range(B)])
    return np.percentile(v, [2.5, 97.5])


def wilcoxon_p(a, b):
    d = np.asarray(b) - np.asarray(a)
    if np.all(np.abs(d) <= TIE):
        return 1.0
    return float(wilcoxon(a, b, zero_method="zsplit").pvalue)


def paired(e_ours, e_base, seed=0, B_rel=5000, B_hl=5000):
    a = np.asarray(e_base, float)
    o = np.asarray(e_ours, float)
    d = o - a
    rng = np.random.default_rng(seed)
    nz = a > TIE
    rel = (o[nz] - a[nz]) / a[nz] * 100
    lo, hi = bootstrap_ci(np.median, rel, B_rel, rng)
    hlo, hhi = bootstrap_ci(hodges_lehmann, d, B_hl, rng)
    return dict(n=len(d), median_pct=float(np.median(rel)), ci_lo=float(lo), ci_hi=float(hi),
                hl_J=hodges_lehmann(d), hl_lo=float(hlo), hl_hi=float(hhi), r=rank_biserial(d),
                lower=int((d < -TIE).sum()), tied=int((np.abs(d) <= TIE).sum()), higher=int((d > TIE).sum()),
                p=wilcoxon_p(a, o))


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def median_change(e_ours, e_base):
    a = np.asarray(e_base, float)
    o = np.asarray(e_ours, float)
    nz = a > TIE
    d = o - a
    return float(np.median((o[nz] - a[nz]) / a[nz] * 100)), int((d < -TIE).sum()), \
        int((np.abs(d) <= TIE).sum()), int((d > TIE).sum())
