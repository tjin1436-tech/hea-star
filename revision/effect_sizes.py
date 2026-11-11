"""Paired effect sizes for the revision: median relative change with bootstrap 95% CI,
Hodges-Lehmann paired difference (J) with bootstrap CI, matched-pairs rank-biserial r,
Wilcoxon p with Holm correction.  Usage: python effect_sizes.py <raw.csv> <label> <out.csv>"""
import csv, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon, rankdata

OURS = "HEA* (ours)"
BASES = ("A*", "TA*", "TA*+turn")
rng = np.random.default_rng(0)


def hl(d):
    w = (d[:, None] + d[None, :]) / 2
    return np.median(w[np.triu_indices(len(d))])


def rank_biserial(d):
    d = d[np.abs(d) > 1e-9]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def boot(f, x, B=5000):
    n = len(x)
    v = np.array([f(x[rng.integers(n, size=n)]) for _ in range(B)])
    return np.percentile(v, [2.5, 97.5])


raw, label, out = sys.argv[1:4]
by = defaultdict(dict)
for r in csv.DictReader(open(raw)):
    by[(r["map"], r["map_id"])][r["method"]] = r
groups = sorted({k for k, _ in by}) + ["all"]
rows = []
for g in groups:
    for b in BASES:
        pr = [(float(v[b]["energy"]), float(v[OURS]["energy"])) for (k, _), v in by.items()
              if (g == "all" or k == g) and v[b]["success"] == "True" and v[OURS]["success"] == "True"]
        a = np.array([x for x, _ in pr]); o = np.array([y for _, y in pr])
        rel = (o - a) / a * 100
        d = o - a
        p = 1.0 if np.allclose(d, 0) else wilcoxon(a, o, zero_method="zsplit").pvalue
        lo, hi = boot(np.median, rel)
        hlo, hhi = boot(hl, d, 2000)
        rows.append(dict(data=label, group=g, baseline=b, n=len(pr),
                         median_rel_pct=round(float(np.median(rel)), 2), ci_lo=round(float(lo), 2), ci_hi=round(float(hi), 2),
                         mean_rel_pct=round(float(rel.mean()), 2),
                         hl_diff_J=round(float(hl(d)), 1), hl_ci_lo=round(float(hlo), 1), hl_ci_hi=round(float(hhi), 1),
                         rank_biserial=round(float(rank_biserial(d)), 3),
                         lower=int((d < -1e-9).sum()), tied=int((np.abs(d) <= 1e-9).sum()), higher=int((d > 1e-9).sum()),
                         p=p))
# Holm over the 'all' comparisons of this data set
alls = sorted([r for r in rows if r["group"] == "all"], key=lambda r: r["p"])
m = len(alls)
for i, r in enumerate(alls):
    r["p_holm"] = min(1.0, max((m - j) * alls[j]["p"] for j in range(i + 1)))
for r in rows:
    r.setdefault("p_holm", "")
    r["p"] = f"{r['p']:.2e}"
    if r["p_holm"] != "":
        r["p_holm"] = f"{r['p_holm']:.2e}"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
