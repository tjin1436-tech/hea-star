"""Fig. 6 (energy box plots per map family) and Fig. 8 (sensitivity) from result CSVs.
Usage: python result_figures.py <energy_raw.csv> <sensitivity_energy.csv> <out_dir>"""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

raw, sens, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
rows = list(csv.DictReader(open(raw)))
methods = ["A*", "TA*", "TA*+turn", "HEA* (ours)"]
fams = sorted({r["map"] for r in rows})
fig, axes = plt.subplots(1, len(fams), figsize=(4 * len(fams), 3.6), sharey=False)
for ax, f in zip(np.atleast_1d(axes), fams):
    data = [[float(r["energy"]) / 1000 for r in rows if r["map"] == f and r["method"] == m] for m in methods]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(methods) + 1), [m.replace(" (ours)", "") for m in methods], rotation=20)
    ax.set_title(f); ax.set_ylabel("energy (kJ)")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig6_energy_boxplots.png"), dpi=200); plt.close(fig)

s = list(csv.DictReader(open(sens)))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for k, (param, xl) in enumerate((("L", "track contact length L (m)"), ("mu_scale", "friction scale"))):
    ss = [r for r in s if r["param"] == param]
    if param == "mu_scale":
        ss = sorted(ss + [dict(r, value="1.0") for r in s if r["param"] == "L" and r["value"] == "0.6"], key=lambda r: float(r["value"]))
    x = [float(r["value"]) for r in ss]
    for b in ("A*", "TA*", "TA*+turn"):
        ax[k].plot(x, [-float(r[f"dE_vs_{b}_median_pct"]) for r in ss], "o-", label=f"vs {b}")
    ax[k].set_xlabel(xl); ax[k].set_ylabel("median energy saving of HEA* (%)")
    ax[k].set_ylim(bottom=-2)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=9, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(out, "fig8_sensitivity.png"), dpi=200); plt.close(fig)
print("ok")
