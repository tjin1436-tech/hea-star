"""Tables 8-15 and Fig. 6 from results/main_raw.csv (and Table 12 from results/primitive_use.csv).

Usage: python experiments/analyze_main.py [--res results] [--figs figures]
Writes results/tables/*.csv and prints the tables.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from heastar import stats as ST          # noqa: E402
from heastar.planners import BASELINES, PLANNERS   # noqa: E402

DATA = (("synthetic", "Synthetic"), ("dem_syn", "DEM, synthetic surfaces"), ("dem_wc", "DEM, WorldCover"))
DECOMP = [("Pricing turns (grid)", "TA*+turn", "TA*"),
          ("Pricing turns (any-angle)", "TA*+turn+S", "TA*+S"),
          ("16 instead of 8 headings", "TA*+turn-16", "TA*+turn"),
          ("Arcs, 8 headings", "HEA*-8", "TA*+turn"),
          ("Arcs, 16 headings", "HEA*", "TA*+turn-16"),
          ("Any-angle smoothing", "TA*+turn+S", "TA*+turn"),
          ("8-heading lattice vs any-angle", "HEA*-8", "TA*+turn+S")]


def wide(df, col="energy"):
    """instances x planners table of one column, keyed by (data, group, idx, kappa)."""
    return df.pivot_table(index=["data", "group", "idx", "kappa"], columns="planner", values=col, aggfunc="first")


def show(title, t):
    print(f"\n=== {title}\n" + t.to_string(index=False))


def main(res, figs):
    df = pd.read_csv(os.path.join(res, "main_raw.csv"))
    df = df[df.success.astype(str) == "True"]
    out = os.path.join(res, "tables")
    os.makedirs(out, exist_ok=True)
    E = wide(df)

    # ---------------------------------------------------------------- Table 8
    s = df[(df.data == "synthetic") & (df.kappa == 1.0)]
    t8 = []
    for p in PLANNERS:
        q = s[s.planner == p]
        t8.append(dict(planner=p, n=len(q), **{c: f"{q[c].mean():.1f} ± {q[c].std(ddof=0):.1f}"
                                                 for c in ("energy", "time_s", "length", "rotation_deg", "plan_ms")}))
    t8 = pd.DataFrame(t8)
    t8.to_csv(os.path.join(out, "table8_synthetic_nominal.csv"), index=False)
    show("Table 8: synthetic maps, kappa = 1, mean ± std", t8)

    # ---------------------------------------------------------------- Table 9
    t9 = []
    for kappa in (1.0, 5.0):
        for data, label in DATA:
            sub = E.xs((data, kappa), level=("data", "kappa"), drop_level=False) if (data in E.index.get_level_values(0)) else None
            if sub is None or len(sub) == 0:
                continue
            rows = []
            for b in BASELINES:
                ok = sub[["HEA*", b]].dropna()
                r = ST.paired(ok["HEA*"].values, ok[b].values)
                rows.append(dict(kappa=kappa, data=label, comparison=f"HEA* vs {b}", **r))
            for r, ph in zip(rows, ST.holm([r["p"] for r in rows])):
                r["p_holm"] = ph
            t9 += rows
    t9 = pd.DataFrame(t9)
    t9.to_csv(os.path.join(out, "table9_paired.csv"), index=False)
    f9 = t9.assign(median=lambda d: d.apply(lambda r: f"{r.median_pct:+.1f}% [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}]", axis=1),
                   HL=lambda d: d.apply(lambda r: f"{r.hl_J:.0f} [{r.hl_lo:.0f}, {r.hl_hi:.0f}]", axis=1),
                   ltg=lambda d: d.apply(lambda r: f"{r.lower}/{r.tied}/{r.higher}", axis=1),
                   r_=lambda d: d.r.round(2), p_holm_=lambda d: d.p_holm.map(lambda v: f"{v:.1e}"))
    show("Table 9: paired comparisons of HEA*", f9[["kappa", "data", "comparison", "n", "median", "HL", "r_", "ltg", "p_holm_"]])

    # ---------------------------------------------------------------- Table 10
    t10 = []
    for data, label in DATA:
        for g in sorted(set(E.loc[data].index.get_level_values("group"))) if data in E.index.get_level_values(0) else []:
            r = dict(data=label, group=g)
            for kappa in (1.0, 5.0):
                sub = E.loc[(data, g, slice(None), kappa), :]
                for b in ("Theta*-E", "TA*+turn+S"):
                    ok = sub[["HEA*", b]].dropna()
                    r[f"vs {b}, k={kappa:g}"] = f"{ST.median_change(ok['HEA*'], ok[b])[0]:+.1f}%"
            t10.append(r)
    t10 = pd.DataFrame(t10)
    t10.to_csv(os.path.join(out, "table10_groups.csv"), index=False)
    show("Table 10: median energy change of HEA* per map family / region", t10)

    # ---------------------------------------------------------------- Table 11
    t11 = []
    for eff, a, b in DECOMP:
        r = dict(effect=eff, planner=a, reference=b)
        for data, label in DATA:
            for kappa in (1.0, 5.0):
                if data not in E.index.get_level_values(0):
                    continue
                sub = E.xs((data, kappa), level=("data", "kappa"))[[a, b]].dropna()
                m, lo, ti, hi = ST.median_change(sub[a], sub[b])
                r[f"{data}, k={kappa:g}"] = f"{m:+.1f}% ({lo}/{ti}/{hi})"
        t11.append(r)
    t11 = pd.DataFrame(t11)
    t11.to_csv(os.path.join(out, "table11_decomposition.csv"), index=False)
    show("Table 11: decomposition (median change, lower/tied/higher)", t11)

    # ---------------------------------------------------------------- Table 12 + arc-gain correlates
    up = os.path.join(res, "primitive_use.csv")
    if os.path.exists(up):
        u = pd.read_csv(up)
        u = u[(u.data == "synthetic") & (u.kappa == 1.0) & (u.planner == "HEA*-8")]
        t12 = u.pivot_table(index="surface", columns="kind", values="count", aggfunc="sum", fill_value=0)
        t12 = t12.reindex(["concrete", "gravel", "grass", "sand"]).fillna(0).astype(int)
        t12.to_csv(os.path.join(out, "table12_primitive_use.csv"))
        show("Table 12: HEA*-8 primitive use by surface (synthetic, kappa = 1)", t12.reset_index())
        arcs = u[u.kind.str.startswith("A")]
        pts = u[u.kind == "point_turn"]
        soft = ("sand", "grass")
        print(f"share of arcs on sand/grass: {arcs[arcs.surface.isin(soft)]['count'].sum() / max(arcs['count'].sum(), 1):.0%};"
              f" of point turns: {pts[pts.surface.isin(soft)]['count'].sum() / max(pts['count'].sum(), 1):.0%}")
    s8 = df[(df.data == "synthetic") & (df.kappa == 1.0)].set_index(["group", "idx", "planner"])
    a8 = s8.xs("HEA*-8", level="planner")
    tt = s8.xs("TA*+turn", level="planner")
    gain = (a8.energy - tt.energy) / tt.energy * 100
    cor = []
    for c, lab in (("arc_share", "share of path on arcs"), ("arcs", "number of arcs"), ("dist", "start-goal distance"),
                   ("relief", "relief"), ("soft_share", "sand+grass share of path")):
        rho, p = spearmanr(gain, a8[c])
        cor.append(dict(feature=lab, spearman_rho=round(rho, 2), p=f"{p:.1e}"))
    cor = pd.DataFrame(cor)
    cor.to_csv(os.path.join(out, "arc_gain_correlates.csv"), index=False)
    show("Arc gain of HEA*-8 over TA*+turn (synthetic, kappa = 1): Spearman correlations", cor)
    hi_soft = a8.soft_share >= 0.6
    print(f"paths with >= 60 % sand+grass: {hi_soft.sum()}, median arc gain {gain[hi_soft].median():+.1f}% "
          f"vs {gain[~hi_soft].median():+.1f}% for the others")

    # ---------------------------------------------------------------- Table 13: tracking
    t13 = []
    for ctrl, col in (("Stop-and-turn", "st_energy"), ("Pure pursuit", "pp_energy"), ("Pure pursuit, 20 deg", "pp20_energy")):
        T = wide(df, col)
        for comp in [("HEA*", b) for b in ("TA*", "TA*+S", "Theta*-E", "TA*+turn+S")] + \
                [("over", p) for p in ("HEA*", "Theta*-E", "TA*+S")]:
            r = dict(controller=ctrl, comparison=f"HEA* vs {comp[1]}" if comp[0] != "over" else f"Overhead of {comp[1]}")
            for data, label in DATA:
                for kappa in (1.0, 5.0):
                    if data not in T.index.get_level_values(0):
                        continue
                    sub = T.xs((data, kappa), level=("data", "kappa"))
                    if comp[0] == "over":
                        pl = E.xs((data, kappa), level=("data", "kappa"))[comp[1]]
                        ok = pl > ST.TIE                 # zero-energy (all-downhill) instances excluded
                        v = float(np.median((sub[comp[1]][ok] / pl[ok] - 1) * 100))
                    else:
                        v = ST.median_change(sub["HEA*"], sub[comp[1]])[0]
                    r[f"{data}, k={kappa:g}"] = f"{v:+.1f}%"
            t13.append(r)
    t13 = pd.DataFrame(t13)
    t13.to_csv(os.path.join(out, "table13_tracking.csv"), index=False)
    show("Table 13: path tracking", t13)
    # supplementary: median tracking overhead (tracked / planned energy - 1) of all nine planners
    ov = df[df.energy > ST.TIE].assign(stop_and_turn=lambda d: (d.st_energy / d.energy - 1) * 100,
                                       pure_pursuit=lambda d: (d.pp_energy / d.energy - 1) * 100)
    ov = ov.pivot_table(index="planner", columns=["data", "kappa"], values=["stop_and_turn", "pure_pursuit"],
                        aggfunc="median").reindex(PLANNERS).round(1)
    ov.columns = [f"{c}, {d}, k={k:g}" for c, d, k in ov.columns]
    ov.to_csv(os.path.join(out, "tracking_overhead_all_planners.csv"))
    show("Median tracking overhead (%) of all nine planners", ov.reset_index())
    coll = df.groupby(["data", "planner"])[["st_collision", "pp_collision"]].agg(
        lambda c: int((c.astype(str) == "True").sum()))
    coll.to_csv(os.path.join(out, "tracking_collisions.csv"))
    show("Tracked paths whose centre entered a blocked cell (both kappas)", coll.reset_index())

    # ---------------------------------------------------------------- Section 6.5: grade, idle power, rotation rate
    g = df[df.kappa == 1.0].groupby("data")[["grade_lon_deg", "grade_lat_deg"]].max()
    n_over = df[(df.kappa == 1.0) & ((df.grade_lon_deg > 20) | (df.grade_lat_deg > 20))]
    show("Maximum grade along the paths (kappa = 1)", g.reset_index())
    print(f"paths exceeding 20 deg anywhere: {len(n_over)} of {len(df[df.kappa == 1.0])}"
          + ("" if n_over.empty else f" ({', '.join(n_over.planner + '@' + n_over.data)})"))
    Tm = wide(df, "time_s")
    t14 = []
    for data, label in DATA:
        if data not in E.index.get_level_values(0):
            continue
        for P_idle in (0, 20, 58):
            r = dict(data=label, P_W=P_idle)
            for b in ("TA*", "TA*+S", "Theta*-E", "TA*+turn+S"):
                vals = []
                for kappa in (1.0, 5.0):
                    e = E.xs((data, kappa), level=("data", "kappa"))
                    t = Tm.xs((data, kappa), level=("data", "kappa"))
                    vals.append(ST.median_change(e["HEA*"] + P_idle * t["HEA*"], e[b] + P_idle * t[b])[0])
                r[f"vs {b}"] = f"{vals[0]:+.1f}% / {vals[1]:+.1f}%"
            t14.append(r)
    t14 = pd.DataFrame(t14)
    t14.to_csv(os.path.join(out, "table14_idle_power.csv"), index=False)
    show("Table 14: HEA* with idle power E + P T (kappa = 1 / kappa = 5)", t14)
    t15 = []
    for p in PLANNERS:
        q = s[s.planner == p]
        r = dict(planner=p)
        for w in (0.3, 0.6, 1.2):
            r[f"omega={w}"] = round(float((q.length / 0.5 + np.radians(q.rot_inplace_deg) / w).mean()), 1)
        t15.append(r)
    t15 = pd.DataFrame(t15)
    t15.to_csv(os.path.join(out, "table15_rotation_rate.csv"), index=False)
    show("Table 15: mean travel time (s), synthetic, kappa = 1", t15)

    if figs:
        fig6(E, figs)


def fig6(E, figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(figs, exist_ok=True)
    # drawn at the printed size (full text width, 5.67 in) so that the fonts match the body text
    fig, axes = plt.subplots(1, 3, figsize=(5.67, 2.7), sharey=True)
    bases = ("TA*", "TA*+S", "Theta*-E", "TA*+turn+S")
    for ax, (data, label) in zip(axes, DATA):
        if data not in E.index.get_level_values(0):
            continue
        n = len(E.xs((data, 1.0), level=("data", "kappa")))
        for j, kappa in enumerate((1.0, 5.0)):
            sub = E.xs((data, kappa), level=("data", "kappa"))
            vals = [((sub["HEA*"] - sub[b]) / sub[b] * 100)[sub[b] > 0].dropna() for b in bases]
            pos = np.arange(len(bases)) * 3 + j
            bp = ax.boxplot(vals, positions=pos, widths=0.8, showfliers=False, patch_artist=True,
                            medianprops=dict(color="k", lw=1), boxprops=dict(lw=0.7),
                            whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
            for b in bp["boxes"]:
                b.set_facecolor(("#7fb2d9", "#e8868a")[j])
            ax.plot([], [], "s", color=("#7fb2d9", "#e8868a")[j], label=f"κ = {kappa:g}")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(np.arange(len(bases)) * 3 + 0.5)
        ax.set_xticklabels([f"vs {b}" for b in bases], rotation=35, ha="right", fontsize=7.5)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.set_title(f"{label.replace('DEM, ', 'DEM, ').replace('WorldCover surfaces', 'WorldCover')}\n({n} instances)",
                     fontsize=8)
    axes[0].set_ylabel("energy change of HEA* (%)", fontsize=8)
    axes[0].legend(loc="lower right", fontsize=7.5, handletextpad=0.2, borderpad=0.3)
    fig.tight_layout(pad=0.3, w_pad=0.5)
    fig.savefig(os.path.join(figs, "fig6_energy_change.png"), dpi=300)
    print(f"saved {figs}/fig6_energy_change.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    ap.add_argument("--figs", default="figures")
    a = ap.parse_args()
    main(a.res, a.figs)
