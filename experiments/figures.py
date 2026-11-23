"""Figures 2, 3, 4, 5 and 7 of the paper (Fig. 6 is drawn by analyze_main.py).

Usage: python experiments/figures.py [--out figures] [--fig5 hills 0] [--fig7 hanksville 0]
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
from matplotlib.colors import ListedColormap   # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from heastar import instances as I          # noqa: E402
from heastar import paths as P              # noqa: E402
from heastar import planners as PL          # noqa: E402
from heastar.energy import NOMINAL, SURFACES, arc_vs_point_ratio  # noqa: E402
from heastar.lattice import make_primitives  # noqa: E402

SURF_COL = {"concrete": "#d4d4d4", "gravel": "#b9b8e0", "grass": "#9fd89a", "sand": "#fdd29b"}
LINE_COL = {"concrete": "tab:blue", "grass": "tab:orange", "gravel": "tab:green", "sand": "tab:red"}
PATH_STYLE = {"A*": ("tab:blue", "-"), "TA*": ("tab:orange", "-"), "Theta*-E": ("tab:green", "-"),
              "TA*+turn+S": ("tab:purple", "--"), "HEA*": ("tab:red", "-")}


def fig2(out):
    m = NOMINAL
    R = np.logspace(-1, 3, 600)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    i = SURFACES.index("concrete")
    po, pi = m.track_power(R, m.mu_r[i], m.mu_l0[i])
    ax[0].plot(R, po, label="outer track")
    ax[0].plot(R, pi, label="inner track")
    brk = (pi < 0) & (R > m.robot.B / 2)
    ax[0].fill_between(R, pi.min() * 1.1, po.max(), where=brk, color="tab:orange", alpha=0.12,
                       label="inner track braking")
    ax[0].axvline(m.robot.B / 2, color="k", ls=":", lw=1)
    ax[0].text(m.robot.B / 2 * 1.1, po.max() * 0.95, "R = B/2", fontsize=8)
    ax[0].axhline(0, color="grey", lw=0.6)
    ax[0].set(xscale="log", xlabel="turning radius R (m)", ylabel="track power (W)",
              title="(a) per-track power, concrete, v = 0.5 m/s, κ = 1")
    ax[0].legend(fontsize=8)
    for s in ("concrete", "grass", "gravel", "sand"):
        j = SURFACES.index(s)
        e = m.drive_energy(1.0, 0.0, m.mu_r[j], m.mu_l0[j], R)
        ax[1].plot(R, e, color=LINE_COL[s], label=s)
        ax[1].axhline(float(m.drive_energy(1.0, 0.0, m.mu_r[j], m.mu_l0[j])), color=LINE_COL[s], ls="--", lw=0.8)
    ax[1].set(xscale="log", xlabel="turning radius R (m)", ylabel="energy per metre (J/m)",
              title="(b) energy per metre (dashed: straight)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_power_energy.png"), dpi=200)
    plt.close(fig)


def fig3(out):
    R = np.linspace(0.3, 52, 800)
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in ("concrete", "grass", "gravel", "sand"):
        ax.plot(R, arc_vs_point_ratio(NOMINAL, s, R), color=LINE_COL[s], label=s)
    ax.axhline(1, color="k", lw=0.6)
    ax.set(xlabel="arc radius R (m)", ylabel="E_arc / E_point-turn+straight", title="45° heading change: arc vs point turn")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_arc_vs_point_turn.png"), dpi=200)
    plt.close(fig)


def fig4(out):
    vecs, ths, prims = make_primitives(16)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    cols = {"S": "k", "A1": "tab:orange", "A2": "tab:green", "A4": "tab:red", "A8": "tab:purple"}
    for ax, h in zip(axes, (0, 1)):
        seen = set()
        for p in prims[h]:
            if p.kind == "R":
                continue
            lab = p.kind if p.kind not in seen else None
            seen.add(p.kind)
            ax.plot(p.samples[:, 0], p.samples[:, 1], color=cols[p.kind], lw=2 if p.kind == "S" else 1.4, label=lab)
            ax.plot(*p.samples[-1], "k.", ms=4)
        ax.add_patch(plt.Circle((0, 0), 0.5, fill=False, ls="--", color="grey"))
        ax.set(title=f"heading {math.degrees(ths[h]):.1f}°", xlabel="x (cells)", ylabel="y (cells)", aspect="equal")
        ax.grid(alpha=0.3)
        ax.text(0.03, 0.97, "S: straight move; point turns\nto the neighbouring headings", transform=ax.transAxes,
                va="top", fontsize=7)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_primitives.png"), dpi=200)
    plt.close(fig)


def draw_map(ax, tm):
    cmap = ListedColormap([SURF_COL[s] for s in SURFACES])
    ax.imshow(tm.surface.T, origin="lower", cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax.imshow(np.where(tm.free, np.nan, 1.0).T, origin="lower", cmap=ListedColormap(["black"]), interpolation="nearest")
    if np.ptp(tm.elev) > 0:
        ax.contour(tm.elev.T, levels=12, colors="k", linewidths=0.3, alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_paths(ax, inst, model, planners=tuple(PATH_STYLE)):
    planned = PL.plan_all(inst, model, PL.build_lattices(inst, model), planners)
    res = PL.score(inst, planned, model)
    for name in planners:
        pl = planned[name]
        if pl is None:
            continue
        Q = P.path_points(pl.elems)
        c, ls = PATH_STYLE[name]
        ax.plot(Q[:, 0], Q[:, 1], color=c, ls=ls, lw=1.4,
                label=f"{name}: {res[name]['energy'] / 1e3:.2f} kJ, {res[name]['rotation_deg']:.0f}°")
    ax.plot(*inst.start, "*", color="darkred", ms=11)
    ax.plot(*inst.goal, "o", color="limegreen", mec="darkgreen", ms=7)
    ax.legend(fontsize=7, loc="upper left")


def fig5(out, kind="hills", idx=0):
    inst = I.synthetic(kind, idx)
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_map(ax, inst.tm)
    draw_paths(ax, inst, NOMINAL)
    ax.set_title(f"Synthetic {kind} map (κ = 1). Grey concrete, lilac gravel, green grass, orange sand, black blocked",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_example_synthetic.png"), dpi=200)
    plt.close(fig)


def fig7(out, region="hanksville", idx=0):
    a, b = I.dem(region, idx)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, inst, lab in ((axes[0], a, "synthetic surfaces"), (axes[1], b, "ESA WorldCover surfaces")):
        draw_map(ax, inst.tm)
        draw_paths(ax, inst, NOMINAL)
        ax.set_title(f"{region}, real elevation (κ = 1), {lab}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig7_example_dem.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--fig5", nargs=2, default=["hills", "0"])
    ap.add_argument("--fig7", nargs=2, default=["hanksville", "0"])
    ap.add_argument("--only", nargs="*", default=None, help="subset of 2 3 4 5 7")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    todo = set(a.only or ["2", "3", "4", "5", "7"])
    if "2" in todo:
        fig2(a.out)
    if "3" in todo:
        fig3(a.out)
    if "4" in todo:
        fig4(a.out)
    if "5" in todo:
        fig5(a.out, a.fig5[0], int(a.fig5[1]))
    if "7" in todo:
        fig7(a.out, a.fig7[0], int(a.fig7[1]))
    print(f"figures saved to {a.out}/")
