"""Figures 1, 2, 3, 4, 5 and 7 of the paper (Fig. 6 is drawn by analyze_main.py).

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

TEXT_W = 5.67   # text width of the Robotica template (in)
SURF_COL = {"concrete": "#d4d4d4", "gravel": "#b9b8e0", "grass": "#9fd89a", "sand": "#fdd29b"}
LINE_COL = {"concrete": "tab:blue", "grass": "tab:orange", "gravel": "tab:green", "sand": "tab:red"}
PATH_STYLE = {"A*": ("tab:blue", "-"), "TA*": ("tab:orange", "-"), "Theta*-E": ("tab:green", "-"),
              "TA*+turn+S": ("tab:purple", "--"), "HEA*": ("tab:red", "-")}


def fig1(out):
    """Fig. 1: overview of the framework, drawn at the printed size (full text width)."""
    from matplotlib.patches import FancyBboxPatch
    boxes = [("Inputs", ["elevation map", "surface classes", "obstacles", "(C-space)", "robot parameters",
                         "$m, B, L, \\eta$"]),
             ("Energy model\n(Sec. 3)", ["rolling + grade", "turning moment", "$M(R)$", "inner-track", "braking (no",
                                         "regeneration)"]),
             ("Edge-cost tables\n(Sec. 4.2)", ["$E[h][p][x, y]$ for", "straight moves,", "4 arc sizes,",
                                               "point turns"]),
             ("HEA* search\n(Sec. 4.3\u20134.4)", ["state $(x, y, \\theta)$", "consistent energy", "heuristic"]),
             ("Output", ["minimum-energy", "path within", "the lattice", "(straight, arcs,", "point turns)"])]
    W, H = TEXT_W, 1.75
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    n, gap = len(boxes), 0.17
    bw = (W - 0.04 - gap * (n - 1)) / n
    for i, (title, lines) in enumerate(boxes):
        x0 = 0.02 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x0, 0.04), bw, H - 0.08, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc="#eef3fb", ec="#3a6fb0", lw=0.9))
        ax.text(x0 + bw / 2, H - 0.13, title, ha="center", va="top", fontsize=7.8, fontweight="bold",
                linespacing=1.15)
        ax.text(x0 + bw / 2, 0.68, "\n".join(lines), ha="center", va="center", fontsize=7.5, linespacing=1.3)
        if i < n - 1:
            ax.annotate("", xy=(x0 + bw + gap - 0.01, H / 2), xytext=(x0 + bw + 0.01, H / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#3a6fb0", lw=1.0, mutation_scale=8))
    fig.savefig(os.path.join(out, "fig1_overview.png"), dpi=300)
    plt.close(fig)


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
    fig.savefig(os.path.join(out, "fig2_power_energy.png"), dpi=200, bbox_inches="tight")
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
    fig.savefig(os.path.join(out, "fig3_arc_vs_point_turn.png"), dpi=200, bbox_inches="tight")
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
    fig.savefig(os.path.join(out, "fig4_primitives.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_map(ax, tm):
    cmap = ListedColormap([SURF_COL[s] for s in SURFACES])
    ax.imshow(tm.surface.T, origin="lower", cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax.imshow(np.where(tm.free, np.nan, 1.0).T, origin="lower", cmap=ListedColormap(["black"]), interpolation="nearest")
    if np.ptp(tm.elev) > 0:
        ax.contour(tm.elev.T, levels=12, colors="k", linewidths=0.3, alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_paths(ax, inst, model, planners=tuple(PATH_STYLE), legend_size=7.5):
    planned = PL.plan_all(inst, model, PL.build_lattices(inst, model), planners)
    res = PL.score(inst, planned, model)
    for name in planners:
        pl = planned[name]
        if pl is None:
            continue
        Q = P.path_points(pl.elems)
        c, ls = PATH_STYLE[name]
        ax.plot(Q[:, 0], Q[:, 1], color=c, ls=ls, lw=1.3,
                label=f"{name}: {res[name]['energy'] / 1e3:.2f} kJ, {res[name]['rotation_deg']:.0f}°")
    ax.plot(*inst.start, "*", color="darkred", ms=9)
    ax.plot(*inst.goal, "o", color="limegreen", mec="darkgreen", ms=6)
    ax.legend(fontsize=legend_size, loc="lower right", framealpha=0.9, borderpad=0.4, handlelength=1.6,
              labelspacing=0.3)


# Figures 1, 5, 6 and 7 are drawn at the size at which they are printed (text width 409.7 pt = 5.67 in),
# so that their fonts (7.5-9 pt) match the body text: Fig. 5 at 0.75 of the text width, the others at full width.


def fig5(out, kind="hills", idx=0):
    inst = I.synthetic(kind, idx)
    w = 0.75 * TEXT_W
    fig, ax = plt.subplots(figsize=(w, w + 0.25))
    draw_map(ax, inst.tm)
    draw_paths(ax, inst, NOMINAL, legend_size=7.5)
    ax.set_title(f"Synthetic {kind} map (κ = 1)", fontsize=9)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "fig5_example_synthetic.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig7(out, region="hanksville", idx=0):
    a, b = I.dem(region, idx)
    name = {"hanksville": "Hanksville, UT", "mojave": "Mojave, CA", "palouse": "Palouse, WA",
            "flinthills": "Flint Hills, KS"}.get(region, region)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, TEXT_W / 2 + 0.2))
    for ax, inst, lab in ((axes[0], a, "synthetic surfaces"), (axes[1], b, "ESA WorldCover surfaces")):
        draw_map(ax, inst.tm)
        draw_paths(ax, inst, NOMINAL, legend_size=6.5)
        ax.set_title(f"{name}: {lab}", fontsize=8.5)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(os.path.join(out, "fig7_example_dem.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--fig5", nargs=2, default=["hills", "0"])
    ap.add_argument("--fig7", nargs=2, default=["hanksville", "0"])
    ap.add_argument("--only", nargs="*", default=None, help="subset of 1 2 3 4 5 7")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    todo = set(a.only or ["1", "2", "3", "4", "5", "7"])
    if "1" in todo:
        fig1(a.out)
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
