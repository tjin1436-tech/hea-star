"""Fig. 4 (motion primitives) and Fig. 7 (example on a real DEM crop).
Usage: python paper_figs_extra.py <results_dem dir with tiles/> <out_dir>"""
import math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import energy_model as EM
from lattice import Lattice, grid_astar, primitives
import run_dem as RD

dem_dir, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
# Fig 4
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, h, t in ((axes[0], 0, "heading 0°"), (axes[1], 1, "heading 45°")):
    for p in primitives(h):
        S = p["samples"]
        if p["name"] == "R":
            continue
        col = {"S": "k", "A1": "tab:orange", "A2": "tab:green", "A4": "tab:red", "A8": "tab:purple"}[p["name"]]
        ax.plot(S[:, 0], S[:, 1], "-", lw=3 if p["name"] == "S" else 1.6, color=col,
                label=p["name"] if p["dh"] >= 0 else None)
        ax.plot(*S[-1], "k.", ms=5)
    ax.add_patch(plt.Circle((0, 0), 0.35, fill=False, ls="--", color="gray"))
    ax.annotate("±45° point turn", (0, 0), (1.5, -1.5) if h == 0 else (3, -0.6), arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_title(t); ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")
axes[0].legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_primitives.png"), dpi=200); plt.close(fig)

# Fig 7: one Hanksville crop
rb = EM.Robot()
lat, lon, _ = RD.REGIONS["hanksville"]
elev, mpp = RD.fetch_region(lat, lon, cache=os.path.join(dem_dir, "tiles"))
e1 = zoom(elev, mpp, order=1)
rng = np.random.default_rng(5)
while True:
    tm = RD.crop_map(e1, 80, rng, rb)
    import run_energy as RE
    ends = RE.pick(tm, rng)
    if ends:
        s, g, h0 = ends
        full = Lattice(tm, rb)
        if grid_astar(full, s, g, energy=False)[0] is not None:
            break
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
cmap = matplotlib.colors.ListedColormap(["#d9d9d9", "#a1d99b", "#bcbddc", "#fdd49e"])
axes[0].imshow(np.where(tm.free, tm.terrain, np.nan).T, origin="lower", cmap=cmap, vmin=-0.5, vmax=3.5)
axes[0].contour(tm.elev.T, levels=12, colors="k", linewidths=0.4, alpha=0.5)
axes[1].imshow(np.where(tm.free, tm.elev, np.nan).T, origin="lower", cmap="terrain")
for m, col in (("A*", "tab:blue"), ("TA*", "tab:orange"), ("HEA* (ours)", "tab:red")):
    if m == "HEA* (ours)":
        acts, _ = full.plan(s, h0, g)
    else:
        p, _ = grid_astar(full, s, g, energy=(m == "TA*"))
        acts = full.grid_path_to_actions(p, h0)
    ev = full.evaluate(acts)
    xs = np.concatenate([x + full.prims[h][pi]["samples"][:, 0] for (x, y, h), pi in acts])
    ys = np.concatenate([y + full.prims[h][pi]["samples"][:, 1] for (x, y, h), pi in acts])
    for ax in axes:
        ax.plot(xs, ys, color=col, lw=1.8, label=f"{m}: {ev['energy'] / 1000:.2f} kJ, {ev['point_turns']} point turns")
for ax in axes:
    ax.plot(*s, "go", ms=8); ax.plot(*g, "r*", ms=12); ax.set_xticks([]); ax.set_yticks([])
axes[0].set_title("Hanksville, UT (real DEM) — surface classes (synthetic)"); axes[1].set_title("elevation (m)")
axes[0].legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig7_dem_example.png"), dpi=200); plt.close(fig)
print("ok")
