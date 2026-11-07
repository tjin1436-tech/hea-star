"""Figures for Section 3: power-vs-radius behaviour of the skid-steer model (qualitative check
against the regimes reported by Yu et al. / Morales et al.) and the 45-deg turn crossover."""
import math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from energy_model import G, TERRAINS, Robot, mu_lat, rotate_energy, track_energy

out = sys.argv[1] if len(sys.argv) > 1 else "paper/figs"
os.makedirs(out, exist_ok=True)
rb = Robot()
w = rb.mass * G
R = np.logspace(-1, 3, 400)

# Fig 2: per-track power vs radius on concrete at v = 0.5 m/s
mu_r, mu_l0 = TERRAINS["concrete"]
base = mu_r * w / 2
M = mu_lat(mu_l0, R, rb.B) * w * rb.L / 4
fo, fi = base + M / rb.B, base - M / rb.B
vo, vi = rb.v * (1 + rb.B / (2 * R)), rb.v * (1 - rb.B / (2 * R))
po, pi = fo * vo, fi * vi
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogx(R, po, label="outer track")
ax[0].semilogx(R, pi, label="inner track")
ax[0].axhline(0, color="k", lw=0.6)
gen = pi < 0
ax[0].fill_between(R, pi.min(), po.max(), where=gen, color="tab:orange", alpha=0.12, label="inner track braking")
ax[0].axvline(rb.B / 2, ls=":", color="gray"); ax[0].text(rb.B / 2 * 1.1, po.max() * 0.9, "R = B/2", fontsize=8)
ax[0].set_xlabel("turning radius R (m)"); ax[0].set_ylabel("track power (W)"); ax[0].set_title("(a) per-track power, concrete, v = 0.5 m/s")
ax[0].legend(fontsize=8)
for t, (mr, ml) in TERRAINS.items():
    e = [float(track_energy(1.0, 0.0, mr, ml, r, rb)) for r in R]
    ax[1].semilogx(R, e, label=t)
    ax[1].axhline(float(track_energy(1.0, 0.0, mr, ml, math.inf, rb)), ls="--", lw=0.7, color=ax[1].lines[-1].get_color())
ax[1].set_xlabel("turning radius R (m)"); ax[1].set_ylabel("energy per metre (J/m)"); ax[1].set_title("(b) energy per metre (dashed: straight)")
ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_power_vs_radius.png"), dpi=200); plt.close(fig)
# regime boundaries
r_lo = R[gen].min() if gen.any() else float("nan"); r_hi = R[gen].max() if gen.any() else float("nan")
print(f"inner track braking for {r_lo:.2f} m < R < {r_hi:.1f} m (concrete)")

# Fig 3: energy of a 45-deg heading change + advancing chord c, arc vs point-turn+straight
fig, ax = plt.subplots(figsize=(6, 4))
chords = np.linspace(0.5, 40, 200)
for t, (mr, ml) in TERRAINS.items():
    Rr = chords / (2 * math.sin(math.pi / 8)); L = Rr * math.pi / 4
    arc = np.array([float(track_energy(l, 0, mr, ml, r, rb)) for l, r in zip(L, Rr)])
    pt = np.array([float(track_energy(c, 0, mr, ml, math.inf, rb)) for c in chords]) + rotate_energy(math.pi / 4, mr, ml, rb)
    ratio = arc / pt
    ax.plot(Rr, ratio, label=t)
    cross = Rr[np.argmax(ratio < 1)] if (ratio < 1).any() else float("nan")
    print(f"{t:9s} arc cheaper than point turn for R > {cross:.1f} m")
ax.axhline(1, color="k", lw=0.6)
ax.set_xlabel("arc radius R (m)"); ax.set_ylabel("E_arc / E_point-turn+straight")
ax.set_title("45° heading change: arc vs point turn"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_turn_crossover.png"), dpi=200); plt.close(fig)
