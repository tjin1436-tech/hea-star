"""Quantitative check of the energy model against the 16 measured runs in Table III of
Morales et al., IEEE T-RO 25(5):1098-1108, 2009 (Auriga-beta, 286 kg, asphalt, ~52 m paths).

Measured energy minus the parts our model and Morales' identified parameters explain directly
(driver power-up 58 W x time, traction resistance K = 342 N x length / eta) leaves the energy
attributable to turning (plus unmodelled driver braking / transients), which is divided by the
total steered angle psi and compared with our model's turning energy per radian.
Usage: python validate_morales.py
"""
import csv, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from energy_model import G, Robot, rotate_energy, track_energy

here = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(here, "morales2009_table3.csv"))))
m, K, eta, P0, mu = 286.0, 342.0, 0.95, 58.0, 0.81       # Morales et al. 2009, Sec. III
rb = Robot(mass=m, B=0.42, L=0.72, eta=eta)
mu_r, mu_l0 = K / (m * G), mu * 0.925                      # so that mu_l(0) = 0.81
E = np.array([float(r["energy_kJ"]) * 1e3 for r in rows])
T = np.array([float(r["time_s"]) for r in rows])
s = np.array([float(r["length_m"]) for r in rows])
psi = np.array([float(r["psi_rad"]) for r in rows])
base = P0 * T + K * s / eta
turn_meas = (E - base)
per_rad_meas = turn_meas / psi
model = {}
model["point turn"] = rotate_energy(1.0, mu_r, mu_l0, rb)
for R in (1.0, 2.0, 3.0):
    model[f"arc R={R:.0f} m"] = float(track_energy(R, 0, mu_r, mu_l0, R, rb)) - float(track_energy(R, 0, mu_r, mu_l0, math.inf, rb))
print(f"measured energy {E.mean() / 1e3:.1f} kJ; explained by power-up + traction {base.mean() / 1e3:.1f} kJ")
print(f"turning share of measured energy: {np.mean(turn_meas / E) * 100:.0f}%")
print(f"turning energy per rad (measured residual): {per_rad_meas.mean():.0f} ± {per_rad_meas.std():.0f} J/rad")
for k, v in model.items():
    print(f"model {k:12s}: {v:.0f} J/rad  -> implied gain {per_rad_meas.mean() / v:.1f}x")
