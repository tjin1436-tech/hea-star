"""Consistency with measured data (Section 3.5, Tables 5-7, Figs. 2-3).

Morales et al. (IEEE T-RO 25(5), 2009, Table III) report the measured energy of 16 runs of the 286-kg tracked
robot Auriga-beta on asphalt (data/morales2009_table3.csv). With their identified parameters (Table 5):
  Step 1  driver power-up (58 W x time) and traction resistance (342 N x length / eta) explain part of the energy;
  Step 2  the remainder divided by the total steered angle psi gives the energy per radian of turning,
which is compared with our model (kappa = 1) for a point turn and arcs of 1 m and 3 m (Table 6).
Table 7: crossover radius above which a 45 deg arc is cheaper than a point turn followed by the chord.

Usage: python experiments/validate_morales.py [--out results]
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from heastar.energy import G, HIGH_LOSS, NOMINAL, SURFACES, Model, Robot, crossover_radius  # noqa: E402

M_KG, K_N, ETA, P0_W, MU = 286.0, 342.0, 0.95, 58.0, 0.81      # Table 5
AURIGA = Model(robot=Robot(mass=M_KG, B=0.42, L=0.72, eta=ETA),
               mu_r=(K_N / (M_KG * G),) * 4, mu_l0=(MU * 0.925,) * 4)   # mu_l(0) = 0.81


def model_per_rad(model=AURIGA):
    mr, ml = model.mu_r[0], model.mu_l0[0]
    out = {"Point turn": float(model.rotate_energy(1.0, mr, ml))}
    for R in (1.0, 3.0):     # arc energy per radian minus the straight energy of the same length
        out[f"Arc, R = {R:g} m"] = float(model.drive_energy(R, 0, mr, ml, R) - model.drive_energy(R, 0, mr, ml))
    return out


def main(out):
    df = pd.read_csv(os.path.join(ROOT, "data", "morales2009_table3.csv"))
    E = df.energy_kJ.values * 1e3
    base = P0_W * df.time_s.values + K_N * df.length_m.values / ETA
    turn = E - base
    per_rad = turn / df.psi_rad.values
    print(f"measured energy (mean): {E.mean() / 1e3:.1f} kJ; power-up + traction: {base.mean() / 1e3:.1f} kJ; "
          f"remainder {turn.mean() / 1e3:.1f} kJ = {np.mean(turn / E):.0%} of the total")
    print(f"turning energy per radian (measured remainder): {per_rad.mean() / 1e3:.1f} ± {per_rad.std() / 1e3:.1f} kJ/rad")
    t6 = pd.DataFrame([dict(manoeuvre=k, model_kJ_per_rad=round(v / 1e3, 2), ratio_measured_to_model=round(per_rad.mean() / v, 1))
                       for k, v in model_per_rad().items()])
    print(t6.to_string(index=False))
    t7 = pd.DataFrame([dict(surface=s, crossover_m_kappa1=round(crossover_radius(NOMINAL, s), 1),
                            crossover_m_kappa5=round(crossover_radius(HIGH_LOSS, s), 1)) for s in SURFACES])
    print(t7.to_string(index=False))
    # regimes of the inner track on concrete (Fig. 2a)
    i = SURFACES.index("concrete")
    R = np.logspace(-1.5, 3, 20000)
    _, p_in = NOMINAL.track_power(R, NOMINAL.mu_r[i], NOMINAL.mu_l0[i])
    brk = R[(p_in < 0) & (R > NOMINAL.robot.B / 2)]
    print(f"inner track brakes for B/2 < R < R* with R* = {brk.max():.1f} m (concrete, kappa = 1)")
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    t6.to_csv(os.path.join(out, "tables", "table6_morales.csv"), index=False)
    t7.to_csv(os.path.join(out, "tables", "table7_crossover.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    main(ap.parse_args().out)
