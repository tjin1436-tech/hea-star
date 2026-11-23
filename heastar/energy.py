"""Energy model of a skid-steer tracked robot (Section 3 of the paper).

All functions are vectorised with NumPy and broadcast over their array arguments.

Straight motion (Eqs. 1-2)
    F0 = (mu_r m g cos(a) + m g sin(a)) / 2                       per-track longitudinal force
    E_str = ds / (eta cos a) * 2 max(0, F0) = max(0, m g (mu_r ds + dz)) / eta

Turning on a circular arc of radius R (Eqs. 3-7)
    M(R)  = kappa * mu_l(R) * m g cos(a) * L / 4                  turning moment
    mu_l(R) = mu_l0 / (0.925 + 0.15 R / B)                        Wong and Chiang's law
    F_o = F0 + M/B,  F_i = F0 - M/B                               force couple on the tracks
    v_o = v (1 + B/2R),  v_i = v (1 - B/2R)                       track speeds
    P(R) = [max(0, F_o v_o) + max(0, F_i v_i)] / eta              braking power is dissipated
    E_arc = P(R) ds / (v cos a)

Point turn (Eq. 8)
    E_rot = (M(0) + mu_r m g cos(a) B/2) |dtheta| / eta

R = inf gives straight motion in the same formula, because mu_l(inf) = 0 and B/2R = 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

G = 9.81

# Table 3: (rolling resistance mu_r, lateral friction coefficient mu_l0). Representative values.
SURFACES = ("concrete", "gravel", "grass", "sand")
SURFACE_COEFFS = {
    "concrete": (0.03, 0.55),
    "gravel": (0.06, 0.75),
    "grass": (0.08, 0.65),
    "sand": (0.15, 0.85),
}


@dataclass(frozen=True)
class Robot:
    """Table 4."""
    mass: float = 60.0            # m (kg)
    B: float = 0.5                # track gauge (m)
    L: float = 0.6                # track-ground contact length (m)
    eta: float = 0.8              # drivetrain efficiency
    v: float = 0.5                # cruise speed (m/s)
    omega: float = 0.6            # in-place rotation rate (rad/s)
    max_grade_deg: float = 20.0   # maximum traversable grade
    length: float = 0.80          # overall length (m), assumed
    width: float = 0.65           # overall width (m), assumed

    @property
    def r_c(self) -> float:
        """Radius of the circle circumscribing the footprint (0.52 m)."""
        return math.hypot(self.length / 2, self.width / 2)


@dataclass(frozen=True)
class Model:
    """Robot + surface coefficients + turning-model parameters.

    kappa  : gain on the turning moment (1 = nominal, 5 = high turning loss, Section 3.5)
    wong_a, wong_b : coefficients of the radius law (4)
    mu_r, mu_l0    : per-surface coefficients, in the order of SURFACES
    """
    robot: Robot = field(default_factory=Robot)
    kappa: float = 1.0
    wong_a: float = 0.925
    wong_b: float = 0.15
    mu_r: tuple = tuple(SURFACE_COEFFS[s][0] for s in SURFACES)
    mu_l0: tuple = tuple(SURFACE_COEFFS[s][1] for s in SURFACES)

    def with_(self, **kw) -> "Model":
        rb_kw = {k: kw.pop(k) for k in list(kw) if k in Robot.__dataclass_fields__}
        m = replace(self, **kw)
        if rb_kw:
            m = replace(m, robot=replace(m.robot, **rb_kw))
        return m

    @property
    def mu_r_arr(self) -> np.ndarray:
        return np.asarray(self.mu_r, float)

    @property
    def mu_l0_arr(self) -> np.ndarray:
        return np.asarray(self.mu_l0, float)

    # ------------------------------------------------------------------ model terms
    def mu_lat(self, mu_l0, R):
        """Eq. (4); R in metres, R = 0 for a point turn, R = inf for a straight move."""
        R = np.asarray(R, float)
        with np.errstate(invalid="ignore", over="ignore"):
            return np.where(np.isinf(R), 0.0, mu_l0 / (self.wong_a + self.wong_b * np.where(np.isinf(R), 0, R) / self.robot.B))

    def drive_energy(self, ds, dz, mu_r, mu_l0, R=math.inf):
        """Energy (J) to drive a sub-segment of horizontal length ds (m) with elevation change dz (m)
        on a circular arc of radius R (m, inf = straight).  Eqs. (1)-(7)."""
        rb = self.robot
        ds = np.asarray(ds, float)
        dz = np.asarray(dz, float)
        R = np.asarray(R, float)
        w = rb.mass * G
        l3 = np.hypot(ds, dz)                                   # = ds / cos(a)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosa = np.where(l3 > 0, ds / l3, 1.0)
            sina = np.where(l3 > 0, dz / l3, 0.0)
        F0 = 0.5 * (mu_r * w * cosa + w * sina)
        M = self.kappa * self.mu_lat(mu_l0, R) * w * cosa * rb.L / 4.0
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(np.isinf(R), 0.0, rb.B / (2.0 * R))       # B / 2R
        fo, fi = F0 + M / rb.B, F0 - M / rb.B
        p_over_v = np.maximum(0.0, fo * (1 + q)) + np.maximum(0.0, fi * (1 - q))
        return l3 * p_over_v / rb.eta

    def rotate_energy(self, dtheta, mu_r, mu_l0, cos_a=1.0):
        """Eq. (8): in-place rotation by dtheta (rad) on a grade with cosine cos_a."""
        rb = self.robot
        w = rb.mass * G
        M0 = self.kappa * self.mu_lat(mu_l0, 0.0) * w * cos_a * rb.L / 4.0
        return (M0 + mu_r * w * cos_a * rb.B / 2.0) * np.abs(dtheta) / rb.eta

    def track_power(self, R, mu_r, mu_l0, slope=0.0):
        """Per-track mechanical power (W) on an arc of radius R (Fig. 2a): (P_outer, P_inner)."""
        rb = self.robot
        w = rb.mass * G
        a = np.arctan(slope)
        F0 = 0.5 * (mu_r * w * np.cos(a) + w * np.sin(a))
        M = self.kappa * self.mu_lat(mu_l0, R) * w * np.cos(a) * rb.L / 4.0
        q = rb.B / (2.0 * np.asarray(R, float))
        return (F0 + M / rb.B) * rb.v * (1 + q), (F0 - M / rb.B) * rb.v * (1 - q)


NOMINAL = Model()
HIGH_LOSS = Model(kappa=5.0)


def scenario(name: str) -> Model:
    return {"nominal": NOMINAL, "k1": NOMINAL, "high": HIGH_LOSS, "k5": HIGH_LOSS}[name]


def arc_vs_point_ratio(model: Model, surface: str, R, dtheta=math.pi / 4):
    """Fig. 3 / Table 7: E(arc of radius R turning dtheta) / E(point turn + straight chord), flat ground."""
    i = SURFACES.index(surface)
    mr, ml = model.mu_r[i], model.mu_l0[i]
    R = np.asarray(R, float)
    arc = model.drive_energy(R * dtheta, 0.0, mr, ml, R)
    chord = 2 * R * math.sin(dtheta / 2)
    pt = model.rotate_energy(dtheta, mr, ml) + model.drive_energy(chord, 0.0, mr, ml)
    return arc / pt


def crossover_radius(model: Model, surface: str, dtheta=math.pi / 4, rmax=200.0):
    """Radius above which a dtheta arc becomes cheaper than a point turn followed by the straight chord
    (first crossing below 1 after the peak of the ratio; inf if the arc is never cheaper)."""
    R = np.linspace(0.05, rmax, 40000)
    r = arc_vs_point_ratio(model, surface, R, dtheta)
    k0 = int(np.argmax(r))
    below = np.nonzero(r[k0:] < 1.0)[0]
    return float(R[k0 + below[0]]) if len(below) else math.inf
