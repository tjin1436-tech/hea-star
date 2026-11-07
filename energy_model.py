"""Skid-steer (tracked) robot energy model and terrain maps.

Power model after Morales et al. (T-RO 2009) and Wong's empirical lateral-friction law:
  * Each track carries half the weight. Its longitudinal force is
        base = (mu_r * m g cos(a) + m g sin(a)) / 2       (rolling resistance + grade)
  * Turning on radius R needs a resistive moment  M = mu_l(R) * m g * L / 4,  supplied as a
    force couple +-M/B on the outer/inner track (B = track gauge, L = contact length).
  * Lateral friction falls with turning radius:  mu_l(R) = mu_l0 / (0.925 + 0.15 R / B).
  * Track speeds are v_o = v (1 + B/2R), v_i = v (1 - B/2R). A track with negative power is
    braking; no energy is recovered (regen = 0 by default), so P = sum max(0, F_j v_j) / eta.
  * In-place rotation by dtheta costs (M(0) + mu_r m g B/2) |dtheta| / eta.
The terrain coefficients below are representative values (assumed, see TERRAINS); the paper must
state them and include a sensitivity study.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

G = 9.81

# name: (rolling resistance mu_r, lateral friction at zero radius mu_l0)
TERRAINS = {
    "concrete": (0.03, 0.55),
    "grass": (0.08, 0.65),
    "gravel": (0.06, 0.75),
    "sand": (0.15, 0.85),
}
TERRAIN_NAMES = list(TERRAINS)


@dataclass
class Robot:
    mass: float = 60.0       # kg
    B: float = 0.5           # track gauge (m)
    L: float = 0.6           # track contact length (m)
    eta: float = 0.8         # drivetrain efficiency
    regen: float = 0.0       # fraction of braking power recovered
    v: float = 0.5           # cruise speed (m/s)
    omega: float = 0.6       # in-place rotation rate (rad/s)
    max_slope_deg: float = 20.0
    turn_gain: float = 1.0   # scales the turning moment M (calibration / sensitivity; 1 = Wong model)


def mu_lat(mu_l0, R, B):
    """Wong's empirical lateral friction coefficient for turning radius R (R=0: in place)."""
    return mu_l0 / (0.925 + 0.15 * R / B)


def track_energy(ds, slope, mu_r, mu_l0, R, rb: Robot, const_mu=False):
    """Energy (J) to drive horizontal distance ds on grade `slope` (dz/dxy) with radius R.
    Arrays broadcast. R = inf means straight. const_mu: planner's simplified model where
    lateral friction does not depend on R (ablation)."""
    a = np.arctan(slope)
    w = rb.mass * G
    base = (mu_r * w * np.cos(a) + w * np.sin(a)) / 2.0
    length = ds / np.cos(a)
    if np.isinf(R):
        p = 2 * np.where(base > 0, base, rb.regen * base)
        return length * p / rb.eta
    mul = mu_l0 / 0.925 if const_mu else mu_lat(mu_l0, R, rb.B)
    M = rb.turn_gain * mul * w * rb.L / 4.0
    fo, fi = base + M / rb.B, base - M / rb.B
    vo, vi = 1 + rb.B / (2 * R), 1 - rb.B / (2 * R)
    po, pi = fo * vo, fi * vi
    p = np.where(po > 0, po, rb.regen * po) + np.where(pi > 0, pi, rb.regen * pi)
    return length * p / rb.eta


def rotate_energy(dtheta, mu_r, mu_l0, rb: Robot):
    w = rb.mass * G
    M0 = rb.turn_gain * mu_lat(mu_l0, 0.0, rb.B) * w * rb.L / 4.0
    return (M0 + mu_r * w * rb.B / 2.0) * abs(dtheta) / rb.eta


# ---------------------------------------------------------------- terrain maps
@dataclass
class TerrainMap:
    elev: np.ndarray      # (N, N) metres, indexed [x, y]
    terrain: np.ndarray   # (N, N) int index into TERRAIN_NAMES
    free: np.ndarray      # (N, N) bool, traversable
    cell: float = 1.0     # metres per cell

    @property
    def mu_r(self):
        return np.array([TERRAINS[n][0] for n in TERRAIN_NAMES])[self.terrain]

    @property
    def mu_l0(self):
        return np.array([TERRAINS[n][1] for n in TERRAIN_NAMES])[self.terrain]


def _noise(n, sigma, rng):
    z = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="reflect")
    return (z - z.min()) / (z.max() - z.min() + 1e-12)


def make_map(kind, n, rng, rb: Robot = Robot()):
    """kind: 'flat' (mixed terrain patches, ~flat), 'hills' (rolling hills + mixed terrain),
    'rocky' (hills + many rocks)."""
    amp = {"flat": 0.5, "hills": 8.0, "rocky": 6.0}[kind]
    elev = amp * _noise(n, n / 8, rng) + 0.3 * amp * _noise(n, n / 25, rng)
    t = _noise(n, n / 12, rng)
    q = np.quantile(t, [0.25, 0.5, 0.75])
    terrain = np.digitize(t, q)  # 4 roughly equal patches
    free = np.ones((n, n), bool)
    gx, gy = np.gradient(elev, 1.0)
    free &= np.degrees(np.arctan(np.hypot(gx, gy))) <= rb.max_slope_deg
    n_rocks = {"flat": 0.004, "hills": 0.004, "rocky": 0.012}[kind] * n * n
    xx, yy = np.mgrid[0:n, 0:n]
    for _ in range(int(n_rocks)):
        cx, cy, r = rng.uniform(0, n), rng.uniform(0, n), rng.uniform(0.8, 2.5)
        free &= (xx - cx) ** 2 + (yy - cy) ** 2 > r * r
    free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
    return TerrainMap(elev, terrain, free)


def load_dem(path, n=None, cell=1.0, rng=None, rb: Robot = Robot()):
    """Real elevation model: a .npy array (metres). Terrain classes are synthesised (no labels in
    a DEM); slope > max_slope is blocked. Crops/downsamples to n x n if given."""
    z = np.load(path).astype(float)
    if n is not None:
        step = max(1, min(z.shape) // n)
        z = z[::step, ::step][:n, :n]
    rng = rng or np.random.default_rng(0)
    m = z.shape[0]
    t = _noise(m, m / 12, rng)
    terrain = np.digitize(t, np.quantile(t, [0.25, 0.5, 0.75]))
    gx, gy = np.gradient(z, cell)
    free = np.degrees(np.arctan(np.hypot(gx, gy))) <= rb.max_slope_deg
    free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
    return TerrainMap(z, terrain, free, cell)
