"""Path representation and evaluation with the energy model.

Every planner returns a path as a list of elements in cell coordinates:
    Rot(x, y, dtheta)              in-place rotation at a point
    Curve(pts, radius, kind, dh)   a sampled curve (straight move, arc, or straight segment of a polyline);
                                   radius[k] is the local turning radius (cells) of sub-segment k (inf = straight)
All planners are scored by `evaluate`, which integrates the same model as the lattice edge tables (Section 4.2):
bilinear elevation at the samples, surface class at the sub-segment midpoints, local radius of curvature.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from .energy import Model
from .maps import TerrainMap

SAMPLE_DS = 0.1        # sampling interval (cells)
COLL_MIN = 0.05        # minimum half-width (cells) of the collision box around every sample


@dataclass
class Rot:
    x: float
    y: float
    dtheta: float


@dataclass
class Curve:
    pts: np.ndarray            # (k, 2) cell coordinates
    radius: np.ndarray         # (k-1,) cells, inf = straight
    kind: str = "S"            # "S" straight move, "A<k>" arc, "L" polyline segment
    dheading: float = 0.0      # heading change along the curve (rad)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.pts, axis=0), axis=1).sum())


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def cell_of(p):
    """Cell index of a point (round half up; identical for tables and evaluation)."""
    return np.floor(np.asarray(p) + 0.5).astype(int)


# ------------------------------------------------------------------------------------------- collision


def coll_margin(pts):
    gap = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).max()) if len(pts) > 1 else 0.0
    return max(COLL_MIN, 0.55 * gap)


def swept_offsets(S, margin=None):
    """Relative cells (closed unit squares centred on integer points) that overlap a square of half-width
    max(0.05, 0.55 * gap) around any sample of S. A primitive is valid only if all of them are traversable:
    the whole curve then stays in traversable cells without touching a blocked boundary or corner."""
    d = coll_margin(S) if margin is None else margin
    cells = set()
    for x, y in S:
        for cx in range(math.ceil(x - d - 0.5), math.floor(x + d + 0.5) + 1):
            for cy in range(math.ceil(y - d - 0.5), math.floor(y + d + 0.5) + 1):
                cells.add((cx, cy))
    return sorted(cells)


def points_blocked(free, P, margin):
    """True if any closed cell overlapping the box of half-width `margin` around any point is blocked
    or outside the map (same rule as swept_offsets, vectorised for arbitrary points)."""
    n, m = free.shape
    base = np.floor(P + 0.5).astype(int)
    r = max(1, int(math.ceil(margin)))
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            cx, cy = base[:, 0] + dx, base[:, 1] + dy
            ov = (np.abs(P[:, 0] - cx) <= 0.5 + margin) & (np.abs(P[:, 1] - cy) <= 0.5 + margin)
            if not ov.any():
                continue
            cx, cy = cx[ov], cy[ov]
            out = (cx < 0) | (cy < 0) | (cx >= n) | (cy >= m)
            if out.any():
                return True
            if not free[cx, cy].all():
                return True
    return False


def line_samples(a, b, ds=SAMPLE_DS):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    k = int(math.ceil(L / ds - 1e-9)) + 1
    return np.linspace(a, b, max(k, 2))


# ------------------------------------------------------------------------------------------- evaluation


class Scorer:
    """Caches per-map arrays for one (map, model) pair."""

    def __init__(self, tm: TerrainMap, model: Model):
        self.tm, self.model = tm, model
        self.mu_r, self.mu_l0 = tm.mu(model)
        self.cos_a = tm.cos_grade()

    def interp(self, P):
        return map_coordinates(self.tm.elev, [P[:, 0], P[:, 1]], order=1, mode="nearest")

    def curve_energy(self, pts, radius):
        tm = self.tm
        z = self.interp(pts)
        ds = np.linalg.norm(np.diff(pts, axis=0), axis=1) * tm.cell
        dz = np.diff(z)
        c = cell_of((pts[:-1] + pts[1:]) / 2)
        cx = np.clip(c[:, 0], 0, tm.shape[0] - 1)
        cy = np.clip(c[:, 1], 0, tm.shape[1] - 1)
        e = self.model.drive_energy(ds, dz, self.mu_r[cx, cy], self.mu_l0[cx, cy], np.asarray(radius) * tm.cell)
        return float(e.sum())

    def rot_energy(self, x, y, dtheta):
        cx, cy = cell_of((x, y))
        cx = min(max(cx, 0), self.tm.shape[0] - 1)
        cy = min(max(cy, 0), self.tm.shape[1] - 1)
        return float(self.model.rotate_energy(dtheta, self.mu_r[cx, cy], self.mu_l0[cx, cy], self.cos_a[cx, cy]))

    def line_energy(self, a, b):
        """Straight drive a -> b (cells); inf if the swept cells touch a blocked cell."""
        P = line_samples(a, b)
        if points_blocked(self.tm.free, P, coll_margin(P)):
            return math.inf
        return self.curve_energy(P, np.full(len(P) - 1, np.inf))

    def evaluate(self, elems):
        rb = self.model.robot
        E = T = L = rot_all = rot_inp = 0.0
        n_pt = n_arc = 0
        prev_rot = False
        for el in elems:
            if isinstance(el, Rot):
                if abs(el.dtheta) < 1e-12:
                    continue
                E += self.rot_energy(el.x, el.y, el.dtheta)
                T += abs(el.dtheta) / rb.omega
                rot_all += abs(el.dtheta)
                rot_inp += abs(el.dtheta)
                if not prev_rot:
                    n_pt += 1
                prev_rot = True
            else:
                E += self.curve_energy(el.pts, el.radius)
                l = el.length * self.tm.cell
                L += l
                T += l / rb.v
                rot_all += abs(el.dheading)
                n_arc += el.kind.startswith("A")
                prev_rot = False
        return dict(energy=E, time_s=T, length=L, rotation_deg=math.degrees(rot_all),
                    rot_inplace_deg=math.degrees(rot_inp), point_turns=n_pt, arcs=n_arc)


# ------------------------------------------------------------------------------------------- polylines


def polyline_elements(pts, th0):
    """Path that rotates in place at every vertex and drives straight between vertices (Section 5.3).
    The goal may be reached with any heading, so there is no final rotation."""
    elems, th = [], th0
    for a, b in zip(pts[:-1], pts[1:]):
        d = np.asarray(b, float) - np.asarray(a, float)
        if np.hypot(*d) < 1e-12:
            continue
        nth = math.atan2(d[1], d[0])
        dth = wrap(nth - th)
        if abs(dth) > 1e-9:
            elems.append(Rot(float(a[0]), float(a[1]), dth))
        P = line_samples(a, b)
        elems.append(Curve(P, np.full(len(P) - 1, np.inf), "L", 0.0))
        th = nth
    return elems


def merge_collinear(pts):
    """Drop interior vertices of a cell path where the direction does not change."""
    pts = [tuple(p) for p in pts]
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        d1 = np.subtract(pts[i], out[-1]); d2 = np.subtract(pts[i + 1], pts[i])
        if abs(d1[0] * d2[1] - d1[1] * d2[0]) > 1e-9 or np.dot(d1, d2) < 0:
            out.append(pts[i])
    out.append(pts[-1])
    return out


def path_points(elems, ds=None):
    """Dense geometry of a path (concatenated curve samples), for plotting and audits."""
    P = [el.pts for el in elems if isinstance(el, Curve)]
    return np.vstack(P) if P else np.zeros((0, 2))


def grade_along(tm: TerrainMap, elems, ds=SAMPLE_DS):
    """Maximum longitudinal and lateral grade (deg) along a path, from the bilinear elevation surface,
    sampled every ds cells (Section 6.5)."""
    z = tm.elev
    n, m = z.shape
    best_lon = best_lat = 0.0
    for el in elems:
        if not isinstance(el, Curve):
            continue
        P = el.pts
        t = np.diff(P, axis=0)
        L = np.linalg.norm(t, axis=1)
        keep = L > 1e-12
        if not keep.any():
            continue
        t = t[keep] / L[keep, None]
        Q = ((P[:-1] + P[1:]) / 2)[keep]
        x0 = np.clip(np.floor(Q[:, 0]).astype(int), 0, n - 2)
        y0 = np.clip(np.floor(Q[:, 1]).astype(int), 0, m - 2)
        fx, fy = Q[:, 0] - x0, Q[:, 1] - y0
        z00, z10, z01, z11 = z[x0, y0], z[x0 + 1, y0], z[x0, y0 + 1], z[x0 + 1, y0 + 1]
        gx = ((1 - fy) * (z10 - z00) + fy * (z11 - z01)) / tm.cell
        gy = ((1 - fx) * (z01 - z00) + fx * (z11 - z10)) / tm.cell
        lon = np.abs(gx * t[:, 0] + gy * t[:, 1])
        lat = np.abs(-gx * t[:, 1] + gy * t[:, 0])
        best_lon = max(best_lon, float(np.degrees(np.arctan(lon.max()))))
        best_lat = max(best_lat, float(np.degrees(np.arctan(lat.max()))))
    return best_lon, best_lat
