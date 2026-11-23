"""Path tracking in a kinematic simulation (Section 6.4).

Two controllers replay a planned path at constant speed v (time step 0.05 s); the energy model is integrated
over the executed motion (local radius of the executed curve, bilinear elevation, surface under the robot).

stop_and_turn   the path is split at every vertex where the planned heading jumps (point turns of lattice paths,
                vertices of grid and any-angle paths). The robot rotates in place to the heading of the next
                segment and follows it, straight or arc, with pure pursuit (look-ahead 1 m). This is the
                execution that the planned energies assume.
pure_pursuit    the robot follows the whole path with pure pursuit (look-ahead 1 m) and rotates in place only
                when the look-ahead point lies more than `rot_threshold` (60 deg) off its heading.
"""
from __future__ import annotations

import math

import numpy as np

from . import paths as P
from .energy import Model
from .maps import TerrainMap

DT = 0.05
LOOKAHEAD_M = 1.0


def _pieces(elems):
    """Split a path at its in-place rotations: list of dense polylines."""
    pieces, cur = [], []
    for el in elems:
        if isinstance(el, P.Rot):
            if cur:
                pieces.append(cur)
                cur = []
        else:
            cur.append(el.pts if not cur else el.pts[1:])
    if cur:
        pieces.append(cur)
    out = []
    for p in pieces:
        Q = np.vstack(p)
        keep = np.concatenate([[True], np.linalg.norm(np.diff(Q, axis=0), axis=1) > 1e-12])
        Q = Q[keep]
        if len(Q) >= 2:
            out.append(Q)
    return out


class _Sim:
    def __init__(self, tm: TerrainMap, model: Model, x, y, th):
        self.tm, self.model = tm, model
        self.x, self.y, self.th = float(x), float(y), float(th)
        self.ds = model.robot.v * DT / tm.cell          # cells per step
        self.dth = model.robot.omega * DT
        self.ld = LOOKAHEAD_M / tm.cell
        self.traj = [(self.x, self.y)]
        self.kappa = []                                  # curvature per driving step (1/cells), aligned with traj
        self.rot = []                                    # (x, y, dtheta)

    def rotate_to(self, target_th):
        d = P.wrap(target_th - self.th)
        while abs(d) > 1e-12:
            s = math.copysign(min(self.dth, abs(d)), d)
            self.rot.append((self.x, self.y, s))
            self.th += s
            d -= s

    def drive(self, k, ds):
        x, y, th = self.x, self.y, self.th
        if abs(k) < 1e-9:
            x += ds * math.cos(th)
            y += ds * math.sin(th)
        else:
            th2 = th + k * ds
            x += (math.sin(th2) - math.sin(th)) / k
            y -= (math.cos(th2) - math.cos(th)) / k
            th = th2
        self.x, self.y, self.th = x, y, th
        self.traj.append((x, y))
        self.kappa.append(k)

    def follow(self, Q, rot_threshold=None, max_steps=200000):
        """Pure pursuit along polyline Q. rot_threshold: rotate in place when the look-ahead point is more
        than this angle off the heading (None: never)."""
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
        Qx, Qy = Q[:, 0].tolist(), Q[:, 1].tolist()
        cl = cum.tolist()
        n = len(Q)
        i = j = 0
        end = (Qx[-1], Qy[-1])
        win = int(2 * self.ld / max(float(np.median(np.diff(cum))), 1e-3)) + 2
        for _ in range(max_steps):
            x, y = self.x, self.y
            # closest point within a window ahead of the previous one (progress never goes back)
            w = min(n, i + win)
            best, bi = (Qx[i] - x) ** 2 + (Qy[i] - y) ** 2, i
            for q in range(i + 1, w):
                dq = (Qx[q] - x) ** 2 + (Qy[q] - y) ** 2
                if dq < best:
                    best, bi = dq, q
            i = bi
            if j < i:
                j = i
            while j + 1 < n and cl[j] - cl[i] < self.ld:
                j += 1
            if j == n - 1 and math.hypot(end[0] - x, end[1] - y) <= self.ds:
                return                    # within one step (2.5 cm) of the end of the path
            tx, ty = Qx[j], Qy[j]
            ang = math.atan2(ty - y, tx - x)
            a = P.wrap(ang - self.th)
            if j == n - 1 and abs(a) > math.pi / 2 and math.hypot(tx - x, ty - y) < 0.5 * self.ld:
                return                    # passed the end of the path (residual below half the look-ahead)
            if rot_threshold is not None and abs(a) > rot_threshold:
                self.rotate_to(ang)
                a = 0.0
            ell = math.hypot(tx - x, ty - y)
            k = 2 * math.sin(a) / max(ell, 1e-9)
            self.drive(k, self.ds)
        raise RuntimeError("tracking did not reach the end of the path")

    def result(self, sc: P.Scorer):
        T = np.array(self.traj)
        k = np.array(self.kappa)
        with np.errstate(divide="ignore"):
            R = np.where(np.abs(k) > 1e-9, 1.0 / np.abs(k), np.inf)
        E = sc.curve_energy(T, R) if len(T) > 1 else 0.0
        for x, y, d in self.rot:
            E += sc.rot_energy(x, y, d)
        c = P.cell_of(T)
        tm = self.tm
        inside = (c[:, 0] >= 0) & (c[:, 1] >= 0) & (c[:, 0] < tm.shape[0]) & (c[:, 1] < tm.shape[1])
        hit = (~inside).any() or (~tm.free[c[inside, 0], c[inside, 1]]).any()
        rot = sum(abs(d) for _, _, d in self.rot)
        L = float(np.linalg.norm(np.diff(T, axis=0), axis=1).sum()) * tm.cell
        rb = self.model.robot
        return dict(energy=E, collision=bool(hit), length=L, time_s=L / rb.v + rot / rb.omega)


def stop_and_turn(tm: TerrainMap, model: Model, elems, start, th0, sc=None):
    sc = sc or P.Scorer(tm, model)
    sim = _Sim(tm, model, start[0], start[1], th0)
    for Q in _pieces(elems):
        d = Q[1] - Q[0]
        sim.rotate_to(math.atan2(d[1], d[0]))
        sim.follow(Q)
    return sim.result(sc)


def pure_pursuit(tm: TerrainMap, model: Model, elems, start, th0, rot_threshold_deg=60.0, sc=None):
    sc = sc or P.Scorer(tm, model)
    sim = _Sim(tm, model, start[0], start[1], th0)
    curves = [el.pts for el in elems if isinstance(el, P.Curve)]
    if curves:
        Q = np.vstack(curves)
        keep = np.concatenate([[True], np.linalg.norm(np.diff(Q, axis=0), axis=1) > 1e-12])
        sim.follow(Q[keep], rot_threshold=math.radians(rot_threshold_deg))
    return sim.result(sc)
