"""Planners for the tracked-robot energy study.

State lattice over (x, y, heading), 8 headings (45 deg). Motion primitives per heading:
  S   straight one cell (1 or sqrt(2) m)
  Ak  45-deg arc to offset k (d_h + d_{h+-1}), k in ARC_SCALES (radius ~2.9 k cells)
  R   in-place rotation by +-45 deg
Edge energies are precomputed for every cell with numpy (energy_model.track_energy), so the
search itself only does table look-ups.

Planners
  A*            shortest distance on the 8-connected grid (executed with point turns)
  TA*           terrain + slope energy A* on the grid, no turning cost (Inal et al. 2024 style)
  TA*+turn      lattice with S + R only: energy-aware, turning cost, but point turns only
  HEA*-constmu  full lattice, but plans with radius-independent lateral friction
  HEA* (ours)   full lattice with the radius-dependent skid-steer model
Every path is evaluated with the same true energy model.
"""
import heapq
import math
import time

import numpy as np
from scipy.ndimage import map_coordinates

from energy_model import G, Robot, rotate_energy, track_energy

DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
STEP = math.pi / 4
ARC_SCALES = (1, 2, 4, 8)   # 45-deg arcs with radii ~2.9, 5.8, 11.7, 23.4 cells


def _bezier(p1, th0, th1, n):
    p0 = np.zeros(2)
    p1 = np.asarray(p1, float)
    c = np.linalg.norm(p1) / 3
    c1 = p0 + c * np.array([math.cos(th0), math.sin(th0)])
    c2 = p1 - c * np.array([math.cos(th1), math.sin(th1)])
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p1


def primitives(h, arcs=True):
    """List of dicts: dh, off, R, length, samples (k,2)."""
    d = np.array(DIRS[h])
    th = h * STEP
    out = [dict(name="S", dh=0, off=tuple(d), R=math.inf, length=float(np.linalg.norm(d)),
                samples=np.linspace([0, 0], d, 5))]
    if arcs:
        for s in (1, -1):
            d2 = np.array(DIRS[(h + s) % 8])
            for k in ARC_SCALES:
                name = f"A{k}"
                off = k * (d + d2)
                chord = float(np.linalg.norm(off))
                R = chord / (2 * math.sin(STEP / 2))
                L = R * STEP
                out.append(dict(name=name, dh=s, off=tuple(off), R=R, length=L,
                                samples=_bezier(off, th, th + s * STEP, int(math.ceil(L / 0.25)) + 1)))
    for s in (1, -1):
        out.append(dict(name="R", dh=s, off=(0, 0), R=0.0, length=0.0, samples=np.zeros((1, 2))))
    return out


class Lattice:
    """Precomputed edge tables for one map. cost[h][p] -> (N,N) energy (inf = invalid)."""

    def __init__(self, tm, rb: Robot = Robot(), arcs=True, const_mu=False):
        self.tm, self.rb, self.arcs = tm, rb, arcs
        n = tm.elev.shape[0]
        self.n = n
        X, Y = np.mgrid[0:n, 0:n].astype(float)
        mu_r, mu_l0 = tm.mu_r, tm.mu_l0
        self.prims = [primitives(h, arcs) for h in range(8)]
        self.cost, self.true_cost, self.dur = [], [], []
        for h in range(8):
            cs, ts, ds_ = [], [], []
            for p in self.prims[h]:
                if p["name"] == "R":
                    e = rotate_energy(STEP, mu_r, mu_l0, rb)
                    e = np.where(tm.free, e, np.inf)
                    cs.append(e); ts.append(e); ds_.append(STEP / rb.omega)
                    continue
                S = p["samples"]
                valid = np.ones((n, n), bool)
                ix = np.rint(X[None] + S[:, 0, None, None]).astype(int)
                iy = np.rint(Y[None] + S[:, 1, None, None]).astype(int)
                inside = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
                valid &= inside.all(0)
                ixc, iyc = np.clip(ix, 0, n - 1), np.clip(iy, 0, n - 1)
                valid &= tm.free[ixc, iyc].all(0)
                z = np.stack([map_coordinates(tm.elev, [X + sx, Y + sy], order=1, mode="nearest")
                              for sx, sy in S])
                seg = np.linalg.norm(np.diff(S, axis=0), axis=1) * tm.cell
                e_plan = np.zeros((n, n))
                e_true = np.zeros((n, n))
                for k in range(len(seg)):
                    mid = (S[k] + S[k + 1]) / 2
                    mx = np.clip(np.rint(X + mid[0]).astype(int), 0, n - 1)
                    my = np.clip(np.rint(Y + mid[1]).astype(int), 0, n - 1)
                    slope = (z[k + 1] - z[k]) / seg[k]
                    e_true += track_energy(seg[k], slope, mu_r[mx, my], mu_l0[mx, my], p["R"] * tm.cell, rb)
                    e_plan += track_energy(seg[k], slope, mu_r[mx, my], mu_l0[mx, my], p["R"] * tm.cell, rb,
                                           const_mu=const_mu)
                cs.append(np.where(valid, e_plan, np.inf))
                ts.append(np.where(valid, e_true, np.inf))
                ds_.append(p["length"] * tm.cell / rb.v)
            self.cost.append(cs); self.true_cost.append(ts); self.dur.append(ds_)
        self.w = rb.mass * G
        self.mu_r_min = float(mu_r.min())

    # admissible: E >= max(0, w (mu_r_min * dist + dz)) / eta (rolling + grade lower bound, turning >= 0)
    def heuristic(self, x, y, goal):
        d = math.hypot(goal[0] - x, goal[1] - y) * self.tm.cell
        dz = self.tm.elev[goal] - self.tm.elev[x, y]
        return max(0.0, self.w * (self.mu_r_min * d + dz)) / self.rb.eta

    def plan(self, start, h0, goal):
        """A* over (x,y,h). Goal: reach goal cell with any heading. Returns (actions, info)."""
        t0 = time.perf_counter()
        n = self.n
        g = {}
        parent = {}
        s = (start[0], start[1], h0)
        g[s] = 0.0
        pq = [(self.heuristic(*start, goal), 0.0, s)]
        expanded = 0
        closed = set()
        found = None
        while pq:
            f, gc, st = heapq.heappop(pq)
            if st in closed:
                continue
            closed.add(st)
            expanded += 1
            x, y, h = st
            if (x, y) == tuple(goal):
                found = st
                break
            for pi, p in enumerate(self.prims[h]):
                c = self.cost[h][pi][x, y]
                if not np.isfinite(c):
                    continue
                nx, ny, nh = x + p["off"][0], y + p["off"][1], (h + p["dh"]) % 8
                ns = (nx, ny, nh)
                ng = gc + c
                if ng < g.get(ns, math.inf) - 1e-9:
                    g[ns] = ng
                    parent[ns] = (st, pi)
                    heapq.heappush(pq, (ng + self.heuristic(nx, ny, goal), ng, ns))
        info = dict(expanded=expanded, time_ms=(time.perf_counter() - t0) * 1e3)
        if found is None:
            return None, info
        acts = []
        st = found
        while st in parent:
            st, pi = parent[st]
            acts.append((st, pi))
        return acts[::-1], info

    def evaluate(self, acts):
        """True energy, time, length, point turns, total rotation of an action sequence."""
        E = T = L = 0.0
        point_turns = 0
        rot = 0.0
        prev_rot = False
        for (x, y, h), pi in acts:
            p = self.prims[h][pi]
            E += self.true_cost[h][pi][x, y]
            T += self.dur[h][pi]
            L += p["length"] * self.tm.cell
            if p["name"] == "R":
                rot += STEP
                if not prev_rot:
                    point_turns += 1
                prev_rot = True
            else:
                if p["dh"]:
                    rot += STEP
                prev_rot = False
        return dict(energy=E, time_s=T, length=L, point_turns=point_turns, rotation_deg=math.degrees(rot))

    def grid_path_to_actions(self, path, h0):
        """Execute an 8-connected cell path with point turns: rotate (shortest way) to each move
        direction, then drive straight. Returns lattice actions."""
        idx_S = 0
        idx_R = {1: len(self.prims[0]) - 2, -1: len(self.prims[0]) - 1}
        acts, h = [], h0
        for a, b in zip(path[:-1], path[1:]):
            want = DIRS.index((int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))))
            diff = (want - h) % 8
            s = 1 if diff <= 4 else -1
            for _ in range(diff if s == 1 else 8 - diff):
                acts.append(((a[0], a[1], h), idx_R[s]))
                h = (h + s) % 8
            acts.append(((a[0], a[1], h), idx_S))
        return acts


def grid_astar(lat: Lattice, start, goal, energy=True):
    """8-connected A*. energy=False: distance cost (A*); True: terrain+slope straight energy (TA*)."""
    t0 = time.perf_counter()
    gs = {tuple(start): 0.0}
    parent = {}
    cell = lat.tm.cell

    def hfun(x, y):
        if energy:
            return lat.heuristic(x, y, goal)
        return math.hypot(goal[0] - x, goal[1] - y) * cell

    pq = [(hfun(*start), 0.0, tuple(start))]
    closed = set()
    expanded = 0
    while pq:
        f, gc, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1
        if u == tuple(goal):
            break
        x, y = u
        for h, (dx, dy) in enumerate(DIRS):
            c = lat.true_cost[h][0][x, y]
            if not np.isfinite(c):
                continue
            if not energy:
                c = math.hypot(dx, dy) * cell
            v = (x + dx, y + dy)
            ng = gc + c
            if ng < gs.get(v, math.inf) - 1e-9:
                gs[v] = ng
                parent[v] = u
                heapq.heappush(pq, (ng + hfun(*v), ng, v))
    info = dict(expanded=expanded, time_ms=(time.perf_counter() - t0) * 1e3)
    if tuple(goal) not in closed:
        return None, info
    path = [tuple(goal)]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    return path[::-1], info
