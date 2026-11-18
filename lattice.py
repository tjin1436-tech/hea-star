"""Planners for the tracked-robot energy study.

State lattice over (x, y, heading), with either 8 or 16 grid-aligned headings.
The 16-heading lattice is the paper's main configuration; 8 headings are retained
for ablations. Each heading has a straight move, arcs to either neighbour, and a turn.
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
DIRS16 = [(1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1),
          (-1, 0), (-2, -1), (-1, -1), (-1, -2), (0, -1), (1, -2), (1, -1), (2, -1)]
GRID_DIRS = DIRS
DIRECTION_SETS = {8: DIRS, 16: DIRS16}
STEP = math.pi / 4  # legacy 8-heading step used by the revision scripts
ARC_SCALES = (1, 2, 4, 8)
SAMPLE_DS = 0.1             # sampling interval (cells) for edge-cost integration and collision checking
COLL_MARGIN = 0.05          # half-width (cells) of the box checked around every sample, >= half the sample gap


def swept_cells(S, margin=COLL_MARGIN):
    """Relative cells (closed unit squares centred on integer points) that the curve sampled by S can touch.
    Every point of the curve lies within half a sample gap of a sample, so checking all cells that overlap a
    box of half-width max(margin, 0.55 * gap) around each sample is conservative: a primitive is valid only if
    it neither enters nor touches a blocked cell (no corner cutting)."""
    gap = float(np.linalg.norm(np.diff(S, axis=0), axis=1).max()) if len(S) > 1 else 0.0
    d = max(margin, 0.55 * gap)
    cells = set()
    for x, y in S:
        for cx in range(math.ceil(x - d - 0.5), math.floor(x + d + 0.5) + 1):
            for cy in range(math.ceil(y - d - 0.5), math.floor(y + d + 0.5) + 1):
                cells.add((cx, cy))
    return sorted(cells)


def _bezier(p1, th0, th1, n):
    p0 = np.zeros(2)
    p1 = np.asarray(p1, float)
    c = np.linalg.norm(p1) / 3
    c1 = p0 + c * np.array([math.cos(th0), math.sin(th0)])
    c2 = p1 - c * np.array([math.cos(th1), math.sin(th1)])
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p1


def _heading_angles(directions):
    return np.array([math.atan2(y, x) for x, y in directions])


def _turn_angle(h, direction, angles):
    n = len(angles)
    if direction > 0:
        return (angles[(h + 1) % n] - angles[h]) % (2 * math.pi)
    return -((angles[h] - angles[(h - 1) % n]) % (2 * math.pi))


def _bezier_radii(p1, th0, th1, n):
    """Curvature radii at the midpoint of each sampled cubic-Bezier segment."""
    p0 = np.zeros(2)
    p1 = np.asarray(p1, float)
    c = np.linalg.norm(p1) / 3
    c1 = p0 + c * np.array([math.cos(th0), math.sin(th0)])
    c2 = p1 - c * np.array([math.cos(th1), math.sin(th1)])
    t = (np.arange(n - 1, dtype=float) + 0.5) / (n - 1)
    omt = 1 - t
    d1 = (3 * omt[:, None] ** 2 * (c1 - p0)
          + 6 * (omt * t)[:, None] * (c2 - c1)
          + 3 * t[:, None] ** 2 * (p1 - c2))
    d2 = (6 * omt[:, None] * (c2 - 2 * c1 + p0)
          + 6 * t[:, None] * (p1 - 2 * c2 + c1))
    cross = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    return np.divide(np.linalg.norm(d1, axis=1) ** 3, cross,
                     out=np.full_like(cross, np.inf), where=cross > 1e-12)


def primitives(h, arcs=True, headings=8):
    """Create primitives for one heading; arc costs use local curve curvature."""
    if headings not in DIRECTION_SETS:
        raise ValueError("headings must be 8 or 16")
    directions = DIRECTION_SETS[headings]
    angles = _heading_angles(directions)
    d = np.array(directions[h])
    th = angles[h]
    Ls = float(np.linalg.norm(d))
    out = [dict(name="S", dh=0, off=tuple(d), R=math.inf, length=Ls,
                turn=0.0, radii=None,
                samples=np.linspace([0, 0], d, int(math.ceil(Ls / SAMPLE_DS)) + 1))]
    if arcs:
        for s in (1, -1):
            delta = _turn_angle(h, s, angles)
            d2 = np.array(directions[(h + s) % headings])
            for k in ARC_SCALES:
                off = k * (d + d2)
                chord = float(np.linalg.norm(off))
                R = chord / (2 * math.sin(abs(delta) / 2))
                nominal_length = R * abs(delta)
                samples = _bezier(off, th, th + delta,
                                  int(math.ceil(nominal_length / SAMPLE_DS)) + 1)
                sampled_length = float(np.linalg.norm(np.diff(samples, axis=0), axis=1).sum())
                out.append(dict(name=f"A{k}", dh=s, off=tuple(off), R=R, length=sampled_length, turn=delta,
                                radii=_bezier_radii(off, th, th + delta, len(samples)), samples=samples))
    for s in (1, -1):
        out.append(dict(name="R", dh=s, off=(0, 0), R=0.0, length=0.0,
                        turn=_turn_angle(h, s, angles), radii=None, samples=np.zeros((1, 2))))
    return out


class Lattice:
    """Precomputed edge tables for one map. cost[h][p] -> (N,N) energy (inf = invalid)."""

    def __init__(self, tm, rb: Robot = Robot(), arcs=True, const_mu=False, headings=8):
        if headings not in DIRECTION_SETS:
            raise ValueError("headings must be 8 or 16")
        self.tm, self.rb, self.arcs = tm, rb, arcs
        self.heading_count = headings
        self.dirs = DIRECTION_SETS[headings]
        self.heading_angles = _heading_angles(self.dirs)
        n = tm.elev.shape[0]
        self.n = n
        X, Y = np.mgrid[0:n, 0:n].astype(float)
        mu_r, mu_l0 = tm.mu_r, tm.mu_l0
        self.prims = [primitives(h, arcs, headings) for h in range(headings)]
        self.cost, self.true_cost, self.dur = [], [], []
        for h in range(headings):
            cs, ts, ds_ = [], [], []
            for p in self.prims[h]:
                if p["name"] == "R":
                    e = rotate_energy(abs(p["turn"]), mu_r, mu_l0, rb)
                    e = np.where(tm.free, e, np.inf)
                    cs.append(e); ts.append(e); ds_.append(abs(p["turn"]) / rb.omega)
                    continue
                S = p["samples"]
                valid = np.ones((n, n), bool)
                Xi, Yi = np.mgrid[0:n, 0:n]
                for dx, dy in swept_cells(S):
                    cx, cy = Xi + dx, Yi + dy
                    inside = (cx >= 0) & (cx < n) & (cy >= 0) & (cy < n)
                    valid &= inside & tm.free[np.clip(cx, 0, n - 1), np.clip(cy, 0, n - 1)]
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
                    radius = math.inf if p["radii"] is None else p["radii"][k] * tm.cell
                    e_true += track_energy(seg[k], slope, mu_r[mx, my], mu_l0[mx, my], radius, rb)
                    e_plan += track_energy(seg[k], slope, mu_r[mx, my], mu_l0[mx, my], radius, rb,
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
                nx, ny, nh = x + p["off"][0], y + p["off"][1], (h + p["dh"]) % self.heading_count
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
                rot += abs(p["turn"])
                if not prev_rot:
                    point_turns += 1
                prev_rot = True
            else:
                if p["dh"]:
                    rot += abs(p["turn"])
                prev_rot = False
        return dict(energy=E, time_s=T, length=L, point_turns=point_turns, rotation_deg=math.degrees(rot))

    def grid_path_to_actions(self, path, h0):
        """Execute an 8-connected cell path with point turns: rotate (shortest way) to each move
        direction, then drive straight. Returns lattice actions."""
        idx_S = 0
        idx_R = {1: len(self.prims[0]) - 2, -1: len(self.prims[0]) - 1}
        acts, h = [], h0
        for a, b in zip(path[:-1], path[1:]):
            want_vec = (int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))
            target = self.dirs.index(want_vec)
            cw_steps = (target - h) % self.heading_count
            ccw_steps = (h - target) % self.heading_count
            cw_angle = sum(abs(_turn_angle((h + j) % self.heading_count, 1, self.heading_angles))
                           for j in range(cw_steps))
            ccw_angle = sum(abs(_turn_angle((h - j) % self.heading_count, -1, self.heading_angles))
                            for j in range(ccw_steps))
            s, nturns = (1, cw_steps) if cw_angle <= ccw_angle else (-1, ccw_steps)
            for _ in range(nturns):
                acts.append(((a[0], a[1], h), idx_R[s]))
                h = (h + s) % self.heading_count
            acts.append(((a[0], a[1], h), idx_S))
        return acts


def grid_astar(lat: Lattice, start, goal, energy=True, cost_lattice=None):
    """8-connected A*. energy=False: distance cost (A*); True: terrain+slope straight energy (TA*)."""
    t0 = time.perf_counter()
    gs = {tuple(start): 0.0}
    parent = {}
    cell = lat.tm.cell
    edge_lat = cost_lattice or lat

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
        for h, (dx, dy) in enumerate(GRID_DIRS):
            c = edge_lat.true_cost[h][0][x, y]
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
