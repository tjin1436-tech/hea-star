"""Grid and any-angle baselines (Section 5.3).

A*         shortest Euclidean path on the 8-connected grid
TA*        A* on the 8-connected grid with the straight-line energy (2) as edge cost (terrain + slope, no turning)
Theta*-E   Theta* (Daniel et al. 2010) with the straight-line energy integrated along each line-of-sight segment
           as edge cost and the heuristic (9)
smooth     greedy energy-decreasing line-of-sight smoothing (TA*+S, TA*+turn+S): waypoints are removed whenever
           the straight line between the remaining waypoints is collision-free and the total energy, including
           the point turns by arbitrary angles at the vertices, decreases
Grid and any-angle paths are executed by rotating in place at every vertex and driving straight in between.
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np

from . import paths as P
from .lattice import Lattice

DIRS8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def _straight_costs(lat: Lattice):
    """(8, n, m) straight-move energies of the 8 grid directions from a lattice of any heading set."""
    idx = [lat.vecs.index(d) for d in DIRS8]
    return lat.cost[idx, :, :, 0]


def grid_astar(lat: Lattice, start, goal, energy=True):
    """energy=False: shortest path (A*); True: straight-line energy (TA*). Validity of every move is the
    collision check of the lattice straight primitive. Returns (cell path, info)."""
    t0 = time.perf_counter()
    C = _straight_costs(lat)
    cl = [C[k].tolist() for k in range(8)]
    cell = lat.tm.cell
    hm = lat.heuristic_map(goal) if energy else None
    hl = hm.tolist() if energy else None
    gx, gy = goal

    def hf(x, y):
        return hl[x][y] if energy else math.hypot(gx - x, gy - y) * cell

    step = [math.hypot(*d) * cell for d in DIRS8]
    g = {tuple(start): 0.0}
    parent = {}
    pq = [(hf(*start), 0.0, tuple(start))]
    closed = set()
    expanded = 0
    while pq:
        f, gc, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1
        if u == (gx, gy):
            break
        x, y = u
        for k, (dx, dy) in enumerate(DIRS8):
            c = cl[k][x][y]
            if c == math.inf:
                continue
            v = (x + dx, y + dy)
            if v in closed:
                continue
            ng = gc + (c if energy else step[k])
            if ng < g.get(v, math.inf):
                g[v] = ng
                parent[v] = u
                heapq.heappush(pq, (ng + hf(*v), ng, v))
    info = dict(expanded=expanded, plan_ms=(time.perf_counter() - t0) * 1e3)
    if (gx, gy) not in closed:
        return None, info
    path = [(gx, gy)]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    return path[::-1], info


class DirectedLineCache:
    """Cached straight-line energies a -> b (direction matters because of the grade)."""

    def __init__(self, sc: P.Scorer):
        self.sc, self.d = sc, {}
        self.blocked = {}

    def __call__(self, a, b):
        v = self.d.get((a, b))
        if v is None:
            key = (a, b) if a <= b else (b, a)
            if self.blocked.get(key):
                v = math.inf
            else:
                v = self.sc.line_energy(a, b)
                if v == math.inf:
                    self.blocked[key] = True
            self.d[(a, b)] = v
        return v


def theta_star_e(lat: Lattice, sc: P.Scorer, start, goal, standard=True):
    """Theta* with energy edge costs: for each neighbour s' of s the path through parent(s) is used whenever the
    line of sight parent(s) -> s' is free (standard Theta*). standard=False keeps the cheaper of the two candidates
    instead (energy is not a metric); this variant keeps more grid heading changes and is not used in the paper.
    Returns (vertex path, info)."""
    t0 = time.perf_counter()
    C = _straight_costs(lat)
    cl = [C[k].tolist() for k in range(8)]
    hl = lat.heuristic_map(goal).tolist()
    line = DirectedLineCache(sc)
    s0, gl = tuple(start), tuple(goal)
    g = {s0: 0.0}
    parent = {s0: s0}
    pq = [(hl[s0[0]][s0[1]], 0.0, s0)]
    closed = set()
    expanded = 0
    while pq:
        f, gc, s = heapq.heappop(pq)
        if s in closed or gc > g[s] + 1e-9:
            continue
        closed.add(s)
        expanded += 1
        if s == gl:
            break
        x, y = s
        ps = parent[s]
        for k, (dx, dy) in enumerate(DIRS8):
            c1 = cl[k][x][y]
            if c1 == math.inf:
                continue
            v = (x + dx, y + dy)
            if v in closed:
                continue
            best, bp = gc + c1, s
            if ps != s:
                c2 = line(ps, v)
                if c2 < math.inf and (standard or g[ps] + c2 < best - 1e-9):
                    best, bp = g[ps] + c2, ps
            if best < g.get(v, math.inf) - 1e-9:
                g[v] = best
                parent[v] = bp
                heapq.heappush(pq, (best + hl[v[0]][v[1]], best, v))
    info = dict(expanded=expanded, plan_ms=(time.perf_counter() - t0) * 1e3)
    if gl not in closed:
        return None, info
    path = [gl]
    while parent[path[-1]] != path[-1]:
        path.append(parent[path[-1]])
    return path[::-1], info


def polyline_energy(sc: P.Scorer, line, pts, th0):
    E, th = 0.0, th0
    for a, b in zip(pts[:-1], pts[1:]):
        e = line(a, b)
        if e == math.inf:
            return math.inf
        nth = math.atan2(b[1] - a[1], b[0] - a[0])
        E += sc.rot_energy(a[0], a[1], P.wrap(nth - th)) + e
        th = nth
    return E


def smooth(sc: P.Scorer, pts, th0):
    """Greedy energy-decreasing line-of-sight smoothing. Returns (vertices, energy, ms)."""
    t0 = time.perf_counter()
    line = DirectedLineCache(sc)
    pts = [tuple(int(v) for v in p) for p in pts]
    best = polyline_energy(sc, line, pts, th0)
    i = 0
    while i < len(pts) - 2:
        for j in range(len(pts) - 1, i + 1, -1):
            if line(pts[i], pts[j]) == math.inf:
                continue
            cand = pts[:i + 1] + pts[j:]
            e = polyline_energy(sc, line, cand, th0)
            if e < best - 1e-9:
                pts, best = cand, e
                break
        else:
            i += 1
    return pts, best, (time.perf_counter() - t0) * 1e3
