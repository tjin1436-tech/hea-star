"""HEA*: heading-lattice energy A* (Section 4).

State (x, y, theta) on a grid of cells. theta takes the directions of the integer grid vectors with components
up to K (K = 1: 8 headings, K = 2: 16 headings, K = 3: 32 headings). From heading h with grid vector d_h:
    S    straight move by d_h
    A_k  cubic Bezier arc to the neighbouring heading h+-1, moving by k (d_h + d_{h+-1}), k in arc_scales
    R    point turn to h+-1
Edge costs (energies under the model of Section 3) are precomputed for every cell, heading and primitive as
array operations; the search only performs table look-ups. The heuristic (9) is consistent (Proposition 1).
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import paths as P
from .energy import G, Model
from .maps import TerrainMap

ARC_SCALES = (1, 2, 4, 8)


def heading_vectors(n_headings: int):
    """Integer grid vectors with gcd 1 and max(|x|, |y|) <= K, sorted by angle from 0."""
    K = {8: 1, 16: 2, 32: 3}[n_headings]
    vs = {(x, y) for x in range(-K, K + 1) for y in range(-K, K + 1)
          if (x, y) != (0, 0) and math.gcd(x, y) == 1}
    vs = sorted(vs, key=lambda v: math.atan2(v[1], v[0]) % (2 * math.pi))
    assert len(vs) == n_headings
    return vs


def _bezier_eval(p0, c1, c2, p1, t):
    t = t[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p1


def bezier_arc(off, th0, th1, ds=P.SAMPLE_DS):
    """Cubic Bezier from the origin to `off`, tangent to th0 at the start and th1 at the end, sampled at
    (approximately) uniform arc length ds. Returns (samples, local radius at sub-segment midpoints, length)."""
    p0 = np.zeros(2)
    p1 = np.asarray(off, float)
    c = np.linalg.norm(p1) / 3.0
    c1 = p0 + c * np.array([math.cos(th0), math.sin(th0)])
    c2 = p1 - c * np.array([math.cos(th1), math.sin(th1)])
    tt = np.linspace(0, 1, 4001)
    dense = _bezier_eval(p0, c1, c2, p1, tt)
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    L = cum[-1]
    k = int(math.ceil(L / ds - 1e-9)) + 1
    t = np.interp(np.linspace(0, L, k), cum, tt)
    S = _bezier_eval(p0, c1, c2, p1, t)
    S[0], S[-1] = p0, p1
    tm = (t[:-1] + t[1:]) / 2
    d1 = (3 * (1 - tm) ** 2)[:, None] * (c1 - p0) + (6 * (1 - tm) * tm)[:, None] * (c2 - c1) + \
        (3 * tm ** 2)[:, None] * (p1 - c2)
    d2 = (6 * (1 - tm))[:, None] * (c2 - 2 * c1 + p0) + (6 * tm)[:, None] * (p1 - 2 * c2 + c1)
    cross = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    speed = np.linalg.norm(d1, axis=1)
    with np.errstate(divide="ignore"):
        radius = np.where(cross > 1e-12, speed ** 3 / np.maximum(cross, 1e-300), np.inf)
    return S, radius, L


@dataclass
class Prim:
    kind: str                  # "S", "A<k>", "R"
    dh: int                    # heading index change (0, +1, -1)
    off: tuple                 # (dx, dy) cells
    dtheta: float              # heading change (rad)
    samples: np.ndarray = field(repr=False, default=None)
    radius: np.ndarray = field(repr=False, default=None)
    length: float = 0.0        # cells
    nominal_R: float = math.inf

    @property
    def is_arc(self):
        return self.kind.startswith("A")


def make_primitives(n_headings=16, arc_scales=ARC_SCALES, arcs=True, ds=P.SAMPLE_DS):
    """Primitive set per heading: [S, A_k (+1) ..., A_k (-1) ..., R(+1), R(-1)]."""
    vecs = heading_vectors(n_headings)
    ths = [math.atan2(v[1], v[0]) for v in vecs]
    prims = []
    for h, d in enumerate(vecs):
        d = np.array(d)
        L = float(np.linalg.norm(d))
        S = P.line_samples((0, 0), d, ds)
        out = [Prim("S", 0, tuple(int(v) for v in d), 0.0, S, np.full(len(S) - 1, np.inf), L)]
        if arcs:
            for s in (1, -1):
                h2 = (h + s) % n_headings
                d2 = np.array(vecs[h2])
                dth = P.wrap(ths[h2] - ths[h])
                for k in arc_scales:
                    off = k * (d + d2)
                    smp, rad, Lc = bezier_arc(off, ths[h], ths[h] + dth, ds)
                    chord = float(np.linalg.norm(off))
                    out.append(Prim(f"A{k}", s, tuple(int(v) for v in off), dth, smp, rad, Lc,
                                    chord / (2 * math.sin(abs(dth) / 2))))
        for s in (1, -1):
            dth = P.wrap(ths[(h + s) % n_headings] - ths[h])
            out.append(Prim("R", s, (0, 0), dth, np.zeros((1, 2)), np.zeros(0), 0.0, 0.0))
        prims.append(out)
    return vecs, ths, prims


def _shift(A, pad, dx, dy, n, m):
    return A[pad + dx:pad + dx + n, pad + dy:pad + dy + m]


class Lattice:
    """Edge tables of one map and model, and the HEA* search.

    cost[h, x, y, p]: energy (J) of primitive p of heading h from cell (x, y); inf if invalid."""

    def __init__(self, tm: TerrainMap, model: Model, n_headings=16, arc_scales=ARC_SCALES, arcs=True,
                 ds=P.SAMPLE_DS, collision_margin=None):
        t0 = time.perf_counter()
        self.tm, self.model, self.H = tm, model, n_headings
        self.vecs, self.ths, self.prims = make_primitives(n_headings, arc_scales, arcs, ds)
        n, m = tm.shape
        self.n, self.m = n, m
        mu_r, mu_l0 = tm.mu(model)
        cos_a = tm.cos_grade()
        pad = max(abs(c) for ps in self.prims for p in ps for c in p.off) + 3
        Zp = np.pad(tm.elev, pad, mode="edge")
        Rp = np.pad(mu_r, pad, mode="edge")
        Lp = np.pad(mu_l0, pad, mode="edge")
        Fp = np.pad(tm.free, pad, constant_values=False)
        npr = len(self.prims[0])
        cost = np.full((n_headings, n, m, npr), np.inf)
        chunk = max(2, int(3e6 // (n * m)))               # samples per chunk
        for h in range(n_headings):
            for pi, p in enumerate(self.prims[h]):
                if p.kind == "R":
                    e = model.rotate_energy(p.dtheta, mu_r, mu_l0, cos_a)
                    cost[h, :, :, pi] = np.where(tm.free, e, np.inf)
                    continue
                S = p.samples
                valid = np.ones((n, m), bool)
                for dx, dy in P.swept_offsets(S, collision_margin):
                    valid &= _shift(Fp, pad, dx, dy, n, m)
                # bilinear elevation at the samples, surface at the sub-segment midpoints, local radius;
                # processed in chunks of samples to bound the memory on large maps
                ix = np.floor(S[:, 0]).astype(int)
                iy = np.floor(S[:, 1]).astype(int)
                fx, fy = S[:, 0] - ix, S[:, 1] - iy

                def bil(k):
                    return ((1 - fx[k]) * (1 - fy[k]) * _shift(Zp, pad, ix[k], iy[k], n, m)
                            + fx[k] * (1 - fy[k]) * _shift(Zp, pad, ix[k] + 1, iy[k], n, m)
                            + (1 - fx[k]) * fy[k] * _shift(Zp, pad, ix[k], iy[k] + 1, n, m)
                            + fx[k] * fy[k] * _shift(Zp, pad, ix[k] + 1, iy[k] + 1, n, m))

                seg = np.linalg.norm(np.diff(S, axis=0), axis=1) * tm.cell
                mid = P.cell_of((S[:-1] + S[1:]) / 2)
                e = np.zeros((n, m))
                z_prev = bil(0)
                for k0 in range(0, len(S) - 1, chunk):            # sub-segments k -> k+1, in chunks
                    ks = np.arange(k0, min(k0 + chunk, len(S) - 1))
                    z = np.stack([z_prev] + [bil(k + 1) for k in ks])
                    mr = np.stack([_shift(Rp, pad, *mid[k], n, m) for k in ks])
                    ml = np.stack([_shift(Lp, pad, *mid[k], n, m) for k in ks])
                    e += model.drive_energy(seg[ks, None, None], np.diff(z, axis=0), mr, ml,
                                            (p.radius[ks] * tm.cell)[:, None, None]).sum(axis=0)
                    z_prev = z[-1]
                cost[h, :, :, pi] = np.where(valid, e, np.inf)
        self.cost = cost
        self.w = model.robot.mass * G
        present = np.unique(tm.surface)
        self.mu_r_min = float(model.mu_r_arr[present].min())
        self.precompute_s = time.perf_counter() - t0

    @property
    def table_mb(self):
        return self.cost.nbytes / 2 ** 20

    def heading_index(self, h8):
        """Lattice heading of one of the 8 axis / diagonal directions (index 0..7)."""
        return h8 * self.H // 8

    def heuristic_map(self, goal):
        """Eq. (9): max(0, m g (mu_r_min d + z_g - z)) / eta for every cell."""
        n, m = self.n, self.m
        X, Y = np.mgrid[0:n, 0:m]
        d = np.hypot(X - goal[0], Y - goal[1]) * self.tm.cell
        dz = self.tm.elev[goal] - self.tm.elev
        return np.maximum(0.0, self.w * (self.mu_r_min * d + dz)) / self.model.robot.eta

    def plan(self, start, h0, goal, arcs=True, heuristic=True):
        """A* over (x, y, theta) from (start, h0) (h0 = lattice heading index) to the goal cell with any heading.
        arcs=False restricts the search to straight moves and point turns (TA*+turn ablations).
        Returns (actions, info): actions is a list of (h, x, y, p)."""
        t0 = time.perf_counter()
        n, m, H = self.n, self.m, self.H
        hm = self.heuristic_map(goal) if heuristic else np.zeros((n, m))
        hl = hm.tolist()
        allowed = []
        for h in range(H):
            allowed.append([(pi, p.off[0], p.off[1], (h + p.dh) % H) for pi, p in enumerate(self.prims[h])
                            if arcs or not p.is_arc])
        cost = self.cost
        g = {}
        parent = {}
        s0 = (h0, start[0], start[1])
        g[s0] = 0.0
        pq = [(hl[start[0]][start[1]], 0.0, s0)]
        closed = set()
        expanded = 0
        found = None
        gx, gy = goal
        while pq:
            f, gc, st = heapq.heappop(pq)
            if st in closed:
                continue
            closed.add(st)
            expanded += 1
            h, x, y = st
            if x == gx and y == gy:
                found = st
                break
            row = cost[h, x, y].tolist()
            for pi, dx, dy, nh in allowed[h]:
                c = row[pi]
                if c == math.inf:
                    continue
                ns = (nh, x + dx, y + dy)
                if ns in closed:
                    continue
                ng = gc + c
                if ng < g.get(ns, math.inf):
                    g[ns] = ng
                    parent[ns] = (st, pi)
                    heapq.heappush(pq, (ng + hl[ns[1]][ns[2]], ng, ns))
        info = dict(expanded=expanded, plan_ms=(time.perf_counter() - t0) * 1e3,
                    cost=g[found] if found else math.inf)
        if found is None:
            return None, info
        acts = []
        st = found
        while st in parent:
            st, pi = parent[st]
            acts.append((st[0], st[1], st[2], pi))
        return acts[::-1], info

    def elements(self, acts):
        """Path elements (paths.Rot / paths.Curve) of an action sequence."""
        out = []
        for h, x, y, pi in acts:
            p = self.prims[h][pi]
            if p.kind == "R":
                out.append(P.Rot(float(x), float(y), p.dtheta))
            else:
                out.append(P.Curve(p.samples + (x, y), p.radius, p.kind, p.dtheta))
        return out

    def waypoints(self, acts):
        """Cells visited by an action sequence (for any-angle smoothing of lattice paths)."""
        pts = []
        for h, x, y, pi in acts:
            if not pts or pts[-1] != (x, y):
                pts.append((x, y))
        h, x, y, pi = acts[-1]
        p = self.prims[h][pi]
        end = (x + p.off[0], y + p.off[1])
        if pts[-1] != end:
            pts.append(end)
        return pts

    # ---------------------------------------------------------------- 8-connected grid (A*, TA*, Theta*-E)
    def straight_table(self):
        """For an 8-heading lattice: cost of the straight move in each of the 8 directions, (8, n, m)."""
        assert self.H == 8
        return self.cost[:, :, :, 0].transpose(0, 1, 2)
