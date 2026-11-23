"""The nine compared planners (Section 5.3) on one instance.

1 A*            shortest path on the 8-connected grid
2 TA*           straight-line energy A* on the 8-connected grid (terrain + slope, no turning)
3 TA*+S         TA* + energy-decreasing line-of-sight smoothing
4 Theta*-E      Theta* with straight-line energy edge costs
5 TA*+turn      8-heading lattice, straight moves + point turns
6 TA*+turn+S    TA*+turn + the same smoothing
7 TA*+turn-16   16-heading lattice, straight moves + point turns
8 HEA*-8        8-heading lattice with arcs
9 HEA*          16-heading lattice with arcs (ours)
Every path is evaluated with the same energy model (paths.Scorer).
"""
from __future__ import annotations

import math
import time

from . import anyangle as AA
from . import paths as P
from .energy import Model
from .lattice import Lattice
from .maps import TerrainMap

PLANNERS = ("A*", "TA*", "TA*+S", "Theta*-E", "TA*+turn", "TA*+turn+S", "TA*+turn-16", "HEA*-8", "HEA*")
BASELINES = ("A*", "TA*", "TA*+S", "Theta*-E", "TA*+turn+S")     # non-ablated baselines (Table 9)


class Instance:
    """One map + start + goal + initial heading (index of the 8 axis / diagonal directions)."""

    def __init__(self, tm: TerrainMap, start, goal, h8, data="", group="", idx=0):
        self.tm, self.start, self.goal, self.h8 = tm, tuple(start), tuple(goal), int(h8)
        self.data, self.group, self.idx = data, group, idx

    @property
    def th0(self):
        return self.h8 * math.pi / 4


class Planned:
    def __init__(self, name, elems, plan_ms, expanded=0, acts=None, lattice=None, vertices=None):
        self.name, self.elems, self.plan_ms, self.expanded = name, elems, plan_ms, expanded
        self.acts, self.lattice, self.vertices = acts, lattice, vertices


def build_lattices(inst: Instance, model: Model, headings=(8, 16), **kw):
    return {H: Lattice(inst.tm, model, H, **kw) for H in headings}


def plan_all(inst: Instance, model: Model, lats=None, planners=PLANNERS):
    """Plan with every planner. `lats` maps the number of headings to a Lattice built with `model`."""
    lats = lats or build_lattices(inst, model)
    sc = P.Scorer(inst.tm, model)
    out = {}
    s, g = inst.start, inst.goal
    L8 = lats[8]
    grid = {}
    for name in planners:
        t0 = time.perf_counter()
        if name in ("A*", "TA*", "TA*+S"):
            key = name != "A*"
            if key not in grid:
                grid[key] = AA.grid_astar(L8, s, g, energy=key)
            path, info = grid[key]
            if path is None:
                out[name] = None
                continue
            if name == "TA*+S":
                v, _, sm = AA.smooth(sc, path, inst.th0)
                out[name] = Planned(name, P.polyline_elements(v, inst.th0), info["plan_ms"] + sm,
                                    info["expanded"], vertices=v)
            else:
                v = P.merge_collinear(path)
                out[name] = Planned(name, P.polyline_elements(v, inst.th0), info["plan_ms"], info["expanded"],
                                    vertices=v)
        elif name == "Theta*-E":
            path, info = AA.theta_star_e(L8, sc, s, g)
            out[name] = None if path is None else Planned(name, P.polyline_elements(path, inst.th0),
                                                          info["plan_ms"], info["expanded"], vertices=path)
        else:
            H = {"TA*+turn": 8, "TA*+turn+S": 8, "HEA*-8": 8, "TA*+turn-16": 16, "HEA*": 16,
                 "TA*+turn-32": 32, "HEA*-32": 32}[name]
            arcs = name.startswith("HEA*")
            lat = lats[H]
            acts, info = lat.plan(s, lat.heading_index(inst.h8), g, arcs=arcs)
            if acts is None:
                out[name] = None
                continue
            if name == "TA*+turn+S":
                v, _, sm = AA.smooth(sc, lat.waypoints(acts), inst.th0)
                out[name] = Planned(name, P.polyline_elements(v, inst.th0), info["plan_ms"] + sm,
                                    info["expanded"], vertices=v)
            else:
                out[name] = Planned(name, lat.elements(acts), info["plan_ms"], info["expanded"], acts, lat)
    return out


def score(inst: Instance, planned: dict, model: Model):
    """Evaluate every planned path with `model` (the true model)."""
    sc = P.Scorer(inst.tm, model)
    rows = {}
    for name, pl in planned.items():
        if pl is None:
            rows[name] = dict(success=False)
            continue
        r = dict(success=True, plan_ms=pl.plan_ms, expanded=pl.expanded)
        r.update(sc.evaluate(pl.elems))
        rows[name] = r
    return rows
