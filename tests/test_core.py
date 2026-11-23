"""Fast checks of the model, the lattice and the planners.  Run: python -m pytest tests"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from heastar import anyangle as AA, maps as M, paths as P, planners as PL, tracking as TR  # noqa: E402
from heastar.energy import HIGH_LOSS, NOMINAL, SURFACES, crossover_radius  # noqa: E402
from heastar.lattice import Lattice, heading_vectors, make_primitives  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))


def small_instance(seed=3, n=30, kind="hills"):
    rng = np.random.default_rng(seed)
    while True:
        tm = M.make_synthetic(kind, n, rng)
        e = M.pick_endpoints(tm, rng)
        if e:
            return PL.Instance(tm, *e)


def test_straight_energy_eq2():
    i = SURFACES.index("grass")
    e = NOMINAL.drive_energy(2.0, 0.3, NOMINAL.mu_r[i], NOMINAL.mu_l0[i])
    w = 60 * 9.81
    assert e == pytest.approx(w * (0.08 * 2.0 + 0.3) / 0.8)
    assert NOMINAL.drive_energy(1.0, -1.0, 0.03, 0.55) == 0.0       # no regeneration


def test_morales_table6():
    from validate_morales import model_per_rad
    v = model_per_rad()
    assert v["Point turn"] / 1e3 == pytest.approx(0.51, abs=0.005)
    assert v["Arc, R = 1 m"] / 1e3 == pytest.approx(0.75, abs=0.005)
    assert v["Arc, R = 3 m"] / 1e3 == pytest.approx(1.02, abs=0.005)


def test_crossover_table7():
    want = {"sand": 6, "grass": 11, "gravel": 19, "concrete": 29}
    for s, r in want.items():
        assert abs(crossover_radius(NOMINAL, s) - r) <= 1.0


def test_headings_and_radii():
    assert len(heading_vectors(8)) == 8 and len(heading_vectors(32)) == 32
    _, ths, prims = make_primitives(16)
    t = np.mod(ths, 2 * math.pi)
    gaps = np.degrees(np.diff(np.append(t, t[0] + 2 * math.pi)))
    assert set(np.round(gaps, 1)) == {26.6, 18.4}
    assert len(prims[0]) == 11
    a1_axis = [p for p in prims[0] if p.kind == "A1"][0]
    a1_other = [p for p in prims[1] if p.kind == "A1"][0]
    assert a1_axis.nominal_R == pytest.approx(6.9, abs=0.05)
    assert a1_other.nominal_R == pytest.approx(11.3, abs=0.1)
    _, _, p8 = make_primitives(8)
    assert [p for p in p8[0] if p.kind == "A1"][0].nominal_R == pytest.approx(2.9, abs=0.05)


@pytest.mark.parametrize("H", [8, 16])
def test_plan_cost_equals_evaluation_and_dijkstra(H):
    inst = small_instance()
    lat = Lattice(inst.tm, NOMINAL, H)
    h0 = lat.heading_index(inst.h8)
    a, i = lat.plan(inst.start, h0, inst.goal)
    b, j = lat.plan(inst.start, h0, inst.goal, heuristic=False)
    e = P.Scorer(inst.tm, NOMINAL).evaluate(lat.elements(a))["energy"]
    assert e == pytest.approx(i["cost"], rel=1e-9)
    assert i["cost"] == pytest.approx(j["cost"], rel=1e-9)
    assert i["expanded"] <= j["expanded"]


def test_heuristic_consistent():
    inst = small_instance(n=24)
    lat = Lattice(inst.tm, NOMINAL, 16)
    hm = lat.heuristic_map(inst.goal)
    for h in range(16):
        for pi, p in enumerate(lat.prims[h]):
            c = lat.cost[h, :, :, pi]
            xs, ys = np.nonzero(np.isfinite(c))
            for x, y in zip(xs, ys):
                assert hm[x, y] <= c[x, y] + hm[x + p.off[0], y + p.off[1]] + 1e-6


def test_planner_ordering_and_tracking():
    inst = small_instance(seed=5, n=40)
    for model in (NOMINAL, HIGH_LOSS):
        out = PL.plan_all(inst, model)
        r = PL.score(inst, out, model)
        assert all(v["success"] for v in r.values())
        assert r["HEA*-8"]["energy"] <= r["TA*+turn"]["energy"] + 1e-6
        assert r["HEA*"]["energy"] <= r["TA*+turn-16"]["energy"] + 1e-6
        assert r["TA*+S"]["energy"] <= r["TA*"]["energy"] + 1e-6
        assert r["TA*+turn+S"]["energy"] <= r["TA*+turn"]["energy"] + 1e-6
        for pl in out.values():
            st = TR.stop_and_turn(inst.tm, model, pl.elems, inst.start, inst.th0)
            assert not st["collision"]
            pp = TR.pure_pursuit(inst.tm, model, pl.elems, inst.start, inst.th0)
            assert pp["energy"] > 0


def test_collision_check_conservative():
    free = np.ones((5, 5), bool)
    free[2, 3] = free[3, 2] = False                                    # diagonal squeeze (2,2) -> (3,3)
    assert P.points_blocked(free, P.line_samples((2, 2), (3, 3)), 0.05)
    assert not P.points_blocked(free, P.line_samples((1, 1), (2, 2)), 0.05)
