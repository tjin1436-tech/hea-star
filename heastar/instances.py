"""Deterministic instance generators shared by all experiments (Sections 5.1 and 5.2)."""
from __future__ import annotations

import numpy as np

from . import maps as M
from .planners import Instance

SYN_SEED = 7
DEM_SEED = 21
SIZE = 80
_SOURCES = {}


def synthetic(kind: str, idx: int, size=SIZE, seed=SYN_SEED) -> Instance:
    """Instance idx (0..49) of map family `kind`: map, start and goal >= 0.6 N apart, initial heading."""
    rng = np.random.default_rng([seed, M.FAMILIES.index(kind), idx])
    while True:
        tm = M.make_synthetic(kind, size, rng)
        ends = M.pick_endpoints(tm, rng)
        if ends is not None:
            return Instance(tm, *ends, data="synthetic", group=kind, idx=idx)


def synthetic_all(n=50, size=SIZE, seed=SYN_SEED):
    return [(k, i) for k in M.FAMILIES for i in range(n)]


def dem_source(region, cache="data", order=1):
    key = (region, order, cache)
    if key not in _SOURCES:
        _SOURCES[key] = M.load_dem_source(region, cache=cache, order=order)
    return _SOURCES[key]


def dem(region: str, idx: int, size=SIZE, seed=DEM_SEED, cache="data", order=1):
    """Crop idx (0..24) of a region, in two versions on the same crop, start, goal and heading:
    (synthetic surfaces, WorldCover surfaces)."""
    src = dem_source(region, cache, order)
    rng = np.random.default_rng([seed, list(M.REGIONS).index(region), idx])
    for _ in range(100):
        c = M.dem_crop(src, size, rng)
        if c is None:
            break
        syn, wc, origin = c
        ends = M.pick_endpoints(syn, rng, extra_free=wc.free)
        if ends is None:
            continue
        a = Instance(syn, *ends, data="dem_syn", group=region, idx=idx)
        b = Instance(wc, *ends, data="dem_wc", group=region, idx=idx)
        a.origin = b.origin = origin
        return a, b
    raise RuntimeError(f"no valid crop for {region} #{idx}")
