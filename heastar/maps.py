"""Terrain maps (Section 5.1 and 5.2): synthetic maps, AWS Terrain Tiles DEMs and ESA WorldCover surfaces.

A map is a grid of square cells indexed [x, y] (x east, y north). Every cell stores the elevation z (m),
a surface class (index into energy.SURFACES) and a traversability flag.
"""
from __future__ import annotations

import io
import math
import os
import urllib.request
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .energy import SURFACES, Model, Robot

# ------------------------------------------------------------------------------------------- map type


@dataclass
class TerrainMap:
    elev: np.ndarray       # (N, M) metres
    surface: np.ndarray    # (N, M) int, index into SURFACES
    free: np.ndarray       # (N, M) bool, traversable
    cell: float = 1.0      # cell size delta (m)
    name: str = ""

    @property
    def shape(self):
        return self.elev.shape

    def mu(self, model: Model):
        """Per-cell (mu_r, mu_l0) arrays for a model."""
        return model.mu_r_arr[self.surface], model.mu_l0_arr[self.surface]

    def cos_grade(self):
        """Cosine of the grade at the cell centres (from the elevation gradient)."""
        gx, gy = np.gradient(self.elev, self.cell)
        return 1.0 / np.sqrt(1.0 + gx * gx + gy * gy)

    def grade_deg(self):
        gx, gy = np.gradient(self.elev, self.cell)
        return np.degrees(np.arctan(np.hypot(gx, gy)))


def steep_mask(elev, cell, max_grade_deg):
    gx, gy = np.gradient(elev, cell)
    return np.degrees(np.arctan(np.hypot(gx, gy))) > max_grade_deg


def _border(free):
    free = free.copy()
    free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
    return free


# ------------------------------------------------------------------------------------------- synthetic


def noise_field(shape, sigma, rng):
    z = gaussian_filter(rng.standard_normal(shape), sigma, mode="reflect")
    return (z - z.min()) / (z.max() - z.min() + 1e-12)


def synthetic_surfaces(shape, rng):
    """Smoothed noise field split into quartiles: each class covers about a quarter in contiguous patches."""
    t = noise_field(shape, shape[0] / 12, rng)
    return np.digitize(t, np.quantile(t, [0.25, 0.5, 0.75])).astype(np.int8)


FAMILIES = ("flat", "hills", "rocky")
_RELIEF = {"flat": 0.5, "hills": 8.0, "rocky": 6.0}
_ROCKS = {"flat": 0.004, "hills": 0.004, "rocky": 0.012}   # rock discs per cell


def make_synthetic(kind: str, n: int, rng, robot: Robot = Robot()) -> TerrainMap:
    """Section 5.1: elevation = two Gaussian-filtered noise fields at different scales; surfaces from a third
    field split into quartiles; rocks are discs of radius 0.8-2.5 cells; cells steeper than 20 deg blocked.
    Blocked cells are generated directly in configuration space (they are not grown by r_c)."""
    amp = _RELIEF[kind]
    elev = amp * noise_field((n, n), n / 8, rng) + 0.3 * amp * noise_field((n, n), n / 25, rng)
    surface = synthetic_surfaces((n, n), rng)
    free = ~steep_mask(elev, 1.0, robot.max_grade_deg)
    xx, yy = np.mgrid[0:n, 0:n]
    for _ in range(int(_ROCKS[kind] * n * n)):
        cx, cy, r = rng.uniform(0, n), rng.uniform(0, n), rng.uniform(0.8, 2.5)
        free &= (xx - cx) ** 2 + (yy - cy) ** 2 > r * r
    return TerrainMap(elev, surface, _border(free), 1.0, kind)


# ------------------------------------------------------------------------------------------- DEM (AWS)

# name: (lat, lon, description)
REGIONS = {
    "flinthills": (38.60, -96.50, "Flint Hills, Kansas: grassland hills"),
    "hanksville": (38.41, -110.79, "Hanksville, Utah: desert, Mars Desert Research Station"),
    "mojave": (35.05, -116.10, "Mojave Desert, California: alluvial fans"),
    "palouse": (46.90, -117.20, "Palouse, Washington: rolling farmland"),
}
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
WORLDCOVER_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                  "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")

# ESA WorldCover class -> surface (None = blocked), Section 5.2
WORLDCOVER_MAP = {
    10: None,         # tree cover
    20: "gravel",     # shrubland
    30: "grass",      # grassland
    40: "grass",      # cropland
    50: "concrete",   # built-up
    60: "gravel",     # bare / sparse vegetation
    70: None,         # snow and ice
    80: None,         # permanent water bodies
    90: None,         # herbaceous wetland
    95: None,         # mangroves
    100: "grass",     # moss and lichen
}


def tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return int(x), int(y)


@dataclass
class Region:
    """Stitched Terrarium tiles. elev[x, y] in metres (x east, y north) at mpp metres per pixel."""
    name: str
    elev: np.ndarray
    mpp: float
    z: int
    x0: int     # first tile column
    y0: int     # first tile row (north)

    def pixel_to_lonlat(self, px, py):
        """Pixel coordinates (centre of pixel [0,0] at 0,0) -> lon, lat (deg)."""
        H = self.elev.shape[1]
        gx = self.x0 * 256 + np.asarray(px) + 0.5
        gy = self.y0 * 256 + (H - 1 - np.asarray(py)) + 0.5
        n = 256 * 2 ** self.z
        lon = gx / n * 360 - 180
        lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * gy / n))))
        return lon, lat


def fetch_region(name, cache="data/tiles", z=15, tiles=2) -> Region:
    lat, lon, _ = REGIONS[name]
    os.makedirs(cache, exist_ok=True)
    x0, y0 = tile_xy(lat, lon, z)
    rows = []
    from PIL import Image
    for ty in range(y0, y0 + tiles):
        row = []
        for tx in range(x0, x0 + tiles):
            f = os.path.join(cache, f"{z}_{tx}_{ty}.png")
            if not os.path.exists(f):
                with urllib.request.urlopen(TERRARIUM_URL.format(z=z, x=tx, y=ty), timeout=60) as r:
                    data = r.read()
                with open(f, "wb") as fh:
                    fh.write(data)
            a = np.asarray(Image.open(f).convert("RGB"), float)
            row.append(a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768)
        rows.append(np.hstack(row))
    img = np.vstack(rows)              # [row north->south, col west->east]
    elev = img[::-1].T.copy()          # [x east, y north]
    mpp = 156543.03392 * math.cos(math.radians(lat)) / 2 ** z
    return Region(name, elev, mpp, z, x0, y0)


def resample_1m(reg: Region, order=1, cell=1.0):
    """Resample the region to `cell`-metre cells (order 0 nearest, 1 bilinear, 3 cubic).
    Returns (elev, px, py): px, py are the source pixel coordinates of every cell centre."""
    nx = int(reg.elev.shape[0] * reg.mpp / cell)
    ny = int(reg.elev.shape[1] * reg.mpp / cell)
    i = (np.arange(nx) + 0.5) * cell / reg.mpp - 0.5
    j = (np.arange(ny) + 0.5) * cell / reg.mpp - 0.5
    PX, PY = np.meshgrid(i, j, indexing="ij")
    if order == 0:
        z = reg.elev[np.clip(np.floor(PX + 0.5).astype(int), 0, reg.elev.shape[0] - 1),
                     np.clip(np.floor(PY + 0.5).astype(int), 0, reg.elev.shape[1] - 1)]
    else:
        z = map_coordinates(reg.elev, [PX, PY], order=order, mode="nearest")
    return z.astype(np.float64), PX, PY


def fetch_worldcover(reg: Region, cache="data/worldcover"):
    """ESA WorldCover 2021 (10 m) window covering the region. Returns (classes, transform function)."""
    os.makedirs(cache, exist_ok=True)
    f = os.path.join(cache, f"{reg.name}.npz")
    if not os.path.exists(f):
        import rasterio
        from rasterio.windows import from_bounds
        lon0, lat0 = reg.pixel_to_lonlat(-1, -1)
        lon1, lat1 = reg.pixel_to_lonlat(reg.elev.shape[0], reg.elev.shape[1])
        tlat = int(math.floor(min(lat0, lat1) / 3) * 3)
        tlon = int(math.floor(min(lon0, lon1) / 3) * 3)
        tile = f"{'N' if tlat >= 0 else 'S'}{abs(tlat):02d}{'E' if tlon >= 0 else 'W'}{abs(tlon):03d}"
        with rasterio.open(WORLDCOVER_URL.format(tile=tile)) as ds:
            win = from_bounds(min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1), ds.transform)
            win = win.round_offsets().round_lengths()
            a = ds.read(1, window=win)
            t = ds.window_transform(win)
        np.savez_compressed(f, cls=a, transform=np.array([t.a, t.b, t.c, t.d, t.e, t.f]))
    d = np.load(f)
    return d["cls"], d["transform"]


def worldcover_surfaces(reg: Region, wc, PX, PY):
    """Surface class and blocked flag per cell, by nearest WorldCover pixel at the cell's position."""
    cls, t = wc
    lon, lat = reg.pixel_to_lonlat(PX, PY)
    col = np.floor((lon - t[2]) / t[0]).astype(int)
    row = np.floor((lat - t[5]) / t[4]).astype(int)
    c = cls[np.clip(row, 0, cls.shape[0] - 1), np.clip(col, 0, cls.shape[1] - 1)]
    surface = np.zeros(c.shape, np.int8)
    blocked = np.ones(c.shape, bool)
    for k, s in WORLDCOVER_MAP.items():
        if s is not None:
            m = c == k
            surface[m] = SURFACES.index(s)
            blocked[m] = False
    return surface, blocked, c


@dataclass
class DemSource:
    """A region resampled to 1 m, with synthetic-noise-free WorldCover surfaces at the same cells."""
    region: Region
    elev: np.ndarray
    wc_surface: np.ndarray
    wc_blocked: np.ndarray
    wc_class: np.ndarray


def load_dem_source(name, cache="data", order=1, worldcover=True) -> DemSource:
    reg = fetch_region(name, cache=os.path.join(cache, "tiles"))
    elev, PX, PY = resample_1m(reg, order=order)
    if worldcover:
        surf, blk, cls = worldcover_surfaces(reg, fetch_worldcover(reg, os.path.join(cache, "worldcover")), PX, PY)
    else:
        surf = np.zeros(elev.shape, np.int8); blk = np.zeros(elev.shape, bool); cls = np.zeros(elev.shape, np.uint8)
    return DemSource(reg, elev, surf, blk, cls)


def dem_crop(src: DemSource, size, rng, robot: Robot = Robot(), min_free=0.7, min_relief=2.0, tries=500):
    """Section 5.2: random size x size crop with >= 70 % traversable cells and >= 2 m relief.
    Returns two maps on the same crop: synthetic surfaces and WorldCover surfaces."""
    nx, ny = src.elev.shape
    for _ in range(tries):
        x, y = rng.integers(0, nx - size), rng.integers(0, ny - size)
        z = np.array(src.elev[x:x + size, y:y + size], float)
        if np.ptp(z) < min_relief:
            continue
        free = _border(~steep_mask(z, 1.0, robot.max_grade_deg))
        wc_free = free & ~src.wc_blocked[x:x + size, y:y + size]
        if free.mean() < min_free:
            continue
        syn = TerrainMap(z, synthetic_surfaces((size, size), rng), free, 1.0, src.region.name)
        wc = TerrainMap(z, src.wc_surface[x:x + size, y:y + size].copy(), wc_free, 1.0, src.region.name)
        return syn, wc, (int(x), int(y))
    return None


# ------------------------------------------------------------------------------------------- instances


def connected(free, a, b):
    """Reachability for the planners. A diagonal move must not touch the two side cells (conservative check),
    so grid moves connect exactly the 4-connected components of the traversable cells."""
    from scipy.ndimage import label
    lab, _ = label(free)
    return lab[a] != 0 and lab[a] == lab[b]


def pick_endpoints(tm: TerrainMap, rng, min_frac=0.6, tries=1000, extra_free=None):
    """Start and goal at least min_frac * N apart, both traversable and connected, and a random initial heading
    among the 8 axis / diagonal directions (index 0..7, multiples of 45 deg)."""
    free = tm.free if extra_free is None else (tm.free & extra_free)
    cells = np.argwhere(free)
    n = tm.shape[0]
    for _ in range(tries):
        a, b = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
        a, b = (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
        if math.dist(a, b) >= min_frac * n and connected(free, a, b):
            return a, b, int(rng.integers(8))
    return None


def upsample(tm: TerrainMap, f=2) -> TerrainMap:
    """Refine a map by f (Section 6.7, 0.5 m cells): bilinear elevation, repeated surfaces / obstacles."""
    n = tm.shape[0] * f
    c = (np.arange(n) + 0.5) / f - 0.5
    X, Y = np.meshgrid(c, c, indexing="ij")
    z = map_coordinates(tm.elev, [X, Y], order=1, mode="nearest")
    s = np.kron(tm.surface, np.ones((f, f), np.int8))
    fr = np.kron(tm.free, np.ones((f, f), bool))
    return TerrainMap(z, s, _border(fr), tm.cell / f, tm.name)
