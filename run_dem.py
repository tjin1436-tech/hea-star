"""Real-terrain experiment on public elevation data (AWS Terrain Tiles, Terrarium encoding,
https://registry.opendata.aws/terrain-tiles/ ; US data from USGS 3DEP/NED).

For each region we download zoom-15 tiles (~3.7 m/pixel at 38 deg N), decode elevation
(R*256 + G + B/256 - 32768), resample to 1 m cells and cut random 80 x 80 m crops. A DEM has no
surface labels, so terrain classes are synthesised as in the generated maps (stated in the paper);
cells steeper than max_slope are blocked. Needs Internet (Kaggle: Settings -> Internet on).

Usage:  python run_dem.py --crops 25 --out results_dem
"""
import argparse
import io
import math
import os
import urllib.request
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy.ndimage import zoom

import energy_model as EM
import run_energy as RE
from lattice import Lattice, grid_astar

# name: (lat, lon, description)
REGIONS = {
    "hanksville": (38.41, -110.79, "Utah desert, Mars-analog research site (MDRS)"),
    "palouse": (46.90, -117.20, "rolling farmland hills, Washington"),
    "flinthills": (38.60, -96.50, "grassland hills, Kansas"),
    "mojave": (35.05, -116.10, "desert alluvial fans, California"),
}
URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return int(x), int(y)


def fetch_region(lat, lon, z=15, tiles=2, cache="dem_cache"):
    """Stitch tiles x tiles Terrarium tiles around (lat, lon); returns (elev [x,y] metres, m/pixel)."""
    os.makedirs(cache, exist_ok=True)
    x0, y0 = tile_xy(lat, lon, z)
    rows = []
    for ty in range(y0, y0 + tiles):
        row = []
        for tx in range(x0, x0 + tiles):
            f = os.path.join(cache, f"{z}_{tx}_{ty}.png")
            if not os.path.exists(f):
                with urllib.request.urlopen(URL.format(z=z, x=tx, y=ty), timeout=60) as r:
                    open(f, "wb").write(r.read())
            a = np.asarray(Image.open(f).convert("RGB"), float)
            row.append(a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768)
        rows.append(np.hstack(row))
    img = np.vstack(rows)            # [row = north->south, col = west->east]
    elev = img[::-1].T               # -> [x = east, y = north]
    mpp = 156543.03 * math.cos(math.radians(lat)) / 2 ** z
    return elev, mpp


def crop_map(elev1m, size, rng, rb):
    m = elev1m.shape[0]
    for _ in range(200):
        x, y = rng.integers(0, m - size, 2)
        z = np.array(elev1m[x:x + size, y:y + size], float)
        t = EM._noise(size, size / 12, rng)
        terrain = np.digitize(t, np.quantile(t, [0.25, 0.5, 0.75]))
        gx, gy = np.gradient(z, 1.0)
        free = np.degrees(np.arctan(np.hypot(gx, gy))) <= rb.max_slope_deg
        free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
        if free.mean() >= 0.7 and np.ptp(z) >= 2.0:   # skip blocked or perfectly flat crops
            return EM.TerrainMap(z, terrain, free)
    return None


def job(args):
    region, idx, size, seed, rb, dem_path = args
    elev1m = np.load(dem_path, mmap_mode="r")
    rng = np.random.default_rng([seed, list(REGIONS).index(region), idx])
    for _ in range(20):
        tm = crop_map(elev1m, size, rng, rb)
        if tm is None:
            return []
        ends = RE.pick(tm, rng)
        if ends is None:
            continue
        s, g, h0 = ends
        full = Lattice(tm, rb)
        if grid_astar(full, s, g, energy=False)[0] is None:
            continue
        rows = RE.compare(tm, full, s, g, h0, rb, region, idx)
        for r in rows:
            r["relief_m"] = round(float(np.ptp(tm.elev)), 1)
        return rows
    return []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=int, default=25, help="random crops per region")
    ap.add_argument("--size", type=int, default=80)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", default="results_dem")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rb = EM.Robot()
    dems = {}
    for name, (lat, lon, desc) in REGIONS.items():
        elev, mpp = fetch_region(lat, lon, cache=os.path.join(a.out, "tiles"))
        e1 = zoom(elev, mpp, order=1)       # resample to 1 m cells
        dems[name] = os.path.join(a.out, f"dem_{name}_1m.npy")
        np.save(dems[name], e1.astype(np.float32))
        print(f"{name:11s} {desc}: {elev.shape} px at {mpp:.2f} m/px -> {e1.shape} m, "
              f"relief {np.ptp(e1):.0f} m", flush=True)
    jobs = [(n, i, a.size, a.seed, rb, dems[n]) for n in REGIONS for i in range(a.crops)]
    with Pool() as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    rows.sort(key=lambda r: (r["map"], r["map_id"], RE.METHODS.index(r["method"])))
    RE.write_csv(rows, os.path.join(a.out, "dem_raw.csv"))
    summ = RE.summarize(rows)
    RE.write_csv(summ, os.path.join(a.out, "dem_summary.csv"))
    st = RE.stats(rows, kinds=REGIONS)
    RE.write_csv(st, os.path.join(a.out, "dem_stats.csv"))
    for r in summ:
        print(f"{r['map']:11s} {r['method']:13s} n={r['cases']} E={r['energy']:>16s} J  T={r['time_s']:>14s} s  "
              f"turns={r['point_turns']:>10s}")
    print()
    for r in st:
        if r["metric"] == "energy":
            print(f"{r['map']:11s} vs {r['baseline']:13s} median dE={r['median_change_pct']:>6s}%  "
                  f"lower in {r['ours_lower_in']}/{r['n']}  p={r['p_value']} {r['verdict']}")
    print(f"Saved to {a.out}/")
