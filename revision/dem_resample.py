"""Major comment 4: effect of resampling the ~3.5 m DEM to 1 m cells (nearest / bilinear / cubic)
on slopes, edge costs and the planner comparison. Uses the cached tiles of results_dem/tiles.
Usage: python dem_resample.py [--crops 25]"""
import argparse, csv, os, sys
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import zoom
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import energy_model as EM   # noqa
import run_dem as RD        # noqa
import run_energy as RE     # noqa

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--crops", type=int, default=25)
    a = ap.parse_args()
    out = os.environ.get("OUTDIR", os.path.join(HERE, "results")); os.makedirs(out, exist_ok=True)
    tiles = os.path.join(HERE, "..", "results_dem", "tiles")
    rb = EM.Robot()
    rows, slopes = [], []
    with Pool() as pool:
        for order, name in ((0, "nearest"), (1, "bilinear"), (3, "cubic")):
            dems = {}
            for reg, (lat, lon, _) in RD.REGIONS.items():
                elev, mpp = RD.fetch_region(lat, lon, cache=tiles)
                e1 = zoom(elev, mpp, order=order)
                p = os.path.join(out, f"_dem_{reg}_{name}.npy"); np.save(p, e1.astype(np.float32)); dems[reg] = p
                gx, gy = np.gradient(e1, 1.0); s = np.degrees(np.arctan(np.hypot(gx, gy)))
                slopes.append(dict(resampling=name, region=reg, slope_mean=float(s.mean()), slope_p95=float(np.percentile(s, 95)),
                                   blocked_frac=float((s > rb.max_slope_deg).mean())))
            jobs = [(r, i, 80, 21, rb, dems[r]) for r in RD.REGIONS for i in range(a.crops)]
            for rs in pool.map(RD.job, jobs):
                for r in rs:
                    r["resampling"] = name
                rows.extend(rs)
            print(name, "done", flush=True)
            for p in dems.values():
                os.remove(p)
    RE.write_csv(rows, os.path.join(out, "dem_resample_raw.csv"))
    RE.write_csv(slopes, os.path.join(out, "dem_resample_slopes.csv"))
