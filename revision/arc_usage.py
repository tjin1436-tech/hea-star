"""Which arc radii does HEA* use, and on which surface (major comment 3)."""
import csv, os, sys
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import energy_model as EM, lattice as LT
import rev_experiments as RX

def job(a):
    kind, idx = a
    c = RX.new_case(kind, idx, 80, 7, EM.Robot())
    tm, full, s, g, h0 = c
    acts, _ = full.plan(s, h0, g)
    out = Counter()
    for (x, y, h), pi in acts:
        p = full.prims[h][pi]
        if p["name"].startswith("A"):
            S = p["samples"]; m = S[len(S) // 2]
            t = EM.TERRAIN_NAMES[tm.terrain[int(round(x + m[0])), int(round(y + m[1]))]]
            out[(p["name"], t)] += 1
        elif p["name"] == "R":
            S = None
            out[("R", EM.TERRAIN_NAMES[tm.terrain[x, y]])] += 1
    return out

if __name__ == "__main__":
    tot = Counter()
    with Pool(2) as pool:
        for c in pool.map(job, [(k, i) for k in RX.KINDS for i in range(50)]):
            tot.update(c)
    rows = [dict(primitive=k[0], surface=k[1], count=v) for k, v in sorted(tot.items())]
    with open(os.path.join(os.environ.get("OUTDIR", "results"), "arc_usage.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["primitive", "surface", "count"]); w.writeheader(); w.writerows(rows)
    for r in rows: print(r)
