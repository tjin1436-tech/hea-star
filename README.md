# HEA*: The Role of Turning Energy in Path Planning for Tracked Robots

Code and data for the paper

> J. Tian and J.-w. Kim, "The Role of Turning Energy in Path Planning for Tracked Robots," submitted to *Robotica*.

HEA* (Heading-lattice Energy A*) plans paths for skid-steer tracked robots on an (x, y, θ) lattice with
16 headings. Its motion primitives are straight moves, point turns and Bézier arcs of four sizes. Each primitive
is priced by a skid-steer energy model with radius-dependent lateral friction, inner-track braking without
regeneration, terrain-specific rolling resistance and slope. Its energy heuristic is consistent, so HEA* returns
energy-optimal paths within the lattice.

The study compares HEA* with grid, smoothed and any-angle planners on 150 synthetic maps and on 100 crops of
real elevation data. The crops use synthetic or ESA WorldCover surfaces. Every comparison runs under a nominal
and a five-fold turning loss.

> **Note.** The original experiment code of the paper was lost. This repository contains a reconstruction of
> the code from the paper (method, baselines, data sources, metrics and every experiment of Sections 3–6).
> Random number streams and some implementation details differ from the original, so the re-run numbers differ
> somewhat from those printed in the paper; `RESULTS.md` compares them. An earlier, 8-heading version of the
> code is in the git history (commit `62d708c`).

## Requirements

Python 3.9 or later, CPU only:

```bash
pip install -r requirements.txt
```

`rasterio` is only needed for the ESA WorldCover surfaces of the DEM experiment.

## Layout

| Path | Content | Paper |
|---|---|---|
| `heastar/energy.py` | Skid-steer energy model, surface coefficients, robot parameters, crossover radius | Sec. 3, Tables 3, 4, 7 |
| `heastar/maps.py` | Synthetic maps, AWS Terrain Tiles DEMs, ESA WorldCover surfaces, start/goal sampling | Sec. 5.1, 5.2 |
| `heastar/lattice.py` | Heading lattice (8/16/32 headings), Bézier arcs, edge-cost tables, HEA* search, heuristic (9) | Sec. 4 |
| `heastar/anyangle.py` | A*, TA*, Theta*-E, energy-decreasing line-of-sight smoothing | Sec. 5.3 |
| `heastar/planners.py` | The nine compared planners on one instance | Sec. 5.3 |
| `heastar/paths.py` | Path elements, conservative collision check, energy evaluation, grade along paths | Sec. 4.1, 4.2, 6.5 |
| `heastar/tracking.py` | Stop-and-turn and pure-pursuit replay in a kinematic simulation | Sec. 6.4 |
| `heastar/stats.py` | Bootstrap CIs, Hodges–Lehmann, rank-biserial r, Wilcoxon with Holm correction | Sec. 5.4 |
| `heastar/instances.py` | Deterministic instance generators shared by all experiments | Sec. 5.1, 5.2 |
| `experiments/` | One script per experiment (below) | Sec. 6 |
| `data/morales2009_table3.csv` | Table III of Morales et al. (2009), transcribed | Sec. 3.5 |
| `results/` | Result files of the re-run (`results/tables/` holds the tables) | |
| `figures/` | Figures of the re-run | |
| `tests/` | Unit tests (`python -m pytest tests`) | |

## Reproducing the paper

All scripts take `--out` (default `results`) and use every CPU core. The approximate run times below are for
4 cores.

```bash
python experiments/validate_morales.py            # Sec. 3.5: Tables 6 and 7 (seconds)
python experiments/run_main.py --data synthetic   # 150 synthetic instances, kappa = 1 and 5 (~20 min)
python experiments/run_main.py --data dem         # 100 DEM crops x 2 surface versions (~35 min, downloads ~10 MB of tiles)
python experiments/analyze_main.py                # Tables 8-15, Fig. 6, arc-usage correlations
python experiments/headings32.py                  # Sec. 6.2: 32 headings (~1 h)
python experiments/sensitivity.py                 # Table 16
python experiments/uncertainty.py                 # Table 17
python experiments/optimality.py                  # Sec. 6.7: Dijkstra check, heuristic vs exact cost-to-go
python experiments/collision_audit.py             # Sec. 6.7: collision audit at 0.01-cell sampling
python experiments/discretisation.py              # Table 18
python experiments/dem_resample.py                # Sec. 6.7: nearest / bilinear / cubic DEM resampling
python experiments/scaling.py                     # Table 19 (~2.5 h; --sizes 500 --merge adds only 500 x 500, ~1.5 h)
python experiments/figures.py                     # Figs. 2, 3, 4, 5, 7
```

`run_main.py` appends to `results/main_raw.csv`, so an interrupted run resumes where it stopped.
Each row is one planner on one instance and model scenario. A row holds the planned energy, travel time,
length, rotation, planning time, the maximum grade along the path, the share of the path on soft ground and
on arcs, and the energies after stop-and-turn and pure-pursuit tracking.

## Implementation notes

* **Lattice.** The headings are the directions of the integer grid vectors with components up to 1, 2 or 3,
  which gives 8, 16 or 32 headings. From heading *h* with grid vector **d**<sub>h</sub>, the arc A<sub>k</sub>
  moves to k(**d**<sub>h</sub> + **d**<sub>h±1</sub>) along a cubic Bézier curve tangent to both headings. Its
  nominal radius is about 6.9k cells for 16-heading arcs that start or end on an axis or diagonal, 11.3k cells
  for the other 16-heading arcs, and 2.9k cells on the 8-heading lattice. Curves are sampled every 0.1 cells.
  Each sub-segment is priced with the local radius of curvature at its midpoint.
* **Collision check.** Around every sample, all cells that overlap a square of half-width
  max(0.05, 0.55Δ) must be traversable, where Δ is the largest sample gap. This check is conservative: a path
  never touches a blocked cell or squeezes diagonally between two blocked cells. On synthetic maps, blocked
  cells are generated directly in configuration space. Grid, any-angle and lattice planners share this check.
* **Evaluation.** Every path is a list of in-place rotations and sampled curves. One evaluator scores all paths:
  bilinear elevation, the surface class at sub-segment midpoints, and the local radius. Lattice paths
  re-evaluated this way match the search cost to ~1e-12.
* **Theta*-E.** Standard Theta* (the line-of-sight path through the parent's parent is taken whenever it is
  collision-free) with the straight-line energy integrated along each segment as edge cost and heuristic (9).
* **Smoothing.** This step gives TA*+S and TA*+turn+S. It starts from the cells of the path and greedily
  removes waypoints whenever the straight line between the remaining ones is collision-free and the total energy,
  including point turns by arbitrary angles, decreases.

## Data sources

* Elevation: [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/), Terrarium encoding, zoom 15,
  2 × 2 tiles per region. In the United States they are derived from USGS 3DEP. The tiles are downloaded to
  `data/tiles/` and are not redistributed.
* Land cover: [ESA WorldCover 2021 v200](https://doi.org/10.5281/zenodo.7254221), 10 m, read by window from
  the public COGs on AWS and cached in `data/worldcover/`.
* `data/morales2009_table3.csv`: Table III of J. Morales et al., "Power consumption modeling of skid-steer
  tracked mobile robots on rigid terrain," *IEEE Trans. Robot.* 25(5), 1098–1108 (2009).

## Citation

If you use this code, please cite the paper above. The full reference will be added after publication.

## License

MIT, see `LICENSE`.
