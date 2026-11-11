# HEA*: Turning-Aware Energy-Efficient Path Planning for Tracked Robots

Code and data for the paper

> J. Tian and J. Kim, "Turning-Aware Energy-Efficient Path Planning for Tracked Robots on Mixed Terrain," submitted to *Robotica*.

HEA* (Heading-lattice Energy A*) plans paths for skid-steer tracked robots on an (x, y, θ) lattice with eight headings. It prices straight motion, 45° arcs of four radii and ±45° point turns with a skid-steer power model. The model covers radius-dependent lateral friction, inner-track braking, terrain-specific rolling resistance and slope. An admissible energy heuristic keeps the search optimal within the lattice.

## Requirements

Python 3.9 or later and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

Everything runs on a CPU. No GPU is needed.

## Files

| File | Content |
|---|---|
| `energy_model.py` | Robot parameters, surface coefficients, skid-steer energy model, synthetic map generator, DEM loader |
| `lattice.py` | Motion primitives, heading lattice, HEA* search, grid A* baselines |
| `run_energy.py` | Main experiment on synthetic maps (A*, TA*, TA*+turn, HEA*) with Wilcoxon tests |
| `sensitivity_energy.py` | Sensitivity to track length, friction scale and turning-moment gain |
| `run_dem.py` | Real-terrain experiment on AWS Terrain Tiles (needs Internet access) |
| `validation/validate_morales.py` | Check of the turning model against the measurements of Morales et al. (2009) |
| `model_figures.py`, `result_figures.py`, `paper_figs_extra.py` | Paper figures |
| `energy_kaggle.ipynb` | All steps in one notebook, tested on Kaggle (CPU) |
| `results/` | The result files reported in the paper |
| `revision/` | Additional analyses: effect sizes, arc usage, joint parameter uncertainty, Dijkstra check, discretisation, DEM resampling, any-angle baseline; results in `revision/results/` |

## Reproducing the paper

```
python run_energy.py --maps 50 --size 80 --out results_energy          # Table 8 and Fig. 6
python sensitivity_energy.py --maps 20 --out results_energy_sens       # Table 14 and Fig. 8
python run_dem.py --crops 25 --out results_dem                          # Table 10 (downloads DEM tiles)
python validation/validate_morales.py                                   # Section 3.5
python result_figures.py results_energy/energy_raw.csv results_energy_sens/sensitivity_energy.csv figs
python model_figures.py figs
python revision/effect_sizes.py results_energy/energy_raw.csv synthetic eff_synthetic.csv   # Table 11
python revision/collision_audit.py . audit_synthetic.csv                # collision audit (Section 6.6)
python revision/rev_experiments.py features --n 50       # Section 6.3
python revision/arc_usage.py                              # Table 13
python revision/rev_experiments.py anyangle --n 50       # Section 6.3
python revision/dem_resample.py                           # Table 12 (needs the DEM tiles of run_dem.py)
CONSERVATIVE=1 python revision/rev_experiments.py uncertainty --n 50   # Table 15
python revision/rev_experiments.py optimality --n 10 --size 40          # Section 6.6
python revision/rev_experiments.py discret --n 20                       # Table 16
```

Random seeds are fixed in the scripts, so the runs reproduce the files in `results/`.

All primitives use a conservative swept-cell collision check (`lattice.swept_cells`, 0.1-cell sampling): a path may neither enter nor touch a blocked cell. An earlier version of the code checked only sample points every 0.25 cells; its results are superseded by the files in `results/`.

## Data sources

- Elevation: [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/) (Terrarium encoding, zoom 15). In the United States these are derived from the USGS 3D Elevation Program. The tiles are downloaded by `run_dem.py` and are not redistributed here.
- `validation/morales2009_table3.csv` was transcribed from Table III of J. Morales et al., "Power consumption modeling of skid-steer tracked mobile robots on rigid terrain," *IEEE Trans. Robot.* 25(5), 1098–1108 (2009).

## Citation

If you use this code, please cite the paper above. The full reference will be added after publication.

## License

MIT, see `LICENSE`.
