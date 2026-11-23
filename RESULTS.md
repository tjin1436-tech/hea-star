# Re-run of the experiments with the reconstructed code

The original experiment code was lost, and the code in this repository was rebuilt from the manuscript. Every
experiment of the paper was run again with it, on a 4-core Linux VM like the one described in Section 5.4.
The result files are in `results/` and the tables in `results/tables/`. This page compares the re-run with the
numbers printed in the manuscript.

The random streams that generate the maps, crops, endpoints and parameter draws are new, so individual numbers
differ. **Every qualitative conclusion of the paper is reproduced**, and most medians agree within about one
percentage point.

## Exactly reproduced (deterministic)

| Item | Paper | Re-run |
|---|---|---|
| Morales check: measured / power-up + traction / remainder | 45.3 / 27.2 / 18.1 kJ (40 %) | 45.3 / 27.2 / 18.1 kJ (40 %) |
| Turning energy per radian (measured remainder) | 5.2 ± 0.7 kJ/rad | 5.2 ± 0.7 kJ/rad |
| Table 6: point turn / arc 1 m / arc 3 m | 0.51 / 0.75 / 1.02 kJ/rad | 0.51 / 0.75 / 1.02 kJ/rad |
| Table 7: crossover radius sand / grass / gravel / concrete | ≈ 6 / 11 / 19 / 29 m | 6.0 / 10.5 / 18.3 / 28.7 m |
| Inner-track braking up to R* (concrete) | ≈ 33 m | 33.6 m |
| Nominal arc radii (16 headings) | 6.9k / 11.3k cells | 6.88k / 11.25k cells |
| WorldCover shares (Hanksville bare, Mojave bare, Palouse crop, Flint Hills grass+crop / tree) | 99.5 / 98.4 / 94.4 / 76 / 22 % | 99.5 / 98.4 / 94.4 / 76 / 22 % |

## Main comparison (Table 9, median change of HEA*)

| Data, κ | vs TA* | vs TA*+S | vs Theta*-E | vs TA*+turn+S |
|---|---|---|---|---|
| Synthetic, κ = 1 | paper −19.8 / re-run −20.5 % | −7.8 / −8.0 % | −6.0 / −5.6 % | −3.8 / −4.2 % |
| DEM synthetic surfaces, κ = 1 | −19.4 / −21.3 % | −11.2 / −11.1 % | −15.2 / −17.0 % | −3.7 / −4.1 % |
| DEM WorldCover, κ = 1 | −26.0 / −25.3 % | +0.2 / +0.1 % | +0.2 / +0.1 % | +0.2 / +0.1 % |
| Synthetic, κ = 5 | −45.1 / −45.5 % | −4.1 / −3.8 % | −4.0 / −4.3 % | −1.4 / −1.6 % |
| DEM synthetic surfaces, κ = 5 | −43.8 / −47.0 % | −4.9 / −5.0 % | −4.6 / −5.0 % | −2.1 / −0.9 % |
| DEM WorldCover, κ = 5 | −53.3 / −47.0 % | +2.2 / +2.6 % | +2.2 / +2.5 % | +2.0 / +2.5 % |

HEA* vs A* is −31 / −37 / −21 % (κ = 1) in the re-run, against −39 / −39 / −32 % in the paper. The grid A* of the
re-run breaks ties towards straighter paths: its mean total rotation on the synthetic maps is 643° against 997°
in Table 8, so it costs less. All other baselines agree closely. For example, Theta*-E turns 186° on average
(paper 184°).

## Decomposition (Table 11, synthetic, κ = 1 / κ = 5)

| Effect | Paper | Re-run |
|---|---|---|
| Pricing turns (grid) | −13.1 / −39.2 % | −13.9 / −40.4 % |
| Pricing turns (any-angle) | −3.5 / −0.8 % | −3.6 / −0.5 % |
| 16 instead of 8 headings | −5.3 / −7.5 % | −4.9 / −6.9 % |
| Arcs, 8 headings | −2.5 / 0.0 % | −2.8 / 0.0 % |
| Arcs, 16 headings | −2.0 / 0.0 % | −1.9 / 0.0 % |
| Any-angle smoothing | −3.3 / −5.7 % | −2.7 / −5.2 % |
| HEA*-8 vs TA*+turn+S | +0.4 / +6.1 % | +0.3 / +5.5 % |

The arc gain correlates with the share of the path driven on arcs (Spearman ρ = −0.61; paper −0.63) and with
the number of arcs (−0.54; paper −0.60). It does not correlate with distance, relief or the soft-ground share.

## Other studies

| Study | Paper | Re-run |
|---|---|---|
| Stop-and-turn tracking overhead | within 0.7 % | within 0.1 % (median) |
| Pure pursuit, κ = 5: HEA* vs TA*+S / TA*+turn+S (synthetic) | +6.4 / +6.8 % | +10.8 / +9.8 % |
| Tracked paths entering a blocked cell (pure pursuit) | ≤ 4 of 150 | ≤ 4 of 150 (TA*+turn+S), 0 on DEM |
| Max grade along paths | 19.9° (one path > 20°) | 20.4° (10 of 3,150 paths marginally > 20°) |
| Idle power 58 W vs TA*+S (synthetic, κ = 1) | −0.3 % | +0.2 % |
| Table 16: L = 1.2 m and κ × 5 remove the arc gain | 0.0 % | 0.0 % |
| Table 16: HEA*-8 vs TA*, L 0.4 → 1.2 m | −13.6 → −22.7 % | −14.1 → −25.6 % |
| Table 17: pricing turns robust (TA*+turn < TA* with nominal model) | 149 / 150 | 149 / 150 |
| Table 17: HEA*-8 worse than TA*+turn with nominal model, κ > 2 | 47 of 68 | 38 of 56 |
| Optimality: HEA* = Dijkstra; heuristic violations | 30/30; 0 | 30/30 and 30/30; 0 of 737,448 states |
| Median h / cost-to-go (8 / 16 headings) | 0.22 / 0.23 | 0.22 / 0.23 |
| Fewer expansions than Dijkstra (8 / 16 headings) | 30 / 32 % | 29 / 32 % |
| Collision audit: conservative check | 0 paths touch | 0 of 2,850 paths touch |
| Collision audit: naive check (0.25 cell, sample cells only) | 16 of 150 enter, ≤ 0.04 cells | 9 of 150 enter, ≤ 0.04 cells |
| Table 18: default / 1.0-cell sampling | −2.4 % / 15 worse | −3.0 % / 15 worse |
| Table 18: A1, A2 only / A1–A16 | −0.1 / −3.0 % | −0.4 / −3.4 % |
| DEM resampling: nearest-neighbour raises the HEA*-8 saving vs TA* | 16.3 → 21.5 % | 17.6 → 22.0 % |
| 32 vs 16 headings, point turns (κ = 1 / 5) | −1.8 / −2.4 % | −1.6 / −2.4 % |
| 32 vs 16 headings, with arcs | −1.2 / −3.1 % | −1.1 / −2.8 % |
| HEA*-32 vs TA*+turn+S | −4.8 / −4.4 % | −5.2 / −4.7 % |
| Table 19, 320 × 320, 16 headings: precompute / tables / peak memory / plan time | 255 s / 288 MB / 1,033 MB / 2.4–41.6 s | 118 s / 138 MB / 626 MB / 3.2–16.9 s |
| Table 19: 32 vs 16 headings precomputation | ×3.4 | ×3.1 |

## Known differences

* **Random streams.** The maps, crops, endpoints and uncertainty draws come from new random streams. Medians
  therefore move by about ±1 percentage point, and counts differ (e.g. the κ bins of Table 17).
* **Grid A* tie-breaking.** A* paths are straighter than in the paper (see above). This affects only the
  A* column.
* **Any-angle paths on DEM crops.** The DEM crops of the re-run have ~95 % traversable cells. Most smoothed and
  Theta*-E paths are one straight segment, so their pure-pursuit overhead on DEM is ~0 %, while the paper
  reports +1–17 %.
* **Speed and memory.** The re-run implementation is about 2–5× faster than the times reported in the paper
  (e.g. HEA* ≈ 0.3–0.5 s per query and ≈ 7 s precomputation on 80 × 80 maps, against 1.6 s and 14 s). It stores
  one edge table instead of two, so the table sizes are half those of Table 19. Table 19 was re-run up to
  320 × 320 cells (`results/tables/table19_scaling.csv`); 500 × 500 needs `--sizes 500` and several hours.
