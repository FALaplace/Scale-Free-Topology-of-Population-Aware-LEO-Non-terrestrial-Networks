# Scale-Free Topology of Population-Aware LEO Non-terrestrial Networks

This repository provides a lightweight reproducibility package for the paper **“Scale-Free Topology of Population-Aware LEO Non-terrestrial Networks.”** It contains the aggregate results used in the revised manuscript, the processed degree distributions needed to reproduce the robust refits in Table 1, and a standalone statistical analysis script.

## Repository contents

```text
analysis/
  reproduce_table1.py
data/processed/table1_degree_distributions/
  GS*_CITY*.txt
results/core/
  attachment_proxy.csv
  attachment_proxy_fit.csv
  table1_robust_refit.csv
  dynamic_snapshot_statistics.csv
  static_ablation.csv
  periodicity_peaks.csv
  cross_constellation_static_statistics.csv
results/diagnostics/
  association_sensitivity.csv
  hub_failure_sensitivity.csv
  degree_population_check.csv
```

The six files under `data/processed/table1_degree_distributions/` contain only aggregate degree-probability pairs. The first row lists degree values and the second row lists their empirical probabilities. They contain no terminal identities, geographic coordinates, orbital states, or edge lists.

## Reproduce the Table 1 refits

Python 3 with NumPy and SciPy is required:

```bash
pip install numpy scipy
python analysis/reproduce_table1.py
```

The script performs the same analysis used for the revised Table 1:

- discrete power-law maximum-likelihood estimation with data-driven `xmin`;
- Kolmogorov–Smirnov goodness-of-fit assessment using 99 parametric bootstrap replicates;
- 95% bootstrap confidence intervals for the exponent using 199 replicates;
- factor-two logarithmic binning for the descriptive fit;
- likelihood comparison with a discrete lognormal alternative.

It verifies the reproduced key statistics against `results/core/table1_robust_refit.csv` and writes the full recomputed table to `results/table1_robust_refit_reproduced.csv`.

## Result files

| File | Analysis represented |
|---|---|
| `attachment_proxy.csv`, `attachment_proxy_fit.csv` | Population-induced attachment-opportunity analysis |
| `table1_robust_refit.csv` | Robust refit of the six degree distributions reported in Table 1 |
| `dynamic_snapshot_statistics.csv` | Statistical results across the selected Starlink snapshots |
| `static_ablation.csv` | Population-aware versus spatially uniform terminal-distribution ablation |
| `periodicity_peaks.csv` | Dominant periodic components in the dynamic topology series |
| `cross_constellation_static_statistics.csv` | Static comparison among Starlink, TeleSat, Kuiper, and OneWeb |
| `association_sensitivity.csv` | Sensitivity to the terminal-association rule |
| `hub_failure_sensitivity.csv` | Targeted-hub versus random-failure diagnostic |
| `degree_population_check.csv` | Degree-population definition diagnostic |

## Data scope

To avoid redistributing large or location-sensitive inputs, this repository does **not** include raw population records, terminal coordinates, orbital trajectories, topology snapshots, or link-level network files. Consequently, it supports independent inspection of the reported aggregate results and reproduction of the Table 1 statistical refits, but not end-to-end regeneration of every network snapshot.

The omitted inputs and complete data-generation workflow are available from the corresponding author upon reasonable request, subject to the applicable data-sharing constraints.
