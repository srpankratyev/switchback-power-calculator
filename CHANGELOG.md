# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

New experimental subpackage `switchback_power_calculator.carryover` for experiments whose treatment effects continue into later periods. It plans how many weeks each of six tests needs at each window length (`lag_moments_from_aggregates`, `compare_designs`) and runs those tests on a finished experiment (`experiment_windows`, `estimate_lags`, `run_tests`). The top-level package does not import it, and nothing in the existing API changes.

Fixes to the data API. Results from `estimate_design_parameters_from_data` and `power_from_data` change for incomplete grids and for small cells; the parameter API is unchanged.

- Incomplete grids: the variance was divided by `num_clusters × num_time_periods` even when some of those cells held no data (empty, or dropped by `min_cell_count`), which understated the standard error by the square root of the grid fill fraction. It is now divided by the number of cells present. `DesignStats` gains `grid_fill_fraction` and `effective_num_time_periods`, `DesignParameters.summarize_kwargs()` passes the effective period count, and `DataDiagnostics.warnings` reports absent cells.
- Variance shares: `between_cell_variance_share` is now the one-way ANOVA variance-component estimate instead of `SS_cell / SS_total`, which overstated the share by about `(1 − share) / mean_obs_per_cell` and so overstated the standard error for small cells (about 10% at 20 observations per cell, 24% at 5). When every cell holds a single observation the raw ratio is returned with a warning.

## 1.0.0

Initial release of `switchback-power-calculator`: power and sample-size calculator for switchback (cluster × time) experiments, implementing Pankratev (2026). Supports direct parameter input and estimation from pre-period data.
