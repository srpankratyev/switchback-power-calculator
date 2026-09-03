# switchback-power-calculator

Power and sample-size calculator for switchback experiments — treatment randomized at the cluster × time cell (e.g. zone × hour). Implements the closed-form variance formula from [Pankratev (2026)](https://arxiv.org/abs/2606.03012); pass design parameters directly or infer them from pre-period data.

## Install

```bash
pip install switchback-power-calculator
```

To work on the package locally (Python 3.10 or newer):

```bash
pip install -e ".[dev]"
```

## Usage

With this package, you can estimate the variance and power of your switchback experiment from known design parameters or from a pre-period dataset.

For the data API, pass a long-format table with one row per observation and three columns: cluster ID, time period ID, and numeric outcome. Column names are configurable; defaults are `cluster_id`, `time_id`, and `outcome`.

Use the following functions depending on your goal:

**From pre-period data**

- Estimate design parameters (cluster/time counts, cell sizes, variance shares, outcome SD): `estimate_design_parameters_from_data`
- Estimate variance shares only: `estimate_variance_shares`
- Estimate variance and power in one call: `power_from_data`

**From inferred design parameters**

- Estimate estimator variance (standard error and approximation warnings): `summarize_estimator_variance_from_design`
- Estimate variance, MDE, and (optionally) power and required cells: `summarize_power_from_design`

**From known design parameters**

- Estimate estimator variance: `summarize_estimator_variance`
- Estimate statistical power: `power`
- Estimate minimum detectable effect: `mde`
- Estimate required number of cells: `num_required_randomizations`
- Estimate variance, MDE, and (optionally) power and required cells together: `summarize_power`

See [API reference](#api-reference) for signatures and defaults. The [examples](#examples--parameter-api) below show typical calls.

## Examples — parameter API

```python
from switchback_power_calculator import summarize_power

result = summarize_power(
    num_clusters=100,
    num_time_periods=168,
    mean_obs_per_cell=20,
    cell_size_cv=1.5,
    between_cell_variance_share=0.20,
    outcome_sd=1000,
    treatment_effect=100,
)
print(round(result.standard_error, 2), round(result.mde, 2), round(result.power, 2))
print(result.approximation_validity, result.warnings)
```

```
12.91 36.17 1.0
approximation accurate ()
```

## Examples — data API

```python
from switchback_power_calculator import (
    estimate_design_parameters_from_data,
    summarize_estimator_variance_from_design,
    summarize_power_from_design,
)

parameters, diagnostics = estimate_design_parameters_from_data(
    df,
    cluster_col="zone",
    time_col="hour_bucket",
    outcome_col="trip_duration_sec",
    min_cell_count=10,
)
variance = summarize_estimator_variance_from_design(parameters)
result = summarize_power_from_design(parameters, treatment_effect=50)
```

Or in one step:

```python
from switchback_power_calculator import power_from_data

result, parameters, diagnostics = power_from_data(df, treatment_effect=50)
```

## API reference

Functions are layered: clean data → infer parameters → estimate variance → estimate power. `within_cell_variance_share` defaults to `1 - between_cell_variance_share` when omitted.

**Data**

```python
prepare_data(
    df,
    cluster_col="cluster_id",
    time_col="time_id",
    outcome_col="outcome",
    min_cell_count=1,
    max_missing_fraction=0.5,
    missing_fraction_warn=0.1,
)  # -> (DataFrame, DataDiagnostics)

compute_design_stats(
    df,
    cluster_col="cluster_id",
    time_col="time_id",
    outcome_col="outcome",
    min_cell_count=1,
)  # -> DesignStats

estimate_variance_shares(
    df,
    cluster_col="cluster_id",
    time_col="time_id",
    outcome_col="outcome",
    min_cell_count=1,
)  # -> VarianceShares

estimate_design_parameters_from_data(
    df,
    cluster_col="cluster_id",
    time_col="time_id",
    outcome_col="outcome",
    min_cell_count=1,
    max_missing_fraction=0.5,
    missing_fraction_warn=0.1,
)  # -> (DesignParameters, DataDiagnostics)

power_from_data(
    df,
    cluster_col="cluster_id",
    time_col="time_id",
    outcome_col="outcome",
    min_cell_count=1,
    max_missing_fraction=0.5,
    missing_fraction_warn=0.1,
    treatment_effect=None,
    alpha=0.05,
    power_level=0.8,
)  # -> (PowerResult, DesignParameters, DataDiagnostics)
```

**Variance** (from known parameters)

```python
estimator_variance(
    num_clusters,
    num_time_periods,
    *,
    mean_obs_per_cell,
    cell_size_cv,
    between_cell_variance_share,
    within_cell_variance_share=None,
    outcome_sd,
)  # -> float, Var(tau_hat)

standard_error(...)  # -> float, same signature as estimator_variance

formula_approximation_validity(
    num_clusters,
    num_time_periods,
    *,
    mean_obs_per_cell,
    cell_size_cv,
)  # -> (ApproximationValidity, tuple[str, ...])

summarize_estimator_variance(
    num_clusters,
    num_time_periods,
    *,
    mean_obs_per_cell,
    cell_size_cv,
    between_cell_variance_share,
    within_cell_variance_share=None,
    outcome_sd,
)  # -> VarianceResult

summarize_estimator_variance_from_design(parameters)  # -> VarianceResult
```

**Power** (from known parameters)

```python
power(
    treatment_effect,
    num_clusters,
    num_time_periods,
    *,
    mean_obs_per_cell,
    cell_size_cv,
    between_cell_variance_share,
    within_cell_variance_share=None,
    outcome_sd,
    alpha=0.05,
)  # -> float

mde(
    num_clusters,
    num_time_periods,
    *,
    mean_obs_per_cell,
    cell_size_cv,
    between_cell_variance_share,
    within_cell_variance_share=None,
    outcome_sd,
    alpha=0.05,
    power_level=0.8,
)  # -> float

num_required_randomizations(
    treatment_effect,
    *,
    mean_obs_per_cell,
    cell_size_cv,
    between_cell_variance_share,
    within_cell_variance_share=None,
    outcome_sd,
    alpha=0.05,
    power_level=0.8,
)  # -> float, required num_clusters * num_time_periods
   # (this is what PowerResult.required_num_cells reports)

summarize_power(
    num_clusters,
    num_time_periods,
    *,
    mean_obs_per_cell,
    cell_size_cv,
    between_cell_variance_share,
    within_cell_variance_share=None,
    outcome_sd,
    treatment_effect=None,
    alpha=0.05,
    power_level=0.8,
)  # -> PowerResult

summarize_power_from_design(
    parameters,
    treatment_effect=None,
    alpha=0.05,
    power_level=0.8,
)  # -> PowerResult
```

Pass `treatment_effect` to `summarize_power`, `summarize_power_from_design`, or `power_from_data` to also get power and `required_num_cells`.

## The formula

```
Var(τ̂) ≈ (4σ²/JH) × [ S_res/n̄ + S_macro × (1/n̄ + 1 + cv²) ]
```

| Paper | Code | Meaning |
|-------|------|---------|
| `τ̂` | — | The estimated treatment effect. This is its variance. |
| `J` | `num_clusters` | Clusters (zones, stores, markets) |
| `H` | `num_time_periods` | Time periods treatment is re-randomized over |
| `n̄` | `mean_obs_per_cell` | Mean observations per cluster × time cell |
| `cv` | `cell_size_cv` | Coefficient of variation of cell sizes |
| `S_macro` | `between_cell_variance_share` | Share of outcome variance between cells |
| `S_res` | `within_cell_variance_share` | Share within cells (the two shares sum to 1) |
| `σ` | `outcome_sd` | Outcome standard deviation |
| `τ` | `treatment_effect` | Effect size you want to detect, in outcome units |

## Assumptions

- Cell-level 50/50 randomization; no stratification or blocking.
- Cell size and cell-level outcome shocks are treated as unrelated. On real marketplace data — where that independence is only approximate — the formula tends to *understate* variance: the NYC placebo study found ~17% underprediction on average. Consider inflating the reported SE by 10–20% when budgeting.
- Separately, the formula tends to *overstate* variance for small or highly imbalanced designs (`cell_size_cv ≥ 2` or `num_clusters × num_time_periods ≤ 10`) — a different regime from the marketplace-data bias above. See `approximation_validity` and `warnings` on every `VarianceResult` / `PowerResult`.

## Tests

Requires Python 3.10+ (see `requires-python` in `pyproject.toml`).

```bash
pip install -e ".[dev]"
pytest
```

## Citation

> Pankratev, S. (2026). *Powerful Switchback Experiments — Or Not?* arXiv:2606.03012. https://arxiv.org/abs/2606.03012

Replication code: https://github.com/srpankratyev/power_paper_repo_simulations_placebo
