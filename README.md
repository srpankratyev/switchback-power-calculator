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

How the data API reads a pre-period extract:

- **Only cells with data count as randomizations.** If some cluster × time cells are empty, or are dropped by `min_cell_count`, the variance is divided by the number of cells present (`DesignStats.n_cells`), not by `num_clusters × num_time_periods`, and `DataDiagnostics.warnings` says how many cells are absent. `min_cell_count` therefore describes an experiment whose analysis also drops those cells. To unpack inferred parameters into the parameter API, use `parameters.summarize_kwargs()`, which passes the periods actually present per cluster.
- **Variance shares are variance components, not raw sums of squares.** The between-cell share is the one-way ANOVA estimate with the cell as the group. The raw ratio `SS_cell / SS_total` counts within-cell noise in the cell means as between-cell variance and overstates the share by about `(1 − share) / mean_obs_per_cell`. The cluster / time / interaction split in `VarianceShares` is the raw, descriptive partition.

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

## Carryover-aware planning and analysis (experimental)

Everything above assumes a treated period affects only itself. The subpackage `switchback_power_calculator.carryover` drops that assumption: a treated period has an effect `l0` in the same period, `l1` in the next, and a tail that fades after that. It covers both ends of an experiment, and the top-level package does not import it.

**Six tests.** All are one-sided at `alpha` (default 0.10), run on per-cluster lag-0 and lag-1 estimates, and differ in what they assume about effects after the next window. `C` bounds those later effects: they have the sign of `l1` and add up to at most `C × l1`.

| # | id | reports an improvement when | assumes |
|---|---|---|---|
| 1 | `lag0` | the same-window effect is significantly below 0 | no carryover |
| 2 | `lag01` | A = same-window + next-window effect is significantly below 0 | nothing after the next window |
| 3 | `two_endpoint` | A and A + μ are both significantly below 0, where μ is the most later windows can add | the bound `C`, and one share of it across clusters |
| 4 | `moment_bound` | every mix of clusters at and away from the bound is rejected | exchangeable next-window effects, known `C`; uses the between-cluster variance τ² as estimated, which can be negative |
| 4.5 | `moment_bound_floored` | the same | the same, with τ² floored at 0; it rejects only when test 4 does |
| 5 | `assumption_free` | every sign pattern of the clusters' next-window effects is rejected | only the bound `C` |

**Before the experiment: how long does each test need?** Read the noise off a pre-period at each candidate window length, state the effect you plan for, and compare:

```python
from switchback_power_calculator.carryover import (
    TailBound, compare_designs, lag_moments_from_aggregates, synthetic_panel,
)

# One row per cluster and hour: cluster_id, period_start, numerator (sum of the
# outcome), denominator (observations), expected (expected sum of the outcome).
pre_period = synthetic_panel(300, 4, seed=3)
origin = pre_period["period_start"].min()
designs = {
    "1 h": lag_moments_from_aggregates(pre_period, expected_col="expected"),
    "4 h": lag_moments_from_aggregates(
        pre_period, expected_col="expected", window_minutes=240, origin=origin
    ),
}
bound = TailBound(c=0.43, source="decay fit", stress_c=0.8)
# Same-hour effect -0.10, 30% of it undone in the next hour, later hours at the bound.
target = bound.scenario(-0.10, 0.3, l1_spread_sd=0.05, label="same-hour -0.10, 30% rebound")
plan = compare_designs(
    designs, [target], direction="lower_is_better", tail_bound=bound,
    tests=("lag0", "two_endpoint", "moment_bound_floored"), metric="outcome", unit="units",
)
print(plan.to_text())
```

```
Q3. Weeks to 80% power: same-hour -0.10, 30% rebound -> total -0.0571 units [tail at bound -0.0571, no tail -0.0700]
    ranked by weeks to target power; not a recommendation
  #   test                                                   design  weeks             if noise +20%     power if spread x2
  1   lag-0 (same-window effect only)                        1 h     <= 1              <= 1
  1   lag-0 (same-window effect only)                        4 h     1.63 -> run 2     2.32 -> run 3
  3   two-endpoint [C]                                       1 h     5.40 -> run 6     7.77 -> run 8
  3   two-endpoint [C]                                       4 h     6.43 -> run 7     9.23 -> run 10
  3   two-endpoint [stress C]                                4 h     6.76 -> run 7     9.69 -> run 10
  4.5 moment bound, tau2 floored at 0 [C], simulated         1 h     7                 10                0.68  fragile to spread
  4.5 moment bound, tau2 floored at 0 [C], simulated         4 h     7                 10                0.82
  4.5 moment bound, tau2 floored at 0 [stress C], simulated  4 h     7                 10                0.80
  3   two-endpoint [stress C]                                1 h     11.26 -> run 12   16.21 -> run 17
  4.5 moment bound, tau2 floored at 0 [stress C], simulated  1 h     19                28                0.29  fragile to spread
```

The full text also shows the noise at each window (Q1) and the smallest detectable effect by rebound share (Q2). `plan.basis`, `plan.curve` and `plan.table` hold the same numbers as data frames. Tests 1 to 3 are planned in closed form (`closed_form_power`, `mde`, `weeks_to_power`); tests 4, 4.5 and 5 are simulated by running the analysis code on simulated experiments (`simulated_power`), and need the scenario's between-cluster spread `l1_spread_sd`. Without pre-period data, `lag_moments_from_parameters` builds the moments from the same inputs as `summarize_power`.

**After the experiment: run the tests.**

```python
from switchback_power_calculator.carryover import estimate_lags, experiment_windows, run_tests

# outcomes: cluster_id, period_start, numerator, denominator, expected (one row per cluster-hour)
# assignment: cluster_id, period_start, treated (1 or 0)
windows = experiment_windows(
    outcomes, assignment, window_minutes=240, origin=experiment_start, expected_col="expected"
)
estimates = estimate_lags(windows, window_minutes=240)
c_window = dict(zip(estimates.cluster_ids, bound.window_values(estimates.cluster_ids, 240)))
results = run_tests(estimates, c_window, direction="lower_is_better", t1=t1)
results["moment_bound_floored"].improves
```

`preregistration(plan, design=..., test=..., scenario=..., weeks=..., cluster_rule=...)` writes the chosen design, test, duration and settings as one line to record before launch, and `check(line, run)` lists where a finished analysis departs from it.

`t1` is the linearisation point of the two moment-bound tests. It must be fixed without looking at the experiment itself: `t1_leave_one_out` computes it from the other experiments of a run, `planning_t1` from the plan.

**What to keep in mind**

- `C` is an assumption; the pre-period cannot estimate it. For windows longer than the base period the planner converts it with a geometric rule that needs each later lag to be at most `C / (1 + C)` times the one before.
- Planned SEs assume the experiment is as noisy as the pre-period. The "if noise +20%" column is a budget, in line with the advice above to inflate the SE by 10–20%.
- A cluster's level drifts from week to week. The analysis centres each cluster over the experiment, so a one-week experiment never sees that drift and a longer one sees part of it. The planner reads the noise off the pre-period both ways, within each week and over the whole pre-period, and joins the two by the planned duration (`ClusterLagMoments.moments_for`). An experiment longer than the pre-period is planned as if it were as long as the pre-period, so give the planner at least as many weeks as you intend to run.
- Planned power is slightly optimistic with few windows per cluster (flagged below 50). Treat differences under about 0.05 power or one week between window choices as ties.
- Each test keeps its false-improvement rate only under its own assumption, and reporting whichever test passes does not. Choose one test before the experiment.

## Tests

Requires Python 3.10+ (see `requires-python` in `pyproject.toml`).

```bash
pip install -e ".[dev]"
pytest
```

## Citation

> Pankratev, S. (2026). *Powerful Switchback Experiments – Or Not?* arXiv:2606.03012. https://arxiv.org/abs/2606.03012

Replication code: https://github.com/srpankratyev/power_paper_repo_simulations_placebo
