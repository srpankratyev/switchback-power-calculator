from __future__ import annotations

import numpy as np
import pandas as pd

from switchback_power_calculator.core import summarize_power_from_design
from switchback_power_calculator.types import (
  DataDiagnostics,
  DesignParameters,
  DesignStats,
  PowerResult,
  VarianceShares,
)

REQUIRED_COLUMNS = ("cluster_id", "time_id", "outcome")
MIN_CLUSTERS = 2
DEFAULT_MAX_MISSING_FRACTION = 0.5
DEFAULT_MISSING_FRACTION_WARN = 0.1


def _validate_cluster_count(num_clusters: int) -> None:
  if num_clusters < MIN_CLUSTERS:
    raise ValueError(
      f"At least {MIN_CLUSTERS} clusters are required; got num_clusters={num_clusters}. "
      "With a single cluster the cluster shock is not identified across units."
    )


def _coerce_outcome(series: pd.Series) -> pd.Series:
  numeric = pd.to_numeric(series, errors="coerce")
  unparseable = series.notna() & numeric.isna()
  if unparseable.any():
    n_bad = int(unparseable.sum())
    raise ValueError(
      f"outcome column must be numeric; {n_bad} value(s) could not be parsed as numbers."
    )
  return numeric


def _normalize_id_column(series: pd.Series) -> pd.Series:
  """Cast IDs to string so 1 and \"1\" identify the same cluster or period."""
  return series.map(str)


def prepare_data(
  df: pd.DataFrame,
  cluster_col: str = "cluster_id",
  time_col: str = "time_id",
  outcome_col: str = "outcome",
  min_cell_count: int = 1,
  max_missing_fraction: float = DEFAULT_MAX_MISSING_FRACTION,
  missing_fraction_warn: float = DEFAULT_MISSING_FRACTION_WARN,
) -> tuple[pd.DataFrame, DataDiagnostics]:
  """Clean a long-format extract for power analysis."""
  if min_cell_count < 1:
    raise ValueError("min_cell_count must be at least 1.")
  if not 0 < max_missing_fraction <= 1:
    raise ValueError("max_missing_fraction must be in (0, 1].")
  if not 0 <= missing_fraction_warn < max_missing_fraction:
    raise ValueError("missing_fraction_warn must be in [0, max_missing_fraction).")

  rename_map = {
    cluster_col: "cluster_id",
    time_col: "time_id",
    outcome_col: "outcome",
  }
  missing_cols = [name for name in rename_map if name not in df.columns]
  if missing_cols:
    raise KeyError(f"Missing required columns: {missing_cols}")

  work = df[list(rename_map)].rename(columns=rename_map).copy()
  rows_input = len(work)
  if rows_input == 0:
    raise ValueError("Input frame is empty.")

  work["outcome"] = _coerce_outcome(work["outcome"])

  complete = work.dropna(subset=list(REQUIRED_COLUMNS))
  rows_dropped_missing = rows_input - len(complete)
  if len(complete) == 0:
    raise ValueError("No rows remain after dropping missing values.")

  complete = complete.copy()
  complete["cluster_id"] = _normalize_id_column(complete["cluster_id"])
  complete["time_id"] = _normalize_id_column(complete["time_id"])

  missing_fraction = rows_dropped_missing / rows_input
  data_warnings: list[str] = []
  if missing_fraction >= max_missing_fraction:
    raise ValueError(
      f"{rows_dropped_missing} of {rows_input} rows ({missing_fraction:.0%}) have missing "
      f"cluster, time, or outcome values, exceeding max_missing_fraction="
      f"{max_missing_fraction:.0%}. Variance shares inferred from the remaining rows "
      "may not describe the full dataset."
    )
  if missing_fraction >= missing_fraction_warn:
    data_warnings.append(
      f"{rows_dropped_missing} of {rows_input} rows ({missing_fraction:.0%}) were dropped "
      "for missing values; check that the remaining data is representative."
    )

  cell_counts = complete.groupby(["cluster_id", "time_id"], observed=True).size()
  cells_before_filter = len(cell_counts)
  keep_cells = cell_counts[cell_counts >= min_cell_count].index
  if len(keep_cells) == 0:
    raise ValueError("No cells remain after min_cell_count filter.")

  filtered = (
    complete.set_index(["cluster_id", "time_id"])
    .loc[complete.set_index(["cluster_id", "time_id"]).index.isin(keep_cells)]
    .reset_index()
  )

  diagnostics = DataDiagnostics(
    rows_input=rows_input,
    rows_used=len(filtered),
    rows_dropped_missing=rows_dropped_missing,
    cells_before_filter=cells_before_filter,
    cells_after_filter=len(keep_cells),
    min_cell_count=min_cell_count,
    warnings=tuple(data_warnings),
  )
  return filtered, diagnostics


def _compute_design_stats_from_clean(clean: pd.DataFrame) -> DesignStats:
  cell_counts = clean.groupby(["cluster_id", "time_id"], observed=True).size()
  mean_obs_per_cell = float(cell_counts.mean())
  if mean_obs_per_cell <= 0:
    raise ValueError("Mean cell size must be positive.")

  cell_size_cv = float(cell_counts.std(ddof=0) / mean_obs_per_cell)
  num_clusters = int(clean["cluster_id"].nunique())
  num_time_periods = int(clean["time_id"].nunique())
  _validate_cluster_count(num_clusters)

  return DesignStats(
    num_clusters=num_clusters,
    num_time_periods=num_time_periods,
    n_cells=len(cell_counts),
    mean_obs_per_cell=mean_obs_per_cell,
    cell_size_cv=cell_size_cv,
    n_observations=len(clean),
  )


def compute_design_stats(
  df: pd.DataFrame,
  cluster_col: str = "cluster_id",
  time_col: str = "time_id",
  outcome_col: str = "outcome",
  min_cell_count: int = 1,
) -> DesignStats:
  """Count clusters, time periods, and cell sizes in the data."""
  clean, _ = prepare_data(
    df,
    cluster_col=cluster_col,
    time_col=time_col,
    outcome_col=outcome_col,
    min_cell_count=min_cell_count,
  )
  return _compute_design_stats_from_clean(clean)


def _estimate_variance_shares_from_clean(clean: pd.DataFrame) -> VarianceShares:
  if len(clean) < 2:
    raise ValueError("At least two observations are required to estimate variance shares.")

  grand_mean = clean["outcome"].mean()
  total_ss = ((clean["outcome"] - grand_mean) ** 2).sum()
  if total_ss <= 0:
    raise ValueError("Total sum of squares is zero; cannot decompose variance.")

  agg_cluster = clean.groupby("cluster_id", observed=True)["outcome"].agg(["mean", "count"])
  ss_cluster = (agg_cluster["count"] * (agg_cluster["mean"] - grand_mean) ** 2).sum()

  agg_time = clean.groupby("time_id", observed=True)["outcome"].agg(["mean", "count"])
  ss_time = (agg_time["count"] * (agg_time["mean"] - grand_mean) ** 2).sum()

  agg_cell = clean.groupby(["cluster_id", "time_id"], observed=True)["outcome"].agg(
    ["mean", "count"]
  )
  ss_cell = (agg_cell["count"] * (agg_cell["mean"] - grand_mean) ** 2).sum()
  ss_int = ss_cell - ss_cluster - ss_time

  warnings: list[str] = []
  if ss_int < -1e-9 * total_ss:
    warnings.append(
      "Interaction share is negative after ANOVA partition; macro share uses max(S_int, 0)."
    )

  s_cl = float(ss_cluster / total_ss)
  s_time = float(ss_time / total_ss)
  s_int = float(max(ss_int, 0.0) / total_ss)
  s_macro = float(ss_cell / total_ss)
  s_res = 1.0 - s_macro

  return VarianceShares(
    cluster_variance_share=s_cl,
    time_period_variance_share=s_time,
    interaction_variance_share=s_int,
    within_cell_variance_share=s_res,
    between_cell_variance_share=s_macro,
    outcome_sd=float(np.sqrt(total_ss / (len(clean) - 1))),
    warnings=tuple(warnings),
  )


def estimate_variance_shares(
  df: pd.DataFrame,
  cluster_col: str = "cluster_id",
  time_col: str = "time_id",
  outcome_col: str = "outcome",
  min_cell_count: int = 1,
) -> VarianceShares:
  """Split outcome variance into between-cell and within-cell parts (layer 1)."""
  clean, _ = prepare_data(
    df,
    cluster_col=cluster_col,
    time_col=time_col,
    outcome_col=outcome_col,
    min_cell_count=min_cell_count,
  )
  return _estimate_variance_shares_from_clean(clean)


def _estimate_design_parameters_from_clean(clean: pd.DataFrame) -> DesignParameters:
  return DesignParameters(
    design=_compute_design_stats_from_clean(clean),
    shares=_estimate_variance_shares_from_clean(clean),
  )


def estimate_design_parameters_from_data(
  df: pd.DataFrame,
  cluster_col: str = "cluster_id",
  time_col: str = "time_id",
  outcome_col: str = "outcome",
  min_cell_count: int = 1,
  max_missing_fraction: float = DEFAULT_MAX_MISSING_FRACTION,
  missing_fraction_warn: float = DEFAULT_MISSING_FRACTION_WARN,
) -> tuple[DesignParameters, DataDiagnostics]:
  """Infer all formula inputs from a pre-period extract (layer 2).

  Returns `DesignParameters` (design counts, cell-size stats, and variance
  shares) plus cleaning diagnostics. Pass the parameters to
  `summarize_estimator_variance_from_design` or `summarize_power_from_design`.
  """
  clean, diagnostics = prepare_data(
    df,
    cluster_col=cluster_col,
    time_col=time_col,
    outcome_col=outcome_col,
    min_cell_count=min_cell_count,
    max_missing_fraction=max_missing_fraction,
    missing_fraction_warn=missing_fraction_warn,
  )
  return _estimate_design_parameters_from_clean(clean), diagnostics


def power_from_data(
  df: pd.DataFrame,
  cluster_col: str = "cluster_id",
  time_col: str = "time_id",
  outcome_col: str = "outcome",
  min_cell_count: int = 1,
  max_missing_fraction: float = DEFAULT_MAX_MISSING_FRACTION,
  missing_fraction_warn: float = DEFAULT_MISSING_FRACTION_WARN,
  treatment_effect: float | None = None,
  alpha: float = 0.05,
  power_level: float = 0.8,
) -> tuple[PowerResult, DesignParameters, DataDiagnostics]:
  """Estimate design parameters from data and return power metrics.

  Convenience wrapper: `estimate_design_parameters_from_data` followed by
  `summarize_power_from_design`.
  """
  parameters, diagnostics = estimate_design_parameters_from_data(
    df,
    cluster_col=cluster_col,
    time_col=time_col,
    outcome_col=outcome_col,
    min_cell_count=min_cell_count,
    max_missing_fraction=max_missing_fraction,
    missing_fraction_warn=missing_fraction_warn,
  )
  result = summarize_power_from_design(
    parameters,
    treatment_effect=treatment_effect,
    alpha=alpha,
    power_level=power_level,
  )
  return result, parameters, diagnostics
