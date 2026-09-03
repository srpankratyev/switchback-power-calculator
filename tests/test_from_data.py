"""Tests for inferring design parameters and variance shares from data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from switchback_power_calculator import (
    estimate_design_parameters_from_data,
    estimate_variance_shares,
    power_from_data,
    prepare_data,
    summarize_estimator_variance_from_design,
    summarize_power_from_design,
)
from switchback_power_calculator.from_data import compute_design_stats

NUM_CLUSTERS_TRUE, NUM_TIME_PERIODS_TRUE, N_TRUE = 25, 40, 30


@pytest.fixture
def balanced_frame() -> pd.DataFrame:
  """Fully populated grid with equal cell sizes, so cell_size_cv is exactly zero."""
  rng = np.random.default_rng(0)
  rows = []
  for j in range(NUM_CLUSTERS_TRUE):
    cluster_mean = rng.normal(0, 40)
    for h in range(NUM_TIME_PERIODS_TRUE):
      cell_shock = rng.normal(0, 20)
      for y in rng.normal(1000 + cluster_mean + cell_shock, 150, size=N_TRUE):
        rows.append({"cluster_id": j, "time_id": h, "outcome": y})
  return pd.DataFrame(rows)


@pytest.fixture
def imbalanced_frame() -> pd.DataFrame:
  rng = np.random.default_rng(42)
  rows = []
  for j in range(10):
    for h in range(12):
      n = max(1, int(rng.poisson(20 + 10 * (j % 3))))
      for y in rng.normal(500 + rng.normal(0, 25), 100, size=n):
        rows.append({"cluster_id": f"c{j}", "time_id": h, "outcome": y})
  return pd.DataFrame(rows)


class TestDesignRecovery:
  def test_recovers_known_design(self, balanced_frame):
    design = compute_design_stats(balanced_frame)
    assert design.num_clusters == NUM_CLUSTERS_TRUE
    assert design.num_time_periods == NUM_TIME_PERIODS_TRUE
    assert design.mean_obs_per_cell == pytest.approx(N_TRUE)
    assert design.cell_size_cv == pytest.approx(0.0, abs=1e-12)
    assert design.n_cells == NUM_CLUSTERS_TRUE * NUM_TIME_PERIODS_TRUE
    assert design.n_observations == NUM_CLUSTERS_TRUE * NUM_TIME_PERIODS_TRUE * N_TRUE

  def test_cell_size_cv_positive_when_cells_uneven(self, imbalanced_frame):
    assert compute_design_stats(imbalanced_frame).cell_size_cv > 0.0

  def test_cell_size_cv_matches_manual_computation(self, imbalanced_frame):
    counts = imbalanced_frame.groupby(["cluster_id", "time_id"]).size()
    expected = counts.std(ddof=0) / counts.mean()
    assert compute_design_stats(imbalanced_frame).cell_size_cv == pytest.approx(expected)


class TestVarianceShares:
  def test_shares_sum_to_one(self, imbalanced_frame):
    shares = estimate_variance_shares(imbalanced_frame)
    assert (
      shares.between_cell_variance_share + shares.within_cell_variance_share
      == pytest.approx(1.0, rel=1e-12)
    )

  def test_components_are_non_negative(self, imbalanced_frame):
    shares = estimate_variance_shares(imbalanced_frame)
    for value in (
      shares.cluster_variance_share,
      shares.time_period_variance_share,
      shares.interaction_variance_share,
      shares.within_cell_variance_share,
    ):
      assert value >= 0.0

  def test_outcome_sd_matches_sample_std(self, imbalanced_frame):
    shares = estimate_variance_shares(imbalanced_frame)
    assert shares.outcome_sd == pytest.approx(imbalanced_frame["outcome"].std(ddof=1))

  def test_pure_cluster_variation_loads_on_cluster_share(self):
    rows = [
      {"cluster_id": j, "time_id": h, "outcome": 100.0 * j}
      for j in range(5)
      for h in range(4)
    ]
    shares = estimate_variance_shares(pd.DataFrame(rows))
    assert shares.cluster_variance_share == pytest.approx(1.0, abs=1e-9)
    assert shares.within_cell_variance_share == pytest.approx(0.0, abs=1e-9)

  def test_pure_within_cell_noise_loads_on_residual(self):
    rng = np.random.default_rng(7)
    rows = [
      {"cluster_id": j, "time_id": h, "outcome": rng.normal(0, 1)}
      for j in range(20)
      for h in range(20)
      for _ in range(30)
    ]
    shares = estimate_variance_shares(pd.DataFrame(rows))
    assert shares.within_cell_variance_share > 0.9

  def test_between_share_exact_under_negative_interaction(self):
    """Regression: with unbalanced cells, SS_interaction can go negative,
    which used to inflate between_cell_variance_share above the true
    SS_cell / SS_total after the max(0, .) clamp and renormalization."""
    rng = np.random.default_rng(29)
    rows = []
    for j in range(6):
      cluster_effect = rng.normal(0, 40)
      for h in range(5):
        n = max(2, int(3 + 90 * (j * h) / (5 * 4)))
        for y in rng.normal(100 + cluster_effect + 20.0 * h, 20.0, size=n):
          rows.append({"cluster_id": j, "time_id": h, "outcome": y})
    df = pd.DataFrame(rows)

    shares = estimate_variance_shares(df)
    assert shares.warnings
    assert "negative" in shares.warnings[0].lower()

    grand_mean = df["outcome"].mean()
    total_ss = ((df["outcome"] - grand_mean) ** 2).sum()
    agg_cell = df.groupby(["cluster_id", "time_id"])["outcome"].agg(["mean", "count"])
    ss_cell = (agg_cell["count"] * (agg_cell["mean"] - grand_mean) ** 2).sum()
    true_between = ss_cell / total_ss

    assert shares.between_cell_variance_share == pytest.approx(true_between, rel=1e-12)
    assert shares.within_cell_variance_share == pytest.approx(1.0 - true_between, rel=1e-12)


class TestColumnCoercion:
  def test_mixed_int_and_string_cluster_ids_merge(self):
    df = pd.DataFrame(
      {
        "c": [1, 1, "1", "1", 2, 2],
        "t": [1, 1, 1, 1, 2, 2],
        "y": [10.0, 11.0, 12.0, 13.0, 20.0, 21.0],
      }
    )
    design = compute_design_stats(
      prepare_data(df, cluster_col="c", time_col="t", outcome_col="y")[0]
    )
    assert design.num_clusters == 2
    assert design.num_time_periods == 2

  def test_string_outcome_values_are_coerced(self):
    df = pd.DataFrame(
      {"c": [1, 1, 2, 2], "t": [1, 1, 2, 2], "y": ["10", "11", "20", "21"]}
    )
    shares = estimate_variance_shares(
      prepare_data(df, cluster_col="c", time_col="t", outcome_col="y")[0]
    )
    assert shares.outcome_sd > 0

  def test_unparseable_outcome_raises(self):
    df = pd.DataFrame(
      {"c": [1, 1, 2, 2], "t": [1, 1, 2, 2], "y": ["10", "bad", "20", "21"]}
    )
    with pytest.raises(ValueError, match="outcome column must be numeric"):
      prepare_data(df, cluster_col="c", time_col="t", outcome_col="y")


class TestDataCleaning:
  def test_missing_outcomes_dropped_and_counted(self, balanced_frame):
    dirty = balanced_frame.copy()
    dirty.loc[::100, "outcome"] = np.nan
    n_missing = int(dirty["outcome"].isna().sum())

    clean, diagnostics = prepare_data(dirty)
    assert diagnostics.rows_input == len(dirty)
    assert diagnostics.rows_dropped_missing == n_missing
    assert diagnostics.rows_used == len(dirty) - n_missing
    assert not clean["outcome"].isna().any()

  def test_sparse_cells_filtered(self, balanced_frame):
    padded = pd.concat([
      balanced_frame,
      pd.DataFrame({"cluster_id": [999] * 3, "time_id": [0, 1, 2], "outcome": [1.0, 2.0, 3.0]}),
    ])
    _, diagnostics = prepare_data(padded, min_cell_count=10)
    assert diagnostics.cells_before_filter == NUM_CLUSTERS_TRUE * NUM_TIME_PERIODS_TRUE + 3
    assert diagnostics.cells_after_filter == NUM_CLUSTERS_TRUE * NUM_TIME_PERIODS_TRUE

  def test_custom_column_names_accepted(self, balanced_frame):
    renamed = balanced_frame.rename(
      columns={"cluster_id": "zone", "time_id": "hour", "outcome": "trips"}
    )
    clean, _ = prepare_data(renamed, cluster_col="zone", time_col="hour", outcome_col="trips")
    assert list(clean.columns) == ["cluster_id", "time_id", "outcome"]

  def test_min_cell_count_one_keeps_everything(self, imbalanced_frame):
    _, diagnostics = prepare_data(imbalanced_frame, min_cell_count=1)
    assert diagnostics.cells_before_filter == diagnostics.cells_after_filter

  def test_compute_design_stats_accepts_custom_column_names(self, balanced_frame):
    """Regression: compute_design_stats used to ignore cluster_col/time_col/
    outcome_col and always look for the default column names."""
    renamed = balanced_frame.rename(
      columns={"cluster_id": "zone", "time_id": "hour", "outcome": "trips"}
    )
    design = compute_design_stats(
      renamed, cluster_col="zone", time_col="hour", outcome_col="trips"
    )
    assert design.num_clusters == NUM_CLUSTERS_TRUE
    assert design.num_time_periods == NUM_TIME_PERIODS_TRUE

  def test_estimate_variance_shares_accepts_custom_column_names(self, imbalanced_frame):
    renamed = imbalanced_frame.rename(
      columns={"cluster_id": "zone", "time_id": "hour", "outcome": "trips"}
    )
    default_shares = estimate_variance_shares(imbalanced_frame)
    renamed_shares = estimate_variance_shares(
      renamed, cluster_col="zone", time_col="hour", outcome_col="trips"
    )
    assert renamed_shares == default_shares


class TestEstimateDesignParameters:
  def test_prepare_data_called_once(self, imbalanced_frame, monkeypatch):
    calls = 0
    original = prepare_data

    def counted_prepare_data(*args, **kwargs):
      nonlocal calls
      calls += 1
      return original(*args, **kwargs)

    monkeypatch.setattr("switchback_power_calculator.from_data.prepare_data", counted_prepare_data)
    estimate_design_parameters_from_data(imbalanced_frame)
    assert calls == 1

  def test_bundles_design_and_shares(self, imbalanced_frame):
    parameters, diagnostics = estimate_design_parameters_from_data(imbalanced_frame)
    assert parameters.num_clusters == parameters.design.num_clusters
    assert parameters.between_cell_variance_share == (
      parameters.shares.between_cell_variance_share
    )
    assert diagnostics.rows_used > 0

  def test_variance_layer_matches_manual(self, imbalanced_frame):
    parameters, _ = estimate_design_parameters_from_data(imbalanced_frame)
    variance = summarize_estimator_variance_from_design(parameters)
    assert variance.variance > 0
    assert variance.standard_error == pytest.approx(variance.variance**0.5)

  def test_power_layer_matches_manual(self, imbalanced_frame):
    parameters, _ = estimate_design_parameters_from_data(imbalanced_frame)
    result = summarize_power_from_design(parameters, treatment_effect=10.0)
    assert result.variance == pytest.approx(
      summarize_estimator_variance_from_design(parameters).variance
    )
    assert result.mde is not None
    assert result.power is not None


class TestPowerFromData:
  def test_matches_manual_pipeline(self, imbalanced_frame):
    result, parameters, _ = power_from_data(imbalanced_frame, treatment_effect=10.0)
    manual = summarize_power_from_design(parameters, treatment_effect=10.0)
    assert result.variance == pytest.approx(manual.variance, rel=1e-12)
    assert result.mde == pytest.approx(manual.mde, rel=1e-12)
    assert result.approximation_validity == manual.approximation_validity

  def test_reports_approximation_validity(self, imbalanced_frame):
    result, _, _ = power_from_data(imbalanced_frame)
    assert result.approximation_validity in {
      "approximation accurate",
      "approximation inaccurate",
    }

  def test_sparse_grid_flagged_as_inaccurate(self):
    rng = np.random.default_rng(1)
    rows = [
      {"cluster_id": j, "time_id": h, "outcome": rng.normal(100, 10)}
      for j in range(3)
      for h in range(2)
      for _ in range(25)
    ]
    result, _, _ = power_from_data(pd.DataFrame(rows))
    assert result.approximation_validity == "approximation inaccurate"


class TestErrorPaths:
  def test_all_outcomes_missing(self):
    df = pd.DataFrame({"cluster_id": [1, 2], "time_id": [1, 2], "outcome": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="No rows remain"):
      prepare_data(df)

  def test_zero_variance_outcome(self):
    df = pd.DataFrame(
      {"cluster_id": [1, 1, 2, 2], "time_id": [1, 2, 1, 2], "outcome": [5.0] * 4}
    )
    with pytest.raises(ValueError, match="Total sum of squares is zero"):
      estimate_variance_shares(df)

  def test_filter_removes_every_cell(self, balanced_frame):
    with pytest.raises(ValueError, match="No cells remain"):
      prepare_data(balanced_frame, min_cell_count=10**6)

  def test_missing_required_column(self):
    with pytest.raises(KeyError, match="Missing required columns"):
      prepare_data(pd.DataFrame({"cluster_id": [1], "outcome": [1.0]}))

  @pytest.mark.parametrize("min_cell_count", [0, -5])
  def test_invalid_min_cell_count(self, balanced_frame, min_cell_count):
    with pytest.raises(ValueError, match="min_cell_count must be at least 1"):
      prepare_data(balanced_frame, min_cell_count=min_cell_count)

  def test_warns_on_moderate_missing_fraction(self, balanced_frame):
    dirty = balanced_frame.copy()
    n_drop = max(1, len(dirty) // 5)
    dirty.loc[: n_drop - 1, "outcome"] = np.nan

    _, diagnostics = prepare_data(dirty, missing_fraction_warn=0.1)
    assert diagnostics.rows_dropped_missing == n_drop
    assert len(diagnostics.warnings) == 1
    assert "missing values" in diagnostics.warnings[0].lower()

  def test_rejects_excessive_missing_fraction(self, balanced_frame):
    dirty = balanced_frame.copy()
    dirty.loc[: len(dirty) // 2, "outcome"] = np.nan

    with pytest.raises(ValueError, match="exceeding max_missing_fraction"):
      prepare_data(dirty, max_missing_fraction=0.5)

  def test_single_cluster_rejected(self):
    df = pd.DataFrame(
      {"cluster_id": [1] * 5, "time_id": [1] * 5, "outcome": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    with pytest.raises(ValueError, match="At least 2 clusters are required"):
      estimate_design_parameters_from_data(df)

  def test_empty_frame(self):
    df = pd.DataFrame({"cluster_id": [], "time_id": [], "outcome": []})
    with pytest.raises(ValueError, match="Input frame is empty"):
      prepare_data(df)


def test_single_cluster_rejected_by_compute_design_stats():
  with pytest.raises(ValueError, match="At least 2 clusters are required"):
    compute_design_stats(
      pd.DataFrame(
        {"cluster_id": [1] * 5, "time_id": [1] * 5, "outcome": [1.0, 2.0, 3.0, 4.0, 5.0]}
      )
    )
