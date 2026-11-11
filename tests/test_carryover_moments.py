"""Tests for pre-period lag moments, the planned covariance and the synthetic panel."""

from __future__ import annotations

import dataclasses
import functools
import math

import numpy as np
import pandas as pd
import pytest

from switchback_power_calculator import core
from switchback_power_calculator.carryover.examples import synthetic_panel
from switchback_power_calculator.carryover.moments import (
  ClusterLagMoments,
  PlannedCovariance,
  ProfileFit,
  aggregate_observations,
  lag_moments_from_aggregates,
  lag_moments_from_parameters,
  leave_one_week_out_expected,
  profile_fingerprint,
  window_totals,
)

START = pd.Timestamp("2030-01-07")
HOUR = pd.Timedelta(hours=1)
MOMENT_FIELDS = (
  "presence",
  "lag_presence",
  "mean_count",
  "mean_count_lag",
  "count_sq_ratio",
  "count_sq_ratio_lag",
  "m00",
  "m11",
  "m01",
  "position_share",
)
PARAMETERS = dict(
  mean_obs_per_cell=37.5,
  cell_size_cv=0.8,
  between_cell_variance_share=0.07,
  outcome_sd=12.3,
)


def frame_from_counts(counts: dict, seed: int = 0, start: pd.Timestamp = START) -> pd.DataFrame:
  """Hand-built hourly rows: `counts[cluster][h]` is the hour's count, or None for no row."""
  rng = np.random.default_rng(seed)
  rows = []
  for cluster, values in counts.items():
    for h, count in enumerate(values):
      if count is None:
        continue
      rows.append({
        "cluster_id": cluster,
        "period_start": start + h * HOUR,
        "numerator": count * (10.0 + rng.normal()),
        "denominator": count,
      })
  return pd.DataFrame(rows)


def assert_same_moments(left: ClusterLagMoments, right: ClusterLagMoments, rel: float = 1e-9) -> None:
  assert left.cluster_ids == right.cluster_ids
  assert left.nominal_windows == right.nominal_windows
  for name in MOMENT_FIELDS:
    np.testing.assert_allclose(getattr(left, name), getattr(right, name), rtol=rel, atol=1e-12, err_msg=name)


def with_week_labels(df: pd.DataFrame) -> pd.DataFrame:
  """Add hour-of-week `slot` and `week` labels counted from the first hour."""
  hours = ((df["period_start"] - df["period_start"].min()) / HOUR).round().astype(int)
  return df.assign(slot=hours % 168, week=hours // 168)


@pytest.fixture(scope="module")
def small_panel() -> pd.DataFrame:
  return synthetic_panel(12, 1, seed=1)


@pytest.fixture(scope="module")
def default_panel() -> pd.DataFrame:
  return synthetic_panel()


# ---------------------------------------------------------------------------
# Re-randomisation: the actual estimator under fresh coins on a fixed panel
# ---------------------------------------------------------------------------


def complete_windows(df: pd.DataFrame, window_minutes: int, origin: pd.Timestamp) -> pd.DataFrame:
  """Window residuals and counts, built here with a groupby rather than by the package."""
  k = window_minutes // 60
  hour = ((df["period_start"] - origin) / HOUR).round().astype(int)
  totals = (
    df.assign(window=hour // k, residual=df["numerator"] - df["expected"])
    .groupby(["cluster_id", "window"])
    .agg(residual=("residual", "sum"), count=("denominator", "sum"), rows=("denominator", "size"))
    .reset_index()
  )
  return totals[totals["rows"] == k]


@dataclasses.dataclass(frozen=True)
class Rerandomised:
  cluster_ids: tuple
  weights: np.ndarray  # cluster observations / all observations
  l0: np.ndarray  # (draws, clusters); nan where an arm had no observations
  l1: np.ndarray

  def se(self, lag1_weight: float) -> tuple[float, float]:
    """SE of sum_i rho_i (l0_i + b l1_i), estimated two ways.

    First the plain SD of the aggregate across draws. Second sum_i rho_i^2
    times each cluster's own variance, which is the same quantity because
    clusters flip independent coins, with far less Monte Carlo noise.
    """
    contrast = self.l0 + lag1_weight * self.l1
    every = np.isfinite(contrast).all(axis=1)
    aggregate = float((contrast[every] @ self.weights).std(ddof=1))
    by_cluster = float(np.sqrt(np.sum(self.weights**2 * np.nanvar(contrast, axis=0, ddof=1))))
    return aggregate, by_cluster

  def corr(self) -> float:
    every = np.isfinite(self.l0).all(axis=1) & np.isfinite(self.l1).all(axis=1)
    return float(np.corrcoef(self.l0[every] @ self.weights, self.l1[every] @ self.weights)[0, 1])


@functools.lru_cache(maxsize=None)
def panel_for(seed: int, weeks: int, gap_share: float) -> pd.DataFrame:
  return synthetic_panel(120, weeks, seed=seed, gap_share=gap_share)


def rerandomised(
  seed: int, weeks: int, gap_share: float, window_minutes: int, n_draws: int = 8000
) -> Rerandomised:
  """Per-cluster l0 and l1 under `n_draws` fresh sets of fair coins.

  l0 is the ratio-of-sums difference in means of residuals between treated
  and control windows. l1 is the same with each window grouped by the
  previous window's coin, over windows whose previous window is present.
  """
  df = panel_for(seed, weeks, gap_share)
  totals = complete_windows(df, window_minutes, df["period_start"].min())
  ids, cluster = np.unique(totals["cluster_id"].to_numpy(dtype=object), return_inverse=True)
  window = totals["window"].to_numpy() - totals["window"].min()
  n_windows = int(window.max()) + 1
  present = np.zeros((len(ids), n_windows), dtype=bool)
  present[cluster, window] = True
  residual = np.zeros(present.shape)
  count = np.zeros(present.shape)
  residual[cluster, window] = totals["residual"]
  count[cluster, window] = totals["count"]
  lag = present[:, 1:] & present[:, :-1]

  rng = np.random.default_rng(1000 + seed)
  l0 = np.empty((n_draws, len(ids)))
  l1 = np.empty((n_draws, len(ids)))
  for i in range(len(ids)):
    bits = np.unpackbits(np.frombuffer(rng.bytes((n_draws * n_windows + 7) // 8), dtype=np.uint8))
    coins = bits[: n_draws * n_windows].reshape(n_draws, n_windows).astype(float)
    # Row t: the window's own residual and count, then the next window's
    # (zero unless both are present), so one product sums the treated arms.
    sums = np.zeros((n_windows, 4))
    sums[:, 0], sums[:, 1] = residual[i], count[i]
    sums[:-1, 2], sums[:-1, 3] = residual[i, 1:] * lag[i], count[i, 1:] * lag[i]
    treated = coins @ sums
    control = sums.sum(axis=0)[None, :] - treated
    with np.errstate(divide="ignore", invalid="ignore"):
      l0[:, i] = treated[:, 0] / treated[:, 1] - control[:, 0] / control[:, 1]
      l1[:, i] = treated[:, 2] / treated[:, 3] - control[:, 2] / control[:, 3]
  unusable = ~(np.isfinite(l0) & np.isfinite(l1))
  l0[unusable] = np.nan
  l1[unusable] = np.nan
  return Rerandomised(tuple(ids), count.sum(axis=1) / count.sum(), l0, l1)


def moments_for(seed: int, weeks: int, gap_share: float, window_minutes: int) -> ClusterLagMoments:
  df = panel_for(seed, weeks, gap_share)
  return lag_moments_from_aggregates(
    df, expected_col="expected", window_minutes=window_minutes, origin=df["period_start"].min()
  )


def without_finite_sample_factor(moments: ClusterLagMoments) -> ClusterLagMoments:
  zeros = np.zeros(moments.num_clusters)
  return dataclasses.replace(moments, count_sq_ratio=zeros, count_sq_ratio_lag=zeros)


class TestPlannedCovarianceAgainstRerandomisation:
  @pytest.mark.parametrize(
    "seed, weeks, gap_share",
    [(11, 2, 0.02), (3, 1, 0.01), (4, 1, 0.01), (5, 1, 0.01)],
  )
  def test_planned_se_matches_rerandomisation(self, seed, weeks, gap_share):
    for window_minutes in (60, 120, 240, 480):
      moments = moments_for(seed, weeks, gap_share, window_minutes)
      assert moments.pre_period_weeks == pytest.approx(weeks)
      plan = moments.plan(moments.pre_period_weeks)
      actual = rerandomised(seed, weeks, gap_share, window_minutes)
      label = f"seed {seed}, {weeks} week(s), {window_minutes} min"

      assert actual.cluster_ids == plan.cluster_ids
      np.testing.assert_allclose(actual.weights, plan.weights, rtol=1e-12)
      assert np.isnan(actual.l0).mean() < 1e-3, label
      for lag1_weight in (0.0, 1.0, 3.0):
        aggregate, by_cluster = actual.se(lag1_weight)
        planned = plan.contrast_se(lag1_weight)
        assert 0.96 < planned / aggregate < 1.04, (label, lag1_weight)
        assert 0.98 < planned / by_cluster < 1.02, (label, lag1_weight)
      assert plan.corr_l0_l1 == pytest.approx(actual.corr(), abs=0.03), label

  @pytest.mark.parametrize("seed", [3, 4, 5])
  def test_finite_sample_factor_matters_at_8h(self, seed):
    moments = moments_for(seed, 1, 0.01, 480)
    plan = moments.plan(1)
    crude = without_finite_sample_factor(moments).plan(1)
    actual = rerandomised(seed, 1, 0.01, 480)
    for lag1_weight in (0.0, 1.0):
      _, by_cluster = actual.se(lag1_weight)
      assert crude.contrast_se(lag1_weight) / by_cluster < 0.96
      assert 0.98 < plan.contrast_se(lag1_weight) / by_cluster < 1.02
    # The factor exceeds 1 by a term in 1/weeks, so one week is where it shows.
    assert plan.se_l0 / crude.se_l0 > 1.03
    four_weeks = moments.plan(4).se_l0 / without_finite_sample_factor(moments).plan(4).se_l0
    assert 1.0 < four_weeks < plan.se_l0 / crude.se_l0

  def test_reference_figures(self, default_panel):
    """One-week SEs on the default panel, next to the design document's figures."""
    origin = default_panel["period_start"].min()
    measured = {}
    for window_minutes in (60, 120, 240):
      plan = lag_moments_from_aggregates(
        default_panel, expected_col="expected", window_minutes=window_minutes, origin=origin
      ).plan(1)
      measured[window_minutes] = (plan.se_l0, plan.se_l0_plus_l1, plan.corr_l0_l1)

    # This implementation, to guard against drift.
    pinned = {
      60: (0.030572, 0.051471, 0.4065),
      120: (0.036767, 0.061058, 0.3577),
      240: (0.043969, 0.070158, 0.2359),
    }
    # The design document's printed figures, from the generator's prototype.
    printed = {
      60: (0.0306, 0.0514, 0.41),
      120: (0.0368, 0.0609, 0.36),
      240: (0.0440, 0.0699, 0.24),
    }
    for window_minutes, (se_l0, se_sum, corr) in measured.items():
      assert se_l0 == pytest.approx(pinned[window_minutes][0], rel=2e-4)
      assert se_sum == pytest.approx(pinned[window_minutes][1], rel=2e-4)
      assert corr == pytest.approx(pinned[window_minutes][2], abs=2e-4)
      assert se_l0 == pytest.approx(printed[window_minutes][0], abs=5e-5)
      assert se_sum == pytest.approx(printed[window_minutes][1], rel=0.005)
      assert corr == pytest.approx(printed[window_minutes][2], abs=0.005)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------


class TestMoments:
  def test_moments_match_hand_computation(self):
    counts = {
      "a": [3, 1, 4, 1, 5, 9, 2, 6, None, 3, 5, 8, 9, 7, 9, 3],
      "b": [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5],
    }
    df = frame_from_counts(counts, seed=4)
    df["expected"] = 9.5 * df["denominator"]
    moments = lag_moments_from_aggregates(df, window_minutes=120, origin=START, expected_col="expected")

    assert moments.cluster_ids == ("a", "b")
    assert moments.nominal_windows == 8
    for i, cluster in enumerate(("a", "b")):
      rows = df[df["cluster_id"] == cluster].set_index("period_start")
      windows = {}
      for t in range(8):
        hours = [START + (2 * t + h) * HOUR for h in range(2)]
        if all(hour in rows.index for hour in hours):
          part = rows.loc[hours]
          windows[t] = (
            float((part["numerator"] - part["expected"]).sum()),
            float(part["denominator"].sum()),
            part["denominator"].to_numpy(dtype=float),
          )
      centre = sum(r for r, _, _ in windows.values()) / sum(c for _, c, _ in windows.values())
      score = {t: r - centre * c for t, (r, c, _) in windows.items()}
      lagged = [t for t in windows if t - 1 in windows]
      mean_count = np.mean([c for _, c, _ in windows.values()])
      mean_count_lag = np.mean([windows[t][1] for t in lagged])

      assert moments.presence[i] == pytest.approx(len(windows) / 8)
      assert moments.lag_presence[i] == pytest.approx(len(lagged) / 7)
      assert moments.mean_count[i] == pytest.approx(mean_count)
      assert moments.mean_count_lag[i] == pytest.approx(mean_count_lag)
      assert moments.count_sq_ratio[i] == pytest.approx(
        np.mean([c**2 for _, c, _ in windows.values()]) / mean_count**2
      )
      assert moments.count_sq_ratio_lag[i] == pytest.approx(
        np.mean([windows[t][1] ** 2 for t in lagged]) / mean_count_lag**2
      )
      assert moments.m00[i] == pytest.approx(np.mean([s**2 for s in score.values()]))
      assert moments.m11[i] == pytest.approx(np.mean([score[t] ** 2 for t in lagged]))
      assert moments.m01[i] == pytest.approx(np.mean([score[t - 1] * score[t] for t in lagged]))
      by_position = sum(c for _, _, c in windows.values())
      np.testing.assert_allclose(moments.position_share[i], by_position / by_position.sum())
    # Cluster a misses hour 8, so window 4 is absent and window 5 has no lag pair.
    assert moments.presence[0] == pytest.approx(7 / 8)
    assert moments.lag_presence[0] == pytest.approx(5 / 7)

  def test_scores_centre_within_cluster(self, small_panel):
    origin = small_panel["period_start"].min()
    base = lag_moments_from_aggregates(
      small_panel, expected_col="expected", window_minutes=120, origin=origin
    )
    level = small_panel["cluster_id"].map(
      {cluster: 7.5 * i - 30.0 for i, cluster in enumerate(small_panel["cluster_id"].unique())}
    )
    shifted = small_panel.assign(numerator=small_panel["numerator"] + level * small_panel["denominator"])
    moved = lag_moments_from_aggregates(
      shifted, expected_col="expected", window_minutes=120, origin=origin
    )
    assert_same_moments(base, moved, rel=1e-7)

  def test_partial_window_is_absent(self):
    counts = {"a": [2, 3] * 20, "b": [4, 1] * 20}
    counts["a"][13] = None  # hour 13 is in the fourth 4 h window (index 3)
    df = frame_from_counts(counts)
    moments = lag_moments_from_aggregates(df, window_minutes=240, origin=START)

    assert moments.nominal_windows == 10
    # a: 9 of 10 windows present; windows 1, 2 and 5 to 9 have a present predecessor.
    np.testing.assert_allclose(moments.presence, [9 / 10, 1.0])
    np.testing.assert_allclose(moments.lag_presence, [7 / 9, 1.0])
    totals = window_totals(df, window_minutes=240, origin=START)
    assert totals[totals["cluster_id"] == "a"]["window"].tolist() == [0, 1, 2, 4, 5, 6, 7, 8, 9]
    assert totals[totals["cluster_id"] == "b"]["window"].tolist() == list(range(10))

  def test_zero_count_present_missing_row_gap(self):
    quiet = {"a": [3, 2, 0, 4, 1, 5], "b": [2, 2, 3, 1, 4, 2]}
    kept = lag_moments_from_aggregates(frame_from_counts(quiet))
    np.testing.assert_allclose(kept.presence, [1.0, 1.0])
    np.testing.assert_allclose(kept.lag_presence, [1.0, 1.0])
    assert kept.mean_count[0] == pytest.approx(15 / 6)

    gap = {"a": [3, 2, None, 4, 1, 5], "b": [2, 2, 3, 1, 4, 2]}
    broken = lag_moments_from_aggregates(frame_from_counts(gap))
    np.testing.assert_allclose(broken.presence, [5 / 6, 1.0])
    # Hours 1, 4 and 5 follow a present hour; hour 3 follows the gap.
    np.testing.assert_allclose(broken.lag_presence, [3 / 5, 1.0])
    assert broken.mean_count[0] == pytest.approx(15 / 5)

  def test_position_share(self):
    counts = {"a": [1, 3, 2, 6, 1, 5, 2, 4, 3, 1], "b": [2, 2, 5, 1, 3, 3, 0, 4, 1, 1]}
    moments = lag_moments_from_aggregates(frame_from_counts(counts), window_minutes=120, origin=START)
    np.testing.assert_allclose(moments.position_share, [[9 / 28, 19 / 28], [11 / 22, 11 / 22]])
    np.testing.assert_allclose(moments.position_share.sum(axis=1), 1.0)

    single = lag_moments_from_aggregates(frame_from_counts(counts))
    np.testing.assert_allclose(single.position_share, [[1.0], [1.0]])

  def test_origin_shift_changes_position_share(self):
    counts = {"a": [1, 3, 2, 6, 1, 5, 2, 4, 3, 1], "b": [2, 2, 5, 1, 3, 3, 0, 4, 1, 1]}
    df = frame_from_counts(counts)
    # From hour 1 the windows are (1, 2), (3, 4), (5, 6), (7, 8); hours 0 and 9 are partial.
    shifted = lag_moments_from_aggregates(df, window_minutes=120, origin=START + HOUR)
    assert shifted.nominal_windows == 4
    np.testing.assert_allclose(shifted.position_share, [[18 / 26, 8 / 26], [10 / 19, 9 / 19]])
    assert shifted.origin == START + HOUR

  def test_origin_a_whole_window_away_changes_nothing(self, small_panel):
    first = small_panel["period_start"].min()
    base = lag_moments_from_aggregates(small_panel, expected_col="expected", window_minutes=240, origin=first)
    for origin in (first + pd.Timedelta(days=30), first - pd.Timedelta(hours=8)):
      moved = lag_moments_from_aggregates(
        small_panel, expected_col="expected", window_minutes=240, origin=origin
      )
      assert_same_moments(base, moved, rel=1e-12)

  def test_origin_required_multiperiod(self, small_panel):
    with pytest.raises(ValueError, match="origin is required for multi-period windows"):
      lag_moments_from_aggregates(small_panel, window_minutes=120)
    with pytest.raises(ValueError, match="origin is required for multi-period windows"):
      window_totals(small_panel, window_minutes=120)
    moments = lag_moments_from_aggregates(small_panel, window_minutes=60)
    assert moments.origin is None
    assert len(window_totals(small_panel, window_minutes=60)) == len(small_panel)

  def test_cluster_ids_normalised(self):
    counts = {1: [3, 2, 4, 4, 1, 5], 2: [2, 2, 3, 1, 4, 2], 10: [1, 0, 2, 2, 3, 1]}
    as_int = frame_from_counts(counts)
    as_str = as_int.assign(cluster_id=as_int["cluster_id"].astype(str))
    left = lag_moments_from_aggregates(as_int)
    right = lag_moments_from_aggregates(as_str)
    assert left.cluster_ids == ("1", "10", "2")
    assert_same_moments(left, right, rel=0)

    picked = left.subset([2, "1"])
    assert picked.cluster_ids == ("2", "1")
    np.testing.assert_array_equal(picked.m00, left.m00[[2, 0]])
    np.testing.assert_array_equal(picked.position_share, left.position_share[[2, 0]])
    assert picked.nominal_windows == left.nominal_windows
    assert picked.num_clusters == 2

  def test_subset_unknown_id(self, small_panel):
    moments = lag_moments_from_aggregates(small_panel, expected_col="expected")
    with pytest.raises(KeyError, match="c999"):
      moments.subset(["c000", "c999"])
    with pytest.raises(ValueError, match="distinct"):
      moments.subset(["c000", "c000"])

  def test_times_on_any_clock_give_the_same_moments(self, small_panel):
    first = small_panel["period_start"].min()
    kwargs = dict(expected_col="expected", window_minutes=180)
    aware = lag_moments_from_aggregates(small_panel, origin=first, **kwargs)

    naive = small_panel.assign(period_start=small_panel["period_start"].dt.tz_localize(None))
    assert_same_moments(
      aware, lag_moments_from_aggregates(naive, origin=first.tz_localize(None), **kwargs), rel=0
    )
    hours = small_panel.assign(period_start=(small_panel["period_start"] - first) / HOUR + 100.0)
    numeric = lag_moments_from_aggregates(hours, origin=100.0, time_unit_minutes=60, **kwargs)
    assert_same_moments(aware, numeric, rel=0)
    assert numeric.origin == 100.0
    assert aware.origin == first

  def test_small_clusters_are_dropped_and_counted(self):
    counts = {
      "a": [3, 2, 4, 4, 1, 5],
      "b": [2, 2, 3, 1, 4, 2],
      "empty": [0, 0, 0, 0, 0, 0],
      "short": [2, 3, None, 1, None, 4],
      "late": [5, 0, 0, None, 0, 0],
    }
    moments = lag_moments_from_aggregates(frame_from_counts(counts))
    assert moments.cluster_ids == ("a", "b")
    dropped = [note for note in moments.warnings if "dropped" in note]
    assert len(dropped) == 3
    assert any("no observations in a complete window" in note for note in dropped)
    assert any("fewer than 2 lag-present windows" in note for note in dropped)
    assert any("no observations in a lag-present window" in note for note in dropped)
    assert all(note.startswith("1 cluster(s) dropped") for note in dropped)

    with pytest.raises(ValueError, match="Fewer than 2 clusters"):
      lag_moments_from_aggregates(frame_from_counts({"a": counts["a"], "short": counts["short"]}))

  def test_fields_are_read_only(self, small_panel):
    moments = lag_moments_from_aggregates(small_panel, expected_col="expected")
    with pytest.raises(ValueError):
      moments.m00[0] = 1.0
    with pytest.raises(ValueError):
      moments.plan(1).v00[0] = 1.0

  def test_variance_factor_scales_the_three_moments(self, small_panel):
    base = lag_moments_from_aggregates(small_panel, expected_col="expected")
    scaled = lag_moments_from_aggregates(small_panel, expected_col="expected", variance_factor=0.75)
    for name in ("m00", "m11", "m01"):
      np.testing.assert_allclose(getattr(scaled, name), 0.75 * getattr(base, name))
    np.testing.assert_array_equal(scaled.mean_count, base.mean_count)
    assert base.residual == "expected:expected"
    assert scaled.residual == "expected:expected, factor 0.75"
    assert scaled.variance_factor == 0.75
    for bad in (0.0, 2.5, float("nan")):
      with pytest.raises(ValueError, match="variance_factor"):
        lag_moments_from_aggregates(small_panel, expected_col="expected", variance_factor=bad)
    with pytest.raises(ValueError, match="need expected_col"):
      lag_moments_from_aggregates(small_panel, variance_factor=0.75)


class TestInputContract:
  @pytest.fixture
  def frame(self) -> pd.DataFrame:
    return frame_from_counts({"a": [3, 2, 4, 4, 1, 5, 2, 2], "b": [2, 2, 3, 1, 4, 2, 6, 1]})

  def test_input_contract(self, frame):
    duplicated = pd.concat([frame, frame.iloc[[3]]], ignore_index=True)
    with pytest.raises(ValueError, match=r"1 \(cluster, base period\) pair\(s\) have more than one row"):
      lag_moments_from_aggregates(duplicated)

    off_grid = frame.copy()
    off_grid.loc[2, "period_start"] += pd.Timedelta(minutes=20)
    with pytest.raises(ValueError, match="1 row.*grid"):
      lag_moments_from_aggregates(off_grid)

    with pytest.raises(ValueError, match="divide a day"):
      lag_moments_from_aggregates(frame, window_minutes=420, origin=START)
    with pytest.raises(ValueError, match="whole multiple"):
      lag_moments_from_aggregates(frame, window_minutes=90, origin=START)

    numeric = frame.assign(period_start=np.tile(np.arange(8.0), 2))
    with pytest.raises(ValueError, match="time_unit_minutes"):
      lag_moments_from_aggregates(numeric)
    assert lag_moments_from_aggregates(numeric, time_unit_minutes=60).num_clusters == 2

    orphan = frame.copy()
    orphan.loc[1, "denominator"] = 0
    with pytest.raises(ValueError, match="zero denominator and a non-zero numerator"):
      lag_moments_from_aggregates(orphan)

  def test_bad_values_raise(self, frame):
    with pytest.raises(KeyError, match="expected"):
      lag_moments_from_aggregates(frame, expected_col="expected")
    with pytest.raises(KeyError, match="numerator"):
      lag_moments_from_aggregates(frame.drop(columns="numerator"))
    with pytest.raises(ValueError, match="empty"):
      lag_moments_from_aggregates(frame.iloc[:0])

    missing = frame.copy()
    missing.loc[0, "numerator"] = np.nan
    with pytest.raises(ValueError, match="numerator must be numeric and finite"):
      lag_moments_from_aggregates(missing)
    text = frame.assign(denominator=["many"] * len(frame))
    with pytest.raises(ValueError, match="denominator must be numeric and finite"):
      lag_moments_from_aggregates(text)
    negative = frame.copy()
    negative.loc[0, "denominator"] = -1
    with pytest.raises(ValueError, match="non-negative"):
      lag_moments_from_aggregates(negative)
    with pytest.raises(ValueError, match="sums to zero"):
      lag_moments_from_aggregates(frame.assign(numerator=0.0, denominator=0))
    no_cluster = frame.astype({"cluster_id": object})
    no_cluster.loc[0, "cluster_id"] = None
    with pytest.raises(ValueError, match="cluster_id has 1 missing"):
      lag_moments_from_aggregates(no_cluster)
    no_time = frame.copy()
    no_time.loc[0, "period_start"] = pd.NaT
    with pytest.raises(ValueError, match="period_start has 1 missing"):
      lag_moments_from_aggregates(no_time)
    labels = frame.assign(period_start=[f"h{h}" for h in range(len(frame))])
    with pytest.raises(ValueError, match="must hold datetimes"):
      lag_moments_from_aggregates(labels)

  def test_origin_must_match_the_time_column(self, frame, small_panel):
    with pytest.raises(ValueError, match="origin must be a time"):
      lag_moments_from_aggregates(frame, window_minutes=120, origin=0)
    with pytest.raises(ValueError, match="tz-aware"):
      lag_moments_from_aggregates(frame, window_minutes=120, origin=START.tz_localize("UTC"))
    with pytest.raises(ValueError, match="tz-aware"):
      lag_moments_from_aggregates(small_panel, window_minutes=120, origin=START)
    with pytest.raises(ValueError, match="numeric times only"):
      lag_moments_from_aggregates(frame, time_unit_minutes=60)

  def test_repeated_fall_back_hour_must_be_merged(self):
    """Naive local times repeat one hour when daylight saving ends."""
    utc = pd.date_range("2030-11-02 12:00", periods=36, freq="h", tz="UTC")
    local = utc.tz_convert("America/New_York").tz_localize(None)
    assert local.duplicated().sum() == 1
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
      "cluster_id": np.repeat(["a", "b"], len(local)),
      "period_start": np.tile(local, 2),
      "numerator": rng.normal(50.0, 5.0, 2 * len(local)),
      "denominator": 5,
    })
    with pytest.raises(ValueError, match="fall-back hour must be merged .* or dropped before calling"):
      lag_moments_from_aggregates(df)

    merged = df.groupby(["cluster_id", "period_start"], as_index=False).sum()
    moments = lag_moments_from_aggregates(merged)
    assert moments.nominal_windows == 35
    np.testing.assert_allclose(moments.presence, 1.0)


class TestWarnings:
  def test_warnings(self):
    def count(moments, text):
      return sum(text in note for note in moments.warnings)

    short = "pre-period shorter than two weeks"
    absent = "of windows are absent; the plan assumes gaps recur at the pre-period rate"
    raw = "no residualisation: correct only if the analysis also uses raw outcomes"
    offset = "blocks on local time shift by one base period against UTC"

    one_week = synthetic_panel(8, 1, seed=2, gap_share=0.15)
    noisy = lag_moments_from_aggregates(one_week)
    assert (count(noisy, short), count(noisy, absent), count(noisy, raw)) == (1, 1, 1)
    share = f"{100 * (1 - np.median(noisy.presence)):.0f}% of windows are absent"
    assert 0.8 < np.median(noisy.presence) < 0.9
    assert any(note.startswith(share) for note in noisy.warnings)
    assert count(noisy, offset) == 0

    two_weeks = synthetic_panel(8, 2, seed=2, start="2030-03-04")
    clean = lag_moments_from_aggregates(two_weeks, expected_col="expected")
    assert clean.warnings == ()
    # Two weeks on a grid that is out of phase lose one window, not the warning.
    shifted = lag_moments_from_aggregates(
      two_weeks, expected_col="expected", window_minutes=480,
      origin=two_weeks["period_start"].min() + 3 * HOUR,
    )
    assert shifted.nominal_windows == 41
    assert count(shifted, short) == 0

    # 2030-03-10 is a daylight-saving change in New York.
    local = two_weeks.assign(period_start=two_weeks["period_start"].dt.tz_convert("America/New_York"))
    origin = local["period_start"].min()
    blocks = lag_moments_from_aggregates(local, expected_col="expected", window_minutes=120, origin=origin)
    assert count(blocks, offset) == 1
    assert len(blocks.warnings) == 1
    assert lag_moments_from_aggregates(local, expected_col="expected", origin=origin).warnings == ()
    in_utc = lag_moments_from_aggregates(
      two_weeks, expected_col="expected", window_minutes=120, origin=two_weeks["period_start"].min()
    )
    assert_same_moments(blocks, in_utc, rel=0)
    as_objects = local.astype({"period_start": object})
    again = lag_moments_from_aggregates(as_objects, expected_col="expected", window_minutes=120, origin=origin)
    assert count(again, offset) == 1
    assert_same_moments(again, in_utc, rel=0)
    assert window_totals(local, window_minutes=120, origin=origin).attrs["warnings"] == blocks.warnings


# ---------------------------------------------------------------------------
# window_totals
# ---------------------------------------------------------------------------


class TestWindowTotals:
  def test_matches_an_independent_groupby(self, small_panel):
    # An origin 26 hours into the data: earlier windows get negative indices.
    origin = small_panel["period_start"].min() + 26 * HOUR
    totals = window_totals(small_panel, window_minutes=240, origin=origin, expected_col="expected")
    reference = complete_windows(small_panel, 240, origin).reset_index(drop=True)

    assert list(totals.columns) == ["cluster_id", "window", "numerator", "denominator", "expected"]
    assert totals["cluster_id"].tolist() == reference["cluster_id"].tolist()
    assert totals["window"].tolist() == reference["window"].tolist()
    # Hours 2 to 25 make windows -6 to -1; hours 0 and 1 are a partial window.
    assert totals["window"].min() == -6
    assert totals["window"].max() == 34
    np.testing.assert_allclose(totals["numerator"] - totals["expected"], reference["residual"], atol=1e-9)
    np.testing.assert_array_equal(totals["denominator"], reference["count"])
    assert totals.attrs["origin"] == origin
    assert totals.attrs["warnings"] == ()

  def test_expected_is_zero_without_a_column(self, small_panel):
    totals = window_totals(small_panel, window_minutes=60)
    assert (totals["expected"] == 0.0).all()
    assert totals["window"].min() == 0
    np.testing.assert_array_equal(totals["numerator"], small_panel["numerator"])

  def test_same_windows_as_the_moments(self, small_panel):
    origin = small_panel["period_start"].min()
    totals = window_totals(small_panel, window_minutes=480, origin=origin, expected_col="expected")
    moments = lag_moments_from_aggregates(small_panel, window_minutes=480, origin=origin, expected_col="expected")
    per_cluster = totals.groupby("cluster_id")
    assert tuple(per_cluster.groups) == moments.cluster_ids
    np.testing.assert_allclose(per_cluster.size() / moments.nominal_windows, moments.presence)
    np.testing.assert_allclose(per_cluster["denominator"].mean(), moments.mean_count)

  def test_validates_like_the_moments(self, small_panel):
    with pytest.raises(ValueError, match="more than one row"):
      window_totals(pd.concat([small_panel, small_panel.iloc[[0]]]), window_minutes=60)
    with pytest.raises(ValueError, match="divide a day"):
      window_totals(small_panel, window_minutes=420, origin=small_panel["period_start"].min())
    with pytest.raises(KeyError, match="period_start"):
      window_totals(small_panel.rename(columns={"period_start": "hour"}), window_minutes=60)


# ---------------------------------------------------------------------------
# plan()
# ---------------------------------------------------------------------------


class TestPlan:
  @pytest.fixture
  def moments(self, small_panel) -> ClusterLagMoments:
    return lag_moments_from_aggregates(
      small_panel, expected_col="expected", window_minutes=240, origin=small_panel["period_start"].min()
    )

  def test_plan_formulas(self, moments):
    plan = moments.plan(3, se_inflation=1.1)
    horizon = 3 * 10080 / 240
    windows = horizon * moments.presence
    lag_windows = (horizon - 1) * moments.lag_presence
    f0 = 1 + 2 * moments.count_sq_ratio / windows
    f1 = 1 + 2 * moments.count_sq_ratio_lag / lag_windows
    np.testing.assert_allclose(plan.v00, 1.21 * 4 * moments.m00 / (windows * moments.mean_count**2) * f0)
    np.testing.assert_allclose(plan.v11, 1.21 * 4 * moments.m11 / (lag_windows * moments.mean_count_lag**2) * f1)
    np.testing.assert_allclose(
      plan.v01,
      1.21 * 4 * moments.m01 / (windows * moments.mean_count * moments.mean_count_lag) * np.sqrt(f0 * f1),
    )
    np.testing.assert_allclose(
      plan.weights, moments.presence * moments.mean_count / np.sum(moments.presence * moments.mean_count)
    )
    assert plan.weights.sum() == pytest.approx(1.0)
    assert isinstance(plan, PlannedCovariance)
    assert (plan.weeks, plan.se_inflation, plan.window_minutes, plan.base_minutes) == (3.0, 1.1, 240.0, 60.0)
    assert plan.cluster_ids == moments.cluster_ids
    np.testing.assert_array_equal(plan.position_share, moments.position_share)

  def test_contrast_se(self, moments):
    plan = moments.plan(2)
    squared = plan.weights**2
    assert plan.se_l0 == pytest.approx(math.sqrt(np.sum(squared * plan.v00)))
    assert plan.se_l0 == plan.contrast_se() == plan.contrast_se(0.0)
    assert plan.se_l0_plus_l1 == plan.contrast_se(1.0)
    assert plan.se_l0_plus_l1 == pytest.approx(
      math.sqrt(np.sum(squared * (plan.v00 + 2 * plan.v01 + plan.v11)))
    )
    assert plan.corr_l0_l1 == pytest.approx(
      np.sum(squared * plan.v01) / math.sqrt(np.sum(squared * plan.v00) * np.sum(squared * plan.v11))
    )
    assert plan.contrast_se(np.full(plan.num_clusters, 2.5)) == pytest.approx(plan.contrast_se(2.5))
    per_cluster = np.linspace(-1.0, 3.0, plan.num_clusters)
    assert plan.contrast_se(per_cluster) == pytest.approx(
      math.sqrt(np.sum(squared * (plan.v00 + 2 * per_cluster * plan.v01 + per_cluster**2 * plan.v11)))
    )
    with pytest.raises(ValueError, match="one value per cluster"):
      plan.contrast_se(np.ones(plan.num_clusters + 1))
    with pytest.raises(ValueError, match="finite"):
      plan.contrast_se(float("nan"))

  def test_se_inflation_and_duration(self, moments):
    base = moments.plan(2)
    inflated = moments.plan(2, se_inflation=1.3)
    assert inflated.se_l0 == pytest.approx(1.3 * base.se_l0)
    assert inflated.se_l0_plus_l1 == pytest.approx(1.3 * base.se_l0_plus_l1)
    assert inflated.corr_l0_l1 == pytest.approx(base.corr_l0_l1)
    assert moments.plan(8).se_l0 < base.se_l0 / 1.9

  def test_plan_validation(self, moments):
    for weeks in (0, -1, float("nan"), float("inf")):
      with pytest.raises(ValueError, match="weeks"):
        moments.plan(weeks)
    for se_inflation in (0, -1.5, float("nan")):
      with pytest.raises(ValueError, match="se_inflation"):
        moments.plan(1, se_inflation=se_inflation)
    daily = lag_moments_from_parameters(20, window_autocorrelation=0.2, window_minutes=1440, **PARAMETERS)
    with pytest.raises(ValueError, match="at least 2"):
      daily.plan(1 / 7)
    assert daily.plan(2 / 7).se_l0 > 0

  def test_few_windows_warning(self, moments):
    few = "few windows per cluster"
    one_week = moments.plan(1)
    assert one_week.windows_per_cluster == pytest.approx(np.median(42 * moments.presence))
    assert sum(few in note for note in one_week.warnings) == 1
    assert any(note.startswith("few windows per cluster (40 over the planned duration") for note in one_week.warnings)
    assert set(moments.warnings) <= set(one_week.warnings)
    two_weeks = moments.plan(2)
    assert two_weeks.windows_per_cluster > 50
    assert not any(few in note for note in two_weeks.warnings)


# ---------------------------------------------------------------------------
# Parameter route
# ---------------------------------------------------------------------------


class TestParameterRoute:
  def test_parameter_route_matches_core(self):
    for window_minutes in (60, 240):
      for rho in (-0.3, 0.45):
        moments = lag_moments_from_parameters(
          40, window_autocorrelation=rho, window_minutes=window_minutes, **PARAMETERS
        )
        for weeks in (1, 3, 8):
          horizon = weeks * 10080 // window_minutes
          plan = moments.plan(weeks)
          assert plan.se_l0 == pytest.approx(core.standard_error(40, horizon, **PARAMETERS), rel=1e-12)
          assert plan.corr_l0_l1 == pytest.approx(rho * math.sqrt((horizon - 1) / horizon), abs=1e-12)

  def test_parameter_route_copies_core_warnings(self):
    for changed in (dict(cell_size_cv=2.5), dict(mean_obs_per_cell=3)):
      parameters = {**PARAMETERS, **changed}
      moments = lag_moments_from_parameters(40, window_autocorrelation=0.3, window_minutes=120, **parameters)
      _, expected = core.formula_approximation_validity(
        40, 84, mean_obs_per_cell=parameters["mean_obs_per_cell"], cell_size_cv=parameters["cell_size_cv"]
      )
      assert len(expected) == 1
      assert moments.warnings == expected
    assert lag_moments_from_parameters(40, window_autocorrelation=0.3, **PARAMETERS).warnings == ()

  def test_parameter_route_fields(self):
    moments = lag_moments_from_parameters(7, window_autocorrelation=0.25, window_minutes=240, **PARAMETERS)
    assert moments.num_clusters == 7
    assert moments.residual == "parameters"
    assert moments.origin is None
    assert moments.nominal_windows == 0
    assert math.isnan(moments.pre_period_weeks)
    assert moments.variance_factor == 1.0
    np.testing.assert_array_equal(moments.presence, 1.0)
    np.testing.assert_array_equal(moments.lag_presence, 1.0)
    np.testing.assert_array_equal(moments.count_sq_ratio, 0.0)
    np.testing.assert_array_equal(moments.mean_count, 37.5)
    np.testing.assert_array_equal(moments.position_share, np.full((7, 4), 0.25))
    np.testing.assert_allclose(moments.m01, 0.25 * moments.m00)
    np.testing.assert_array_equal(moments.m11, moments.m00)
    assert moments.subset(["3", 5]).cluster_ids == ("3", "5")

  def test_parameter_route_validation(self):
    for rho in (-1.0, 1.0, float("nan")):
      with pytest.raises(ValueError, match="window_autocorrelation"):
        lag_moments_from_parameters(40, window_autocorrelation=rho, **PARAMETERS)
    with pytest.raises(TypeError):
      lag_moments_from_parameters(40, **PARAMETERS)
    with pytest.raises(ValueError, match="num_clusters"):
      lag_moments_from_parameters(1, window_autocorrelation=0.2, **PARAMETERS)
    with pytest.raises(ValueError, match="divide a day"):
      lag_moments_from_parameters(40, window_autocorrelation=0.2, window_minutes=420, **PARAMETERS)
    with pytest.raises(ValueError, match="outcome_sd"):
      lag_moments_from_parameters(40, window_autocorrelation=0.2, **{**PARAMETERS, "outcome_sd": 0})


# ---------------------------------------------------------------------------
# Synthetic panel and observation rows
# ---------------------------------------------------------------------------


def explode(panel: pd.DataFrame) -> pd.DataFrame:
  """One row per unit: an hour with count c becomes c rows of numerator / c."""
  count = panel["denominator"].to_numpy()
  safe = np.maximum(count, 1)
  return pd.DataFrame({
    "cluster_id": np.repeat(panel["cluster_id"].to_numpy(dtype=object), count),
    "timestamp": np.repeat(panel["period_start"].to_numpy(), count),
    "outcome": np.repeat(panel["numerator"].to_numpy() / safe, count),
    "expected": np.repeat(panel["expected"].to_numpy() / safe, count),
  })


class TestSyntheticPanel:
  def test_synthetic_panel_pinned(self, default_panel):
    """Pinned to this implementation's own output for seed 3."""
    df = synthetic_panel(5, 1, seed=3)
    assert df.shape == (824, 5)
    assert list(df.columns) == ["cluster_id", "period_start", "numerator", "denominator", "expected"]
    assert df["cluster_id"].unique().tolist() == ["c000", "c001", "c002", "c003", "c004"]
    first = df.head(3)
    assert first["cluster_id"].tolist() == ["c000"] * 3
    assert first["period_start"].tolist() == [
      pd.Timestamp("2030-01-07", tz="UTC") + h * HOUR for h in range(3)
    ]
    assert first["denominator"].tolist() == [6, 8, 10]
    np.testing.assert_allclose(
      first["numerator"] - first["expected"],
      [10.147239467429017, -29.889332847573996, -7.30791664349249],
      rtol=1e-9,
      atol=1e-9 * df["numerator"].abs().max(),
    )
    assert int(df["denominator"].sum()) == 24545

    assert len(default_panel) == 199496
    assert 0.009 <= 1 - len(default_panel) / (300 * 672) <= 0.011

  def test_shape_of_the_panel(self):
    df = synthetic_panel(40, 2, seed=8, gap_share=0.0, mean_units=12.0)
    assert len(df) == 40 * 336
    assert str(df["period_start"].dt.tz) == "UTC"
    assert df["period_start"].min() == pd.Timestamp("2030-01-07", tz="UTC")
    assert df["period_start"].max() == pd.Timestamp("2030-01-20 23:00", tz="UTC")
    assert df["denominator"].mean() == pytest.approx(12.0, rel=0.02)
    quiet = df[df["denominator"] == 0]
    assert len(quiet) > 0
    assert (quiet["numerator"] == 0).all() and (quiet["expected"] == 0).all()
    # The expected value per unit is the level 100 plus a daily wave and a cluster level.
    busy = df[df["denominator"] > 0]
    assert (busy["expected"] / busy["denominator"]).mean() == pytest.approx(100.0, abs=2.0)
    pd.testing.assert_frame_equal(df, synthetic_panel(40, 2, seed=8, gap_share=0.0, mean_units=12.0))
    assert not df["numerator"].equals(synthetic_panel(40, 2, seed=9, gap_share=0.0, mean_units=12.0)["numerator"])

  def test_arguments_are_checked(self):
    for kwargs in (dict(n_clusters=0), dict(weeks=0), dict(weeks=1.5), dict(gap_share=1.0), dict(mean_units=0)):
      with pytest.raises(ValueError):
        synthetic_panel(**kwargs)


class TestAggregateObservations:
  def test_aggregate_observations_round_trip(self):
    full = synthetic_panel(120, 1, gap_share=0)
    back = aggregate_observations(explode(full), expected_col="expected", empty_periods="zero")
    assert list(back.columns) == ["cluster_id", "period_start", "numerator", "denominator", "expected"]
    assert back["cluster_id"].tolist() == full["cluster_id"].tolist()
    assert (back["period_start"] == full["period_start"]).all()
    np.testing.assert_array_equal(back["denominator"], full["denominator"])
    np.testing.assert_allclose(back["numerator"], full["numerator"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(back["expected"], full["expected"], rtol=1e-9, atol=1e-9)

    gappy = synthetic_panel(60, 1)
    busy = gappy[gappy["denominator"] > 0].reset_index(drop=True)
    back = aggregate_observations(explode(gappy), expected_col="expected", empty_periods="gap")
    assert back["cluster_id"].tolist() == busy["cluster_id"].tolist()
    assert (back["period_start"] == busy["period_start"]).all()
    np.testing.assert_array_equal(back["denominator"], busy["denominator"])
    np.testing.assert_allclose(back["numerator"], busy["numerator"], rtol=1e-9)

  def test_zero_and_gap(self):
    obs = pd.DataFrame({
      "cluster_id": [1, 1, 1, "2", 2],
      "timestamp": pd.to_datetime([
        "2030-01-07 00:10", "2030-01-07 00:50", "2030-01-07 03:00", "2030-01-07 01:59", "2030-01-07 02:00",
      ]),
      "outcome": [1.0, 2.0, 4.0, 8.0, 16.0],
    })
    gap = aggregate_observations(obs, origin="2030-01-07", empty_periods="gap")
    assert list(gap.columns) == ["cluster_id", "period_start", "numerator", "denominator"]
    assert gap["cluster_id"].tolist() == ["1", "1", "2", "2"]
    assert gap["period_start"].tolist() == [START, START + 3 * HOUR, START + HOUR, START + 2 * HOUR]
    assert gap["numerator"].tolist() == [3.0, 4.0, 8.0, 16.0]
    assert gap["denominator"].tolist() == [2, 1, 1, 1]

    zero = aggregate_observations(obs, origin="2030-01-07", empty_periods="zero")
    assert zero["cluster_id"].tolist() == ["1"] * 4 + ["2"] * 4
    assert zero["period_start"].tolist() == [START + h * HOUR for h in range(4)] * 2
    assert zero["numerator"].tolist() == [3.0, 0.0, 0.0, 4.0, 0.0, 8.0, 16.0, 0.0]
    assert zero["denominator"].tolist() == [2, 0, 0, 1, 0, 1, 1, 0]

    half_hours = aggregate_observations(obs, origin="2030-01-07", base_minutes=30, empty_periods="gap")
    assert half_hours["period_start"].tolist()[:2] == [START, START + pd.Timedelta(minutes=30)]

    numeric = obs.assign(timestamp=[0.2, 0.8, 3.0, 1.99, 2.0])
    by_number = aggregate_observations(numeric, origin=0, time_unit_minutes=60, empty_periods="gap")
    assert by_number["period_start"].tolist() == [0.0, 3.0, 1.0, 2.0]
    assert by_number["denominator"].tolist() == gap["denominator"].tolist()

  def test_feeds_the_moments(self):
    panel = synthetic_panel(10, 1, seed=6)
    back = aggregate_observations(explode(panel), expected_col="expected", empty_periods="zero")
    moments = lag_moments_from_aggregates(back, expected_col="expected")
    assert moments.num_clusters == 10
    np.testing.assert_allclose(moments.presence, 1.0)

  def test_nothing_is_dropped_silently(self):
    obs = pd.DataFrame({
      "cluster_id": ["a", "a", "b"],
      "timestamp": pd.to_datetime(["2030-01-07 00:10", "2030-01-07 01:10", "2030-01-07 00:20"]),
      "outcome": [1.0, 2.0, 3.0],
    })
    with pytest.raises(TypeError):
      aggregate_observations(obs)
    with pytest.raises(ValueError, match="empty_periods"):
      aggregate_observations(obs, empty_periods="drop")
    with pytest.raises(KeyError, match="expected"):
      aggregate_observations(obs, expected_col="expected", empty_periods="gap")
    for column, value in (("outcome", np.nan), ("timestamp", pd.NaT), ("cluster_id", None)):
      broken = obs.astype({"cluster_id": object})
      broken.loc[1, column] = value
      with pytest.raises(ValueError, match=column):
        aggregate_observations(broken, empty_periods="gap")
    with pytest.raises(ValueError, match="time_unit_minutes"):
      aggregate_observations(obs.assign(timestamp=[0.0, 1.0, 2.0]), empty_periods="gap")


# ---------------------------------------------------------------------------
# Fitted profile
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def sparse_weeks(num_weeks: int) -> pd.DataFrame:
  """A sparse panel of `num_weeks` pre-period weeks plus one held-out week."""
  return with_week_labels(synthetic_panel(300, num_weeks + 1, seed=3, mean_units=5.0))


def one_week_variance(df: pd.DataFrame, clusters: list, **kwargs) -> float:
  return lag_moments_from_aggregates(df, **kwargs).subset(clusters).plan(1).se_l0 ** 2


class TestLeaveOneWeekOut:
  def test_expected_values_by_hand(self):
    df = pd.DataFrame({
      "cluster_id": ["a"] * 6 + ["b"] * 6,
      "slot": [0, 1] * 6,
      "week": [0, 0, 1, 1, 2, 2] * 2,
      "numerator": [10.0, 30.0, 14.0, 0.0, 18.0, 44.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
      "denominator": [2, 3, 2, 0, 3, 4, 1, 1, 1, 1, 1, 1],
    })
    fit = leave_one_week_out_expected(df, slot_col="slot", week_col="week")
    assert isinstance(fit, ProfileFit)
    assert fit.num_weeks == 3
    assert fit.variance_factor == pytest.approx(8 / 9)
    assert fit.expected.index.equals(df.index)
    assert fit.warnings == ()
    # Cluster a, slot 0, week 0: the other weeks have (14 + 18) / (2 + 3) per unit.
    assert fit.expected[0] == pytest.approx(2 * 32 / 5)
    # Cluster a, slot 1, week 0: (0 + 44) / (0 + 4).
    assert fit.expected[1] == pytest.approx(3 * 11.0)
    # Zero-count row: expected 0.
    assert fit.expected[3] == 0.0
    # Cluster b, slot 0, week 2: (5 + 7) / 2.
    assert fit.expected[10] == pytest.approx(6.0)

    profile = fit.profile.set_index(["cluster_id", "slot"])["value"]
    assert len(profile) == 4
    assert profile[("a", 0)] == pytest.approx(42 / 7)
    assert profile[("a", 1)] == pytest.approx(74 / 7)
    assert profile[("b", 1)] == pytest.approx(8.0)
    assert fit.fingerprint == profile_fingerprint(fit.profile)

  def test_unsupported_slot_falls_back_to_the_cluster(self):
    df = pd.DataFrame({
      "cluster_id": ["a"] * 6 + ["b"] * 6,
      "slot": [0, 1] * 6,
      "week": [0, 0, 1, 1, 2, 2] * 2,
      "numerator": [10.0, 30.0, 14.0, 0.0, 18.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
      "denominator": [2, 3, 2, 0, 3, 0, 1, 1, 1, 1, 1, 1],
    })
    fit = leave_one_week_out_expected(df, slot_col="slot", week_col="week")
    # Cluster a has slot-1 observations in week 0 only, so that row uses the
    # cluster's ratio over weeks 1 and 2: (14 + 18) / (2 + 3).
    assert fit.expected[1] == pytest.approx(3 * 32 / 5)
    unsupported = [note for note in fit.warnings if "no observations in the other weeks" in note]
    assert len(unsupported) == 1
    assert unsupported[0].startswith("18.8% of observations")  # 3 of 16

  def test_lowo_week_labels(self):
    panel = with_week_labels(synthetic_panel(6, 3, seed=5, gap_share=0.0))
    with pytest.raises(ValueError, match="at least 2 weeks"):
      leave_one_week_out_expected(panel[panel["week"] == 0], slot_col="slot", week_col="week")

    stub = panel[(panel["week"] < 2) | (panel["slot"] < 72)]
    with pytest.raises(ValueError, match="week labels must mark whole weeks: week 2 covers 43%"):
      leave_one_week_out_expected(stub, slot_col="slot", week_col="week")

    spring = panel[~((panel["week"] == 1) & (panel["slot"] == 50))]
    fit = leave_one_week_out_expected(spring, slot_col="slot", week_col="week")
    lacking = [note for note in fit.warnings if "lacks" in note]
    assert lacking == [
      "week 1 lacks 1 of 168 slots (for example a daylight-saving change); handled by fallback"
    ]
    assert fit.variance_factor == pytest.approx(8 / 9)
    assert np.isfinite(fit.expected).all()
    moments = lag_moments_from_aggregates(spring, profile_fit=fit)
    assert moments.num_clusters == 6
    assert lacking[0] in moments.warnings

    two = leave_one_week_out_expected(panel[panel["week"] < 2], slot_col="slot", week_col="week")
    assert sum("profile fitted on 2 weeks" in note for note in two.warnings) == 1
    assert two.variance_factor == pytest.approx(3 / 4)
    three = leave_one_week_out_expected(panel, slot_col="slot", week_col="week")
    assert not any("profile fitted on" in note for note in three.warnings)

  def test_profile_fit_link(self):
    panel = with_week_labels(synthetic_panel(8, 3, seed=5))
    fit = leave_one_week_out_expected(panel, slot_col="slot", week_col="week")
    moments = lag_moments_from_aggregates(panel, profile_fit=fit)
    assert moments.residual == f"lowo profile fp {fit.fingerprint}; factor {fit.variance_factor!r}"
    assert moments.residual.endswith("; factor 0.8888888888888888")
    assert moments.variance_factor == fit.variance_factor
    by_column = lag_moments_from_aggregates(
      panel.assign(lowo=fit.expected), expected_col="lowo", variance_factor=fit.variance_factor
    )
    assert_same_moments(moments, by_column, rel=1e-12)

    with pytest.raises(ValueError, match="profile_fit was computed on a different frame"):
      lag_moments_from_aggregates(panel.sort_values("numerator"), profile_fit=fit)
    with pytest.raises(ValueError, match="profile_fit was computed on a different frame"):
      lag_moments_from_aggregates(panel[panel["week"] < 2], profile_fit=fit)
    with pytest.raises(ValueError, match="do not pass"):
      lag_moments_from_aggregates(panel, profile_fit=fit, variance_factor=0.9)
    with pytest.raises(ValueError, match="do not pass"):
      lag_moments_from_aggregates(panel, profile_fit=fit, expected_col="expected")

    external = lag_moments_from_aggregates(panel, expected_col="expected", profile_fingerprint=fit.fingerprint)
    assert external.residual == f"external profile fp {fit.fingerprint}"
    for bad in ("abc", fit.fingerprint.upper() + "!", 12):
      with pytest.raises(ValueError, match="12 hex digits"):
        lag_moments_from_aggregates(panel, expected_col="expected", profile_fingerprint=bad)
    with pytest.raises(ValueError, match="need expected_col"):
      lag_moments_from_aggregates(panel, profile_fingerprint=fit.fingerprint)

  def test_profile_fingerprint_invariance(self):
    profile = pd.DataFrame({
      "cluster_id": [1, 1, 2, 2],
      "slot": [0, 1, 0, 1],
      "value": [10.5, 11.25, 9.0, 8.75],
    })
    fingerprint = profile_fingerprint(profile)
    assert len(fingerprint) == 12 and set(fingerprint) <= set("0123456789abcdef")
    assert profile_fingerprint(profile.sample(frac=1.0, random_state=0)) == fingerprint
    as_text = profile.assign(cluster_id=profile["cluster_id"].astype(str), slot=profile["slot"].astype(str))
    assert profile_fingerprint(as_text) == fingerprint
    renamed = profile.rename(columns={"cluster_id": "zone", "slot": "hour", "value": "mean"})
    assert profile_fingerprint(renamed, cluster_col="zone", slot_col="hour", value_col="mean") == fingerprint

    changed = profile.copy()
    changed.loc[2, "value"] = 9.000001
    assert profile_fingerprint(changed) != fingerprint
    with pytest.raises(ValueError, match="more than once"):
      profile_fingerprint(pd.concat([profile, profile.iloc[[0]]]))

  @pytest.mark.parametrize("num_weeks", [3, 4])
  def test_lowo_profile_factor(self, num_weeks):
    """A plan from leave-one-week-out residuals matches a held-out week analysed
    with the profile fitted on all pre-period weeks."""
    panel = sparse_weeks(num_weeks)
    pre = panel[panel["week"] < num_weeks]
    held = panel[panel["week"] == num_weeks]
    fit = leave_one_week_out_expected(pre, slot_col="slot", week_col="week")
    value = fit.profile.set_index(["cluster_id", "slot"])["value"]

    def fitted(df: pd.DataFrame) -> pd.DataFrame:
      rows = pd.MultiIndex.from_frame(df[["cluster_id", "slot"]])
      return df.assign(fitted=value.reindex(rows).to_numpy() * df["denominator"])

    held_moments = lag_moments_from_aggregates(fitted(held), expected_col="fitted")
    planned_moments = lag_moments_from_aggregates(pre, profile_fit=fit)
    clusters = [c for c in planned_moments.cluster_ids if c in set(held_moments.cluster_ids)]
    assert len(clusters) >= 295

    actual = held_moments.subset(clusters).plan(1).se_l0 ** 2
    planned = planned_moments.subset(clusters).plan(1).se_l0 ** 2
    assert 0.94 < planned / actual < 1.06

    if num_weeks == 3:
      # Leaving the factor out overstates the variance by about K^2 / (K^2 - 1) = 1.125.
      no_factor = one_week_variance(pre.assign(lowo=fit.expected), clusters, expected_col="lowo")
      assert no_factor / planned == pytest.approx(9 / 8)
      assert no_factor / actual > 1.10
      # Residuals from the profile's own weeks understate it by far more.
      in_sample = one_week_variance(fitted(pre), clusters, expected_col="fitted")
      assert actual / in_sample > 1.5
