"""Synthetic data for the carryover examples and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

HOURS_PER_WEEK = 168


def synthetic_panel(
  n_clusters: int = 300,
  weeks: int = 4,
  *,
  seed: int = 3,
  gap_share: float = 0.01,
  mean_units: float = 30.0,
  start: str = "2030-01-07",
) -> pd.DataFrame:
  """An hourly pre-period panel with a daily shape, AR(1) shocks and gaps.

  Each cluster has its own size and level. Unit counts follow a two-peak daily
  shape that is 20% higher on the last two days of each week. The outcome per
  unit is a known expected value plus an hourly shock that carries over from
  hour to hour (autocorrelation 0.6) plus unit noise. A share `gap_share` of
  the (cluster, hour) rows is missing.

  The numbers are round and uncalibrated. The same `seed` always gives the
  same frame, because the order of the random draws is fixed.

  Args:
    n_clusters: Number of clusters.
    weeks: Number of whole weeks, starting at `start`.
    seed: Seed of the one random generator used.
    gap_share: Probability that a (cluster, hour) row is missing.
    mean_units: Mean number of units per cluster and hour.
    start: First hour; a time-zone-free value is taken as UTC.

  Returns:
    One row per present (cluster, hour), ordered by cluster then hour:
    `cluster_id` ("c000", "c001", ...), `period_start` (UTC), `numerator` (sum
    of the outcome over the hour's units), `denominator` (number of units) and
    `expected` (expected outcome per unit times the number of units).
  """
  if int(n_clusters) != n_clusters or n_clusters < 1:
    raise ValueError("n_clusters must be a positive whole number.")
  if int(weeks) != weeks or weeks < 1:
    raise ValueError("weeks must be a positive whole number.")
  if not 0 <= gap_share < 1:
    raise ValueError("gap_share must lie in [0, 1).")
  if not mean_units > 0:
    raise ValueError("mean_units must be positive.")
  n_clusters, weeks = int(n_clusters), int(weeks)

  rng = np.random.default_rng(seed)
  n_hours = HOURS_PER_WEEK * weeks
  hour = np.arange(n_hours)
  hour_of_day = hour % 24
  day_of_week = (hour // 24) % 7
  weekend = day_of_week >= 5

  shape = (
    0.05
    + 0.95 * np.exp(-0.5 * ((hour_of_day - 12) / 3.5) ** 2)
    + 0.7 * np.exp(-0.5 * ((hour_of_day - 19) / 2.0) ** 2)
  )
  shape = shape * np.where(weekend, 1.2, 1.0)
  shape = shape / shape.mean()

  # The order of the draws below is fixed: size, count, level, shocks (one
  # draw per hour), unit noise, gaps.
  size = rng.lognormal(0.0, 0.6, n_clusters)
  size = size * mean_units / size.mean()
  count = rng.poisson(size[:, None] * shape[None, :])
  level = rng.normal(0.0, 3.0, n_clusters)
  shock = np.empty((n_clusters, n_hours))
  shock[:, 0] = rng.normal(0.0, 2.0, n_clusters)
  innovation_scale = np.sqrt(1.0 - 0.36)
  for t in range(1, n_hours):
    shock[:, t] = 0.6 * shock[:, t - 1] + innovation_scale * rng.normal(0.0, 2.0, n_clusters)
  unit_noise = rng.normal(0.0, 10.0, (n_clusters, n_hours))
  present = rng.random((n_clusters, n_hours)) >= gap_share

  slot = day_of_week * 24 + hour_of_day
  expected_per_unit = (
    100.0
    + 4.0 * np.sin(2.0 * np.pi * slot / 24.0)
    + 1.5 * weekend
  )[None, :] + level[:, None]
  numerator = count * (expected_per_unit + shock) + np.sqrt(count) * unit_noise
  expected = count * expected_per_unit

  first = pd.Timestamp(start)
  first = first.tz_localize("UTC") if first.tzinfo is None else first.tz_convert("UTC")
  cluster_index, hour_index = np.nonzero(present)
  ids = np.array([f"c{i:03d}" for i in range(n_clusters)], dtype=object)
  return pd.DataFrame({
    "cluster_id": ids[cluster_index],
    "period_start": first + pd.to_timedelta(hour_index, unit="h"),
    "numerator": numerator[present],
    "denominator": count[present],
    "expected": expected[present],
  })
