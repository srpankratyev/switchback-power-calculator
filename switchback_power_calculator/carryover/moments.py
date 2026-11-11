"""Pre-period data to per-cluster lag moments, and the covariance they plan.

A switchback experiment flips a fair coin for every (cluster, window). The
same-window effect l0 compares windows by their own coin; the next-window
effect l1 compares them by the previous window's coin. How noisy both
estimates will be depends only on second moments of the window scores, and
those can be read off a pre-period. `lag_moments_from_aggregates` reads them,
and `ClusterLagMoments.plan` turns them into the variances and covariance of
the two estimates for an experiment of a given length.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import numbers
import re
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from switchback_power_calculator.carryover.types import (
  WEEK_MINUTES,
  normalize_ids,
  read_only,
  validate_window,
)
from switchback_power_calculator.core import (
  estimator_variance,
  formula_approximation_validity,
)

__all__ = [
  "ClusterLagMoments",
  "PlannedCovariance",
  "ProfileFit",
  "aggregate_observations",
  "lag_moments_from_aggregates",
  "lag_moments_from_parameters",
  "leave_one_week_out_expected",
  "profile_fingerprint",
  "window_totals",
]

GRID_TOLERANCE = 1e-6
FEW_WINDOWS_THRESHOLD = 50.0
LOW_PRESENCE_THRESHOLD = 0.9
MIN_WEEK_COVERAGE = 0.95
UNSUPPORTED_SHARE_WARNING = 0.01

_ORIGIN_REQUIRED = (
  "origin is required for multi-period windows: pass the start of an experiment block"
)
_DUPLICATE_HINT = (
  "with naive local times the repeated daylight-saving fall-back hour must be merged "
  "(sum numerator, denominator and expected) or dropped before calling"
)
_PER_CLUSTER_FIELDS = (
  "presence",
  "lag_presence",
  "mean_count",
  "mean_count_lag",
  "count_sq_ratio",
  "count_sq_ratio_lag",
  "m00",
  "m11",
  "m01",
)
_WEEK_MOMENT_FIELDS = ("m00_week", "m11_week", "m01_week")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PlannedCovariance:
  """Planned variances of the per-cluster lag estimates at a given duration.

  Attributes:
    window_minutes: Length of the randomization window.
    base_minutes: Length of the base period.
    weeks: Planned duration.
    se_inflation: Factor applied to every standard error.
    cluster_ids: Cluster ids, as strings.
    weights: Weights rho_i, the planned share of observations; they sum to 1.
    v00: Planned variance of each cluster's same-window estimate l0.
    v01: Planned covariance of each cluster's l0 and l1 estimates.
    v11: Planned variance of each cluster's next-window estimate l1.
    position_share: (clusters, k) share of observations by base-period
      position within a window, carried from the moments.
    windows_per_cluster: Median over clusters of the present windows
      expected over the planned duration.
    warnings: The moments' notes, plus a few-windows note when it applies.
  """

  window_minutes: float
  base_minutes: float
  weeks: float
  se_inflation: float
  cluster_ids: tuple[str, ...]
  weights: np.ndarray
  v00: np.ndarray
  v01: np.ndarray
  v11: np.ndarray
  position_share: np.ndarray
  windows_per_cluster: float
  warnings: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    object.__setattr__(self, "cluster_ids", normalize_ids(self.cluster_ids))
    n = len(self.cluster_ids)
    for name in ("weights", "v00", "v01", "v11"):
      array = read_only(getattr(self, name))
      if array.shape != (n,):
        raise ValueError(f"{name} must have one value per cluster ({n}); got shape {array.shape}.")
      object.__setattr__(self, name, array)
    shares = read_only(self.position_share)
    if shares.ndim != 2 or shares.shape[0] != n:
      raise ValueError(f"position_share must have one row per cluster ({n}); got shape {shares.shape}.")
    object.__setattr__(self, "position_share", shares)
    object.__setattr__(self, "warnings", tuple(self.warnings))

  @property
  def num_clusters(self) -> int:
    return len(self.cluster_ids)

  def contrast_se(self, lag1_weight: float | np.ndarray = 0.0) -> float:
    """Standard error of sum_i rho_i (l0_i + b_i l1_i).

    Args:
      lag1_weight: The weight b on the next-window effect: one number, or one
        value per cluster in the order of `cluster_ids`.
    """
    weight = np.asarray(lag1_weight, dtype=float)
    if weight.ndim != 0 and weight.shape != (self.num_clusters,):
      raise ValueError(
        f"lag1_weight must be one number or one value per cluster ({self.num_clusters}); "
        f"got shape {weight.shape}."
      )
    if not np.all(np.isfinite(weight)):
      raise ValueError("lag1_weight must be finite.")
    variance = float(
      np.sum(self.weights**2 * (self.v00 + 2.0 * weight * self.v01 + weight**2 * self.v11))
    )
    return math.sqrt(max(variance, 0.0))

  @property
  def se_l0(self) -> float:
    """Standard error of the weighted same-window effect, sum_i rho_i l0_i."""
    return self.contrast_se(0.0)

  @property
  def se_l0_plus_l1(self) -> float:
    """Standard error of sum_i rho_i (l0_i + l1_i)."""
    return self.contrast_se(1.0)

  @property
  def corr_l0_l1(self) -> float:
    """Correlation of the two aggregates sum_i rho_i l0_i and sum_i rho_i l1_i."""
    squared = self.weights**2
    covariance = float(np.sum(squared * self.v01))
    return covariance / math.sqrt(float(np.sum(squared * self.v00)) * float(np.sum(squared * self.v11)))


@dataclass(frozen=True, eq=False)
class ClusterLagMoments:
  """Per-cluster second moments of window scores read off a pre-period.

  A window is present when every one of its base periods has a row, and
  lag-present when the window before it is present too. The score of a
  window is its residual minus the cluster's mean residual per observation
  times its observation count.

  The analysis centres each cluster over the experiment, so how much of a
  cluster's week-to-week level shift counts as noise depends on how long the
  experiment runs. The moments are therefore kept twice: with the mean taken
  over the whole pre-period (`m00`, `m11`, `m01`) and with it taken within
  each week of the pre-period (`m00_week`, `m11_week`, `m01_week`). `plan`
  moves from the second to the first as the planned duration grows.

  Attributes:
    window_minutes: Length of the randomization window.
    base_minutes: Length of the base period.
    origin: Origin of the window grid (a time or a number). `None` when it
      was not given for single-period windows, and on the parameter route.
    cluster_ids: Cluster ids, as strings.
    presence: Present windows over nominal windows.
    lag_presence: Lag-present windows over nominal windows minus one.
    mean_count: Mean observation count over present windows.
    mean_count_lag: Mean observation count over lag-present windows.
    count_sq_ratio: Mean squared count over the squared mean count, over
      present windows (0 on the parameter route).
    count_sq_ratio_lag: The same over lag-present windows.
    m00: Mean squared score over present windows, times `variance_factor`.
    m11: Mean squared score over lag-present windows, times `variance_factor`.
    m01: Mean product of a window's score and the previous window's score
      over lag-present windows, times `variance_factor`.
    position_share: (clusters, k) share of observations by base-period
      position within a window.
    nominal_windows: Windows from the first to the last complete window
      across all clusters (0 on the parameter route).
    residual: How outcomes were residualised: "none", "expected:<column>",
      "lowo profile fp <hex>; factor <value>", "external profile fp <hex>"
      or "parameters".
    variance_factor: Factor applied to the three moments.
    warnings: Notes about the data behind the moments.
    m00_week, m11_week, m01_week: The three moments with each cluster's mean
      taken within each week of the pre-period (weeks counted from the first
      window; windows after the last whole week join it). `None` on the
      parameter route, where `plan` uses `m00`, `m11` and `m01` as they are.
  """

  window_minutes: float
  base_minutes: float
  origin: object
  cluster_ids: tuple[str, ...]
  presence: np.ndarray
  lag_presence: np.ndarray
  mean_count: np.ndarray
  mean_count_lag: np.ndarray
  count_sq_ratio: np.ndarray
  count_sq_ratio_lag: np.ndarray
  m00: np.ndarray
  m11: np.ndarray
  m01: np.ndarray
  position_share: np.ndarray
  nominal_windows: int
  residual: str
  variance_factor: float
  warnings: tuple[str, ...] = ()
  m00_week: np.ndarray | None = None
  m11_week: np.ndarray | None = None
  m01_week: np.ndarray | None = None

  def __post_init__(self) -> None:
    object.__setattr__(self, "cluster_ids", normalize_ids(self.cluster_ids))
    n = len(self.cluster_ids)
    given = [name for name in _WEEK_MOMENT_FIELDS if getattr(self, name) is not None]
    if given and len(given) != len(_WEEK_MOMENT_FIELDS):
      raise ValueError("m00_week, m11_week and m01_week must be given together.")
    for name in (*_PER_CLUSTER_FIELDS, *given):
      array = read_only(getattr(self, name))
      if array.shape != (n,):
        raise ValueError(f"{name} must have one value per cluster ({n}); got shape {array.shape}.")
      object.__setattr__(self, name, array)
    shares = read_only(self.position_share)
    if shares.ndim != 2 or shares.shape[0] != n:
      raise ValueError(f"position_share must have one row per cluster ({n}); got shape {shares.shape}.")
    object.__setattr__(self, "position_share", shares)
    object.__setattr__(self, "warnings", tuple(self.warnings))

  @property
  def num_clusters(self) -> int:
    return len(self.cluster_ids)

  @property
  def pre_period_weeks(self) -> float:
    """Length of the pre-period in weeks; nan on the parameter route."""
    if self.nominal_windows <= 0:
      return float("nan")
    return self.nominal_windows * self.window_minutes / WEEK_MINUTES

  def subset(self, cluster_ids: Iterable[object]) -> "ClusterLagMoments":
    """The moments of the given clusters, in the order given.

    Raises:
      KeyError: An id is not among `cluster_ids`.
    """
    wanted = normalize_ids(cluster_ids)
    position = {cluster: i for i, cluster in enumerate(self.cluster_ids)}
    unknown = [cluster for cluster in wanted if cluster not in position]
    if unknown:
      raise KeyError(f"{len(unknown)} cluster id(s) not in these moments, e.g. {unknown[:3]}.")
    if not wanted:
      raise ValueError("subset needs at least one cluster id.")
    if len(set(wanted)) != len(wanted):
      raise ValueError("subset cluster ids must be distinct.")
    rows = np.array([position[cluster] for cluster in wanted], dtype=np.intp)
    changes: dict[str, object] = {name: getattr(self, name)[rows] for name in _PER_CLUSTER_FIELDS}
    if self.m00_week is not None:
      changes.update({name: getattr(self, name)[rows] for name in _WEEK_MOMENT_FIELDS})
    changes["position_share"] = self.position_share[rows]
    changes["cluster_ids"] = wanted
    return dataclasses.replace(self, **changes)

  def moments_for(self, weeks: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(m00, m11, m01)` for an experiment of `weeks`.

    Among W weeks whose level shifts are exchangeable, a share 1 - 1/W of
    their variance is left after centring over the W weeks. The within-week
    moments (W = 1) and the whole-pre-period moments (W = the pre-period's
    K weeks) are joined on that scale:

      m(W) = m_week + s (m - m_week),  s = (1 - 1/W) / (1 - 1/K), kept in [0, 1]

    A one-week experiment gets the within-week moments, and an experiment as
    long as the pre-period or longer gets the whole-pre-period moments: the
    pre-period holds no evidence about variation slower than itself.
    """
    if self.m00_week is None:
      return self.m00, self.m11, self.m01
    pre = self.pre_period_weeks
    if not pre > 1.0 or weeks >= pre:
      return self.m00, self.m11, self.m01
    share = min(max((1.0 - 1.0 / float(weeks)) / (1.0 - 1.0 / pre), 0.0), 1.0)
    return (
      self.m00_week + share * (self.m00 - self.m00_week),
      self.m11_week + share * (self.m11 - self.m11_week),
      self.m01_week + share * (self.m01 - self.m01_week),
    )

  def plan(self, weeks: float, *, se_inflation: float = 1.0) -> PlannedCovariance:
    """Planned variances of the lag estimates for an experiment of `weeks`.

    With H = weeks * 10080 / window_minutes nominal windows, each cluster's
    estimate is a ratio-of-sums difference in means under an independent fair
    coin per window. Linearising it gives

      V00 = 4 m00 / (H p nbar^2)            * f0,  f0 = 1 + 2 q  / (H p)
      V11 = 4 m11 / ((H - 1) p1 nbar1^2)    * f1,  f1 = 1 + 2 q1 / ((H - 1) p1)
      V01 = 4 m01 / (H p nbar nbar1)        * sqrt(f0 f1)

    where f is the second-order term for the random number of observations
    in each arm. Gaps are assumed to recur at the pre-period rate. The
    moments are those of an experiment of this length (`moments_for`): a
    cluster's week-to-week level shift is noise only as far as the experiment
    spans more than one week.

    Args:
      weeks: Planned duration; the experiment must span at least two windows.
      se_inflation: Factor applied to every standard error, to budget for an
        experiment that is noisier than the pre-period.
    """
    if not (np.isfinite(weeks) and weeks > 0):
      raise ValueError("weeks must be positive.")
    if not (np.isfinite(se_inflation) and se_inflation > 0):
      raise ValueError("se_inflation must be positive.")
    horizon = float(weeks) * WEEK_MINUTES / self.window_minutes
    if horizon < 2.0 - 1e-9:
      raise ValueError(
        f"weeks={weeks:g} gives {horizon:g} windows of {self.window_minutes:g} minutes; "
        "at least 2 are needed."
      )
    windows = horizon * self.presence
    lag_windows = (horizon - 1.0) * self.lag_presence
    f0 = 1.0 + 2.0 * self.count_sq_ratio / windows
    f1 = 1.0 + 2.0 * self.count_sq_ratio_lag / lag_windows
    scale = float(se_inflation) ** 2
    m00, m11, m01 = self.moments_for(weeks)
    v00 = 4.0 * m00 / (windows * self.mean_count**2) * f0 * scale
    v11 = 4.0 * m11 / (lag_windows * self.mean_count_lag**2) * f1 * scale
    v01 = (
      4.0 * m01 / (windows * self.mean_count * self.mean_count_lag)
      * np.sqrt(f0 * f1) * scale
    )
    planned_counts = self.presence * self.mean_count
    windows_per_cluster = float(np.median(windows))
    notes = list(self.warnings)
    if windows_per_cluster < FEW_WINDOWS_THRESHOLD:
      notes.append(
        f"few windows per cluster ({windows_per_cluster:.0f} over the planned duration at "
        "this window): planned power may be optimistic by up to about 0.05"
      )
    return PlannedCovariance(
      window_minutes=self.window_minutes,
      base_minutes=self.base_minutes,
      weeks=float(weeks),
      se_inflation=float(se_inflation),
      cluster_ids=self.cluster_ids,
      weights=planned_counts / planned_counts.sum(),
      v00=v00,
      v01=v01,
      v11=v11,
      position_share=self.position_share,
      windows_per_cluster=windows_per_cluster,
      warnings=tuple(notes),
    )


@dataclass(frozen=True, eq=False)
class ProfileFit:
  """A per-(cluster, slot) profile fitted on pre-period weeks.

  Attributes:
    expected: Leave-one-week-out expected value times the denominator, one
      value per input row, on the input frame's index.
    variance_factor: (K^2 - 1) / K^2 for K weeks. Residuals from a profile
      that left the row's week out are noisier than residuals from the
      profile fitted on all K weeks by exactly this factor.
    profile: `cluster_id`, `slot`, `value`: the profile fitted on all K
      weeks, which is what the analysis subtracts.
    fingerprint: `profile_fingerprint(profile)`.
    num_weeks: K, the number of week labels.
    warnings: Notes about the fit.
  """

  expected: pd.Series
  variance_factor: float
  profile: pd.DataFrame
  fingerprint: str
  num_weeks: int
  warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Reading rows onto the base grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Clock:
  """Row times as minutes from the origin, and the way back to times."""

  minutes: np.ndarray
  origin: object  # pd.Timestamp (UTC when the input was tz-aware) or float
  unit_minutes: float | None  # length of one numeric time unit; None for datetimes
  utc_offsets: int  # distinct UTC offsets in tz-aware input, else 0

  def at(self, minutes: np.ndarray) -> object:
    """Times at the given minutes from the origin."""
    if self.unit_minutes is None:
      return self.origin + pd.to_timedelta(np.asarray(minutes, dtype=float), unit="m")
    return self.origin + np.asarray(minutes, dtype=float) / self.unit_minutes


@dataclass(frozen=True)
class _Rows:
  """Validated base-period rows on the grid from the origin."""

  cluster_ids: tuple[str, ...]
  cluster: np.ndarray  # row -> index into cluster_ids
  period: np.ndarray  # row -> base-period index from the origin
  numerator: np.ndarray
  denominator: np.ndarray
  expected: np.ndarray
  k: int
  origin: object
  utc_offsets: int


@dataclass(frozen=True)
class _Windows:
  """Complete windows, ordered by (cluster, window)."""

  cluster: np.ndarray
  window: np.ndarray
  numerator: np.ndarray
  denominator: np.ndarray
  expected: np.ndarray
  position_counts: np.ndarray  # (windows, k): denominator by base-period position


def _require_columns(df: object, columns: Iterable[str]) -> None:
  if not isinstance(df, pd.DataFrame):
    raise TypeError("df must be a pandas DataFrame.")
  missing = [name for name in columns if name not in df.columns]
  if missing:
    raise KeyError(f"Missing required columns: {missing}")
  if len(df) == 0:
    raise ValueError("Input frame is empty.")


def _numeric(series: pd.Series, name: str) -> np.ndarray:
  """A column as finite floats; anything else raises."""
  if pd.api.types.is_bool_dtype(series.dtype):
    raise ValueError(f"{name} must be numeric.")
  values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
  bad = ~np.isfinite(values)
  if bad.any():
    raise ValueError(
      f"{name} must be numeric and finite; {int(bad.sum())} value(s) are missing, "
      "infinite or not numbers."
    )
  return values


def _cluster_codes(series: pd.Series, name: str) -> tuple[np.ndarray, tuple[str, ...]]:
  """Row -> cluster index, and the cluster ids as sorted strings."""
  codes, uniques = pd.factorize(series)
  if (codes < 0).any():
    raise ValueError(f"{name} has {int((codes < 0).sum())} missing value(s).")
  labels = normalize_ids(uniques)
  ordered = tuple(sorted(set(labels)))
  position = {label: i for i, label in enumerate(ordered)}
  remap = np.array([position[label] for label in labels], dtype=np.int64)
  return remap[codes], ordered


def _object_times(times: pd.Series, time_col: str) -> tuple[pd.Series, int]:
  """Datetime objects (possibly with mixed UTC offsets) as a datetime column."""
  offsets = set()
  for value in times:
    if not hasattr(value, "utcoffset"):
      raise ValueError(
        f"{time_col} must hold datetimes, or numbers with time_unit_minutes; "
        f"got a {type(value).__name__}."
      )
    offsets.add(value.utcoffset())
  if None in offsets:
    if len(offsets) > 1:
      raise ValueError(f"{time_col} mixes tz-aware and naive times.")
    return pd.to_datetime(times), 0
  return pd.to_datetime(times, utc=True), len(offsets)


def _time_origin(origin: object, aware: bool) -> pd.Timestamp:
  if isinstance(origin, (bool, numbers.Number)):
    raise ValueError("origin must be a time, like the time column.")
  try:
    stamp = pd.Timestamp(origin)
  except (ValueError, TypeError) as error:
    raise ValueError(f"origin {origin!r} is not a valid time.") from error
  if stamp is pd.NaT:
    raise ValueError("origin must not be missing.")
  if aware != (stamp.tzinfo is not None):
    raise ValueError(
      "origin must be tz-aware when the times are tz-aware, and naive when they are naive."
    )
  return stamp.tz_convert("UTC") if aware else stamp


def _clock(
  times: pd.Series,
  origin: object,
  time_unit_minutes: float | None,
  *,
  time_col: str,
) -> _Clock:
  """Put a time column on one clock, in minutes from the origin.

  tz-aware times are converted to UTC; naive times are used as given;
  numbers are multiplied by `time_unit_minutes`. The origin gets the same
  handling and defaults to the earliest time.
  """
  if times.isna().any():
    raise ValueError(f"{time_col} has {int(times.isna().sum())} missing value(s).")
  utc_offsets = 0
  if times.dtype == object:
    times, utc_offsets = _object_times(times, time_col)

  if pd.api.types.is_datetime64_any_dtype(times.dtype):
    if time_unit_minutes is not None:
      raise ValueError("time_unit_minutes applies to numeric times only.")
    aware = times.dt.tz is not None
    if aware:
      utc = times.dt.tz_convert("UTC")
      if utc_offsets == 0:
        offset = times.dt.tz_localize(None) - utc.dt.tz_localize(None)
        utc_offsets = int(offset.nunique())
      times = utc
    start = times.min() if origin is None else _time_origin(origin, aware)
    minutes = ((times - start) / pd.Timedelta(minutes=1)).to_numpy(dtype=float)
    return _Clock(minutes=minutes, origin=start, unit_minutes=None, utc_offsets=utc_offsets)

  if pd.api.types.is_numeric_dtype(times.dtype) and not pd.api.types.is_bool_dtype(times.dtype):
    if time_unit_minutes is None:
      raise ValueError(
        f"{time_col} is numeric: pass time_unit_minutes, the length of one time unit in minutes."
      )
    if not (np.isfinite(time_unit_minutes) and time_unit_minutes > 0):
      raise ValueError("time_unit_minutes must be positive.")
    values = _numeric(times, time_col)
    if origin is None:
      start = float(values.min())
    elif isinstance(origin, numbers.Real) and not isinstance(origin, bool) and np.isfinite(origin):
      start = float(origin)
    else:
      raise ValueError("origin must be a finite number when the times are numeric.")
    minutes = (values - start) * float(time_unit_minutes)
    return _Clock(
      minutes=minutes, origin=start, unit_minutes=float(time_unit_minutes), utc_offsets=0
    )

  raise ValueError(
    f"{time_col} must hold datetimes, or numbers with time_unit_minutes; got dtype {times.dtype}."
  )


def _read_rows(
  df: pd.DataFrame,
  *,
  window_minutes: float,
  base_minutes: float,
  cluster_col: str,
  time_col: str,
  numerator_col: str,
  denominator_col: str,
  expected_col: str | None,
  expected_values: np.ndarray | None,
  origin: object,
  time_unit_minutes: float | None,
) -> _Rows:
  """Validate base-period rows and put them on the grid from the origin."""
  columns = [cluster_col, time_col, numerator_col, denominator_col]
  if expected_col is not None:
    columns.append(expected_col)
  _require_columns(df, columns)
  k = validate_window(window_minutes, base_minutes)
  if k > 1 and origin is None:
    raise ValueError(_ORIGIN_REQUIRED)

  numerator = _numeric(df[numerator_col], numerator_col)
  denominator = _numeric(df[denominator_col], denominator_col)
  if expected_col is not None:
    expected = _numeric(df[expected_col], expected_col)
  elif expected_values is not None:
    expected = np.asarray(expected_values, dtype=float)
    if expected.shape != numerator.shape or not np.all(np.isfinite(expected)):
      raise ValueError("expected values must be finite, one per row.")
  else:
    expected = np.zeros_like(numerator)
  if (denominator < 0).any():
    raise ValueError(
      f"{denominator_col} must be non-negative; {int((denominator < 0).sum())} value(s) are negative."
    )
  if not denominator.sum() > 0:
    raise ValueError(f"{denominator_col} sums to zero: there are no observations.")
  empty = denominator == 0
  if (empty & (numerator != 0)).any():
    raise ValueError(
      f"{int((empty & (numerator != 0)).sum())} row(s) have a zero {denominator_col} "
      f"and a non-zero {numerator_col}."
    )
  if (empty & (expected != 0)).any():
    raise ValueError(
      f"{int((empty & (expected != 0)).sum())} row(s) have a zero {denominator_col} "
      "and a non-zero expected value."
    )

  cluster, cluster_ids = _cluster_codes(df[cluster_col], cluster_col)
  clock = _clock(df[time_col], origin, time_unit_minutes, time_col=time_col)
  position = clock.minutes / float(base_minutes)
  period = np.rint(position)
  off_grid = np.abs(position - period) > GRID_TOLERANCE
  if off_grid.any():
    raise ValueError(
      f"{int(off_grid.sum())} row(s) do not start on the {float(base_minutes):g}-minute "
      "grid from origin."
    )
  period = period.astype(np.int64)

  span = int(period.max() - period.min()) + 1
  key = cluster * span + (period - period.min())
  _, key_counts = np.unique(key, return_counts=True)
  repeated = int((key_counts > 1).sum())
  if repeated:
    raise ValueError(
      f"{repeated} (cluster, base period) pair(s) have more than one row: {_DUPLICATE_HINT}."
    )
  return _Rows(
    cluster_ids=cluster_ids,
    cluster=cluster,
    period=period,
    numerator=numerator,
    denominator=denominator,
    expected=expected,
    k=k,
    origin=clock.origin,
    utc_offsets=clock.utc_offsets,
  )


def _complete_windows(rows: _Rows) -> _Windows:
  """Sum base rows into windows, keeping windows with all k base rows."""
  k = rows.k
  window = np.floor_divide(rows.period, k)
  order = np.lexsort((rows.period, rows.cluster))
  cluster = rows.cluster[order]
  window = window[order]
  numerator = rows.numerator[order]
  denominator = rows.denominator[order]
  expected = rows.expected[order]

  first = np.ones(len(order), dtype=bool)
  first[1:] = (cluster[1:] != cluster[:-1]) | (window[1:] != window[:-1])
  starts = np.flatnonzero(first)
  sizes = np.diff(np.append(starts, len(order)))
  # Duplicates were rejected, so k rows in a window are its k base periods.
  complete = sizes == k
  full = starts[complete]
  return _Windows(
    cluster=cluster[full],
    window=window[full],
    numerator=np.add.reduceat(numerator, starts)[complete],
    denominator=np.add.reduceat(denominator, starts)[complete],
    expected=np.add.reduceat(expected, starts)[complete],
    position_counts=denominator[full[:, None] + np.arange(k)[None, :]],
  )


def _offset_warning(rows: _Rows) -> list[str]:
  if rows.k > 1 and rows.utc_offsets > 1:
    return [
      f"the UTC offset changes within the data ({rows.utc_offsets} offsets); times were "
      "converted to UTC, and blocks on local time shift by one base period against UTC at "
      "each daylight-saving change: pass naive local times if the blocks follow local time"
    ]
  return []


def window_totals(
  df: pd.DataFrame,
  *,
  window_minutes: float,
  origin: object = None,
  base_minutes: float = 60,
  cluster_col: str = "cluster_id",
  time_col: str = "period_start",
  numerator_col: str = "numerator",
  denominator_col: str = "denominator",
  expected_col: str | None = None,
  time_unit_minutes: float | None = None,
) -> pd.DataFrame:
  """Sum base-period rows into complete windows on the grid from `origin`.

  This is the window grid `lag_moments_from_aggregates` uses, so a plan and
  an analysis that both go through it put every row in the same window.

  Args:
    df: One row per (cluster, base period).
    window_minutes: Length of the window, a whole multiple of `base_minutes`
      that divides a day.
    origin: Start of a window. Required when the window is longer than the
      base period; otherwise it defaults to the earliest time.
    base_minutes: Length of the base period.
    cluster_col: Cluster id column.
    time_col: Start of each base period: datetimes (tz-aware ones are
      converted to UTC), or numbers together with `time_unit_minutes`.
    numerator_col: Sum of the outcome over the base period.
    denominator_col: Number of observations in the base period.
    expected_col: Expected sum of the outcome, if there is one.
    time_unit_minutes: Length of one unit of a numeric time column.

  Returns:
    One row per complete (cluster, window), ordered by cluster then window:
    `cluster_id` (str), `window` (index from the origin; window t covers base
    periods t*k to t*k + k - 1), `numerator`, `denominator` and `expected`
    (0.0 without `expected_col`). A window missing any base row is left out.
    `attrs["origin"]` holds the origin used and `attrs["warnings"]` any notes.
  """
  rows = _read_rows(
    df,
    window_minutes=window_minutes,
    base_minutes=base_minutes,
    cluster_col=cluster_col,
    time_col=time_col,
    numerator_col=numerator_col,
    denominator_col=denominator_col,
    expected_col=expected_col,
    expected_values=None,
    origin=origin,
    time_unit_minutes=time_unit_minutes,
  )
  windows = _complete_windows(rows)
  ids = np.array(rows.cluster_ids, dtype=object)
  totals = pd.DataFrame({
    "cluster_id": ids[windows.cluster],
    "window": windows.window.astype(np.int64),
    "numerator": windows.numerator,
    "denominator": windows.denominator,
    "expected": windows.expected,
  })
  totals.attrs["origin"] = rows.origin
  totals.attrs["warnings"] = tuple(_offset_warning(rows))
  return totals


# ---------------------------------------------------------------------------
# Moments from pre-period aggregates
# ---------------------------------------------------------------------------


def lag_moments_from_aggregates(
  df: pd.DataFrame,
  *,
  window_minutes: float = 60,
  base_minutes: float = 60,
  cluster_col: str = "cluster_id",
  time_col: str = "period_start",
  numerator_col: str = "numerator",
  denominator_col: str = "denominator",
  expected_col: str | None = None,
  variance_factor: float = 1.0,
  profile_fit: ProfileFit | None = None,
  profile_fingerprint: str | None = None,
  origin: object = None,
  time_unit_minutes: float | None = None,
) -> ClusterLagMoments:
  """Read per-cluster lag moments off pre-period aggregates.

  Residualise the same way the analysis will, by exactly one route:
  `profile_fit` (from `leave_one_week_out_expected`), `expected_col` (an
  expected value from outside the pre-period), or neither (raw outcomes).

  For windows longer than the base period, the times and `origin` must be on
  the clock the experiment's blocks follow. For blocks on local wall-clock
  time, pass naive local times, with the repeated daylight-saving fall-back
  hour merged or dropped.

  Args:
    df: One row per (cluster, base period). A missing row is a gap; a row
      with a zero denominator is a present period with no observations.
    window_minutes: Length of the randomization window, a whole multiple of
      `base_minutes` that divides a day.
    base_minutes: Length of the base period.
    cluster_col: Cluster id column.
    time_col: Start of each base period: datetimes (tz-aware ones are
      converted to UTC), or numbers together with `time_unit_minutes`.
    numerator_col: Sum of the outcome over the base period.
    denominator_col: Number of observations in the base period.
    expected_col: Expected sum of the outcome over the base period.
    variance_factor: Factor in (0, 2] applied to the moments; only with
      `expected_col`.
    profile_fit: A fit made on this same frame. It supplies the expected
      values and the variance factor.
    profile_fingerprint: `profile_fingerprint()` of an external profile
      behind `expected_col`, recorded in `residual`.
    origin: Start of an experiment block. Required when the window is longer
      than the base period; otherwise it defaults to the earliest time.
    time_unit_minutes: Length of one unit of a numeric time column.

  Returns:
    The moments of every cluster that has observations and at least two
    lag-present windows. Dropped clusters are counted in `warnings`.

  Raises:
    KeyError: A column is missing.
    ValueError: The frame is empty; a value is missing or not a number; a
      denominator is negative or they sum to zero; a row has a zero
      denominator and a non-zero numerator; a row is off the base grid; a
      (cluster, base period) is repeated; `origin` is missing for a
      multi-period window; fewer than two clusters remain.
  """
  if not isinstance(df, pd.DataFrame):
    raise TypeError("df must be a pandas DataFrame.")
  factor = float(variance_factor)
  if not (np.isfinite(factor) and 0 < factor <= 2):
    raise ValueError("variance_factor must lie in (0, 2].")
  if profile_fingerprint is not None and not (
    isinstance(profile_fingerprint, str) and re.fullmatch(r"[0-9a-f]{12}", profile_fingerprint)
  ):
    raise ValueError("profile_fingerprint must be the 12 hex digits from profile_fingerprint().")

  notes: list[str] = []
  fit_notes: tuple[str, ...] = ()
  expected_values = None
  if profile_fit is not None:
    if not isinstance(profile_fit, ProfileFit):
      raise TypeError("profile_fit must be a ProfileFit from leave_one_week_out_expected.")
    if expected_col is not None or profile_fingerprint is not None or factor != 1.0:
      raise ValueError(
        "profile_fit brings its own expected values, variance factor and fingerprint: "
        "do not pass expected_col, variance_factor or profile_fingerprint with it."
      )
    if not profile_fit.expected.index.equals(df.index):
      raise ValueError("profile_fit was computed on a different frame")
    expected_values = profile_fit.expected.to_numpy(dtype=float)
    factor = float(profile_fit.variance_factor)
    residual = f"lowo profile fp {profile_fit.fingerprint}; factor {factor!r}"
    fit_notes = tuple(profile_fit.warnings)
  elif expected_col is not None:
    if profile_fingerprint is not None:
      residual = f"external profile fp {profile_fingerprint}"
    else:
      residual = f"expected:{expected_col}"
    if factor != 1.0:
      residual += f", factor {factor!r}"
  else:
    if profile_fingerprint is not None or factor != 1.0:
      raise ValueError("profile_fingerprint and variance_factor need expected_col.")
    residual = "none"

  rows = _read_rows(
    df,
    window_minutes=window_minutes,
    base_minutes=base_minutes,
    cluster_col=cluster_col,
    time_col=time_col,
    numerator_col=numerator_col,
    denominator_col=denominator_col,
    expected_col=expected_col,
    expected_values=expected_values,
    origin=origin,
    time_unit_minutes=time_unit_minutes,
  )
  notes.extend(_offset_warning(rows))
  windows = _complete_windows(rows)
  if len(windows.window) == 0:
    raise ValueError(
      f"No complete {float(window_minutes):g}-minute window: every window is missing a base row."
    )

  n_input = len(rows.cluster_ids)
  nominal = int(windows.window.max() - windows.window.min()) + 1
  cluster = windows.cluster
  count = windows.denominator
  residual_sum = windows.numerator - windows.expected
  lag = np.zeros(len(cluster), dtype=bool)
  lag[1:] = (cluster[1:] == cluster[:-1]) & (windows.window[1:] == windows.window[:-1] + 1)
  at = np.flatnonzero(lag)

  def per_cluster(values: np.ndarray, where: np.ndarray | None = None) -> np.ndarray:
    index = cluster if where is None else cluster[where]
    return np.bincount(index, weights=values, minlength=n_input)

  n_present = np.bincount(cluster, minlength=n_input).astype(float)
  n_lag = np.bincount(cluster[at], minlength=n_input).astype(float)
  total_count = per_cluster(count)
  total_count_lag = per_cluster(count[at], at)

  no_count = total_count <= 0
  few_lags = ~no_count & (n_lag < 2)
  no_lag_count = ~no_count & ~few_lags & (total_count_lag <= 0)
  keep = ~(no_count | few_lags | no_lag_count)
  all_counts = total_count.sum()
  for dropped, reason in (
    (no_count, "no observations in a complete window"),
    (few_lags, "fewer than 2 lag-present windows"),
    (no_lag_count, "no observations in a lag-present window"),
  ):
    if dropped.any():
      share = total_count[dropped].sum() / all_counts if all_counts > 0 else 0.0
      notes.append(
        f"{int(dropped.sum())} cluster(s) dropped ({share:.2%} of observations): {reason}"
      )
  if int(keep.sum()) < 2:
    raise ValueError(
      f"Fewer than 2 clusters have observations and 2 lag-present windows "
      f"({int(keep.sum())} of {n_input}); at least 2 are needed."
    )

  with np.errstate(divide="ignore", invalid="ignore"):
    centre = per_cluster(residual_sum) / total_count
    score = residual_sum - np.where(keep, centre, 0.0)[cluster] * count
    mean_count = total_count / n_present
    mean_count_lag = total_count_lag / n_lag
    m00 = per_cluster(score**2) / n_present * factor
    m11 = per_cluster(score[at] ** 2, at) / n_lag * factor
    m01 = per_cluster(score[at - 1] * score[at], at) / n_lag * factor
    # The same three moments with the mean taken within each week of the
    # pre-period. Windows after the last whole week join that week.
    per_week = int(round(WEEK_MINUTES / float(window_minutes)))
    n_weeks = max(1, nominal // per_week)
    week = np.minimum((windows.window - windows.window.min()) // per_week, n_weeks - 1)
    cell = cluster * n_weeks + week
    cell_count = np.bincount(cell, weights=count, minlength=n_input * n_weeks)
    cell_residual = np.bincount(cell, weights=residual_sum, minlength=n_input * n_weeks)
    cell_centre = np.where(cell_count > 0, cell_residual / cell_count, 0.0)
    score_week = residual_sum - cell_centre[cell] * count
    m00_week = per_cluster(score_week**2) / n_present * factor
    m11_week = per_cluster(score_week[at] ** 2, at) / n_lag * factor
    m01_week = per_cluster(score_week[at - 1] * score_week[at], at) / n_lag * factor
    count_sq_ratio = per_cluster(count**2) / n_present / mean_count**2
    count_sq_ratio_lag = per_cluster(count[at] ** 2, at) / n_lag / mean_count_lag**2
    position_totals = np.zeros((n_input, rows.k))
    np.add.at(position_totals, cluster, windows.position_counts)
    position_share = position_totals / total_count[:, None]
  presence = n_present / nominal
  lag_presence = n_lag / (nominal - 1)

  notes.extend(fit_notes)
  # One window of slack: a grid that is out of phase with the data loses a window.
  if (nominal + 1) * float(window_minutes) < 2 * WEEK_MINUTES:
    notes.append(
      "pre-period shorter than two weeks: no week-to-week check is possible; "
      "plans from one week are less reliable"
    )
  median_presence = float(np.median(presence[keep]))
  if median_presence < LOW_PRESENCE_THRESHOLD:
    notes.append(
      f"{100 * (1 - median_presence):.0f}% of windows are absent; the plan assumes gaps "
      "recur at the pre-period rate"
    )
  if residual == "none":
    notes.append(
      "no residualisation: correct only if the analysis also uses raw outcomes; "
      "subtracting a good expected value usually lowers SE a lot"
    )

  return ClusterLagMoments(
    window_minutes=float(window_minutes),
    base_minutes=float(base_minutes),
    origin=None if origin is None else rows.origin,
    cluster_ids=tuple(np.array(rows.cluster_ids, dtype=object)[keep]),
    presence=presence[keep],
    lag_presence=lag_presence[keep],
    mean_count=mean_count[keep],
    mean_count_lag=mean_count_lag[keep],
    count_sq_ratio=count_sq_ratio[keep],
    count_sq_ratio_lag=count_sq_ratio_lag[keep],
    m00=m00[keep],
    m11=m11[keep],
    m01=m01[keep],
    position_share=position_share[keep],
    nominal_windows=nominal,
    residual=residual,
    variance_factor=factor,
    warnings=tuple(dict.fromkeys(notes)),
    m00_week=m00_week[keep],
    m11_week=m11_week[keep],
    m01_week=m01_week[keep],
  )


# ---------------------------------------------------------------------------
# Observation rows to aggregates
# ---------------------------------------------------------------------------


def aggregate_observations(
  df: pd.DataFrame,
  *,
  cluster_col: str = "cluster_id",
  time_col: str = "timestamp",
  outcome_col: str = "outcome",
  expected_col: str | None = None,
  base_minutes: float = 60,
  origin: object = None,
  time_unit_minutes: float | None = None,
  empty_periods: Literal["zero", "gap"],
) -> pd.DataFrame:
  """Sum one-row-per-observation data into (cluster, base period) rows.

  Each observation goes to the base period that contains its time, counted
  from `origin`. Pass an `origin` on the hour (or on whatever boundary the
  base periods follow); the default is the earliest time in the data.

  Args:
    df: One row per observation.
    cluster_col: Cluster id column.
    time_col: Time of the observation: datetimes (tz-aware ones are
      converted to UTC), or numbers together with `time_unit_minutes`.
    outcome_col: The outcome.
    expected_col: Expected outcome of the observation, if there is one.
    base_minutes: Length of the base period.
    origin: Start of a base period; defaults to the earliest time.
    time_unit_minutes: Length of one unit of a numeric time column.
    empty_periods: What a base period without observations means. "zero"
      writes a zero-count row for every cluster and every base period
      between the first and last periods in the data, so lags run through
      quiet periods. "gap" leaves such periods out, so they break lags; use
      it when a missing period means the cluster was not operating or not
      logged.

  Returns:
    One row per (cluster, base period), ordered by cluster then period:
    `cluster_id` (str), `period_start`, `numerator` (sum of the outcome),
    `denominator` (number of observations), and `expected` when
    `expected_col` is given (0 in the rows written by "zero").

  Raises:
    KeyError: A column is missing.
    ValueError: The frame is empty, or an outcome, cluster, time or expected
      value is missing. No row is ever dropped silently, because a dropped
      row changes a lag pair.
  """
  if empty_periods not in ("zero", "gap"):
    raise ValueError("empty_periods must be 'zero' or 'gap'.")
  if not (np.isfinite(base_minutes) and base_minutes > 0):
    raise ValueError("base_minutes must be positive.")
  columns = [cluster_col, time_col, outcome_col]
  if expected_col is not None:
    columns.append(expected_col)
  _require_columns(df, columns)

  outcome = _numeric(df[outcome_col], outcome_col)
  expected = None if expected_col is None else _numeric(df[expected_col], expected_col)
  cluster, cluster_ids = _cluster_codes(df[cluster_col], cluster_col)
  clock = _clock(df[time_col], origin, time_unit_minutes, time_col=time_col)
  period = np.floor(clock.minutes / float(base_minutes) + 1e-9).astype(np.int64)

  first = int(period.min())
  span = int(period.max()) - first + 1
  key = cluster * span + (period - first)
  if empty_periods == "zero":
    cells = np.arange(len(cluster_ids) * span, dtype=np.int64)
    slot = key
  else:
    cells, slot = np.unique(key, return_inverse=True)
  n_cells = len(cells)
  result = {
    "cluster_id": np.array(cluster_ids, dtype=object)[cells // span],
    "period_start": clock.at((cells % span + first) * float(base_minutes)),
    "numerator": np.bincount(slot, weights=outcome, minlength=n_cells),
    "denominator": np.bincount(slot, minlength=n_cells).astype(np.int64),
  }
  if expected is not None:
    result["expected"] = np.bincount(slot, weights=expected, minlength=n_cells)
  return pd.DataFrame(result)


# ---------------------------------------------------------------------------
# Fitted profile: leave-one-week-out expected values
# ---------------------------------------------------------------------------


def _sum_of_others(values: np.ndarray) -> np.ndarray:
  """Along the last axis, the sum of every entry except the one in place.

  Built from running sums on each side, so a sum of zeros is exactly zero.
  """
  zero = np.zeros(values.shape[:-1] + (1,))
  running = np.cumsum(values, axis=-1)
  reverse = np.cumsum(values[..., ::-1], axis=-1)[..., ::-1]
  before = np.concatenate([zero, running[..., :-1]], axis=-1)
  after = np.concatenate([reverse[..., 1:], zero], axis=-1)
  return before + after


def _ratio(numerator: np.ndarray, denominator: np.ndarray, fallback: np.ndarray | float) -> np.ndarray:
  """numerator / denominator where the denominator is positive, else fallback."""
  with np.errstate(divide="ignore", invalid="ignore"):
    return np.where(denominator > 0, numerator / denominator, fallback)


def _codes(series: pd.Series, name: str) -> tuple[np.ndarray, pd.Index]:
  """Row -> index into the distinct values (sorted when they can be)."""
  if series.isna().any():
    raise ValueError(f"{name} has {int(series.isna().sum())} missing value(s).")
  try:
    codes, uniques = pd.factorize(series, sort=True)
  except TypeError:
    codes, uniques = pd.factorize(series)
  return codes, pd.Index(uniques)


def leave_one_week_out_expected(
  df: pd.DataFrame,
  *,
  slot_col: str,
  week_col: str,
  cluster_col: str = "cluster_id",
  numerator_col: str = "numerator",
  denominator_col: str = "denominator",
) -> ProfileFit:
  """Expected values from a per-(cluster, slot) profile fitted on other weeks.

  Use this when the analysis will subtract a profile fitted on all K
  pre-period weeks and there is no profile from outside the pre-period. Each
  row's expected value is its denominator times the mean outcome of the same
  cluster and slot over the other weeks. Pass the result to
  `lag_moments_from_aggregates(profile_fit=...)`, which also applies the
  variance factor (K^2 - 1) / K^2.

  The package never computes a clock or a week: `slot_col` and `week_col`
  are yours. With local-clock slots, a daylight-saving week lacks a slot (or
  repeats one), which is handled and noted in `warnings`.

  Args:
    df: One row per (cluster, base period).
    slot_col: A label that recurs every week, for example hour of week.
    week_col: A label of the week the row belongs to; weeks must be whole.
    cluster_col: Cluster id column.
    numerator_col: Sum of the outcome over the base period.
    denominator_col: Number of observations in the base period.

  Raises:
    KeyError: A column is missing.
    ValueError: The frame is empty; a value is missing or not a number;
      there are fewer than two week labels; a week covers less than 95% of
      the slots seen in all weeks together.
  """
  _require_columns(df, [cluster_col, slot_col, week_col, numerator_col, denominator_col])
  numerator = _numeric(df[numerator_col], numerator_col)
  denominator = _numeric(df[denominator_col], denominator_col)
  if (denominator < 0).any():
    raise ValueError(
      f"{denominator_col} must be non-negative; {int((denominator < 0).sum())} value(s) are negative."
    )
  if not denominator.sum() > 0:
    raise ValueError(f"{denominator_col} sums to zero: there are no observations.")
  cluster, cluster_ids = _cluster_codes(df[cluster_col], cluster_col)
  slot, slots = _codes(df[slot_col], slot_col)
  week, week_labels = _codes(df[week_col], week_col)
  n_clusters, n_slots, n_weeks = len(cluster_ids), len(slots), len(week_labels)
  if n_weeks < 2:
    raise ValueError(
      f"{week_col} has {n_weeks} distinct label(s); a leave-one-week-out profile needs at least 2 weeks."
    )

  notes: list[str] = []
  covered = np.zeros((n_weeks, n_slots), dtype=bool)
  covered[week, slot] = True
  coverage = covered.mean(axis=1)
  short = np.flatnonzero(coverage < MIN_WEEK_COVERAGE)
  if len(short):
    b = int(short[0])
    raise ValueError(
      f"week labels must mark whole weeks: week {week_labels[b]} covers "
      f"{coverage[b]:.0%} of the {n_slots} slots."
    )
  for b in np.flatnonzero(coverage < 1):
    notes.append(
      f"week {week_labels[b]} lacks {int((~covered[b]).sum())} of {n_slots} slots "
      "(for example a daylight-saving change); handled by fallback"
    )

  shape = (n_clusters, n_slots, n_weeks)
  cell = np.ravel_multi_index((cluster, slot, week), shape)
  size = n_clusters * n_slots * n_weeks
  outcome = np.bincount(cell, weights=numerator, minlength=size).reshape(shape)
  count = np.bincount(cell, weights=denominator, minlength=size).reshape(shape)

  # Leave-one-week-out: same cluster and slot in the other weeks; if those
  # have no observations, the cluster's overall ratio in the other weeks;
  # if the cluster has none there either, the ratio over all clusters.
  outcome_other, count_other = _sum_of_others(outcome), _sum_of_others(count)
  cluster_other = _ratio(
    _sum_of_others(outcome.sum(axis=1)),
    _sum_of_others(count.sum(axis=1)),
    _ratio(_sum_of_others(outcome.sum(axis=(0, 1))), _sum_of_others(count.sum(axis=(0, 1))), 0.0)[None, :],
  )
  supported = count_other[cluster, slot, week] > 0
  value = np.where(
    supported,
    _ratio(outcome_other, count_other, 0.0)[cluster, slot, week],
    cluster_other[cluster, week],
  )
  expected = pd.Series(value * denominator, index=df.index, name="expected")

  unsupported_share = float(denominator[~supported].sum() / denominator.sum())
  if n_weeks == 2:
    notes.append(
      "profile fitted on 2 weeks: with sparse counts the plan can understate variance by "
      "up to ~17%; prefer 3+ weeks or an external profile"
    )
  if unsupported_share > UNSUPPORTED_SHARE_WARNING:
    notes.append(
      f"{unsupported_share:.1%} of observations are in slots with no observations in the "
      "other weeks; their expected value is the cluster's overall mean"
    )

  # The profile the analysis subtracts: all K weeks, with the same fallbacks.
  cluster_all = _ratio(
    outcome.sum(axis=(1, 2)),
    count.sum(axis=(1, 2)),
    float(_ratio(np.array(outcome.sum()), np.array(count.sum()), 0.0)),
  )
  full = _ratio(outcome.sum(axis=2), count.sum(axis=2), cluster_all[:, None])
  profile = pd.DataFrame({
    "cluster_id": np.repeat(np.array(cluster_ids, dtype=object), n_slots),
    "slot": np.tile(np.asarray(slots), n_clusters),
    "value": full.ravel(),
  })
  return ProfileFit(
    expected=expected,
    variance_factor=(n_weeks**2 - 1.0) / n_weeks**2,
    profile=profile,
    fingerprint=profile_fingerprint(profile),
    num_weeks=n_weeks,
    warnings=tuple(notes),
  )


def profile_fingerprint(
  profile: pd.DataFrame,
  *,
  cluster_col: str = "cluster_id",
  slot_col: str = "slot",
  value_col: str = "value",
) -> str:
  """A short, order-free fingerprint of a per-(cluster, slot) profile.

  The first 12 hex digits of sha256 over the sorted lines
  "cluster|slot|repr(float(value))", with cluster ids and slots as strings.
  Row order and int-versus-str labels do not change it; any value does.
  """
  _require_columns(profile, [cluster_col, slot_col, value_col])
  values = _numeric(profile[value_col], value_col)
  keys = [
    f"{cluster}|{slot}"
    for cluster, slot in zip(normalize_ids(profile[cluster_col]), normalize_ids(profile[slot_col]))
  ]
  if len(set(keys)) != len(keys):
    raise ValueError(
      f"{len(keys) - len(set(keys))} (cluster, slot) pair(s) appear more than once in the profile."
    )
  lines = sorted(f"{key}|{float(value)!r}" for key, value in zip(keys, values))
  return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Moments from known parameters
# ---------------------------------------------------------------------------


def lag_moments_from_parameters(
  num_clusters: int,
  *,
  mean_obs_per_cell: float,
  cell_size_cv: float,
  between_cell_variance_share: float,
  within_cell_variance_share: float | None = None,
  outcome_sd: float,
  window_autocorrelation: float,
  window_minutes: float = 60,
  base_minutes: float = 60.0,
) -> ClusterLagMoments:
  """Lag moments for identical clusters, from the package's own parameters.

  The inputs are those of `summarize_power`, per window cell: a cell is one
  cluster in one window. To compare windows, call this once per window with
  that window's own parameters and the same `base_minutes`.

  `plan(weeks).se_l0` then equals `standard_error(num_clusters, H, ...)` for
  H = weeks * 10080 / window_minutes windows.

  Args:
    num_clusters: Number of clusters.
    mean_obs_per_cell: Mean observations per (cluster, window).
    cell_size_cv: Coefficient of variation of the cell sizes.
    between_cell_variance_share: Share of outcome variance between cells.
    within_cell_variance_share: Share within cells; defaults to the rest.
    outcome_sd: Standard deviation of the outcome.
    window_autocorrelation: Correlation between a cluster's scores in
      consecutive windows, strictly between -1 and 1.
    window_minutes: Length of the randomization window.
    base_minutes: Length of the base period.
  """
  k = validate_window(window_minutes, base_minutes)
  if isinstance(num_clusters, bool) or int(num_clusters) != num_clusters:
    raise ValueError("num_clusters must be a whole number.")
  num_clusters = int(num_clusters)
  if not -1 < window_autocorrelation < 1:
    raise ValueError("window_autocorrelation must lie strictly between -1 and 1.")
  shared = dict(mean_obs_per_cell=mean_obs_per_cell, cell_size_cv=cell_size_cv)
  # Variance of one observation-weighted cell mean: outcome_sd^2 times the
  # design factor of the closed-form formula.
  cell_variance = (
    estimator_variance(
      num_clusters,
      1,
      between_cell_variance_share=between_cell_variance_share,
      within_cell_variance_share=within_cell_variance_share,
      outcome_sd=outcome_sd,
      **shared,
    )
    * num_clusters
    / 4.0
  )
  _, notes = formula_approximation_validity(
    num_clusters, round(WEEK_MINUTES / window_minutes), **shared
  )
  ones = np.ones(num_clusters)
  zeros = np.zeros(num_clusters)
  m00 = cell_variance * float(mean_obs_per_cell) ** 2 * ones
  return ClusterLagMoments(
    window_minutes=float(window_minutes),
    base_minutes=float(base_minutes),
    origin=None,
    cluster_ids=tuple(str(i) for i in range(num_clusters)),
    presence=ones,
    lag_presence=ones,
    mean_count=float(mean_obs_per_cell) * ones,
    mean_count_lag=float(mean_obs_per_cell) * ones,
    count_sq_ratio=zeros,
    count_sq_ratio_lag=zeros,
    m00=m00,
    m11=m00,
    m01=float(window_autocorrelation) * m00,
    position_share=np.full((num_clusters, k), 1.0 / k),
    nominal_windows=0,
    residual="parameters",
    variance_factor=1.0,
    warnings=tuple(notes),
  )
