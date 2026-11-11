"""Shared types and validators for carryover-aware planning and analysis."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

import numpy as np

Direction = Literal["lower_is_better", "higher_is_better"]
WEEK_MINUTES = 10080.0
DAY_MINUTES = 1440.0


def orientation(direction: str) -> float:
  """+1 when a lower metric is an improvement, -1 when a higher one is.

  Multiplying an effect by this sign makes "improvement" mean "negative"
  everywhere downstream.
  """
  if direction == "lower_is_better":
    return 1.0
  if direction == "higher_is_better":
    return -1.0
  raise ValueError(
    f"direction must be 'lower_is_better' or 'higher_is_better'; got {direction!r}."
  )


def validate_one_sided_alpha(alpha: float) -> None:
  """Check a one-sided significance level."""
  if not 0 < alpha < 0.5:
    raise ValueError("alpha is one-sided and must lie strictly between 0 and 0.5.")


def validate_target_power(power_level: float, alpha: float) -> None:
  """Check a target power against the one-sided level it must exceed."""
  validate_one_sided_alpha(alpha)
  if not alpha < power_level < 1:
    raise ValueError("power_level must lie strictly between alpha and 1.")


def validate_window(window_minutes: float, base_minutes: float) -> int:
  """Return k for a window of k base periods; the window must divide a day."""
  if base_minutes <= 0 or window_minutes <= 0:
    raise ValueError("window_minutes and base_minutes must be positive.")
  ratio = window_minutes / base_minutes
  k = int(round(ratio))
  if k < 1 or abs(ratio - k) > 1e-9:
    raise ValueError(
      f"window_minutes={window_minutes:g} must be a whole multiple of "
      f"base_minutes={base_minutes:g}."
    )
  days = DAY_MINUTES / window_minutes
  if abs(days - round(days)) > 1e-9:
    raise ValueError(
      f"window_minutes={window_minutes:g} must divide a day (1440 minutes) evenly."
    )
  return k


def normalize_ids(values: Iterable[object]) -> tuple[str, ...]:
  """Cluster ids as strings, so 1 and "1" name the same cluster."""
  return tuple(str(value) for value in values)


def read_only(values: object) -> np.ndarray:
  """A float64 copy that cannot be written to."""
  array = np.array(values, dtype=np.float64)
  array.setflags(write=False)
  return array


def fingerprint(pairs: Mapping[str, float] | Iterable[tuple[str, float]]) -> str:
  """First 12 hex digits of sha256 over sorted "id=repr(value)" lines."""
  items = pairs.items() if isinstance(pairs, Mapping) else pairs
  lines = sorted(f"{key}={float(value)!r}" for key, value in items)
  return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:12]


@dataclass(frozen=True, eq=False)
class LagEstimates:
  """Per-cluster lag-0 and lag-1 effect estimates with their covariance.

  Effects are in metric units and raw sign (treatment minus control). The
  same type carries the estimates of a real experiment (`estimate_lags`)
  and of a simulated one (`simulate`), so every test statistic runs on both.

  Attributes:
    cluster_ids: Cluster ids, as strings.
    weights: Order-share weights rho_i; they sum to 1.
    l0: Same-window effect per cluster.
    l1: Next-window effect per cluster.
    s00, s01, s11: Design-based variance of l0, covariance, variance of l1.
    window_minutes: Length of the randomization window.
    orders: Observations behind each cluster's estimate, when known.
    n_input_clusters: Clusters handed to the estimator before drops.
    n_dropped: Clusters dropped by the estimator.
    dropped_orders_share: Share of input orders in the dropped clusters.
    warnings: Notes from the estimator.
  """

  cluster_ids: tuple[str, ...]
  weights: np.ndarray
  l0: np.ndarray
  l1: np.ndarray
  s00: np.ndarray
  s01: np.ndarray
  s11: np.ndarray
  window_minutes: float
  orders: np.ndarray | None = None
  n_input_clusters: int | None = None
  n_dropped: int = 0
  dropped_orders_share: float = 0.0
  warnings: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    object.__setattr__(self, "cluster_ids", normalize_ids(self.cluster_ids))
    n = len(self.cluster_ids)
    for name in ("weights", "l0", "l1", "s00", "s01", "s11"):
      array = read_only(getattr(self, name))
      if array.shape != (n,):
        raise ValueError(f"{name} must have one value per cluster ({n}); got shape {array.shape}.")
      object.__setattr__(self, name, array)
    if self.orders is not None:
      object.__setattr__(self, "orders", read_only(self.orders))

  @property
  def num_clusters(self) -> int:
    return len(self.cluster_ids)


@dataclass(frozen=True, eq=False)
class Scenario:
  """A true effect to plan for, at the base resolution, in raw sign.

  The effect of one treated base period is `l0` in that period, `l1` in the
  next, and `l1 * tail_decay**(q - 1)` in the q-th period after it, so the
  total effect of sustained treatment is `l0 + l1 / (1 - tail_decay)`.

  Attributes:
    label: Name shown in tables.
    l0: Same-period effect.
    l1: Next-period effect (order-weighted mean across clusters).
    tail_decay: Ratio of each later lag to the one before, in [0, 1): one
      number, or a value per cluster id. 0 means nothing after the next period.
    l1_spread_sd: Between-cluster SD of the next-period effect. Needed by the
      simulated tests; `None` means not stated.
    base_minutes: Length of the base period.
  """

  label: str
  l0: float
  l1: float
  tail_decay: float | Mapping[str, float]
  l1_spread_sd: float | None = None
  base_minutes: float = 60.0

  def __post_init__(self) -> None:
    if not (np.isfinite(self.l0) and np.isfinite(self.l1)):
      raise ValueError("l0 and l1 must be finite.")
    if isinstance(self.tail_decay, Mapping):
      decay = {str(key): float(value) for key, value in self.tail_decay.items()}
      object.__setattr__(self, "tail_decay", decay)
      values = np.array(list(decay.values()), dtype=float)
    else:
      values = np.array([float(self.tail_decay)])
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values >= 1):
      raise ValueError("tail_decay must lie in [0, 1) for every cluster.")
    if self.l1_spread_sd is not None and not (np.isfinite(self.l1_spread_sd) and self.l1_spread_sd >= 0):
      raise ValueError("l1_spread_sd must be non-negative when given.")
    if self.base_minutes <= 0:
      raise ValueError("base_minutes must be positive.")

  @classmethod
  def from_rebound_share(
    cls,
    l0: float,
    rebound_share: float,
    *,
    tail_decay: float | Mapping[str, float],
    l1_spread_sd: float | None = None,
    label: str | None = None,
    base_minutes: float = 60.0,
  ) -> "Scenario":
    """Scenario whose next-period effect undoes `rebound_share` of `l0`."""
    scenario = cls(
      label=label or "",
      l0=l0,
      l1=-rebound_share * l0,
      tail_decay=tail_decay,
      l1_spread_sd=l1_spread_sd,
      base_minutes=base_minutes,
    )
    if not label:
      object.__setattr__(
        scenario,
        "label",
        f"l0={l0:g}, share={rebound_share:g}, decay={scenario.decay_label}",
      )
    return scenario

  @property
  def rebound_share(self) -> float:
    """Share of the same-period effect undone in the next period, -l1 / l0."""
    return float("nan") if self.l0 == 0 else -self.l1 / self.l0

  @property
  def decay_label(self) -> str:
    if isinstance(self.tail_decay, Mapping):
      return f"per-cluster:{fingerprint(self.tail_decay)[:8]}"
    return f"{float(self.tail_decay):g}"

  def decay_values(self, cluster_ids: Iterable[object]) -> np.ndarray:
    """Tail decay per cluster, in the order of `cluster_ids`."""
    ids = normalize_ids(cluster_ids)
    if not isinstance(self.tail_decay, Mapping):
      return np.full(len(ids), float(self.tail_decay))
    missing = [cluster for cluster in ids if cluster not in self.tail_decay]
    if missing:
      raise KeyError(
        f"tail_decay has no value for {len(missing)} cluster(s), e.g. {missing[:3]}."
      )
    return np.array([self.tail_decay[cluster] for cluster in ids], dtype=float)
