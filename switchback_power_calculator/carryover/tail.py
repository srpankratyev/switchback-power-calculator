"""Bound on the carryover that continues after the next period."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

import numpy as np

from switchback_power_calculator.carryover.types import (
  Scenario,
  fingerprint,
  normalize_ids,
  validate_window,
)

ASSUMPTION = (
  "each later lag has the same sign as the next-period effect and is at most "
  "g = C/(1+C) times the lag before it (so everything after the next period "
  "adds up to at most C times the next-period effect)"
)


def _as_values(c: float | Mapping[object, float]) -> float | dict[str, float]:
  if isinstance(c, Mapping):
    values = {str(key): float(value) for key, value in c.items()}
    array = np.array(list(values.values()), dtype=float)
    if array.size == 0:
      raise ValueError("C mapping is empty.")
  else:
    values = float(c)
    array = np.array([values])
  if not np.all(np.isfinite(array)) or np.any(array < 0):
    raise ValueError("C must be finite and non-negative.")
  return values


def _gamma_to_c(gamma: float | Mapping[object, float]) -> float | dict[str, float]:
  def convert(value: float) -> float:
    value = float(value)
    if not 0 <= value < 1:
      raise ValueError("gamma must lie in [0, 1).")
    return value / (1.0 - value)

  if isinstance(gamma, Mapping):
    return {str(key): convert(value) for key, value in gamma.items()}
  return convert(gamma)


@dataclass(frozen=True, eq=False)
class TailBound:
  """Per-period bound on carryover after the next period.

  `c` is defined at the base resolution: with decay factor gamma per base
  period, C = gamma / (1 - gamma). The tests that use it assume that
  ASSUMPTION holds. At a window of one base period only its parenthetical
  consequence is needed; longer windows need the per-lag statement, which is
  what `window_values` relies on.

  Attributes:
    c: One C, or a value per cluster id.
    source: Where C comes from, e.g. "decay fit, March".
    stress_c: A more pessimistic C to plan against, optional.
    stress_source: Where the stress value comes from.
    base_minutes: Length of the base period C refers to.
    missing: "error" raises for a cluster without a C; "mean" gives it the
      mean of the mapping, which is what the legacy tool does.
  """

  c: float | Mapping[str, float]
  source: str
  stress_c: float | Mapping[str, float] | None = None
  stress_source: str = "stress"
  base_minutes: float = 60.0
  missing: Literal["error", "mean"] = "error"

  def __post_init__(self) -> None:
    if not self.source:
      raise ValueError("source must say where C comes from.")
    if self.missing not in ("error", "mean"):
      raise ValueError("missing must be 'error' or 'mean'.")
    if self.base_minutes <= 0:
      raise ValueError("base_minutes must be positive.")
    object.__setattr__(self, "c", _as_values(self.c))
    if self.stress_c is not None:
      object.__setattr__(self, "stress_c", _as_values(self.stress_c))

  @classmethod
  def from_gamma(
    cls,
    gamma: float | Mapping[object, float],
    *,
    source: str,
    stress_gamma: float | Mapping[object, float] | None = None,
    stress_source: str = "stress",
    base_minutes: float = 60.0,
    missing: Literal["error", "mean"] = "error",
  ) -> "TailBound":
    """Build from decay factors gamma in [0, 1): C = gamma / (1 - gamma)."""
    return cls(
      c=_gamma_to_c(gamma),
      source=source,
      stress_c=None if stress_gamma is None else _gamma_to_c(stress_gamma),
      stress_source=stress_source,
      base_minutes=base_minutes,
      missing=missing,
    )

  def _values(self, stress: bool) -> float | Mapping[str, float]:
    if stress:
      if self.stress_c is None:
        raise ValueError("no stress_c was given.")
      return self.stress_c
    return self.c

  def base_values(self, cluster_ids: Iterable[object], *, stress: bool = False) -> np.ndarray:
    """C per cluster at the base resolution, in the order of `cluster_ids`."""
    ids = normalize_ids(cluster_ids)
    values = self._values(stress)
    if not isinstance(values, Mapping):
      return np.full(len(ids), float(values))
    absent = [cluster for cluster in ids if cluster not in values]
    if absent and self.missing == "error":
      raise KeyError(f"no C for {len(absent)} cluster(s), e.g. {absent[:3]}.")
    fill = float(np.mean(list(values.values())))
    return np.array([values.get(cluster, fill) for cluster in ids], dtype=float)

  def decay_values(self, cluster_ids: Iterable[object], *, stress: bool = False) -> np.ndarray:
    """Per-period decay g = C / (1 + C) per cluster."""
    c = self.base_values(cluster_ids, stress=stress)
    return c / (1.0 + c)

  def window_values(
    self, cluster_ids: Iterable[object], window_minutes: float, *, stress: bool = False
  ) -> np.ndarray:
    """C per cluster for windows of k base periods: g**k / (1 - g**k).

    This is the largest ratio of the carryover after the next window to the
    next-window effect that ASSUMPTION allows, whatever the mix of orders
    within a window. At k = 1 it is C itself.
    """
    k = validate_window(window_minutes, self.base_minutes)
    decay = self.decay_values(cluster_ids, stress=stress) ** k
    return decay / (1.0 - decay)

  def scenario(
    self,
    l0: float,
    rebound_share: float,
    *,
    cluster_ids: Iterable[object] | None = None,
    tail_share: float = 1.0,
    l1_spread_sd: float | None = None,
    label: str | None = None,
  ) -> Scenario:
    """A true effect whose later carryover sits at `tail_share` of the bound.

    Uses the planning C, never the stress value. `cluster_ids` is needed
    when C is per cluster.
    """
    if not 0 <= tail_share <= 1:
      raise ValueError("tail_share must lie in [0, 1].")
    if isinstance(self.c, Mapping):
      ids = normalize_ids(self.c.keys() if cluster_ids is None else cluster_ids)
      c = tail_share * self.base_values(ids)
      decay: float | dict[str, float] = dict(zip(ids, c / (1.0 + c)))
    else:
      c = tail_share * float(self.c)
      decay = c / (1.0 + c)
    return Scenario.from_rebound_share(
      l0,
      rebound_share,
      tail_decay=decay,
      l1_spread_sd=l1_spread_sd,
      label=label,
      base_minutes=self.base_minutes,
    )

  def fingerprint(self, *, stress: bool = False) -> str:
    """Checksum that pins the C values without printing them."""
    values = self._values(stress)
    if isinstance(values, Mapping):
      return fingerprint(values)
    return f"scalar:{float(values)!r}"

  def label(self, *, stress: bool = False) -> str:
    values = self._values(stress)
    source = self.stress_source if stress else self.source
    if isinstance(values, Mapping):
      return f"C per cluster, mean {np.mean(list(values.values())):.2f} ({source})"
    return f"C = {float(values):.2f} ({source})"
