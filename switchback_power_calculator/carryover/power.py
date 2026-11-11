"""Closed-form power, minimum detectable effect and duration for tests 1 to 3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import owens_t
from scipy.stats import norm

from switchback_power_calculator.carryover.tail import TailBound
from switchback_power_calculator.carryover.types import (
  Direction,
  Scenario,
  orientation,
  validate_one_sided_alpha,
  validate_target_power,
)

CLOSED_FORM_TESTS = ("lag0", "lag01", "two_endpoint")
WeeksNote = Literal[
  "reached", "at_or_below_one_week", "beyond_max_weeks", "not_reachable", "not_computed"
]


def lag_coefficients(position_share: np.ndarray, decay: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """How base-resolution carryover shows up at a window of k base periods.

  An observation at position h of its window (h = 0 .. k-1) feels lags 1..h
  of its own window's treatment, lags h+1..h+k of the previous window's, and
  everything later from older windows. With lag q equal to l1 * d**(q-1) and
  `position_share[i, h]` the share of cluster i's observations at position h:

    window lag-0 effect = l0 + a0 * l1
    window lag-1 effect = a1 * l1
    later windows       = tail_ratio * a1 * l1

  Returns `(a0, a1, tail_ratio)` per cluster. At k = 1: (0, 1, d / (1 - d)).
  """
  share = np.asarray(position_share, dtype=float)
  d = np.asarray(decay, dtype=float)[:, None]
  k = share.shape[1]
  h = np.arange(k)[None, :]
  a0 = np.sum(share * (1 - d**h) / (1 - d), axis=1)
  a1 = np.sum(share * d**h * (1 - d**k) / (1 - d), axis=1)
  dk = d[:, 0] ** k
  return a0, a1, dk / (1 - dk)


def _check_base(cov, scenario: Scenario) -> None:
  if scenario.base_minutes != cov.base_minutes:
    raise ValueError(
      f"scenario is stated at {scenario.base_minutes:g}-minute periods but the design's "
      f"base period is {cov.base_minutes:g} minutes."
    )


def _coefficients(cov, scenario: Scenario) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  _check_base(cov, scenario)
  decay = scenario.decay_values(cov.cluster_ids)
  return (decay, *lag_coefficients(cov.position_share, decay))


def window_means(cov, scenario: Scenario) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Per-cluster true (lag-0, lag-1, later) effects at the design's window, raw sign."""
  _, a0, a1, tail_ratio = _coefficients(cov, scenario)
  L1 = a1 * scenario.l1
  return scenario.l0 + a0 * scenario.l1, L1, L1 * tail_ratio


def window_l1_spread_sd(cov, scenario: Scenario) -> float:
  """Between-cluster SD of the window-level lag-1 effect implied by the scenario."""
  if scenario.l1_spread_sd is None:
    raise ValueError("the scenario does not state l1_spread_sd.")
  _, _, a1, _ = _coefficients(cov, scenario)
  return float(scenario.l1_spread_sd * np.sqrt(np.sum(cov.weights * a1**2)))


def total_effect(cov, scenario: Scenario) -> float:
  """Order-weighted total effect of sustained treatment, raw sign.

  It is the same at every window length: l0 + l1 / (1 - decay).
  """
  _check_base(cov, scenario)
  decay = scenario.decay_values(cov.cluster_ids)
  return float(np.sum(cov.weights * (scenario.l0 + scenario.l1 / (1 - decay))))


def phi2(a: float, b: float, r: float) -> float:
  """P(X < a, Y < b) for standard normals with correlation r, by Owen's T.

  Deterministic, and accurate as r approaches 1, which long windows reach.
  """
  if r >= 1 - 1e-12:
    return float(norm.cdf(min(a, b)))
  if r <= -(1 - 1e-12):
    return float(max(0.0, norm.cdf(a) - norm.cdf(-b)))
  a = 1e-300 if a == 0 else a
  b = 1e-300 if b == 0 else b
  s = math.sqrt(1 - r * r)
  beta = 0.0 if (a * b > 0 or (a * b == 0 and a + b >= 0)) else 0.5
  value = (
    0.5 * norm.cdf(a)
    + 0.5 * norm.cdf(b)
    - owens_t(a, (b - r * a) / (a * s))
    - owens_t(b, (a - r * b) / (b * s))
    - beta
  )
  return float(min(max(value, 0.0), 1.0))


def lag1_weights(test: str, cov, tail_bound: TailBound | None = None, *, stress: bool = False) -> tuple:
  """Lag-1 weights of the contrasts a closed-form test must all reject.

  A contrast is sum rho (l0 + b l1): lag0 has b = 0, lag01 has b = 1, and the
  two-endpoint test needs both b = 1 and b = 1 + C at the design's window.
  """
  if test == "lag0":
    return (0.0,)
  if test == "lag01":
    return (1.0,)
  if test == "two_endpoint":
    if tail_bound is None:
      raise ValueError("the two-endpoint test needs a tail_bound.")
    if tail_bound.base_minutes != cov.base_minutes:
      raise ValueError("tail_bound and design must share the same base period.")
    c = tail_bound.window_values(cov.cluster_ids, cov.window_minutes, stress=stress)
    return (1.0, 1.0 + c)
  raise ValueError(f"{test!r} has no closed form; choose from {CLOSED_FORM_TESTS} or simulate it.")


def _correlation(cov, b1, b2) -> float:
  covariance = np.sum(cov.weights**2 * (cov.v00 + (b1 + b2) * cov.v01 + b1 * b2 * cov.v11))
  return float(covariance / (cov.contrast_se(b1) * cov.contrast_se(b2)))


def _joint_power(cov, weights: tuple, thetas: Sequence[float], z_alpha: float) -> float:
  cut = [-z_alpha - theta / cov.contrast_se(b) for theta, b in zip(thetas, weights)]
  if len(cut) == 1:
    return float(norm.cdf(cut[0]))
  return phi2(cut[0], cut[1], _correlation(cov, weights[0], weights[1]))


def closed_form_power(
  cov,
  test: str,
  scenario: Scenario,
  *,
  direction: Direction,
  alpha: float = 0.10,
  tail_bound: TailBound | None = None,
  stress: bool = False,
) -> float:
  """Probability that a closed-form test reports an improvement under `scenario`.

  `cov` is a `PlannedCovariance` (`moments.plan(weeks)`). `stress=True` runs
  the two-endpoint test with the bound's stress C.
  """
  validate_one_sided_alpha(alpha)
  sign = orientation(direction)
  L0, L1, _ = window_means(cov, scenario)
  weights = lag1_weights(test, cov, tail_bound, stress=stress)
  thetas = [sign * float(np.sum(cov.weights * (L0 + b * L1))) for b in weights]
  return _joint_power(cov, weights, thetas, norm.ppf(1 - alpha))


@dataclass(frozen=True)
class MdeResult:
  """Smallest same-period effect a test detects, for one shape of carryover.

  The shape is a rebound share s (the next period undoes s of the same-period
  effect) and a tail decay. Effects of that shape scale with the same-period
  effect m, so the MDE is the m at which power reaches the target.

  Attributes:
    test: Test id.
    rebound_share: s, at the base resolution.
    tail_decay: Label of the decay.
    mde_same_period: m, the magnitude of the same-period effect. NaN if unreachable.
    total_effect_at_mde: Total effect implied by m under the decay, raw sign.
    total_improves: Whether that total is an improvement.
    reachable: False when the test cannot reach the target at any effect size.
    limit_share: Smallest rebound share at which the test can never pass.
    total_range_lo: Total with later carryover at the tail bound, raw sign (NaN without a bound).
    total_range_hi: Total with nothing after the next period, raw sign (NaN without a bound).
  """

  test: str
  rebound_share: float
  tail_decay: str
  mde_same_period: float
  total_effect_at_mde: float
  total_improves: bool
  reachable: bool
  limit_share: float
  total_range_lo: float = float("nan")
  total_range_hi: float = float("nan")


def mde(
  cov,
  test: str,
  *,
  rebound_share: float,
  tail_decay: float | Mapping[str, float],
  direction: Direction,
  alpha: float = 0.10,
  power_level: float = 0.80,
  tail_bound: TailBound | None = None,
  stress: bool = False,
) -> MdeResult:
  """Minimum detectable same-period effect of a closed-form test."""
  validate_target_power(power_level, alpha)
  sign = orientation(direction)
  z_alpha, z_power = norm.ppf(1 - alpha), norm.ppf(power_level)
  shape = Scenario(
    label="shape", l0=-1.0, l1=rebound_share, tail_decay=tail_decay, base_minutes=cov.base_minutes
  )
  decay, a0, a1, _ = _coefficients(cov, shape)
  weights = lag1_weights(test, cov, tail_bound, stress=stress)
  rho = cov.weights
  # Oriented mean of contrast j is -m * kappa_j.
  loadings = [float(np.sum(rho * (a0 + b * a1))) for b in weights]
  kappas = [1 - rebound_share * loading for loading in loadings]
  ses = [cov.contrast_se(b) for b in weights]
  limits = [1 / loading if loading > 0 else math.inf for loading in loadings]

  def power_at(m: float) -> float:
    return _joint_power(cov, weights, [-m * kappa for kappa in kappas], z_alpha)

  reachable = all(kappa > 0 for kappa in kappas)
  if not reachable:
    m_star = math.nan
  elif len(weights) == 1:
    m_star = (z_alpha + z_power) * ses[0] / kappas[0]
  else:
    # Joint power is at most the smaller marginal and at least 1 - sum of shortfalls.
    lo = max((z_alpha + z_power) * se / kappa for se, kappa in zip(ses, kappas))
    upper = norm.ppf(1 - (1 - power_level) / 2)
    hi = max((z_alpha + upper) * se / kappa for se, kappa in zip(ses, kappas))
    if power_at(lo) >= power_level - 1e-10:
      m_star = lo
    else:
      m_star = brentq(lambda m: power_at(m) - power_level, lo, hi, xtol=1e-12 * hi)

  total_factor = float(np.sum(rho * (1 - rebound_share / (1 - decay))))
  total = -sign * m_star * total_factor
  lo_total = hi_total = math.nan
  if tail_bound is not None and reachable:
    c_bar = float(np.sum(rho * tail_bound.base_values(cov.cluster_ids)))
    lo_total = -sign * m_star * (1 - rebound_share * (1 + c_bar))
    hi_total = -sign * m_star * (1 - rebound_share)
  return MdeResult(
    test=test,
    rebound_share=float(rebound_share),
    tail_decay=shape.decay_label,
    mde_same_period=float(m_star),
    total_effect_at_mde=float(total),
    total_improves=bool(reachable and sign * total < 0),
    reachable=reachable,
    limit_share=float(min(limits)),
    total_range_lo=float(lo_total),
    total_range_hi=float(hi_total),
  )


@dataclass(frozen=True)
class WeeksResult:
  """Duration a test needs to reach the target power.

  Attributes:
    weeks: Exact duration (closed form) or whole weeks (simulation); inf if not reached.
    weeks_runnable: `weeks` rounded up to whole weeks, at least 1; None if not reached.
    note: "reached", "at_or_below_one_week", "beyond_max_weeks", "not_reachable"
      (no duration is enough) or "not_computed".
    power_at_runnable: Power at `weeks_runnable`; NaN if not reached.
  """

  weeks: float
  weeks_runnable: int | None
  note: WeeksNote
  power_at_runnable: float


def weeks_to_power(
  moments,
  test: str,
  scenario: Scenario,
  *,
  direction: Direction,
  alpha: float = 0.10,
  power_level: float = 0.80,
  se_inflation: float = 1.0,
  max_weeks: int = 52,
  tail_bound: TailBound | None = None,
  stress: bool = False,
) -> WeeksResult:
  """Weeks a closed-form test needs to reach `power_level` under `scenario`.

  `moments` is a `ClusterLagMoments`. Durations below one week are not
  reported, because a partial week has a different mix of hours from the
  pre-period the moments were read off.
  """
  validate_target_power(power_level, alpha)
  if max_weeks < 1:
    raise ValueError("max_weeks must be at least 1.")

  def power_at(weeks: float) -> float:
    return closed_form_power(
      moments.plan(weeks, se_inflation=se_inflation),
      test,
      scenario,
      direction=direction,
      alpha=alpha,
      tail_bound=tail_bound,
      stress=stress,
    )

  sign = orientation(direction)
  cov = moments.plan(1.0, se_inflation=se_inflation)
  L0, L1, _ = window_means(cov, scenario)
  weights = lag1_weights(test, cov, tail_bound, stress=stress)
  if any(sign * float(np.sum(cov.weights * (L0 + b * L1))) >= 0 for b in weights):
    return WeeksResult(math.inf, None, "not_reachable", math.nan)
  if power_at(float(max_weeks)) < power_level:
    return WeeksResult(math.inf, None, "beyond_max_weeks", math.nan)
  first = power_at(1.0)
  if first >= power_level:
    return WeeksResult(1.0, 1, "at_or_below_one_week", first)
  weeks = brentq(lambda d: power_at(d) - power_level, 1.0, float(max_weeks), xtol=1e-6)
  runnable = max(1, math.ceil(weeks - 1e-9))
  return WeeksResult(float(weeks), runnable, "reached", power_at(float(runnable)))


def rebound_curve(
  cov,
  tests: Sequence[str] = ("lag0", "lag01"),
  *,
  tail_decay: float | Mapping[str, float],
  direction: Direction,
  shares: Sequence[float] = (-0.5, -0.25, 0.0, 0.1, 0.2, 0.3, 0.4),
  alpha: float = 0.10,
  power_level: float = 0.80,
  tail_bound: TailBound | None = None,
  stress: bool = False,
) -> pd.DataFrame:
  """MDE of each closed-form test across rebound shares, one row per pair."""
  rows = [
    mde(
      cov,
      test,
      rebound_share=share,
      tail_decay=tail_decay,
      direction=direction,
      alpha=alpha,
      power_level=power_level,
      tail_bound=tail_bound,
      stress=stress,
    )
    for test in tests
    for share in shares
  ]
  return pd.DataFrame([row.__dict__ for row in rows])


def scenario_grid(
  l0_values: Sequence[float],
  rebound_shares: Sequence[float],
  *,
  tail_decay: float | Mapping[str, float],
  l1_spread_sd: float | None = None,
  base_minutes: float = 60.0,
) -> list[Scenario]:
  """Scenarios for every combination of same-period effect and rebound share."""
  return [
    Scenario.from_rebound_share(
      l0, share, tail_decay=tail_decay, l1_spread_sd=l1_spread_sd, base_minutes=base_minutes
    )
    for l0 in l0_values
    for share in rebound_shares
  ]
