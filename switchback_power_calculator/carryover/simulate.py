"""Simulated power for tests without a closed form (4, 4.5 and 5, or any statistic).

Each draw is one hypothetical experiment: true per-cluster lag effects from
the scenario, including a between-cluster spread of the next-period effect,
plus estimation noise with the planned covariance. The tests then run on it
with the same code that analyses a real experiment.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from switchback_power_calculator.carryover import analysis
from switchback_power_calculator.carryover.power import (
  WeeksResult,
  _coefficients,
  window_l1_spread_sd,
)
from switchback_power_calculator.carryover.tail import TailBound
from switchback_power_calculator.carryover.types import (
  Direction,
  LagEstimates,
  Scenario,
  orientation,
  validate_one_sided_alpha,
  validate_target_power,
)

Statistic = Callable[[LagEstimates], float]


@dataclass(frozen=True)
class SimulatedPower:
  """Monte Carlo estimate of a test's power.

  Attributes:
    power: Share of draws in which the test reported an improvement.
    mc_se: Monte Carlo standard error of `power`.
    n_draws: Number of draws.
    seed: Seed of the draws.
    n_failed: Draws where the statistic was not finite (counted as no rejection).
  """

  power: float
  mc_se: float
  n_draws: int
  seed: int
  n_failed: int


def cluster_normals(seed: int, cluster_ids: Sequence[str], n_draws: int) -> np.ndarray:
  """Standard normal draws of shape (n_draws, clusters, 3), keyed by cluster id.

  A cluster's draws depend only on the seed and its own id, so designs that
  share clusters are compared on the same noise, and a longer run extends a
  shorter one draw by draw.
  """
  out = np.empty((n_draws, len(cluster_ids), 3))
  for j, cluster in enumerate(cluster_ids):
    key = int(hashlib.sha256(str(cluster).encode()).hexdigest()[:16], 16)
    out[:, j, :] = np.random.default_rng([seed, key]).standard_normal((n_draws, 3))
  return out


def _resolve_c(cov, tail_bound: TailBound | None, stress: bool) -> np.ndarray:
  if tail_bound is None:
    return np.zeros(len(cov.cluster_ids))
  if tail_bound.base_minutes != cov.base_minutes:
    raise ValueError("tail_bound and design must share the same base period.")
  return tail_bound.window_values(cov.cluster_ids, cov.window_minutes, stress=stress)


def _draw_estimates(cov, scenario: Scenario, normals: np.ndarray, spread_factor: float):
  """Raw-sign simulated (l0, l1) estimates, each of shape (n_draws, clusters)."""
  if scenario.l1_spread_sd is None:
    raise ValueError(
      "simulation needs l1_spread_sd; pass 0 explicitly to assume identical "
      "next-period effects across clusters."
    )
  _, a0, a1, _ = _coefficients(cov, scenario)
  rho = cov.weights
  z = normals[:, :, 0]
  # Centred so the order-weighted next-period effect equals the scenario's in every draw.
  deviation = scenario.l1_spread_sd * spread_factor * (z - (z @ rho)[:, None])
  l1_true = scenario.l1 + deviation
  sd0 = np.sqrt(cov.v00)
  slope = cov.v01 / sd0
  rest = np.sqrt(np.maximum(cov.v11 - slope**2, 0.0))
  l0 = scenario.l0 + a0 * l1_true + sd0 * normals[:, :, 1]
  l1 = a1 * l1_true + slope * normals[:, :, 1] + rest * normals[:, :, 2]
  return l0, l1


def simulated_power(
  cov,
  test: str | Statistic,
  scenario: Scenario,
  *,
  direction: Direction,
  alpha: float = 0.10,
  n_draws: int = 1000,
  seed: int = 0,
  tail_bound: TailBound | None = None,
  stress: bool = False,
  t1: float | None = None,
  spread_factor: float = 1.0,
  normals: np.ndarray | None = None,
) -> SimulatedPower:
  """Power of a test under `scenario`, by simulation.

  Args:
    cov: A `PlannedCovariance` (`moments.plan(weeks)`).
    test: One of `analysis.TESTS`, or a function of a `LagEstimates` (raw
      sign) that returns a z-statistic rejecting when below -z_(1 - alpha).
    scenario: True effect; must state `l1_spread_sd`.
    direction: "lower_is_better" or "higher_is_better".
    alpha: One-sided level.
    n_draws: Simulated experiments.
    seed: Seed of the draws (keyed by cluster id, see `cluster_normals`).
    tail_bound: Needed by every built-in test except lag0 and lag01.
    stress: Run the test with the bound's stress C.
    t1: Linearisation point of the moment-bound tests. Default: the planning
      rule (`analysis.planning_t1`) at the scenario's spread and the planned
      variances, which is what would be fixed before the experiment.
    spread_factor: Multiplies the TRUE spread only; the test settings (t1)
      stay at the scenario's spread. Shows what an understated spread costs.
    normals: Draws from `cluster_normals`, to reuse across calls.
  """
  validate_one_sided_alpha(alpha)
  if n_draws < 1:
    raise ValueError("n_draws must be at least 1.")
  if spread_factor < 0:
    raise ValueError("spread_factor must be non-negative.")
  if normals is None:
    normals = cluster_normals(seed, cov.cluster_ids, n_draws)
  elif normals.shape != (n_draws, len(cov.cluster_ids), 3):
    raise ValueError("normals must have shape (n_draws, clusters, 3).")
  l0, l1 = _draw_estimates(cov, scenario, normals, spread_factor)
  z_alpha = norm.ppf(1 - alpha)

  if callable(test):
    z = np.empty(n_draws)
    for r in range(n_draws):
      z[r] = test(
        LagEstimates(
          cluster_ids=cov.cluster_ids,
          weights=cov.weights,
          l0=l0[r],
          l1=l1[r],
          s00=cov.v00,
          s01=cov.v01,
          s11=cov.v11,
          window_minutes=cov.window_minutes,
        )
      )
  else:
    if test not in analysis.TESTS:
      raise ValueError(f"unknown test {test!r}; choose from {analysis.TESTS} or pass a function.")
    if tail_bound is None and test in analysis.TESTS_NEEDING_C:
      raise ValueError(f"test {test!r} needs a tail_bound.")
    c = _resolve_c(cov, tail_bound, stress)
    if test in analysis.MOMENT_BOUND_TESTS and t1 is None:
      t1 = analysis.planning_t1(window_l1_spread_sd(cov, scenario), cov.v11)
    z = _builtin_statistics(cov, test, orientation(direction) * l0, orientation(direction) * l1, c, t1, direction)

  finite = np.isfinite(z)
  power = float(np.mean(finite & (z < -z_alpha)))
  return SimulatedPower(
    power=power,
    mc_se=math.sqrt(max(power * (1 - power), 1 / n_draws) / n_draws),
    n_draws=n_draws,
    seed=seed,
    n_failed=int((~finite).sum()),
  )


def _builtin_statistics(cov, test, e0, e1, c, t1, direction) -> np.ndarray:
  """z of a built-in test for every draw; e0 and e1 are oriented estimates.

  The variances are the planned ones in every draw, so every aggregate that
  does not involve the estimates is computed once.
  """
  rho = cov.weights
  v00, v01, v11 = cov.v00, cov.v01, cov.v11
  n_clusters = len(rho)
  V0 = float(np.sum(rho**2 * v00))
  VA = float(np.sum(rho**2 * (v00 + 2 * v01 + v11)))
  Vmu = float(np.sum(rho**2 * c**2 * v11))
  CAmu = float(np.sum(rho**2 * c * (v01 + v11)))
  VB = VA + 2 * CAmu + Vmu
  c_bar = float(np.sum(rho * c))
  dv = rho**2 * c * (2 * v01 + (2 + c) * v11)

  A0 = e0 @ rho
  A = (e0 + e1) @ rho
  mu = e1 @ (rho * c)
  if test == "lag0":
    return A0 / math.sqrt(V0)
  if test == "lag01":
    return A / math.sqrt(VA)
  if test == "two_endpoint":
    return np.maximum(A / math.sqrt(VA), (A + mu) / math.sqrt(VB))

  if n_clusters >= 2:
    w = 1.0 / v11
    S1, S2, S3 = w.sum(), np.sum(w**2), np.sum(w**3)
    cc = S1 - S2 / S1
    centre = (e1 @ w) / S1
    Q = ((e1 - centre[:, None]) ** 2) @ w
    tau2 = (Q - (n_clusters - 1)) / cc
    floored = np.maximum(tau2, 0.0)
    var_tau2 = (
      2 * (n_clusters - 1) + 4 * cc * floored + 2 * (S2 - 2 * S3 / S1 + S2**2 / S1**2) * floored**2
    ) / cc**2
  else:
    tau2 = var_tau2 = np.full(len(A), np.nan)

  if test in analysis.MOMENT_BOUND_TESTS:
    if not t1 > 0:
      raise ValueError("t1 must be positive.")
    if n_clusters < 2:
      raise ValueError("the moment-bound tests need at least two clusters.")
    return analysis.moment_bound_batch(
      A, mu, VA, VB, CAmu, Vmu, c_bar, tau2, var_tau2, t1,
      floor=test == "moment_bound_floored",
    )

  z = np.empty(len(A))
  weight_c = rho * c
  for r in range(len(A)):
    summary = analysis.Summary(
      direction=direction,
      num_clusters=n_clusters,
      A0=float(A0[r]),
      V0=V0,
      A=float(A[r]),
      VA=VA,
      mu=float(mu[r]),
      Vmu=Vmu,
      CAmu=CAmu,
      VB=VB,
      Cbar=c_bar,
      tau2=float(tau2[r]),
      var_tau2=float(var_tau2[r]),
      B=weight_c * e1[r],
      dv=dv,
    )
    z[r] = analysis._statistic(test, summary, t1, "same")
  return z


def simulated_weeks(
  moments,
  test: str | Statistic,
  scenario: Scenario,
  *,
  direction: Direction,
  alpha: float = 0.10,
  power_level: float = 0.80,
  se_inflation: float = 1.0,
  max_weeks: int = 52,
  n_draws: int = 1000,
  seed: int = 0,
  tail_bound: TailBound | None = None,
  stress: bool = False,
  start_weeks: int = 1,
  normals: np.ndarray | None = None,
) -> WeeksResult:
  """Whole weeks a simulated test needs to reach `power_level`.

  Bisects on whole weeks with the same draws at every duration, so power is
  close to monotone in the duration. A test whose power is still short at 100
  times `max_weeks` is reported as "not_reachable": some tests carry a term
  that does not shrink with duration.
  """
  validate_target_power(power_level, alpha)
  if max_weeks < 1 or not 1 <= start_weeks <= max_weeks:
    raise ValueError("need 1 <= start_weeks <= max_weeks.")
  if normals is None:
    normals = cluster_normals(seed, moments.cluster_ids, n_draws)

  def power_at(weeks: float) -> float:
    return simulated_power(
      moments.plan(weeks, se_inflation=se_inflation),
      test,
      scenario,
      direction=direction,
      alpha=alpha,
      n_draws=n_draws,
      seed=seed,
      tail_bound=tail_bound,
      stress=stress,
      normals=normals,
    ).power

  if power_at(max_weeks) < power_level:
    if power_at(100 * max_weeks) < power_level:
      return WeeksResult(math.inf, None, "not_reachable", math.nan)
    return WeeksResult(math.inf, None, "beyond_max_weeks", math.nan)
  first = power_at(start_weeks)
  if first >= power_level:
    note = "at_or_below_one_week" if start_weeks == 1 else "reached"
    return WeeksResult(float(start_weeks), start_weeks, note, first)
  lo, hi, power_hi = start_weeks, max_weeks, math.nan
  while hi - lo > 1:
    mid = (lo + hi) // 2
    power_mid = power_at(mid)
    if power_mid >= power_level:
      hi, power_hi = mid, power_mid
    else:
      lo = mid
  if math.isnan(power_hi):
    power_hi = power_at(hi)
  return WeeksResult(float(hi), hi, "reached", power_hi)


def spread_sensitivity(
  moments,
  test: str | Statistic,
  scenario: Scenario,
  *,
  weeks: float,
  direction: Direction,
  factors: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
  alpha: float = 0.10,
  n_draws: int = 1000,
  seed: int = 0,
  se_inflation: float = 1.0,
  tail_bound: TailBound | None = None,
  stress: bool = False,
) -> pd.DataFrame:
  """Power when the true spread is `factor` times the one the test was set for."""
  cov = moments.plan(weeks, se_inflation=se_inflation)
  normals = cluster_normals(seed, cov.cluster_ids, n_draws)
  rows = []
  for factor in factors:
    result = simulated_power(
      cov,
      test,
      scenario,
      direction=direction,
      alpha=alpha,
      n_draws=n_draws,
      seed=seed,
      tail_bound=tail_bound,
      stress=stress,
      spread_factor=factor,
      normals=normals,
    )
    rows.append(
      {
        "factor": float(factor),
        "l1_spread_sd": factor * scenario.l1_spread_sd,
        "power": result.power,
        "mc_se": result.mc_se,
      }
    )
  return pd.DataFrame(rows)
