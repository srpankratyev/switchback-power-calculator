"""Six one-sided tests of a switchback experiment's effect, allowing for carryover.

All six start from per-cluster lag-0 and lag-1 estimates (`LagEstimates`) and
a bound C on what follows the next window. With rho the order-share weights:

  A  = sum rho (l0 + l1)        the effect over the same and the next window
  mu = sum rho C l1             the most that later windows can add

| #   | id                   | rejects when                                   | assumes                              |
|-----|----------------------|------------------------------------------------|--------------------------------------|
| 1   | lag0                 | sum rho l0 significantly < 0                   | no carryover                         |
| 2   | lag01                | A significantly < 0                            | nothing after the next window        |
| 3   | two_endpoint         | A and A + mu both significantly < 0            | later effects have the sign of l1    |
|     |                      |                                                | and total at most C l1               |
| 4   | moment_bound         | every piece of A + p mu + sqrt(p(1-p) v) < 0   | exchangeable lag-1 effects; v from   |
|     |                      | is rejected, p in [0, 1]                       | the DerSimonian-Laird tau2, as is    |
| 4.5 | moment_bound_floored | the same                                       | the same, with tau2 floored at zero  |
| 5   | assumption_free      | every sign pattern of the lag-1 effects is     | only the bound                       |
|     |                      | rejected                                       |                                      |

"Significantly < 0" is after orientation: effects are multiplied by +1 when a
lower metric is better and by -1 when a higher one is, so improvement is
always negative. A test reports an improvement when its z is below
-z_(1 - alpha). Tests 3 to 5 take a maximum over pieces, so their z is not a
p-value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
from numpy.polynomial import polynomial as P
from scipy.stats import norm

from switchback_power_calculator.carryover.types import (
  Direction,
  LagEstimates,
  normalize_ids,
  orientation,
  validate_one_sided_alpha,
)

TESTS = (
  "lag0",
  "lag01",
  "two_endpoint",
  "moment_bound",
  "moment_bound_floored",
  "assumption_free",
)
TEST_NUMBERS = {
  "lag0": "1",
  "lag01": "2",
  "two_endpoint": "3",
  "moment_bound": "4",
  "moment_bound_floored": "4.5",
  "assumption_free": "5",
}
TEST_LABELS = {
  "lag0": "lag-0",
  "lag01": "lag-0 + lag-1",
  "two_endpoint": "two-endpoint",
  "moment_bound": "moment bound",
  "moment_bound_floored": "moment bound, tau2 floored at 0",
  "assumption_free": "assumption-free",
}
ASSUMES = {
  "lag0": "no carryover",
  "lag01": "nothing after the next window",
  "two_endpoint": "later effects have the sign of the next-window effect and total at most C times it",
  "moment_bound": "exchangeable next-window effects across clusters, known C",
  "moment_bound_floored": "exchangeable next-window effects across clusters, known C",
  "assumption_free": "only the bound C",
}
MOMENT_BOUND_TESTS = ("moment_bound", "moment_bound_floored")
TESTS_NEEDING_C = ("two_endpoint", "moment_bound", "moment_bound_floored", "assumption_free")

CWindow = float | Mapping[object, float]
TailSign = Literal["same", "either"]


def resolve_c(c_window: CWindow, cluster_ids: Iterable[object]) -> np.ndarray:
  """C per cluster at the analysis window, in the order of `cluster_ids`.

  A mapping is keyed by cluster id; keys for clusters that are not present
  are ignored, because the estimator may have dropped some. A bare array is
  refused: nothing would tie its order to the clusters.
  """
  ids = normalize_ids(cluster_ids)
  if isinstance(c_window, np.ndarray):
    raise TypeError("pass C as one number or a mapping keyed by cluster id, not an array.")
  if isinstance(c_window, Mapping):
    lookup = {str(key): float(value) for key, value in c_window.items()}
    absent = [cluster for cluster in ids if cluster not in lookup]
    if absent:
      raise KeyError(f"no C for {len(absent)} cluster(s), e.g. {absent[:3]}.")
    values = np.array([lookup[cluster] for cluster in ids], dtype=float)
  else:
    values = np.full(len(ids), float(c_window))
  if not np.all(np.isfinite(values)) or np.any(values < 0):
    raise ValueError("C must be finite and non-negative.")
  return values


def dl_variance(tau2: float, s2: np.ndarray) -> float:
  """Variance of the DerSimonian-Laird tau2 estimate when the truth is `tau2`."""
  w = 1.0 / np.asarray(s2, dtype=float)
  n = w.size
  S1 = w.sum()
  S2 = np.sum(w**2)
  S3 = np.sum(w**3)
  cc = S1 - S2 / S1
  return float(
    (2 * (n - 1) + 4 * cc * tau2 + 2 * (S2 - 2 * S3 / S1 + S2**2 / S1**2) * tau2**2) / cc**2
  )


def dl_tau2(x: np.ndarray, s2: np.ndarray) -> tuple[float, float]:
  """DerSimonian-Laird between-cluster variance of `x` with noise variances `s2`.

  Returns `(tau2, variance)`. `tau2` is not floored and can be negative; the
  variance is evaluated at `max(tau2, 0)`.
  """
  x = np.asarray(x, dtype=float)
  w = 1.0 / np.asarray(s2, dtype=float)
  n = w.size
  if n < 2:
    raise ValueError("at least two clusters are needed to estimate a between-cluster variance.")
  S1 = w.sum()
  S2 = np.sum(w**2)
  xw = np.sum(w * x) / S1
  Q = np.sum(w * (x - xw) ** 2)
  cc = S1 - S2 / S1
  tau2 = float((Q - (n - 1)) / cc)
  return tau2, dl_variance(max(tau2, 0.0), s2)


@dataclass(frozen=True, eq=False)
class Summary:
  """Order-weighted aggregates of one experiment, oriented so improvement < 0.

  Attributes:
    direction: The direction the aggregates are oriented for.
    num_clusters: Clusters behind the aggregates.
    A0, V0: sum rho l0 and its variance.
    A, VA: sum rho (l0 + l1) and its variance.
    mu, Vmu: sum rho C l1 and its variance.
    CAmu: Covariance of A and mu.
    VB: Variance of A + mu.
    Cbar: sum rho C.
    tau2, var_tau2: DerSimonian-Laird between-cluster variance of l1 (not
      floored) and its variance (evaluated at max(tau2, 0)). NaN with fewer
      than two clusters.
    B: rho C l1 per cluster.
    dv: Variance added per cluster when its later carryover is at the bound.
  """

  direction: Direction
  num_clusters: int
  A0: float
  V0: float
  A: float
  VA: float
  mu: float
  Vmu: float
  CAmu: float
  VB: float
  Cbar: float
  tau2: float
  var_tau2: float
  B: np.ndarray
  dv: np.ndarray


def _summarize(est: LagEstimates, c: np.ndarray, direction: Direction) -> Summary:
  sign = orientation(direction)
  rho = est.weights
  e0 = sign * est.l0
  e1 = sign * est.l1
  s00, s01, s11 = est.s00, est.s01, est.s11
  A0 = np.sum(rho * e0)
  V0 = np.sum(rho**2 * s00)
  A = np.sum(rho * (e0 + e1))
  VA = np.sum(rho**2 * (s00 + 2 * s01 + s11))
  mu = np.sum(rho * c * e1)
  Vmu = np.sum(rho**2 * c**2 * s11)
  CAmu = np.sum(rho**2 * c * (s01 + s11))
  if est.num_clusters >= 2:
    tau2, var_tau2 = dl_tau2(e1, s11)
  else:
    tau2, var_tau2 = float("nan"), float("nan")
  return Summary(
    direction=direction,
    num_clusters=est.num_clusters,
    A0=float(A0),
    V0=float(V0),
    A=float(A),
    VA=float(VA),
    mu=float(mu),
    Vmu=float(Vmu),
    CAmu=float(CAmu),
    VB=float(VA + 2 * CAmu + Vmu),
    Cbar=float(np.sum(rho * c)),
    tau2=tau2,
    var_tau2=var_tau2,
    B=rho * c * e1,
    dv=rho**2 * c * (2 * s01 + (2 + c) * s11),
  )


def summarize(est: LagEstimates, c_window: CWindow, *, direction: Direction) -> Summary:
  """Aggregate per-cluster estimates into the quantities the tests use."""
  return _summarize(est, resolve_c(c_window, est.cluster_ids), direction)


def lag0_z(summary: Summary) -> float:
  return summary.A0 / np.sqrt(summary.V0)


def lag01_z(summary: Summary) -> float:
  return summary.A / np.sqrt(summary.VA)


def two_endpoint_z(summary: Summary, *, tail_sign: TailSign = "same") -> float:
  """Larger of the z-statistics at the two ends of the later carryover.

  "same": later carryover has the sign of the next-window effect, so the
  ends are A (none) and A + mu (at the bound). "either": it may have either
  sign, so the ends are A + mu and A - mu.
  """
  z_bound = (summary.A + summary.mu) / np.sqrt(summary.VB)
  if tail_sign == "same":
    return float(max(lag01_z(summary), z_bound))
  if tail_sign == "either":
    v_minus = summary.VA - 2 * summary.CAmu + summary.Vmu
    return float(max(z_bound, (summary.A - summary.mu) / np.sqrt(v_minus)))
  raise ValueError("tail_sign must be 'same' or 'either'.")


def _zmax_over_p(A, mu, VA, CAmu, Vmu, k, a2) -> np.ndarray:
  """Exact maximum over p in [0, 1] of one linearised moment-bound piece.

  The piece is z(p) = (A + p mu + k sqrt(p (1 - p))) / sqrt(V(p)) with
  V(p) = VA + 2 p CAmu + p^2 Vmu + a2 p (1 - p). Its stationary points are
  roots of p (1 - p) L(p)^2 - k^2 R(p)^2, where L = 2 mu V - (A + mu p) V'
  is linear and R = p (1 - p) V' - (1 - 2 p) V is quadratic, so the
  polynomial has degree at most 4. Squaring adds spurious roots, which are
  harmless because z is evaluated at every candidate and at both ends.

  Every argument is an array of the same length (one entry per experiment);
  the result has that length too.
  """
  V0, V1, V2 = VA, 2 * CAmu + a2, Vmu - a2
  L0 = 2 * mu * V0 - A * V1
  L1 = mu * V1 - 2 * A * V2
  R0, R1, R2 = -V0, 2 * V0, V1 + V2
  k2 = k * k
  coefficients = np.stack(
    [
      -k2 * R0 * R0,
      L0 * L0 - 2 * k2 * R0 * R1,
      2 * L0 * L1 - L0 * L0 - k2 * (R1 * R1 + 2 * R0 * R2),
      L1 * L1 - 2 * L0 * L1 - 2 * k2 * R1 * R2,
      -L1 * L1 - k2 * R2 * R2,
    ],
    axis=1,
  )
  n = len(coefficients)
  candidates = np.zeros((n, 6))
  candidates[:, 1] = 1.0

  def admissible(roots: np.ndarray) -> np.ndarray:
    inside = (np.abs(roots.imag) < 1e-8) & (roots.real > 1e-12) & (roots.real < 1.0 - 1e-12)
    return np.where(inside, roots.real, 0.0)

  active = ~np.all(np.abs(coefficients) <= 1e-8, axis=1)
  quartic = active & (coefficients[:, 4] != 0)
  if quartic.any():
    c = coefficients[quartic]
    companion = np.zeros((len(c), 4, 4))
    companion[:, 1, 0] = companion[:, 2, 1] = companion[:, 3, 2] = 1.0
    companion[:, :, 3] -= c[:, :4] / c[:, 4:5]
    candidates[quartic, 2:] = admissible(np.linalg.eigvals(companion[:, ::-1, ::-1]))
  for i in np.flatnonzero(active & ~quartic):
    roots = admissible(np.atleast_1d(P.polyroots(coefficients[i])).astype(complex))
    candidates[i, 2:2 + len(roots)] = roots

  p = candidates
  with np.errstate(invalid="ignore"):
    g = A[:, None] + p * mu[:, None] + k[:, None] * np.sqrt(p * (1 - p))
    V = VA[:, None] + 2 * p * CAmu[:, None] + p * p * Vmu[:, None] + a2[:, None] * p * (1 - p)
    return np.max(g / np.sqrt(V), axis=1)


def moment_bound_batch(
  A, mu, VA, VB, CAmu, Vmu, Cbar: float, tau2, var_tau2, t1: float, *, floor: bool
) -> np.ndarray:
  """Moment-bound z for many experiments at once; see `moment_bound_z`.

  `A`, `mu`, `tau2` and `var_tau2` are arrays with one entry per experiment;
  the variances and `Cbar` may be scalars shared by all of them.
  """
  A, mu, VA, VB, CAmu, Vmu, tau2, var_tau2 = np.broadcast_arrays(
    *(np.atleast_1d(np.asarray(x, dtype=float)) for x in (A, mu, VA, VB, CAmu, Vmu, tau2, var_tau2))
  )
  if Cbar == 0:
    return A / np.sqrt(VA)
  if floor:
    tau2 = np.maximum(tau2, 0.0)
  t = Cbar * t1
  k = Cbar**2 * tau2 / (2 * t) + t / 2
  a2 = Cbar**4 * var_tau2 / (4 * t * t)
  # z does not depend on the metric's unit, but the search for stationary
  # points does unless it runs on unit-free numbers: with small variances
  # (a long test, or a metric measured on a small scale) every polynomial
  # coefficient falls below the zero tolerance, the interior of [0, 1] is
  # skipped, and the test rejects too easily. So divide effects by the
  # standard error of A and variances by its square first.
  unit = np.sqrt(VA)
  A, mu, k = A / unit, mu / unit, k / unit
  one = np.ones_like(A)
  VB, CAmu, Vmu, a2 = VB / unit**2, CAmu / unit**2, Vmu / unit**2, a2 / unit**2
  z1 = _zmax_over_p(A, mu, one, CAmu, Vmu, k, a2)
  z2 = _zmax_over_p(A + mu, -mu, VB, -(CAmu + Vmu), Vmu, k, a2)
  return np.maximum(z1, z2)


def moment_bound_z(summary: Summary, t1: float, *, floor: bool = False) -> float:
  """Largest z over the pieces of the moment-bound null, each linearised at Cbar * t1.

  The pieces are A + p mu + sqrt(p (1 - p) v) for p in [0, 1], on both
  branches, with v = Cbar^2 tau2 the spread of the later carryover implied by
  the between-cluster variance tau2 of the lag-1 effects.

  `floor=False` (test 4) uses the DerSimonian-Laird tau2 as estimated, which
  is negative in a good share of experiments; `floor=True` (test 4.5) uses
  max(tau2, 0). The two agree when tau2 >= 0, and otherwise the floored z is
  at least as large, so test 4.5 rejects only when test 4 does. The level
  holds for any `t1` fixed before the data; `t1` moves only power.
  """
  if not t1 > 0:
    raise ValueError("t1 must be positive.")
  if summary.num_clusters < 2:
    raise ValueError("the moment-bound tests need at least two clusters.")
  return float(
    moment_bound_batch(
      summary.A, summary.mu, summary.VA, summary.VB, summary.CAmu, summary.Vmu,
      summary.Cbar, summary.tau2, summary.var_tau2, t1, floor=floor,
    )[0]
  )


def assumption_free_z(summary: Summary) -> float:
  """Maximum over d in [0, 1]^J of (A + d.B) / sqrt(VA + d.dv).

  d_i is the share of cluster i's bound that its later carryover realises;
  the maximum bounds the z of every sign pattern of the lag-1 effects. When
  some pattern has a non-negative numerator no rejection is possible, and the
  z of the pattern d = 1[B > 0] is returned (non-negative, not the maximum).

  The maximiser has the form d_i = 1[B_i - lam dv_i > 0] for a multiplier
  lam < 0, so it is found by sweeping lam through the breakpoints B_i / dv_i
  in order, also checking the interior optimum between consecutive sets.
  """
  A, V0, B, dv = summary.A, summary.VA, summary.B, summary.dv
  if A + np.maximum(B, 0).sum() >= 0:
    d = (B > 0).astype(float)
    return float((A + d @ B) / np.sqrt(V0 + d @ dv))

  with np.errstate(divide="ignore", invalid="ignore"):
    breakpoint_ = np.where(dv != 0, B / dv, np.nan)
  moving = np.isfinite(breakpoint_) & (breakpoint_ < 0)
  # Below every breakpoint the set holds the clusters with dv > 0, and those
  # with dv == 0 and B > 0. As lam rises past B_i / dv_i a cluster with
  # dv_i > 0 leaves the set and one with dv_i < 0 enters it.
  start = ((dv > 0) | ((dv == 0) & (B > 0))).astype(float)
  N = np.array([A + start @ B])
  V = np.array([V0 + start @ dv])
  if moving.any():
    points, group = np.unique(breakpoint_[moving], return_inverse=True)
    step = -np.sign(dv[moving])
    b = np.bincount(group, weights=step * B[moving], minlength=len(points))
    v = np.bincount(group, weights=step * dv[moving], minlength=len(points))
    N = N[0] + np.concatenate([[0.0], np.cumsum(b)])
    V = V[0] + np.concatenate([[0.0], np.cumsum(v)])
  with np.errstate(divide="ignore", invalid="ignore"):
    best = float(np.max(np.where(V > 0, N / np.sqrt(np.where(V > 0, V, 1.0)), -np.inf)))
    if moving.any():
      Nj, Vj = N[:-1], V[:-1]
      usable = (b != 0) & (v != 0)
      t = np.clip((Nj * v - 2 * b * Vj) / np.where(usable, b * v, 1.0), 0.0, 1.0)
      Vt = Vj + t * v
      inside = usable & (Vt > 0)
      segment = np.where(inside, (Nj + t * b) / np.sqrt(np.where(inside, Vt, 1.0)), -np.inf)
      best = max(best, float(segment.max()))
  return best


@dataclass(frozen=True)
class TestResult:
  """One test's outcome in the requested direction and in the opposite one.

  Attributes:
    name: Test id, one of `TESTS`.
    number: The test's number ("1" to "5", and "4.5").
    z: Statistic for an improvement; improves when below -z_(1 - alpha).
    improves: Whether the test reports an improvement.
    z_mirror: The statistic recomputed for the opposite direction.
    worsens: Whether the test reports a worsening.
  """

  __test__ = False  # not a pytest class

  name: str
  number: str
  z: float
  improves: bool
  z_mirror: float
  worsens: bool


@dataclass(frozen=True)
class TestResults:
  """Outcome of the requested tests for one experiment.

  Attributes:
    direction: Direction in which "improves" is read.
    alpha: One-sided level.
    results: One `TestResult` per requested test, in `TESTS` order.
    tau2: DerSimonian-Laird between-cluster variance of l1, not floored.
    var_tau2: Its variance, evaluated at max(tau2, 0).
    t1: Linearisation point used by the moment-bound tests, if any ran.
    summary: Aggregates in the requested direction.
    warnings: Notes.
  """

  __test__ = False  # not a pytest class

  direction: Direction
  alpha: float
  results: tuple[TestResult, ...]
  tau2: float
  var_tau2: float
  t1: float | None
  summary: Summary
  warnings: tuple[str, ...] = ()

  def __getitem__(self, name: str) -> TestResult:
    for result in self.results:
      if result.name == name:
        return result
    raise KeyError(name)

  @property
  def improves(self) -> dict[str, bool]:
    return {result.name: result.improves for result in self.results}

  @property
  def worsens(self) -> dict[str, bool]:
    return {result.name: result.worsens for result in self.results}


def _statistic(name: str, summary: Summary, t1: float | None, tail_sign: TailSign) -> float:
  if name == "lag0":
    return float(lag0_z(summary))
  if name == "lag01":
    return float(lag01_z(summary))
  if name == "two_endpoint":
    return two_endpoint_z(summary, tail_sign=tail_sign)
  if name == "moment_bound":
    return moment_bound_z(summary, t1, floor=False)
  if name == "moment_bound_floored":
    return moment_bound_z(summary, t1, floor=True)
  if name == "assumption_free":
    return assumption_free_z(summary)
  raise ValueError(f"unknown test {name!r}; choose from {TESTS}.")


def run_tests(
  est: LagEstimates,
  c_window: CWindow | None = None,
  *,
  direction: Direction,
  tests: Sequence[str] = TESTS,
  alpha: float = 0.10,
  t1: float | None = None,
  tail_sign: TailSign = "same",
) -> TestResults:
  """Run the requested tests on one experiment's lag estimates.

  Args:
    est: Per-cluster lag estimates, from `estimate_lags`.
    c_window: Bound C at the analysis window: one number or a mapping keyed
      by cluster id (`TailBound.window_values` gives it). Needed by every
      test except lag0 and lag01.
    direction: "lower_is_better" or "higher_is_better".
    tests: Test ids to run; results come back in `TESTS` order.
    alpha: One-sided level.
    t1: Linearisation point of the moment-bound tests, fixed before looking
      at this experiment (`t1_leave_one_out` or `planning_t1`).
    tail_sign: Two-endpoint variant, "same" (default) or "either".
  """
  validate_one_sided_alpha(alpha)
  unknown = [name for name in tests if name not in TESTS]
  if unknown:
    raise ValueError(f"unknown test(s) {unknown}; choose from {TESTS}.")
  requested = [name for name in TESTS if name in tests]
  if c_window is None:
    needing = [name for name in requested if name in TESTS_NEEDING_C]
    if needing:
      raise ValueError(f"c_window is required for {needing}.")
    c_window = 0.0
  if t1 is None and any(name in MOMENT_BOUND_TESTS for name in requested):
    raise ValueError("t1 is required for the moment-bound tests.")

  c = resolve_c(c_window, est.cluster_ids)
  opposite: Direction = (
    "higher_is_better" if direction == "lower_is_better" else "lower_is_better"
  )
  forward = _summarize(est, c, direction)
  mirror = _summarize(est, c, opposite)
  z_alpha = norm.ppf(1 - alpha)

  results = []
  for name in requested:
    z = _statistic(name, forward, t1, tail_sign)
    z_mirror = _statistic(name, mirror, t1, tail_sign)
    results.append(
      TestResult(
        name=name,
        number=TEST_NUMBERS[name],
        z=z,
        improves=bool(z < -z_alpha),
        z_mirror=z_mirror,
        worsens=bool(z_mirror < -z_alpha),
      )
    )
  warnings = []
  if tail_sign == "either" and "two_endpoint" in requested:
    warnings.append("two_endpoint: either-sign tail")
  uses_t1 = any(name in MOMENT_BOUND_TESTS for name in requested)
  return TestResults(
    direction=direction,
    alpha=alpha,
    results=tuple(results),
    tau2=forward.tau2,
    var_tau2=forward.var_tau2,
    t1=t1 if uses_t1 else None,
    summary=forward,
    warnings=tuple(warnings),
  )


def t1_leave_one_out(tau2s: Sequence[float], var_tau2s: Sequence[float]) -> list[float]:
  """t1 for each experiment of a run, from the other experiments only.

  t1_i = sqrt(max(pooled tau2 without i, 0) + its standard error), the pool
  being the inverse-variance weighted mean of the unfloored tau2 estimates.
  It never uses experiment i's own data, which keeps the moment-bound tests
  valid. Both moment-bound tests use this same value.
  """
  tau2 = np.asarray(tau2s, dtype=float)
  w = 1.0 / np.asarray(var_tau2s, dtype=float)
  if tau2.size < 2:
    raise ValueError("leave-one-out needs at least two experiments; pass t1 directly otherwise.")
  out = []
  for i in range(tau2.size):
    others = np.arange(tau2.size) != i
    pooled = np.sum(w[others] * tau2[others]) / w[others].sum()
    out.append(float(np.sqrt(max(pooled, 0.0) + np.sqrt(1 / w[others].sum()))))
  return out


def planning_t1(window_spread_sd: float, v11: np.ndarray) -> float:
  """t1 for an experiment that has not run yet.

  Applies the leave-one-out formula to what is known at planning: the
  assumed between-cluster SD of the next-window effect and the planned
  variances `v11` of the per-cluster lag-1 estimates.
  """
  if not window_spread_sd >= 0:
    raise ValueError("window_spread_sd must be non-negative.")
  tau2 = float(window_spread_sd) ** 2
  return float(np.sqrt(tau2 + np.sqrt(dl_variance(tau2, v11))))
