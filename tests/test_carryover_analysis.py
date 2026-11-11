"""Tests for the per-cluster lag estimator, the tail bound and the six tests."""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from switchback_power_calculator.carryover.analysis import (
    TEST_NUMBERS,
    TESTS,
    assumption_free_z,
    dl_tau2,
    dl_variance,
    lag01_z,
    moment_bound_z,
    planning_t1,
    resolve_c,
    run_tests,
    summarize,
    t1_leave_one_out,
    two_endpoint_z,
)
from switchback_power_calculator.carryover.estimate import estimate_lags, experiment_windows
from switchback_power_calculator.carryover.tail import TailBound
from switchback_power_calculator.carryover.types import LagEstimates


def make_estimates(rng, n=40, *, l0=-1.0, l1=0.3, spread=0.0, noise=1.0) -> LagEstimates:
  """Random per-cluster estimates around a common truth, with a valid covariance."""
  orders = rng.lognormal(3.0, 0.6, n)
  s00 = noise**2 * rng.uniform(0.5, 2.0, n) / orders * 30
  s11 = noise**2 * rng.uniform(0.5, 2.0, n) / orders * 30
  s01 = rng.uniform(-0.2, 0.7, n) * np.sqrt(s00 * s11)
  z0, z1 = rng.normal(size=n), rng.normal(size=n)
  corr = s01 / np.sqrt(s00 * s11)
  return LagEstimates(
    cluster_ids=tuple(f"c{i:03d}" for i in range(n)),
    weights=orders / orders.sum(),
    l0=l0 + np.sqrt(s00) * z0,
    l1=l1 + spread * rng.normal(size=n) + np.sqrt(s11) * (corr * z0 + np.sqrt(1 - corr**2) * z1),
    s00=s00,
    s01=s01,
    s11=s11,
    window_minutes=60.0,
    orders=orders,
  )


def c_map(est: LagEstimates, rng, low=0.2, high=2.5) -> dict[str, float]:
  return dict(zip(est.cluster_ids, rng.uniform(low, high, est.num_clusters)))


class TestSummary:
  def test_aggregates_match_hand_computation(self):
    rng = np.random.default_rng(0)
    est = make_estimates(rng, 12)
    c = c_map(est, rng)
    cv = np.array([c[k] for k in est.cluster_ids])
    s = summarize(est, c, direction="lower_is_better")
    rho = est.weights
    assert s.A0 == pytest.approx(np.sum(rho * est.l0))
    assert s.A == pytest.approx(np.sum(rho * (est.l0 + est.l1)))
    assert s.mu == pytest.approx(np.sum(rho * cv * est.l1))
    assert s.VA == pytest.approx(np.sum(rho**2 * (est.s00 + 2 * est.s01 + est.s11)))
    assert s.VB == pytest.approx(
      np.sum(rho**2 * (est.s00 + 2 * (1 + cv) * est.s01 + (1 + cv) ** 2 * est.s11))
    )
    assert s.Cbar == pytest.approx(np.sum(rho * cv))

  def test_direction_flips_effects_but_not_variances(self):
    rng = np.random.default_rng(1)
    est = make_estimates(rng, 12)
    low = summarize(est, 0.5, direction="lower_is_better")
    high = summarize(est, 0.5, direction="higher_is_better")
    assert high.A == pytest.approx(-low.A)
    assert high.mu == pytest.approx(-low.mu)
    assert high.VA == pytest.approx(low.VA)
    assert high.tau2 == pytest.approx(low.tau2)

  def test_resolve_c_rejects_bare_arrays_and_missing_ids(self):
    with pytest.raises(TypeError, match="mapping keyed by cluster id"):
      resolve_c(np.array([0.5, 0.5]), ["a", "b"])
    with pytest.raises(KeyError, match="no C for 1 cluster"):
      resolve_c({"a": 0.5}, ["a", "b"])
    with pytest.raises(ValueError, match="non-negative"):
      resolve_c({"a": -0.5}, ["a"])
    assert resolve_c({1: 0.5, "extra": 9.0}, ["1"]).tolist() == [0.5]


class TestDerSimonianLaird:
  def test_equal_variances_reduce_to_sample_variance_minus_noise(self):
    rng = np.random.default_rng(2)
    x = rng.normal(0, 2.0, 50)
    tau2, _ = dl_tau2(x, np.full(50, 1.5))
    assert tau2 == pytest.approx(x.var(ddof=1) - 1.5)

  def test_estimate_is_not_floored_and_variance_uses_the_floor(self):
    x = np.array([0.01, -0.01, 0.0, 0.005])
    s2 = np.full(4, 1.0)
    tau2, variance = dl_tau2(x, s2)
    assert tau2 < 0
    assert variance == pytest.approx(dl_variance(0.0, s2))

  def test_unbiased_for_a_known_spread(self):
    rng = np.random.default_rng(3)
    s2 = rng.uniform(0.5, 2.0, 60)
    draws = [
      dl_tau2(rng.normal(0, 0.8, 60) + np.sqrt(s2) * rng.normal(size=60), s2)[0]
      for _ in range(2000)
    ]
    assert np.mean(draws) == pytest.approx(0.64, abs=0.03)

  def test_needs_two_clusters(self):
    with pytest.raises(ValueError, match="at least two clusters"):
      dl_tau2(np.array([1.0]), np.array([1.0]))


class TestTwoEndpoint:
  def test_same_sign_is_the_larger_of_the_two_ends(self):
    rng = np.random.default_rng(4)
    est = make_estimates(rng)
    s = summarize(est, 0.8, direction="lower_is_better")
    expected = max(s.A / np.sqrt(s.VA), (s.A + s.mu) / np.sqrt(s.VB))
    assert two_endpoint_z(s) == pytest.approx(expected)

  def test_either_sign_uses_both_bounds(self):
    rng = np.random.default_rng(5)
    est = make_estimates(rng)
    s = summarize(est, 0.8, direction="lower_is_better")
    v_minus = s.VA - 2 * s.CAmu + s.Vmu
    expected = max((s.A + s.mu) / np.sqrt(s.VB), (s.A - s.mu) / np.sqrt(v_minus))
    assert two_endpoint_z(s, tail_sign="either") == pytest.approx(expected)
    with pytest.raises(ValueError, match="tail_sign"):
      two_endpoint_z(s, tail_sign="both")

  def test_zero_bound_reduces_to_lag01(self):
    rng = np.random.default_rng(6)
    s = summarize(make_estimates(rng), 0.0, direction="lower_is_better")
    assert two_endpoint_z(s) == pytest.approx(lag01_z(s))


def _moment_bound_grid(s, t1, floor, n=20001):
  """Brute-force maximum of both moment-bound pieces over a grid of p."""
  tau2 = max(s.tau2, 0.0) if floor else s.tau2
  t = s.Cbar * t1
  k = s.Cbar**2 * tau2 / (2 * t) + t / 2
  a2 = s.Cbar**4 * s.var_tau2 / (4 * t * t)
  p = np.linspace(0.0, 1.0, n)
  root = np.sqrt(p * (1 - p))
  piece1 = (s.A + p * s.mu + k * root) / np.sqrt(
    s.VA + 2 * p * s.CAmu + p**2 * s.Vmu + a2 * p * (1 - p)
  )
  piece2 = (s.A + s.mu - p * s.mu + k * root) / np.sqrt(
    s.VB - 2 * p * (s.CAmu + s.Vmu) + p**2 * s.Vmu + a2 * p * (1 - p)
  )
  return max(piece1.max(), piece2.max())


class TestMomentBound:
  @pytest.mark.parametrize("seed", range(8))
  @pytest.mark.parametrize("floor", [False, True])
  def test_exact_maximum_matches_a_fine_grid(self, seed, floor):
    rng = np.random.default_rng(seed)
    est = make_estimates(rng, spread=rng.choice([0.0, 0.3]))
    s = summarize(est, c_map(est, rng), direction="lower_is_better")
    exact = moment_bound_z(s, t1=0.5, floor=floor)
    grid = _moment_bound_grid(s, 0.5, floor)
    assert exact >= grid - 1e-12
    assert exact == pytest.approx(grid, abs=1e-6)

  def test_both_pieces_are_mirror_images(self):
    """The second piece is the first under p -> 1 - p."""
    rng = np.random.default_rng(9)
    est = make_estimates(rng, spread=0.2)
    s = summarize(est, c_map(est, rng), direction="lower_is_better")
    swapped = dataclasses.replace(
      s, A=s.A + s.mu, mu=-s.mu, VA=s.VB, CAmu=-(s.CAmu + s.Vmu), VB=s.VA
    )
    assert moment_bound_z(swapped, 0.4) == pytest.approx(moment_bound_z(s, 0.4), abs=1e-12)

  def test_zero_bound_reduces_to_lag01(self):
    rng = np.random.default_rng(10)
    s = summarize(make_estimates(rng), 0.0, direction="lower_is_better")
    assert moment_bound_z(s, 0.5) == pytest.approx(lag01_z(s))
    assert moment_bound_z(s, 0.5, floor=True) == pytest.approx(lag01_z(s))

  @pytest.mark.parametrize("unit", [1e-4, 1e-2, 1e3])
  def test_every_statistic_is_unit_free(self, unit):
    """Regression: measuring the metric in another unit must not move any z.
    The legacy search for the moment bound's interior maximum compared raw
    polynomial coefficients with a fixed zero tolerance, so with small
    variances it looked only at p = 0 and p = 1 and rejected too easily."""
    rng = np.random.default_rng(23)
    est = make_estimates(rng, l0=-0.4, l1=0.1, spread=0.3)
    c = c_map(est, rng)
    scaled = dataclasses.replace(
      est, l0=est.l0 * unit, l1=est.l1 * unit,
      s00=est.s00 * unit**2, s01=est.s01 * unit**2, s11=est.s11 * unit**2,
    )
    base = run_tests(est, c, direction="lower_is_better", t1=0.4)
    other = run_tests(scaled, c, direction="lower_is_better", t1=0.4 * unit)
    for name in TESTS:
      assert other[name].z == pytest.approx(base[name].z, rel=1e-9, abs=1e-9)
      assert other[name].z_mirror == pytest.approx(base[name].z_mirror, rel=1e-9, abs=1e-9)
    # The case is one where the interior maximum decides the outcome.
    s = summarize(est, c, direction="lower_is_better")
    ends = max(s.A / np.sqrt(s.VA), (s.A + s.mu) / np.sqrt(s.VB))
    assert base["moment_bound"].z > ends + 0.05

  def test_validates_t1_and_cluster_count(self):
    rng = np.random.default_rng(11)
    s = summarize(make_estimates(rng), 0.5, direction="lower_is_better")
    with pytest.raises(ValueError, match="t1 must be positive"):
      moment_bound_z(s, 0.0)
    with pytest.raises(ValueError, match="at least two clusters"):
      moment_bound_z(dataclasses.replace(s, num_clusters=1), 0.5)


class TestFlooredMomentBound:
  """Test 4.5: the moment bound with the DerSimonian-Laird tau2 floored at zero."""

  def _summaries(self, n_wanted=30):
    rng = np.random.default_rng(12)
    out = []
    while len(out) < n_wanted:
      est = make_estimates(rng, spread=rng.choice([0.0, 0.0, 0.3]))
      out.append(summarize(est, c_map(est, rng), direction="lower_is_better"))
    return out

  def test_equals_test_4_when_tau2_is_not_negative(self):
    cases = [s for s in self._summaries() if s.tau2 >= 0]
    assert cases
    for s in cases:
      assert moment_bound_z(s, 0.5, floor=True) == moment_bound_z(s, 0.5, floor=False)

  def test_never_below_test_4_and_strictly_above_when_tau2_is_negative(self):
    cases = [s for s in self._summaries() if s.tau2 < 0]
    assert len(cases) >= 5
    for s in cases:
      floored = moment_bound_z(s, 0.5, floor=True)
      legacy = moment_bound_z(s, 0.5, floor=False)
      assert floored > legacy

  def test_equals_test_4_on_inputs_with_tau2_set_to_zero(self):
    for s in self._summaries():
      if s.tau2 < 0:
        zeroed = dataclasses.replace(s, tau2=0.0)
        assert moment_bound_z(s, 0.5, floor=True) == moment_bound_z(zeroed, 0.5, floor=False)

  def test_rejects_only_when_test_4_rejects(self):
    rng = np.random.default_rng(13)
    for _ in range(40):
      est = make_estimates(rng, l0=-0.6, l1=0.05)
      results = run_tests(
        est, 0.6, direction="lower_is_better", t1=0.3,
        tests=("moment_bound", "moment_bound_floored"),
      )
      if results["moment_bound_floored"].improves:
        assert results["moment_bound"].improves
      if results["moment_bound_floored"].worsens:
        assert results["moment_bound"].worsens


def _facet_reference(A, V0, B, dv):
  """O(J^2) reference: evaluate the threshold set at a multiplier between every
  pair of consecutive breakpoints, and the interior optimum between sets."""
  if A + np.maximum(B, 0).sum() >= 0:
    d = (B > 0).astype(float)
    return float((A + d @ B) / np.sqrt(V0 + d @ dv))
  with np.errstate(divide="ignore", invalid="ignore"):
    bp = np.where(dv != 0, B / dv, np.nan)
  pts = np.unique(bp[np.isfinite(bp) & (bp < 0)])
  if len(pts):
    lams = np.concatenate([[pts[0] - 1.0], (pts[:-1] + pts[1:]) / 2, [pts[-1] / 2]])
  else:
    lams = np.array([-1.0])
  sets = [(B - lam * dv) > 0 for lam in lams]
  best = -np.inf
  for i, S in enumerate(sets):
    d = S.astype(float)
    N, V = A + d @ B, V0 + d @ dv
    if V > 0:
      best = max(best, N / np.sqrt(V))
    if i + 1 < len(sets):
      changed = np.where(S != sets[i + 1])[0]
      sg = np.where(sets[i + 1][changed], 1.0, -1.0)
      b, v = float(np.sum(B[changed] * sg)), float(np.sum(dv[changed] * sg))
      if b != 0 and v != 0:
        t = np.clip((N * v - 2 * b * V) / (b * v), 0, 1)
        if V + t * v > 0:
          best = max(best, (N + t * b) / np.sqrt(V + t * v))
  return float(best)


class TestAssumptionFree:
  @pytest.mark.parametrize("seed", range(12))
  def test_sweep_matches_quadratic_reference(self, seed):
    rng = np.random.default_rng(100 + seed)
    est = make_estimates(rng, n=int(rng.integers(3, 80)), l0=-3.0, l1=rng.choice([-0.2, 0.0, 0.2]))
    s = summarize(est, c_map(est, rng), direction="lower_is_better")
    assert assumption_free_z(s) == pytest.approx(
      _facet_reference(s.A, s.VA, s.B, s.dv), abs=1e-12
    )

  def test_sweep_handles_non_positive_variance_increments(self):
    """dv can be zero (C = 0) or negative (strongly negative lag covariance)."""
    rng = np.random.default_rng(7)
    est = make_estimates(rng, n=30, l0=-3.0)
    s = summarize(est, c_map(est, rng), direction="lower_is_better")
    dv = s.dv.copy()
    dv[:5] = 0.0
    dv[5:10] *= -0.2
    s = dataclasses.replace(s, dv=dv)
    assert assumption_free_z(s) == pytest.approx(
      _facet_reference(s.A, s.VA, s.B, s.dv), abs=1e-12
    )

  def test_bounds_every_sign_pattern(self):
    rng = np.random.default_rng(8)
    est = make_estimates(rng, n=10, l0=-3.0)
    s = summarize(est, c_map(est, rng), direction="lower_is_better")
    z = assumption_free_z(s)
    for pattern in itertools.product([0.0, 1.0], repeat=10):
      d = np.array(pattern)
      assert (s.A + d @ s.B) / np.sqrt(s.VA + d @ s.dv) <= z + 1e-12
    for _ in range(2000):
      d = rng.random(10)
      assert (s.A + d @ s.B) / np.sqrt(s.VA + d @ s.dv) <= z + 1e-12

  def test_no_rejection_possible_returns_the_first_pattern(self):
    rng = np.random.default_rng(14)
    est = make_estimates(rng, n=10, l0=-0.05, l1=1.0)
    s = summarize(est, 2.0, direction="lower_is_better")
    assert s.A + np.maximum(s.B, 0).sum() >= 0
    d = (s.B > 0).astype(float)
    z = assumption_free_z(s)
    assert z == pytest.approx((s.A + d @ s.B) / np.sqrt(s.VA + d @ s.dv))
    assert z >= 0


class TestRunTests:
  def test_six_tests_in_order_with_numbers(self):
    rng = np.random.default_rng(15)
    est = make_estimates(rng)
    results = run_tests(est, 0.5, direction="lower_is_better", t1=0.4)
    assert tuple(r.name for r in results.results) == TESTS
    assert [r.number for r in results.results] == ["1", "2", "3", "4", "4.5", "5"]
    assert TEST_NUMBERS["moment_bound_floored"] == "4.5"
    assert results.t1 == 0.4

  def test_weaker_assumptions_never_reject_more(self):
    """z is ordered: lag-0+1 <= two-endpoint <= assumption-free, and both
    moment bounds sit between lag-0+1 and... at least lag-0+1."""
    rng = np.random.default_rng(16)
    for _ in range(30):
      est = make_estimates(rng, l0=-1.5, l1=rng.choice([-0.2, 0.2]))
      r = run_tests(est, c_map(est, rng), direction="lower_is_better", t1=0.4)
      assert r["lag01"].z <= r["two_endpoint"].z + 1e-12
      assert r["two_endpoint"].z <= r["assumption_free"].z + 1e-12
      assert r["lag01"].z <= r["moment_bound_floored"].z + 1e-12
      assert r["moment_bound"].z <= r["moment_bound_floored"].z

  def test_mirror_is_the_test_on_negated_estimates(self):
    rng = np.random.default_rng(17)
    est = make_estimates(rng)
    negated = dataclasses.replace(est, l0=-est.l0, l1=-est.l1)
    c = c_map(est, rng)
    forward = run_tests(est, c, direction="lower_is_better", t1=0.4)
    backward = run_tests(negated, c, direction="lower_is_better", t1=0.4)
    for name in TESTS:
      assert forward[name].z_mirror == pytest.approx(backward[name].z, abs=1e-12)
      assert forward[name].worsens == backward[name].improves

  def test_higher_is_better_swaps_improves_and_worsens(self):
    rng = np.random.default_rng(18)
    est = make_estimates(rng)
    low = run_tests(est, 0.5, direction="lower_is_better", t1=0.4)
    high = run_tests(est, 0.5, direction="higher_is_better", t1=0.4)
    for name in TESTS:
      assert high[name].z == pytest.approx(low[name].z_mirror, abs=1e-12)
      assert high[name].improves == low[name].worsens

  def test_decision_uses_one_sided_alpha(self):
    rng = np.random.default_rng(19)
    est = make_estimates(rng, l0=-0.9)
    r = run_tests(est, direction="lower_is_better", tests=("lag0",), alpha=0.05)
    assert r["lag0"].improves == (r["lag0"].z < -norm.ppf(0.95))

  def test_requirements_are_checked(self):
    rng = np.random.default_rng(20)
    est = make_estimates(rng)
    with pytest.raises(ValueError, match="t1 is required"):
      run_tests(est, 0.5, direction="lower_is_better")
    with pytest.raises(ValueError, match="c_window is required"):
      run_tests(est, direction="lower_is_better", tests=("lag0", "two_endpoint"))
    with pytest.raises(ValueError, match="unknown test"):
      run_tests(est, 0.5, direction="lower_is_better", tests=("lag2",))
    with pytest.raises(ValueError, match="direction"):
      run_tests(est, 0.5, direction="down", tests=("lag0",))
    assert run_tests(est, direction="lower_is_better", tests=("lag0", "lag01")).t1 is None

  def test_either_sign_is_recorded(self):
    rng = np.random.default_rng(21)
    est = make_estimates(rng)
    r = run_tests(est, 0.5, direction="lower_is_better", tests=("two_endpoint",), tail_sign="either")
    assert r.warnings == ("two_endpoint: either-sign tail",)

  @pytest.mark.parametrize("name", ["moment_bound", "moment_bound_floored"])
  def test_moment_bound_size_at_the_null_boundary(self, name):
    """With no heterogeneity and the total effect exactly zero at the bound
    (A + mu = 0), the rejection rate must not exceed alpha."""
    rng = np.random.default_rng(22)
    c, l1 = 0.6, 0.25
    rejections = 0
    n_sim = 1500
    for _ in range(n_sim):
      est = make_estimates(rng, n=60, l0=-(1 + c) * l1, l1=l1, noise=0.4)
      r = run_tests(est, c, direction="lower_is_better", tests=(name,), t1=0.1)
      rejections += r[name].improves
    assert rejections / n_sim <= 0.10 + 3 * np.sqrt(0.1 * 0.9 / n_sim)


class TestT1:
  def test_leave_one_out_excludes_own_experiment(self):
    tau2 = [1.0, 4.0, -2.0]
    var = [1.0, 1.0, 1.0]
    t1 = t1_leave_one_out(tau2, var)
    assert t1[0] == pytest.approx(np.sqrt(1.0 + np.sqrt(0.5)))
    assert t1[1] == pytest.approx(np.sqrt(0.0 + np.sqrt(0.5)))  # pooled -0.5 floored at 0
    assert t1[2] == pytest.approx(np.sqrt(2.5 + np.sqrt(0.5)))
    with pytest.raises(ValueError, match="at least two experiments"):
      t1_leave_one_out([1.0], [1.0])

  def test_planning_t1_applies_the_same_formula(self):
    v11 = np.full(50, 0.04)
    expected = np.sqrt(0.09 + np.sqrt(dl_variance(0.09, v11)))
    assert planning_t1(0.3, v11) == pytest.approx(expected)
    assert planning_t1(0.0, v11) > 0


def _panel(rng, n_clusters=60, n_windows=168, l0=0.0, l1=0.0):
  """Hourly window totals with AR(1) shocks and a fair coin per cluster-window."""
  size = rng.lognormal(3.0, 0.5, n_clusters)
  count = rng.poisson(size[:, None] * np.ones(n_windows)).astype(float)
  shock = np.zeros((n_clusters, n_windows))
  shock[:, 0] = rng.normal(0, 1.0, n_clusters)
  for t in range(1, n_windows):
    shock[:, t] = 0.5 * shock[:, t - 1] + np.sqrt(0.75) * rng.normal(0, 1.0, n_clusters)
  noise = np.sqrt(count) * rng.normal(0, 5.0, (n_clusters, n_windows))
  treated = (rng.random((n_clusters, n_windows)) < 0.5).astype(float)
  previous = np.zeros_like(treated)
  previous[:, 1:] = treated[:, :-1]
  residual = count * (shock + l0 * treated + l1 * previous) + noise
  return count, residual, treated


def _frame(count, residual, treated):
  n_clusters, n_windows = count.shape
  return pd.DataFrame({
    "cluster_id": np.repeat([f"c{i:03d}" for i in range(n_clusters)], n_windows),
    "window": np.tile(np.arange(n_windows), n_clusters),
    "residual": residual.ravel(),
    "count": count.ravel(),
    "treated": treated.ravel(),
  })


def _matrix_reference(W, r, c):
  """Lag estimator for a complete panel, written directly on matrices."""
  W1, r1, c1 = W[:, :-1], r[:, 1:], c[:, 1:]
  Bt, D = (W * c).sum(1), ((1 - W) * c).sum(1)
  B1, D1 = (W1 * c1).sum(1), ((1 - W1) * c1).sum(1)
  muT, muC = (W * r).sum(1) / Bt, ((1 - W) * r).sum(1) / D
  muT1, muC1 = (W1 * r1).sum(1) / B1, ((1 - W1) * r1).sum(1) / D1
  psi0 = (r - muT[:, None] * c) / Bt[:, None] + (r - muC[:, None] * c) / D[:, None]
  psi1 = (r1 - muT1[:, None] * c1) / B1[:, None] + (r1 - muC1[:, None] * c1) / D1[:, None]
  return dict(
    l0=muT - muC, l1=muT1 - muC1, s00=0.25 * (psi0**2).sum(1),
    s11=0.25 * (psi1**2).sum(1), s01=0.25 * (psi0[:, :-1] * psi1).sum(1),
  )


class TestEstimateLags:
  def test_complete_panel_matches_matrix_reference(self):
    rng = np.random.default_rng(30)
    count, residual, treated = _panel(rng)
    est = estimate_lags(_frame(count, residual, treated), window_minutes=60)
    ref = _matrix_reference(treated, residual, count)
    for name, values in ref.items():
      np.testing.assert_allclose(getattr(est, name), values, rtol=1e-12)
    np.testing.assert_allclose(est.weights, count.sum(1) / count.sum(), rtol=1e-12)
    assert est.weights.sum() == pytest.approx(1.0)
    assert est.n_dropped == 0 and est.warnings == ()

  def test_recovers_injected_lag_effects(self):
    rng = np.random.default_rng(31)
    count, residual, treated = _panel(rng, n_clusters=300, l0=-1.0, l1=0.4)
    est = estimate_lags(_frame(count, residual, treated), window_minutes=60)
    se0 = np.sqrt(np.sum(est.weights**2 * est.s00))
    se1 = np.sqrt(np.sum(est.weights**2 * est.s11))
    assert np.sum(est.weights * est.l0) == pytest.approx(-1.0, abs=4 * se0)
    assert np.sum(est.weights * est.l1) == pytest.approx(0.4, abs=4 * se1)

  def test_covariance_matches_rerandomization(self):
    """Hold outcomes fixed, redraw the coins: the spread of the aggregate
    estimates must match the reported design-based covariance."""
    rng = np.random.default_rng(32)
    count, residual, _ = _panel(rng, n_clusters=40, n_windows=336)
    draws = []
    for _ in range(600):
      treated = (rng.random(count.shape) < 0.5).astype(float)
      ref = _matrix_reference(treated, residual, count)
      rho = count.sum(1) / count.sum()
      draws.append((np.sum(rho * ref["l0"]), np.sum(rho * ref["l1"])))
    draws = np.array(draws)
    est = estimate_lags(
      _frame(count, residual, (rng.random(count.shape) < 0.5).astype(float)), window_minutes=60
    )
    rho = est.weights
    assert np.sqrt(np.sum(rho**2 * est.s00)) == pytest.approx(draws[:, 0].std(ddof=1), rel=0.10)
    assert np.sqrt(np.sum(rho**2 * est.s11)) == pytest.approx(draws[:, 1].std(ddof=1), rel=0.10)
    corr = np.sum(rho**2 * est.s01) / np.sqrt(np.sum(rho**2 * est.s00) * np.sum(rho**2 * est.s11))
    assert corr == pytest.approx(np.corrcoef(draws.T)[0, 1], abs=0.10)

  def test_unknown_assignment_and_absent_windows_break_lag_pairs(self):
    rng = np.random.default_rng(33)
    count, residual, treated = _panel(rng, n_clusters=3, n_windows=60)
    frame = _frame(count, residual, treated)
    frame.loc[(frame.cluster_id == "c000") & (frame.window == 20), "treated"] = np.nan
    frame = frame[~((frame.cluster_id == "c001") & (frame.window == 30))]
    est = estimate_lags(frame, window_minutes=60, min_windows_per_arm=1)

    for cluster, hole in (("c000", 20), ("c001", 30)):
      i = int(cluster[1:])
      keep = np.arange(60) != hole
      W, r, c = treated[i, keep], residual[i, keep], count[i, keep]
      l0 = (W * r).sum() / (W * c).sum() - ((1 - W) * r).sum() / ((1 - W) * c).sum()
      pairs = [t for t in range(59) if t != hole and t + 1 != hole]
      Wp, rp, cp = treated[i, pairs], residual[i, [t + 1 for t in pairs]], count[i, [t + 1 for t in pairs]]
      l1 = (Wp * rp).sum() / (Wp * cp).sum() - ((1 - Wp) * rp).sum() / ((1 - Wp) * cp).sum()
      k = est.cluster_ids.index(cluster)
      assert est.l0[k] == pytest.approx(l0)
      assert est.l1[k] == pytest.approx(l1)
      assert est.orders[k] == pytest.approx(c.sum())

  def test_clusters_with_too_few_windows_per_arm_are_dropped_and_counted(self):
    rng = np.random.default_rng(34)
    count, residual, treated = _panel(rng, n_clusters=5, n_windows=40)
    treated[0, :] = 0.0
    treated[0, :3] = 1.0
    est = estimate_lags(_frame(count, residual, treated), window_minutes=60, min_windows_per_arm=5)
    assert "c000" not in est.cluster_ids
    assert est.n_input_clusters == 5 and est.n_dropped == 1
    assert est.dropped_orders_share == pytest.approx(count[0].sum() / count.sum())
    assert len(est.warnings) == 1 and "fewer than 5 windows" in est.warnings[0]
    assert est.weights.sum() == pytest.approx(1.0)

  def test_input_validation(self):
    rng = np.random.default_rng(35)
    frame = _frame(*_panel(rng, n_clusters=3, n_windows=30))
    with pytest.raises(KeyError, match="Missing required columns"):
      estimate_lags(frame.drop(columns="treated"), window_minutes=60)
    with pytest.raises(ValueError, match="at most one row per window"):
      estimate_lags(pd.concat([frame, frame.iloc[:1]]), window_minutes=60)
    bad = frame.copy()
    bad.loc[0, "treated"] = 2.0
    with pytest.raises(ValueError, match="must be 1"):
      estimate_lags(bad, window_minutes=60)
    with pytest.raises(ValueError, match="min_windows_per_arm"):
      estimate_lags(frame, window_minutes=60, min_windows_per_arm=0)
    with pytest.raises(ValueError, match="no cluster has enough windows"):
      estimate_lags(frame, window_minutes=60, min_windows_per_arm=100)


def _hourly_frames(rng, n_clusters=8, n_hours=96, block=1):
  """Base-period outcome and assignment frames, randomized in blocks of `block` hours."""
  count, residual, _ = _panel(rng, n_clusters=n_clusters, n_windows=n_hours)
  coins = (rng.random((n_clusters, n_hours // block)) < 0.5).astype(float)
  treated = np.repeat(coins, block, axis=1)
  start = pd.Timestamp("2030-01-07", tz="UTC")
  times = start + pd.to_timedelta(np.arange(n_hours), unit="h")
  ids = [f"c{i:03d}" for i in range(n_clusters)]
  outcomes = pd.DataFrame({
    "cluster_id": np.repeat(ids, n_hours),
    "period_start": np.tile(times, n_clusters),
    "numerator": (residual + 100.0 * count).ravel(),
    "denominator": count.ravel(),
    "expected": (100.0 * count).ravel(),
  })
  assignment = outcomes[["cluster_id", "period_start"]].assign(treated=treated.ravel())
  return outcomes, assignment, count, residual, treated, start


class TestExperimentWindows:
  def test_hourly_experiment_round_trip(self):
    rng = np.random.default_rng(50)
    outcomes, assignment, count, residual, treated, _ = _hourly_frames(rng)
    windows = experiment_windows(outcomes, assignment, window_minutes=60, expected_col="expected")
    assert list(windows.columns) == ["cluster_id", "window", "residual", "count", "treated"]
    est = estimate_lags(windows, window_minutes=60)
    ref = _matrix_reference(treated, residual, count)
    np.testing.assert_allclose(est.l0, ref["l0"], rtol=1e-9)
    np.testing.assert_allclose(est.s01, ref["s01"], rtol=1e-9)

  def test_four_hour_windows_sum_their_hours(self):
    rng = np.random.default_rng(51)
    outcomes, assignment, count, residual, treated, start = _hourly_frames(rng, block=4)
    windows = experiment_windows(
      outcomes, assignment, window_minutes=240, origin=start, expected_col="expected"
    )
    assert len(windows) == 8 * 24
    first = windows[(windows.cluster_id == "c000") & (windows.window == 0)].iloc[0]
    assert first["count"] == pytest.approx(count[0, :4].sum())
    assert first["residual"] == pytest.approx(residual[0, :4].sum())
    assert first["treated"] == treated[0, 0]
    est = estimate_lags(windows, window_minutes=240, min_windows_per_arm=2)
    block_count = count.reshape(8, 24, 4).sum(2)
    block_residual = residual.reshape(8, 24, 4).sum(2)
    ref = _matrix_reference(treated[:, ::4], block_residual, block_count)
    np.testing.assert_allclose(est.l1, ref["l1"], rtol=1e-9)

  def test_assignment_starting_later_keeps_the_outcome_grid(self):
    """The assignment frame must land on the outcomes' window grid even when
    its own first row is later."""
    rng = np.random.default_rng(52)
    outcomes, assignment, *_ = _hourly_frames(rng)
    late = assignment[assignment.period_start >= assignment.period_start.min() + pd.Timedelta(hours=5)]
    windows = experiment_windows(outcomes, late, window_minutes=60, expected_col="expected")
    unknown = windows[windows.treated.isna()]
    assert set(unknown.window) == set(range(5))
    full = experiment_windows(outcomes, assignment, window_minutes=60, expected_col="expected")
    merged = windows.merge(full, on=["cluster_id", "window"], suffixes=("", "_full"))
    known = merged[merged.treated.notna()]
    assert (known.treated == known.treated_full).all()

  def test_mixed_arms_within_a_window_raise(self):
    rng = np.random.default_rng(53)
    outcomes, assignment, _, _, _, start = _hourly_frames(rng, block=1)
    with pytest.raises(ValueError, match="hold both arms"):
      experiment_windows(outcomes, assignment, window_minutes=240, origin=start)

  def test_missing_assignment_is_unknown_not_filled(self):
    rng = np.random.default_rng(54)
    outcomes, assignment, *_ = _hourly_frames(rng)
    assignment = assignment.copy()
    assignment.loc[3, "treated"] = np.nan
    windows = experiment_windows(outcomes, assignment, window_minutes=60)
    assert windows["treated"].isna().sum() == 1
    with pytest.raises(ValueError, match="must be 1"):
      experiment_windows(outcomes, assignment.assign(treated=2.0), window_minutes=60)


class TestTailBound:
  def test_window_rule_is_geometric(self):
    bound = TailBound(c=1.0, source="example")
    ids = ["a", "b"]
    assert bound.window_values(ids, 60).tolist() == [1.0, 1.0]
    np.testing.assert_allclose(bound.window_values(ids, 120), 0.25 / 0.75)
    np.testing.assert_allclose(bound.window_values(ids, 240), 0.5**4 / (1 - 0.5**4))

  @pytest.mark.parametrize("k", [2, 4, 8])
  def test_window_rule_bounds_every_admissible_tail(self, k):
    """Any lag sequence with one sign that fades by at least g per period,
    under any mix of orders within the window, has a window-level tail of at
    most C_k times its window-level lag-1 effect; geometric fading attains it."""
    rng = np.random.default_rng(40 + k)
    c = 0.8
    g = c / (1 + c)
    c_k = TailBound(c=c, source="example").window_values(["a"], 60 * k)[0]
    n_lags = 12 * k
    worst = 0.0
    for trial in range(300):
      ratios = g * rng.random(n_lags) if trial else np.full(n_lags, g)
      lags = np.concatenate([[1.0], np.cumprod(ratios)])  # lags 1, 2, ... of a unit lag-1
      share = rng.dirichlet(np.ones(k))
      # an order at position h feels lags h+1 .. h+k from the previous window
      # and everything beyond from earlier windows
      window_l1 = sum(share[h] * lags[h:h + k].sum() for h in range(k))
      tail = sum(share[h] * lags[h + k:].sum() for h in range(k))
      worst = max(worst, tail / window_l1)
      assert tail / window_l1 <= c_k + 1e-9
    assert worst == pytest.approx(c_k, rel=1e-3)

  def test_from_gamma_and_per_cluster_values(self):
    bound = TailBound.from_gamma({"a": 0.5, "b": 0.2}, source="fit", stress_gamma=0.6)
    np.testing.assert_allclose(bound.base_values(["b", "a"]), [0.25, 1.0])
    np.testing.assert_allclose(bound.decay_values(["a", "b"]), [0.5, 0.2])
    np.testing.assert_allclose(bound.base_values(["a"], stress=True), [1.5])
    with pytest.raises(KeyError, match="no C for 1 cluster"):
      bound.base_values(["a", "z"])
    with pytest.raises(ValueError, match="gamma must lie"):
      TailBound.from_gamma(1.0, source="fit")

  def test_missing_mean_fills_unknown_clusters(self):
    bound = TailBound(c={"a": 1.0, "b": 3.0}, source="fit", missing="mean")
    np.testing.assert_allclose(bound.base_values(["a", "z"]), [1.0, 2.0])

  def test_scenario_puts_the_tail_at_the_bound(self):
    bound = TailBound(c=0.43, source="example")
    scenario = bound.scenario(-0.10, 0.3)
    assert scenario.tail_decay == pytest.approx(0.43 / 1.43)
    assert scenario.l1 == pytest.approx(0.03)
    half = bound.scenario(-0.10, 0.3, tail_share=0.5)
    assert half.tail_decay == pytest.approx(0.215 / 1.215)

  def test_validation_and_fingerprint(self):
    with pytest.raises(ValueError, match="source"):
      TailBound(c=0.5, source="")
    with pytest.raises(ValueError, match="non-negative"):
      TailBound(c=-0.1, source="x")
    with pytest.raises(ValueError, match="no stress_c"):
      TailBound(c=0.5, source="x").base_values(["a"], stress=True)
    assert TailBound(c=0.5, source="x").fingerprint() == "scalar:0.5"
    per_cluster = TailBound(c={"a": 0.5, "b": 1.0}, source="x")
    assert len(per_cluster.fingerprint()) == 12
    assert per_cluster.fingerprint() == TailBound(c={"b": 1.0, "a": 0.5}, source="y").fingerprint()
