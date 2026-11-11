"""Tests for closed-form power, simulated power and the design comparison."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal, norm

from switchback_power_calculator.carryover import analysis
from switchback_power_calculator.carryover.examples import synthetic_panel
from switchback_power_calculator.carryover.moments import lag_moments_from_aggregates
from switchback_power_calculator.carryover.plan import compare_designs
from switchback_power_calculator.carryover.power import (
    closed_form_power,
    lag_coefficients,
    mde,
    phi2,
    rebound_curve,
    scenario_grid,
    total_effect,
    weeks_to_power,
    window_l1_spread_sd,
    window_means,
)
from switchback_power_calculator.carryover.simulate import (
    _builtin_statistics,
    _draw_estimates,
    cluster_normals,
    simulated_power,
    simulated_weeks,
    spread_sensitivity,
)
from switchback_power_calculator.carryover.tail import TailBound
from switchback_power_calculator.carryover.types import LagEstimates, Scenario, orientation

LOWER = "lower_is_better"


@pytest.fixture(scope="module")
def panel():
  return synthetic_panel(120, 2, seed=5)


@pytest.fixture(scope="module")
def hourly(panel):
  return lag_moments_from_aggregates(panel, expected_col="expected")


@pytest.fixture(scope="module")
def four_hourly(panel):
  return lag_moments_from_aggregates(
    panel, expected_col="expected", window_minutes=240, origin=panel["period_start"].min()
  )


@pytest.fixture(scope="module")
def bound():
  return TailBound(c=0.43, source="example", stress_c=0.8)


@pytest.fixture(scope="module")
def target(bound):
  """Same-period effect -0.10, 30% undone next period, later carryover at the bound."""
  return bound.scenario(-0.10, 0.3, l1_spread_sd=0.05, label="target")


class TestLagCoefficients:
  def test_one_period_windows(self):
    a0, a1, tail = lag_coefficients(np.ones((3, 1)), np.array([0.0, 0.3, 0.8]))
    np.testing.assert_allclose(a0, 0.0)
    np.testing.assert_allclose(a1, 1.0)
    np.testing.assert_allclose(tail, [0.0, 0.3 / 0.7, 4.0])

  def test_no_decay_splits_lag_one_by_position(self):
    share = np.array([[0.4, 0.3, 0.2, 0.1]])
    a0, a1, tail = lag_coefficients(share, np.array([0.0]))
    assert a0[0] == pytest.approx(0.6)   # every position but the first feels lag 1 in its own window
    assert a1[0] == pytest.approx(0.4)   # only the first position feels it from the previous window
    assert tail[0] == 0.0

  @pytest.mark.parametrize("k", [1, 2, 4, 8])
  def test_pieces_add_up_to_the_total(self, k):
    rng = np.random.default_rng(k)
    share = rng.dirichlet(np.ones(k), size=20)
    decay = rng.uniform(0.0, 0.9, 20)
    a0, a1, tail = lag_coefficients(share, decay)
    np.testing.assert_allclose(a0 + a1 * (1 + tail), 1 / (1 - decay), rtol=1e-12)

  def test_total_is_the_same_at_every_window(self, hourly, four_hourly, target):
    one = hourly.plan(2)
    four = four_hourly.plan(2)
    assert total_effect(one, target) == pytest.approx(-0.10 + 0.03 / (1 - 0.43 / 1.43))
    L0, L1, tail = window_means(four, target)
    assert np.sum(four.weights * (L0 + L1 + tail)) == pytest.approx(total_effect(one, target), rel=1e-3)

  def test_base_period_must_match(self, hourly):
    scenario = Scenario("x", -0.1, 0.0, tail_decay=0.0, base_minutes=30.0)
    with pytest.raises(ValueError, match="base period"):
      window_means(hourly.plan(1), scenario)


class TestPhi2:
  @pytest.mark.parametrize("seed", range(6))
  def test_matches_bivariate_normal_cdf(self, seed):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(0, 1.5, 2)
    r = rng.uniform(-0.95, 0.95)
    reference = multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([a, b])
    assert phi2(a, b, r) == pytest.approx(reference, abs=2e-4)

  def test_limits(self):
    assert phi2(0.3, -0.4, 0.0) == pytest.approx(norm.cdf(0.3) * norm.cdf(-0.4))
    assert phi2(0.3, -0.4, 1.0) == pytest.approx(norm.cdf(-0.4))
    assert phi2(0.3, -0.4, -1.0) == pytest.approx(max(0.0, norm.cdf(0.3) - norm.cdf(0.4)))
    assert phi2(0.0, 0.0, 0.5) == pytest.approx(0.25 + math.asin(0.5) / (2 * math.pi))
    assert phi2(1.0, 1.2, 1 - 1e-9) == pytest.approx(norm.cdf(1.0), abs=1e-4)


class TestClosedFormPower:
  def test_lag0_matches_hand_computation(self, hourly, target):
    cov = hourly.plan(2)
    expected = norm.cdf(-norm.ppf(0.9) + 0.10 / cov.se_l0)
    assert closed_form_power(cov, "lag0", target, direction=LOWER) == pytest.approx(expected)

  def test_zero_effect_gives_alpha(self, hourly):
    null = Scenario("null", 0.0, 0.0, tail_decay=0.0)
    cov = hourly.plan(2)
    assert closed_form_power(cov, "lag0", null, direction=LOWER, alpha=0.05) == pytest.approx(0.05)
    assert closed_form_power(cov, "lag01", null, direction=LOWER, alpha=0.05) == pytest.approx(0.05)

  def test_power_grows_with_duration_and_falls_with_noise(self, hourly, target, bound):
    for test in ("lag0", "lag01", "two_endpoint"):
      powers = [
        closed_form_power(hourly.plan(w), test, target, direction=LOWER, tail_bound=bound)
        for w in (1, 2, 4, 8)
      ]
      assert powers == sorted(powers)
      noisy = closed_form_power(
        hourly.plan(4, se_inflation=1.3), test, target, direction=LOWER, tail_bound=bound
      )
      assert noisy < powers[2]

  def test_two_endpoint_is_bounded_by_its_two_contrasts(self, hourly, target, bound):
    cov = hourly.plan(6)
    z_alpha = norm.ppf(0.9)
    L0, L1, _ = window_means(cov, target)
    c = bound.window_values(cov.cluster_ids, 60)
    at_none = norm.cdf(-z_alpha - np.sum(cov.weights * (L0 + L1)) / cov.contrast_se(1.0))
    at_bound = norm.cdf(-z_alpha - np.sum(cov.weights * (L0 + (1 + c) * L1)) / cov.contrast_se(1 + c))
    joint = closed_form_power(cov, "two_endpoint", target, direction=LOWER, tail_bound=bound)
    assert joint <= min(at_none, at_bound) + 1e-12
    assert joint >= at_none + at_bound - 1 - 1e-12
    assert closed_form_power(cov, "lag01", target, direction=LOWER) == pytest.approx(at_none)

  def test_stress_bound_costs_power(self, hourly, target, bound):
    cov = hourly.plan(6)
    planning = closed_form_power(cov, "two_endpoint", target, direction=LOWER, tail_bound=bound)
    stress = closed_form_power(
      cov, "two_endpoint", target, direction=LOWER, tail_bound=bound, stress=True
    )
    assert stress < planning

  def test_direction_mirrors(self, hourly, target):
    cov = hourly.plan(2)
    flipped = Scenario("flipped", -target.l0, -target.l1, tail_decay=target.tail_decay)
    assert closed_form_power(cov, "lag01", flipped, direction="higher_is_better") == pytest.approx(
      closed_form_power(cov, "lag01", target, direction=LOWER)
    )

  def test_requirements(self, hourly, target):
    cov = hourly.plan(2)
    with pytest.raises(ValueError, match="needs a tail_bound"):
      closed_form_power(cov, "two_endpoint", target, direction=LOWER)
    with pytest.raises(ValueError, match="no closed form"):
      closed_form_power(cov, "moment_bound", target, direction=LOWER)
    with pytest.raises(ValueError, match="alpha is one-sided"):
      closed_form_power(cov, "lag0", target, direction=LOWER, alpha=0.6)


class TestMde:
  @pytest.mark.parametrize("test", ["lag0", "lag01", "two_endpoint"])
  @pytest.mark.parametrize("share", [-0.25, 0.0, 0.3])
  def test_power_at_the_mde_is_the_target(self, hourly, bound, test, share):
    cov = hourly.plan(3)
    result = mde(
      cov, test, rebound_share=share, tail_decay=0.3, direction=LOWER, tail_bound=bound
    )
    assert result.reachable
    at_mde = Scenario.from_rebound_share(-result.mde_same_period, share, tail_decay=0.3)
    assert closed_form_power(
      cov, test, at_mde, direction=LOWER, tail_bound=bound
    ) == pytest.approx(0.80, abs=1e-6)
    assert result.total_effect_at_mde == pytest.approx(total_effect(cov, at_mde))

  def test_limit_shares_at_one_period_windows(self, hourly, bound):
    cov = hourly.plan(3)
    kwargs = dict(rebound_share=0.2, tail_decay=0.3, direction=LOWER, tail_bound=bound)
    assert math.isinf(mde(cov, "lag0", **kwargs).limit_share)
    assert mde(cov, "lag01", **kwargs).limit_share == pytest.approx(1.0)
    assert mde(cov, "two_endpoint", **kwargs).limit_share == pytest.approx(1 / 1.43)
    assert mde(cov, "two_endpoint", stress=True, **kwargs).limit_share == pytest.approx(1 / 1.8)

  def test_unreachable_past_the_limit_share(self, hourly, bound):
    result = mde(
      hourly.plan(3), "two_endpoint", rebound_share=0.75, tail_decay=0.3,
      direction=LOWER, tail_bound=bound,
    )
    assert not result.reachable and math.isnan(result.mde_same_period)
    assert not result.total_improves

  def test_total_can_stop_improving_before_the_limit(self, hourly):
    """With decay 0.8 the total changes sign from share 0.2, while lag-0 + lag-1
    stays reachable up to share 1."""
    result = mde(hourly.plan(3), "lag01", rebound_share=0.3, tail_decay=0.8, direction=LOWER)
    assert result.reachable
    assert not result.total_improves
    assert result.total_effect_at_mde > 0

  def test_longer_windows_can_give_lag0_a_finite_limit(self, four_hourly):
    result = mde(four_hourly.plan(3), "lag0", rebound_share=0.2, tail_decay=0.3, direction=LOWER)
    assert 1.0 < result.limit_share < math.inf

  def test_total_range_brackets_the_tail(self, hourly, bound):
    result = mde(
      hourly.plan(3), "two_endpoint", rebound_share=0.3, tail_decay=0.43 / 1.43,
      direction=LOWER, tail_bound=bound,
    )
    m = result.mde_same_period
    assert result.total_range_lo == pytest.approx(-m * (1 - 0.3 * 1.43))
    assert result.total_range_hi == pytest.approx(-m * 0.7)
    assert result.total_effect_at_mde == pytest.approx(result.total_range_lo)

  def test_higher_is_better_reports_a_positive_total(self, hourly):
    result = mde(hourly.plan(3), "lag0", rebound_share=0.0, tail_decay=0.0, direction="higher_is_better")
    assert result.total_effect_at_mde == pytest.approx(result.mde_same_period)
    assert result.total_improves

  def test_rebound_curve_and_scenario_grid(self, hourly, bound):
    curve = rebound_curve(
      hourly.plan(2), ("lag0", "lag01", "two_endpoint"), tail_decay=0.3,
      direction=LOWER, shares=(0.0, 0.3), tail_bound=bound,
    )
    assert len(curve) == 6
    assert list(curve.columns[:3]) == ["test", "rebound_share", "tail_decay"]
    grid = scenario_grid([-0.1, -0.2], [0.0, 0.3], tail_decay=0.3)
    assert len(grid) == 4 and len({s.label for s in grid}) == 4


class TestWeeksToPower:
  def test_power_at_the_reported_duration_is_the_target(self, hourly, target, bound):
    result = weeks_to_power(hourly, "two_endpoint", target, direction=LOWER, tail_bound=bound)
    assert result.note == "reached"
    power = closed_form_power(
      hourly.plan(result.weeks), "two_endpoint", target, direction=LOWER, tail_bound=bound
    )
    assert power == pytest.approx(0.80, abs=1e-5)
    assert result.weeks_runnable == math.ceil(result.weeks)
    assert result.power_at_runnable >= 0.80

  def test_notes(self, hourly, target):
    big = Scenario("big", -5.0, 0.0, tail_decay=0.0)
    assert weeks_to_power(hourly, "lag0", big, direction=LOWER).note == "at_or_below_one_week"
    tiny = Scenario("tiny", -1e-4, 0.0, tail_decay=0.0)
    assert weeks_to_power(hourly, "lag0", tiny, direction=LOWER).note == "beyond_max_weeks"
    wrong_way = Scenario("wrong", 0.1, 0.0, tail_decay=0.0)
    result = weeks_to_power(hourly, "lag0", wrong_way, direction=LOWER)
    assert result.note == "not_reachable" and math.isinf(result.weeks) and result.weeks_runnable is None

  def test_noise_budget_lengthens_the_test(self, hourly, target):
    base = weeks_to_power(hourly, "lag01", target, direction=LOWER)
    noisy = weeks_to_power(hourly, "lag01", target, direction=LOWER, se_inflation=1.2)
    assert noisy.weeks == pytest.approx(base.weeks * 1.44, rel=0.05)


def _run(est, name, c, t1):
  return analysis.run_tests(est, c, direction=LOWER, tests=(name,), t1=t1)[name].z


class TestSimulation:
  def test_draws_are_keyed_by_cluster_id(self):
    a = cluster_normals(7, ("x", "y", "z"), 50)
    b = cluster_normals(7, ("z", "x"), 80)
    np.testing.assert_array_equal(a[:, 2, :], b[:50, 0, :])
    np.testing.assert_array_equal(a[:, 0, :], b[:50, 1, :])
    assert not np.array_equal(a, cluster_normals(8, ("x", "y", "z"), 50))

  @pytest.mark.parametrize("name", analysis.TESTS)
  def test_fast_path_equals_the_analysis_kernel(self, hourly, target, bound, name):
    """The simulator's vectorised statistics must be the z that run_tests gives
    on the same simulated estimates."""
    cov = hourly.plan(4)
    normals = cluster_normals(3, cov.cluster_ids, 12)
    l0, l1 = _draw_estimates(cov, target, normals, 1.0)
    c = bound.window_values(cov.cluster_ids, cov.window_minutes)
    t1 = 0.07
    fast = _builtin_statistics(cov, name, l0, l1, c, t1, LOWER)
    for r in range(12):
      est = LagEstimates(
        cluster_ids=cov.cluster_ids, weights=cov.weights, l0=l0[r], l1=l1[r],
        s00=cov.v00, s01=cov.v01, s11=cov.v11, window_minutes=cov.window_minutes,
      )
      assert fast[r] == pytest.approx(_run(est, name, dict(zip(cov.cluster_ids, c)), t1), abs=1e-10)

  @pytest.mark.parametrize("name", ["lag0", "lag01", "two_endpoint"])
  def test_simulation_agrees_with_closed_form(self, hourly, bound, name):
    scenario = bound.scenario(-0.10, 0.3, l1_spread_sd=0.0, label="no spread")
    cov = hourly.plan(4)
    closed = closed_form_power(cov, name, scenario, direction=LOWER, tail_bound=bound)
    sim = simulated_power(
      cov, name, scenario, direction=LOWER, tail_bound=bound, n_draws=4000, seed=1
    )
    assert 0.05 < closed < 0.999
    assert sim.power == pytest.approx(closed, abs=4 * sim.mc_se)
    assert sim.n_failed == 0

  def test_custom_statistic_matches_the_built_in(self, hourly, target):
    cov = hourly.plan(4)

    def statistic(est):
      return analysis.run_tests(est, direction=LOWER, tests=("lag01",))["lag01"].z

    custom = simulated_power(cov, statistic, target, direction=LOWER, n_draws=200, seed=2)
    builtin = simulated_power(cov, "lag01", target, direction=LOWER, n_draws=200, seed=2)
    assert custom.power == builtin.power

  def test_floored_moment_bound_never_has_more_power(self, hourly):
    """With no true spread the unfloored tau2 is negative in about half the
    draws. The floor then costs power wherever the spread penalty matters:
    a large C and a test that is not yet long."""
    wide = TailBound(c=1.5, source="example")
    scenario = wide.scenario(-0.10, 0.1, l1_spread_sd=0.0, label="no spread")
    for weeks in (4, 10):
      cov = hourly.plan(weeks)
      kwargs = dict(direction=LOWER, tail_bound=wide, n_draws=800, seed=4)
      legacy = simulated_power(cov, "moment_bound", scenario, **kwargs)
      floored = simulated_power(cov, "moment_bound_floored", scenario, **kwargs)
      assert floored.power < legacy.power

  def test_wider_true_spread_costs_the_moment_bound_power(self, hourly, target, bound):
    table = spread_sensitivity(
      hourly, "moment_bound_floored", target, weeks=12, direction=LOWER,
      tail_bound=bound, n_draws=400, seed=5, factors=(0.0, 1.0, 3.0),
    )
    assert list(table["factor"]) == [0.0, 1.0, 3.0]
    assert table["power"].iloc[0] >= table["power"].iloc[1] >= table["power"].iloc[2]
    assert table["power"].iloc[0] > table["power"].iloc[2]

  def test_spread_must_be_stated(self, hourly, bound):
    scenario = bound.scenario(-0.10, 0.3)
    with pytest.raises(ValueError, match="simulation needs l1_spread_sd"):
      simulated_power(hourly.plan(2), "assumption_free", scenario, direction=LOWER, tail_bound=bound)
    with pytest.raises(ValueError, match="needs a tail_bound"):
      simulated_power(hourly.plan(2), "moment_bound", bound.scenario(-0.1, 0.3, l1_spread_sd=0.0), direction=LOWER)

  def test_simulated_weeks_brackets_the_target(self, hourly, target, bound):
    kwargs = dict(direction=LOWER, tail_bound=bound, n_draws=400, seed=6)
    result = simulated_weeks(hourly, "moment_bound_floored", target, **kwargs)
    assert result.note == "reached" and result.weeks == result.weeks_runnable
    assert result.power_at_runnable >= 0.80
    before = simulated_power(hourly.plan(result.weeks_runnable - 1), "moment_bound_floored", target, **kwargs)
    assert before.power < 0.80

  def test_spread_floor_is_reported_as_unreachable(self, hourly):
    """At a large C the moment bound carries a spread term that does not shrink
    with duration, so a wide enough spread makes it unreachable for good."""
    wide = TailBound(c=1.0, source="example")
    scenario = wide.scenario(-0.10, 0.2, l1_spread_sd=0.30, label="wide spread")
    result = simulated_weeks(
      hourly, "moment_bound_floored", scenario, direction=LOWER, tail_bound=wide,
      n_draws=200, seed=7, max_weeks=20,
    )
    assert result.note == "not_reachable" and result.weeks_runnable is None

  def test_window_spread_shrinks_with_longer_windows(self, hourly, four_hourly):
    scenario = Scenario("s", -0.1, 0.03, tail_decay=0.0, l1_spread_sd=0.08)
    assert window_l1_spread_sd(hourly.plan(1), scenario) == pytest.approx(0.08)
    assert window_l1_spread_sd(four_hourly.plan(1), scenario) < 0.04


@pytest.fixture(scope="module")
def comparison(hourly, four_hourly, target, bound):
  return compare_designs(
    {"1 h": hourly, "4 h": four_hourly},
    [target],
    direction=LOWER,
    tail_bound=bound,
    n_draws=300,
    max_weeks=40,
    metric="outcome",
    unit="units",
  )


class TestCompareDesigns:
  def test_tables_cover_every_design_test_and_bound(self, comparison):
    table = comparison.table
    assert set(table["design"]) == {"1 h", "4 h"}
    per_design = table[table["design"] == "1 h"]
    assert list(per_design["test_number"]) == ["1", "2", "3", "3", "4", "4", "4.5", "4.5", "5", "5"]
    assert set(per_design.loc[per_design["test"] == "lag0", "analysis_c"]) == {""}
    assert set(per_design.loc[per_design["test"] == "moment_bound_floored", "analysis_c"]) == {"C", "stress"}
    assert set(table.loc[table["method"] == "simulation", "test"]) == {
      "moment_bound", "moment_bound_floored", "assumption_free"
    }
    assert len(comparison.basis) == 2 and "se_at_bound_stress_1wk" in comparison.basis
    assert set(comparison.curve["test"]) == {"lag0", "lag01", "two_endpoint"}

  def test_closed_form_rows_match_direct_calls(self, comparison, hourly, target, bound):
    direct = weeks_to_power(hourly, "two_endpoint", target, direction=LOWER, tail_bound=bound, max_weeks=40)
    row = comparison.table.query("design == '1 h' and test == 'two_endpoint' and analysis_c == 'C'").iloc[0]
    assert row["weeks"] == pytest.approx(direct.weeks)
    assert row["weeks_runnable"] == direct.weeks_runnable
    assert row["weeks_budget"] > row["weeks"]
    assert row["power_at_4wk"] == pytest.approx(
      closed_form_power(hourly.plan(4), "two_endpoint", target, direction=LOWER, tail_bound=bound)
    )

  def test_floored_moment_bound_needs_at_least_as_long(self, comparison):
    table = comparison.table
    for design in ("1 h", "4 h"):
      for variant in ("C", "stress"):
        rows = table[(table["design"] == design) & (table["analysis_c"] == variant)].set_index("test")
        legacy, floored = rows.loc["moment_bound"], rows.loc["moment_bound_floored"]
        if floored["weeks_runnable"] is not None and not pd_isna(floored["weeks_runnable"]):
          assert floored["weeks_runnable"] >= legacy["weeks_runnable"]

  def test_ranking_is_sorted_and_text_follows_the_wording_rules(self, comparison):
    ranked = comparison.ranking("target")
    weeks = ranked["weeks_runnable"].fillna(math.inf).tolist()
    assert weeks == sorted(weeks)
    text = comparison.to_text()
    assert "not a recommendation" in text
    assert "lag-0 (same-window effect only)" in text
    assert "moment bound, tau2 floored at 0" in text
    assert "pre-register one test" in text
    assert "C is an assumption" in text
    lowered = text.lower().replace("not a recommendation", "")
    for word in ("recommend", "best design", "optimal"):
      assert word not in lowered

  def test_defaults_without_a_bound_are_the_two_lag_tests(self, hourly, target):
    comparison = compare_designs({"1 h": hourly}, [target], direction=LOWER)
    assert list(comparison.table["test"]) == ["lag0", "lag01"]
    assert comparison.notes == ()

  def test_simulated_rows_need_a_spread(self, hourly, bound):
    scenario = bound.scenario(-0.10, 0.3, label="no spread stated")
    comparison = compare_designs({"1 h": hourly}, [scenario], direction=LOWER, tail_bound=bound, n_draws=100)
    simulated = comparison.table[comparison.table["method"] == "simulation"]
    assert set(simulated["weeks_note"]) == {"not_computed"}
    assert any("needs l1_spread_sd" in warning for warning in comparison.warnings)
    assert "not computed" in comparison.to_text()

  def test_input_validation(self, hourly, target, bound):
    with pytest.raises(ValueError, match="labels must be distinct"):
      compare_designs({"1 h": hourly}, [target, target], direction=LOWER)
    with pytest.raises(ValueError, match="tail_bound is required"):
      compare_designs({"1 h": hourly}, [target], direction=LOWER, tests=("two_endpoint",))
    with pytest.raises(ValueError, match="unknown test"):
      compare_designs({"1 h": hourly}, [target], direction=LOWER, tests=("lag2",))
    with pytest.raises(ValueError, match="direction"):
      compare_designs({"1 h": hourly}, [target], direction="down")
    with pytest.raises(KeyError, match="no scenario"):
      compare_designs({"1 h": hourly}, [target], direction=LOWER).ranking("missing")


def pd_isna(value) -> bool:
  return value is None or (isinstance(value, float) and math.isnan(value))


def test_orientation_values():
  assert orientation("lower_is_better") == 1.0 and orientation("higher_is_better") == -1.0


class TestPreRegistration:
  def test_line_round_trips_and_escapes(self, comparison):
    from switchback_power_calculator.carryover.prereg import PreRegistration, preregistration

    planned = preregistration(
      comparison, design="4 h", test="moment_bound_floored", scenario="target", weeks=8,
      cluster_rule="top 120; orders >= 5 = kept", analysis_c="stress",
    )
    line = planned.to_line()
    assert line.startswith("carryover-prereg/1; ")
    assert "\n" not in line
    assert PreRegistration.from_line(line) == planned
    assert planned.test_id == "moment_bound_floored" and planned.analysis_c == "stress"
    assert planned.window_origin.endswith("min after 1970-01-01T00:00 (UTC)")
    assert "120 clusters, ids sha256:" in planned.clusters
    assert "rule: top 120; orders >= 5 = kept" in planned.clusters
    assert float(planned.parameters["t1"]) > 0
    assert planned.parameters["C_fp"] == "scalar:0.8"

  def test_closed_form_tests_carry_no_simulation_settings(self, comparison):
    from switchback_power_calculator.carryover.prereg import preregistration

    planned = preregistration(
      comparison, design="1 h", test="lag0", scenario="target", weeks=2, cluster_rule="all"
    )
    assert planned.window_origin == "n/a" and planned.seed == 0 and planned.params == ()
    assert planned.analysis_c == ""
    two = preregistration(
      comparison, design="1 h", test="two_endpoint", scenario="target", weeks=6, cluster_rule="all"
    )
    assert two.analysis_c == "C" and two.parameters["tail_sign"] == "same"
    assert two.target.startswith("power 0.") and "total -0.0571" in two.target

  def test_check_reports_deviations_and_notes(self, comparison, hourly):
    from switchback_power_calculator.carryover.prereg import check, preregistration

    planned = preregistration(
      comparison, design="1 h", test="moment_bound", scenario="target", weeks=7, cluster_rule="all"
    )
    line = planned.to_line()
    faithful = dict(
      metric="outcome", direction=LOWER, alpha=0.10, test_id="moment_bound", analysis_c="C",
      window_minutes=60, window_origin="n/a", weeks_run=7.0, cluster_ids=hourly.cluster_ids,
      min_windows_per_arm=5, residual=hourly.residual,
      params={"t1": planned.parameters["t1"], "C_fp": "scalar:0.43"},
    )
    assert check(line, faithful) == ((), ())

    changed = dict(
      faithful, test_id="moment_bound_floored", weeks_run=5.0, alpha=0.05,
      cluster_ids=hourly.cluster_ids[:-1], params={"t1": "0.5", "tau2_floor": "0"},
    )
    deviations, _ = check(line, changed)
    text = " ".join(deviations)
    for fragment in ("test_id", "weeks: ran 5", "alpha", "clusters", "param t1", "param tau2_floor"):
      assert fragment in text
    assert len(deviations) == 6

    est = LagEstimates(
      cluster_ids=("a", "b"), weights=[0.5, 0.5], l0=[0, 0], l1=[0, 0], s00=[1, 1], s01=[0, 0],
      s11=[1, 1], window_minutes=60, n_input_clusters=3, n_dropped=1, dropped_orders_share=0.02,
    )
    _, notes = check(line, dict(faithful, estimates=est))
    assert notes == ("1 of 3 clusters dropped by the estimator's rules (2.0% of orders).",)

  def test_validation(self, comparison):
    from switchback_power_calculator.carryover.prereg import (
      PreRegistration, check, preregistration, window_origin_label,
    )

    kwargs = dict(design="1 h", test="lag0", scenario="target", weeks=2, cluster_rule="all")
    with pytest.raises(KeyError, match="no design"):
      preregistration(comparison, **{**kwargs, "design": "9 h"})
    with pytest.raises(KeyError, match="no scenario"):
      preregistration(comparison, **{**kwargs, "scenario": "other"})
    with pytest.raises(ValueError, match="whole number"):
      preregistration(comparison, **{**kwargs, "weeks": 2.5})
    with pytest.raises(ValueError, match="does not use a bound"):
      preregistration(comparison, analysis_c="stress", **kwargs)
    with pytest.raises(ValueError, match="not a pre-registration line"):
      PreRegistration.from_line("something else")
    with pytest.raises(KeyError, match="test_id"):
      check(preregistration(comparison, **kwargs), {})
    with pytest.raises(ValueError, match="need an origin"):
      window_origin_label(None, 240, 60)
    shifted = window_origin_label(pd.Timestamp("2030-01-08 04:00", tz="UTC"), 240, 60)
    assert shifted == window_origin_label(pd.Timestamp("2030-01-07 00:00", tz="UTC"), 240, 60)
    assert shifted != window_origin_label(pd.Timestamp("2030-01-07 01:00", tz="UTC"), 240, 60)
