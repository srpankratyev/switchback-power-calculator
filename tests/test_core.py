"""Tests for the closed-form variance formula and derived power metrics."""

from __future__ import annotations

import math

import pytest
from scipy.stats import norm

from switchback_power_calculator import (
    estimator_variance,
    formula_approximation_validity,
    mde,
    num_required_randomizations,
    power,
    summarize_estimator_variance,
    summarize_power,
)

# Paper example: dense but imbalanced marketplace.
PAPER = dict(
  mean_obs_per_cell=20.0,
  cell_size_cv=1.5,
  between_cell_variance_share=0.20,
  within_cell_variance_share=0.80,
  outcome_sd=1000.0,
)


def bracket(
  mean_obs_per_cell: float,
  cell_size_cv: float,
  between_cell_variance_share: float,
  within_cell_variance_share: float,
) -> float:
  return within_cell_variance_share / mean_obs_per_cell + between_cell_variance_share * (
    1.0 / mean_obs_per_cell + 1.0 + cell_size_cv**2
  )


class TestAnalyticalAnchors:
  """Special cases where the formula must reduce to an independently known result."""

  def test_no_macro_variance_recovers_naive_ab(self):
    num_clusters, num_time_periods, n, sigma = 40, 24, 50.0, 300.0
    got = estimator_variance(
      num_clusters,
      num_time_periods,
      mean_obs_per_cell=n,
      cell_size_cv=0.0,
      between_cell_variance_share=0.0,
      within_cell_variance_share=1.0,
      outcome_sd=sigma,
    )
    expected = 4.0 * sigma**2 / (num_clusters * num_time_periods * n)
    assert got == pytest.approx(expected, rel=1e-12)

  def test_eldridge_imbalance_penalty(self):
    """Pure macro variance: the cv ratio must match (1/n + 1 + cv^2) / (1/n + 1)."""
    num_clusters, num_time_periods, n, sigma = 40, 24, 50.0, 300.0
    v0 = estimator_variance(
      num_clusters,
      num_time_periods,
      mean_obs_per_cell=n,
      cell_size_cv=0.0,
      between_cell_variance_share=1.0,
      within_cell_variance_share=0.0,
      outcome_sd=sigma,
    )
    v2 = estimator_variance(
      num_clusters,
      num_time_periods,
      mean_obs_per_cell=n,
      cell_size_cv=2.0,
      between_cell_variance_share=1.0,
      within_cell_variance_share=0.0,
      outcome_sd=sigma,
    )
    expected = (1.0 / n + 1.0 + 4.0) / (1.0 / n + 1.0)
    assert v2 / v0 == pytest.approx(expected, rel=1e-12)

  @pytest.mark.parametrize("m, tol", [(50, 2e-2), (500, 2e-3), (5000, 2e-4)])
  def test_killip_design_effect_converges(self, m, tol):
    """H=1 cluster-randomized case approaches the Killip design effect 1 + rho(m-1).

    Our formula exceeds it by exactly rho, the vanishing mean_obs_per_cell term,
    so the two agree in relative terms only as cluster size grows.
    """
    num_clusters, sigma, rho = 40, 300.0, 0.10
    var = estimator_variance(
      num_clusters,
      1,
      mean_obs_per_cell=m,
      cell_size_cv=0.0,
      between_cell_variance_share=rho,
      within_cell_variance_share=1 - rho,
      outcome_sd=sigma,
    )
    de_ours = var / (4.0 * sigma**2 / (num_clusters * m))
    de_killip = 1.0 + rho * (m - 1)
    assert de_ours / de_killip == pytest.approx(1.0, rel=tol)
    assert de_ours - de_killip == pytest.approx(rho, abs=1e-9)


class TestScaling:
  def test_variance_inversely_proportional_to_num_cells(self):
    v1 = estimator_variance(50, 20, **PAPER)
    v2 = estimator_variance(100, 20, **PAPER)
    assert v2 == pytest.approx(v1 / 2.0, rel=1e-12)

  def test_variance_proportional_to_outcome_sd_squared(self):
    args = {k: v for k, v in PAPER.items() if k != "outcome_sd"}
    v1 = estimator_variance(50, 20, outcome_sd=1000.0, **args)
    v2 = estimator_variance(50, 20, outcome_sd=2000.0, **args)
    assert v2 == pytest.approx(4.0 * v1, rel=1e-12)

  def test_variance_matches_hand_computed_bracket(self):
    num_clusters, num_time_periods = 100, 168
    expected = (
      4.0
      * PAPER["outcome_sd"] ** 2
      / (num_clusters * num_time_periods)
      * bracket(
        PAPER["mean_obs_per_cell"],
        PAPER["cell_size_cv"],
        PAPER["between_cell_variance_share"],
        PAPER["within_cell_variance_share"],
      )
    )
    assert estimator_variance(num_clusters, num_time_periods, **PAPER) == pytest.approx(
      expected, rel=1e-12
    )


class TestRoundTrip:
  """Regression guard: num_required_randomizations once divided by the bracket
  instead of multiplying, inflating required sample size by bracket squared."""

  @pytest.mark.parametrize("treatment_effect", [25.0, 50.0, 100.0, 250.0])
  @pytest.mark.parametrize("target", [0.7, 0.8, 0.9])
  def test_required_randomizations_round_trip(self, treatment_effect, target):
    num_cells = num_required_randomizations(
      treatment_effect, power_level=target, **PAPER
    )
    achieved = power(
      treatment_effect,
      num_clusters=num_cells,
      num_time_periods=1,
      **PAPER,
    )
    assert achieved == pytest.approx(target, abs=1e-4)

  def test_required_randomizations_closed_form(self):
    treatment_effect = 100.0
    z = norm.ppf(0.975) + norm.ppf(0.8)
    expected = (
      4.0
      * PAPER["outcome_sd"] ** 2
      * z**2
      * bracket(
        PAPER["mean_obs_per_cell"],
        PAPER["cell_size_cv"],
        PAPER["between_cell_variance_share"],
        PAPER["within_cell_variance_share"],
      )
      / treatment_effect**2
    )
    assert num_required_randomizations(treatment_effect, **PAPER) == pytest.approx(
      expected, rel=1e-12
    )

  @pytest.mark.parametrize("treatment_effect", [25.0, 100.0, 250.0])
  def test_sign_of_treatment_effect_does_not_change_sample_size(self, treatment_effect):
    assert num_required_randomizations(-treatment_effect, **PAPER) == pytest.approx(
      num_required_randomizations(treatment_effect, **PAPER), rel=1e-12
    )

  @pytest.mark.parametrize("target", [0.7, 0.8, 0.9])
  def test_round_trip_holds_for_negative_treatment_effect(self, target):
    treatment_effect = -75.0
    num_cells = num_required_randomizations(treatment_effect, power_level=target, **PAPER)
    assert power(
      treatment_effect,
      num_clusters=num_cells,
      num_time_periods=1,
      **PAPER,
    ) == pytest.approx(target, abs=1e-4)

  def test_mde_achieves_target_power(self):
    detectable = mde(100, 168, **PAPER)
    assert power(detectable, 100, 168, **PAPER) == pytest.approx(0.8, abs=1e-4)

  def test_mde_equals_z_times_se(self):
    se = math.sqrt(estimator_variance(100, 168, **PAPER))
    z = norm.ppf(0.975) + norm.ppf(0.8)
    assert mde(100, 168, **PAPER) == pytest.approx(z * se, rel=1e-12)


class TestPower:
  def test_zero_effect_returns_alpha(self):
    assert power(0.0, 100, 168, alpha=0.05, **PAPER) == 0.05

  def test_power_increases_with_effect_size(self):
    values = [power(t, 100, 168, **PAPER) for t in (10.0, 20.0, 40.0, 80.0)]
    assert values == sorted(values)

  def test_power_symmetric_in_sign_of_treatment_effect(self):
    assert power(30.0, 100, 168, **PAPER) == pytest.approx(
      power(-30.0, 100, 168, **PAPER), rel=1e-12
    )

  def test_larger_alpha_gives_more_power(self):
    strict = power(30.0, 100, 168, alpha=0.01, **PAPER)
    loose = power(30.0, 100, 168, alpha=0.10, **PAPER)
    assert loose > strict


class TestApproximationValidity:
  def test_paper_example_is_accurate(self):
    validity, warnings = formula_approximation_validity(
      num_clusters=100,
      num_time_periods=168,
      mean_obs_per_cell=20.0,
      cell_size_cv=1.5,
    )
    assert validity == "approximation accurate"
    assert warnings == ()

  @pytest.mark.parametrize("cell_size_cv", [2.0, 2.5, 4.0])
  def test_high_imbalance_is_inaccurate(self, cell_size_cv):
    validity, warnings = formula_approximation_validity(
      num_clusters=100,
      num_time_periods=168,
      mean_obs_per_cell=20.0,
      cell_size_cv=cell_size_cv,
    )
    assert validity == "approximation inaccurate"
    assert len(warnings) == 1

  @pytest.mark.parametrize("num_clusters, num_time_periods", [(1, 1), (5, 2), (10, 1)])
  def test_few_randomizations_is_inaccurate(self, num_clusters, num_time_periods):
    validity, _ = formula_approximation_validity(
      num_clusters=num_clusters,
      num_time_periods=num_time_periods,
      mean_obs_per_cell=20.0,
      cell_size_cv=1.0,
    )
    assert validity == "approximation inaccurate"

  def test_just_above_threshold_is_accurate(self):
    validity, _ = formula_approximation_validity(
      num_clusters=11,
      num_time_periods=1,
      mean_obs_per_cell=20.0,
      cell_size_cv=1.99,
    )
    assert validity == "approximation accurate"

  def test_sparse_cells_warn_without_flipping_validity(self):
    validity, warnings = formula_approximation_validity(
      num_clusters=100,
      num_time_periods=168,
      mean_obs_per_cell=3.0,
      cell_size_cv=1.0,
    )
    assert validity == "approximation accurate"
    assert len(warnings) == 1
    assert "sparse" in warnings[0].lower()

  def test_inaccuracy_warning_supersedes_sparse_warning(self):
    validity, warnings = formula_approximation_validity(
      num_clusters=2,
      num_time_periods=2,
      mean_obs_per_cell=3.0,
      cell_size_cv=3.0,
    )
    assert validity == "approximation inaccurate"
    assert len(warnings) == 1
    assert "Approximation inaccurate" in warnings[0]

  def test_summarize_power_reports_validity(self):
    accurate = summarize_power(
      num_clusters=100,
      num_time_periods=168,
      treatment_effect=100.0,
      **PAPER,
    )
    assert accurate.approximation_validity == "approximation accurate"
    assert accurate.warnings == ()

    inaccurate = summarize_power(
      num_clusters=5,
      num_time_periods=2,
      mean_obs_per_cell=20.0,
      cell_size_cv=2.5,
      between_cell_variance_share=0.20,
      within_cell_variance_share=0.80,
      outcome_sd=1000.0,
    )
    assert inaccurate.approximation_validity == "approximation inaccurate"
    assert inaccurate.warnings


class TestSummarizeEstimatorVariance:
  def test_matches_estimator_variance_and_validity(self):
    summary = summarize_estimator_variance(100, 168, **PAPER)
    assert summary.variance == pytest.approx(estimator_variance(100, 168, **PAPER))
    assert summary.standard_error == pytest.approx(summary.variance**0.5)
    validity, warnings = formula_approximation_validity(
      num_clusters=100,
      num_time_periods=168,
      mean_obs_per_cell=PAPER["mean_obs_per_cell"],
      cell_size_cv=PAPER["cell_size_cv"],
    )
    assert summary.approximation_validity == validity
    assert summary.warnings == warnings


class TestSummarizePower:
  def test_optional_fields_absent_without_treatment_effect(self):
    result = summarize_power(num_clusters=100, num_time_periods=168, **PAPER)
    assert result.power is None
    assert result.required_num_cells is None
    assert result.mde is not None

  def test_fields_agree_with_standalone_functions(self):
    result = summarize_power(
      num_clusters=100,
      num_time_periods=168,
      treatment_effect=100.0,
      **PAPER,
    )
    variance = summarize_estimator_variance(100, 168, **PAPER)
    assert result.variance_summary == variance
    assert result.variance == pytest.approx(variance.variance)
    assert result.standard_error == pytest.approx(variance.standard_error)
    assert result.mde == pytest.approx(mde(100, 168, **PAPER))
    assert result.power == pytest.approx(power(100.0, 100, 168, **PAPER))
    assert result.required_num_cells == pytest.approx(
      num_required_randomizations(100.0, **PAPER)
    )

  def test_within_cell_share_defaults_to_complement(self):
    with_share = estimator_variance(
      100,
      168,
      mean_obs_per_cell=20.0,
      cell_size_cv=1.5,
      between_cell_variance_share=0.20,
      within_cell_variance_share=0.80,
      outcome_sd=1000.0,
    )
    without_share = estimator_variance(
      100,
      168,
      mean_obs_per_cell=20.0,
      cell_size_cv=1.5,
      between_cell_variance_share=0.20,
      outcome_sd=1000.0,
    )
    assert without_share == pytest.approx(with_share, rel=1e-12)


class TestInputValidation:
  @pytest.mark.parametrize("num_clusters, num_time_periods", [(0, 10), (1, 10), (-1, 10)])
  def test_insufficient_clusters_rejected(self, num_clusters, num_time_periods):
    with pytest.raises(ValueError, match="num_clusters must be at least 2"):
      estimator_variance(num_clusters, num_time_periods, **PAPER)

  def test_non_positive_num_time_periods_rejected(self):
    with pytest.raises(ValueError, match="num_time_periods must be positive"):
      estimator_variance(10, 0, **PAPER)

  def test_non_positive_mean_obs_per_cell_rejected(self):
    with pytest.raises(ValueError, match="mean_obs_per_cell must be positive"):
      estimator_variance(
        10,
        10,
        mean_obs_per_cell=0.0,
        cell_size_cv=1.0,
        between_cell_variance_share=0.2,
        within_cell_variance_share=0.8,
        outcome_sd=100.0,
      )

  def test_negative_cell_size_cv_rejected(self):
    with pytest.raises(ValueError, match="cell_size_cv must be non-negative"):
      estimator_variance(
        10,
        10,
        mean_obs_per_cell=20.0,
        cell_size_cv=-0.1,
        between_cell_variance_share=0.2,
        within_cell_variance_share=0.8,
        outcome_sd=100.0,
      )

  def test_shares_must_sum_to_one(self):
    with pytest.raises(ValueError, match="must sum to 1"):
      estimator_variance(
        10,
        10,
        mean_obs_per_cell=20.0,
        cell_size_cv=1.0,
        between_cell_variance_share=0.5,
        within_cell_variance_share=0.9,
        outcome_sd=100.0,
      )

  def test_shares_must_lie_in_unit_interval(self):
    with pytest.raises(ValueError, match="Variance shares"):
      estimator_variance(
        10,
        10,
        mean_obs_per_cell=20.0,
        cell_size_cv=1.0,
        between_cell_variance_share=-0.5,
        within_cell_variance_share=1.5,
        outcome_sd=100.0,
      )

  def test_non_positive_outcome_sd_rejected(self):
    with pytest.raises(ValueError, match="outcome_sd must be positive"):
      estimator_variance(
        10,
        10,
        mean_obs_per_cell=20.0,
        cell_size_cv=1.0,
        between_cell_variance_share=0.2,
        within_cell_variance_share=0.8,
        outcome_sd=0.0,
      )

  def test_zero_treatment_effect_rejected_for_sample_size(self):
    with pytest.raises(ValueError, match="treatment_effect must be non-zero"):
      num_required_randomizations(0.0, **PAPER)

  def test_sample_size_validates_shares(self):
    with pytest.raises(ValueError, match="must sum to 1"):
      num_required_randomizations(
        100.0,
        mean_obs_per_cell=20.0,
        cell_size_cv=1.0,
        between_cell_variance_share=0.5,
        within_cell_variance_share=0.9,
        outcome_sd=100.0,
      )
