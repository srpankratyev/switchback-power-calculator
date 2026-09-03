from __future__ import annotations

import math

from scipy.stats import norm

from switchback_power_calculator.types import (
  ApproximationValidity,
  DesignParameters,
  PowerInputs,
  PowerResult,
  VarianceResult,
  validate_variance_inputs,
)

CELL_SIZE_CV_INACCURATE_THRESHOLD = 2.0
MIN_CELLS_INACCURATE_THRESHOLD = 10
MEAN_OBS_PER_CELL_LOW_DENSITY_WARNING = 5.0


def formula_approximation_validity(
  num_clusters: int,
  num_time_periods: int,
  *,
  mean_obs_per_cell: float,
  cell_size_cv: float,
) -> tuple[ApproximationValidity, tuple[str, ...]]:
  """Judge whether the formula can be trusted for a given design.

  Prefer `summarize_estimator_variance`, which bundles this with the variance.
  """
  warnings: list[str] = []
  triggers: list[str] = []
  num_cells = num_clusters * num_time_periods

  if cell_size_cv >= CELL_SIZE_CV_INACCURATE_THRESHOLD:
    triggers.append(
      f"cell_size_cv={cell_size_cv:.2f} >= {CELL_SIZE_CV_INACCURATE_THRESHOLD:g}"
    )
  if num_cells <= MIN_CELLS_INACCURATE_THRESHOLD:
    triggers.append(
      f"num_clusters*num_time_periods={num_cells} "
      f"<= {MIN_CELLS_INACCURATE_THRESHOLD:g}"
    )

  if triggers:
    warnings.append(
      "Approximation inaccurate (" + ", ".join(triggers) + "): the formula's "
      "accuracy degrades for these inputs. In the paper's synthetic validation "
      "it overpredicts variance here."
    )
  elif mean_obs_per_cell <= MEAN_OBS_PER_CELL_LOW_DENSITY_WARNING:
    warnings.append(
      f"Sparse cells (mean_obs_per_cell={mean_obs_per_cell:.2f} "
      f"<= {MEAN_OBS_PER_CELL_LOW_DENSITY_WARNING:g}): approximation error "
      "may be modestly larger than in typical marketplace settings."
    )

  validity = "approximation inaccurate" if triggers else "approximation accurate"
  return validity, tuple(warnings)


def estimator_variance(
  num_clusters: int,
  num_time_periods: int,
  *,
  mean_obs_per_cell: float,
  cell_size_cv: float,
  between_cell_variance_share: float,
  within_cell_variance_share: float | None = None,
  outcome_sd: float,
) -> float:
  """Variance of the estimated treatment effect, Var(tau_hat).

  For standard error and approximation warnings together, use
  `summarize_estimator_variance`.
  """
  between, within = validate_variance_inputs(
    mean_obs_per_cell,
    cell_size_cv,
    between_cell_variance_share,
    within_cell_variance_share,
    outcome_sd,
  )
  inputs = PowerInputs(
    num_clusters=num_clusters,
    num_time_periods=num_time_periods,
    mean_obs_per_cell=mean_obs_per_cell,
    cell_size_cv=cell_size_cv,
    between_cell_variance_share=between,
    within_cell_variance_share=within,
    outcome_sd=outcome_sd,
  )
  scale = (
    4.0
    * inputs.outcome_sd**2
    / (inputs.num_clusters * inputs.num_time_periods)
  )
  return scale * _variance_bracket(
    mean_obs_per_cell=inputs.mean_obs_per_cell,
    cell_size_cv=inputs.cell_size_cv,
    between_cell_variance_share=inputs.between_cell_variance_share,
    within_cell_variance_share=inputs.within_cell_variance_share,
  )


def _variance_bracket(
  *,
  mean_obs_per_cell: float,
  cell_size_cv: float,
  between_cell_variance_share: float,
  within_cell_variance_share: float,
) -> float:
  """Design-dependent factor: within/n + between*(1/n + 1 + cv^2)."""
  return within_cell_variance_share / mean_obs_per_cell + between_cell_variance_share * (
    1.0 / mean_obs_per_cell + 1.0 + cell_size_cv**2
  )


def standard_error(
  num_clusters: int,
  num_time_periods: int,
  *,
  mean_obs_per_cell: float,
  cell_size_cv: float,
  between_cell_variance_share: float,
  within_cell_variance_share: float | None = None,
  outcome_sd: float,
) -> float:
  """Standard error of the estimated treatment effect, in outcome units."""
  return math.sqrt(
    estimator_variance(
      num_clusters,
      num_time_periods,
      mean_obs_per_cell=mean_obs_per_cell,
      cell_size_cv=cell_size_cv,
      between_cell_variance_share=between_cell_variance_share,
      within_cell_variance_share=within_cell_variance_share,
      outcome_sd=outcome_sd,
    )
  )


def summarize_estimator_variance(
  num_clusters: int,
  num_time_periods: int,
  *,
  mean_obs_per_cell: float,
  cell_size_cv: float,
  between_cell_variance_share: float,
  within_cell_variance_share: float | None = None,
  outcome_sd: float,
) -> VarianceResult:
  """Variance, standard error, and approximation verdict for one design.

  This is the variance-layer summary. Use `summarize_power` when you also
  want MDE, power, and required cell count.
  """
  shared = dict(
    mean_obs_per_cell=mean_obs_per_cell,
    cell_size_cv=cell_size_cv,
    between_cell_variance_share=between_cell_variance_share,
    within_cell_variance_share=within_cell_variance_share,
    outcome_sd=outcome_sd,
  )
  var = estimator_variance(num_clusters, num_time_periods, **shared)
  validity, warnings = formula_approximation_validity(
    num_clusters,
    num_time_periods,
    mean_obs_per_cell=mean_obs_per_cell,
    cell_size_cv=cell_size_cv,
  )
  return VarianceResult(
    variance=var,
    standard_error=math.sqrt(var),
    approximation_validity=validity,
    warnings=warnings,
  )


def summarize_estimator_variance_from_design(
  parameters: DesignParameters,
) -> VarianceResult:
  """Like `summarize_estimator_variance`, using inferred `DesignParameters`."""
  return summarize_estimator_variance(**parameters.summarize_kwargs())


def mde(
  num_clusters: int,
  num_time_periods: int,
  *,
  mean_obs_per_cell: float,
  cell_size_cv: float,
  between_cell_variance_share: float,
  within_cell_variance_share: float | None = None,
  outcome_sd: float,
  alpha: float = 0.05,
  power_level: float = 0.8,
) -> float:
  """Minimum detectable effect at two-sided `alpha` and target `power_level`."""
  se = standard_error(
    num_clusters,
    num_time_periods,
    mean_obs_per_cell=mean_obs_per_cell,
    cell_size_cv=cell_size_cv,
    between_cell_variance_share=between_cell_variance_share,
    within_cell_variance_share=within_cell_variance_share,
    outcome_sd=outcome_sd,
  )
  return (norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power_level)) * se


def power(
  treatment_effect: float,
  num_clusters: int,
  num_time_periods: int,
  *,
  mean_obs_per_cell: float,
  cell_size_cv: float,
  between_cell_variance_share: float,
  within_cell_variance_share: float | None = None,
  outcome_sd: float,
  alpha: float = 0.05,
) -> float:
  """Two-sided power to detect `treatment_effect`. Sign does not matter."""
  if treatment_effect == 0:
    return alpha
  se = standard_error(
    num_clusters,
    num_time_periods,
    mean_obs_per_cell=mean_obs_per_cell,
    cell_size_cv=cell_size_cv,
    between_cell_variance_share=between_cell_variance_share,
    within_cell_variance_share=within_cell_variance_share,
    outcome_sd=outcome_sd,
  )
  z_alpha = norm.ppf(1.0 - alpha / 2.0)
  z_stat = abs(treatment_effect) / se
  return float(1.0 - norm.cdf(z_alpha - z_stat) + norm.cdf(-z_alpha - z_stat))


def num_required_randomizations(
  treatment_effect: float,
  *,
  mean_obs_per_cell: float,
  cell_size_cv: float,
  between_cell_variance_share: float,
  within_cell_variance_share: float | None = None,
  outcome_sd: float,
  alpha: float = 0.05,
  power_level: float = 0.8,
) -> float:
  """Required `num_clusters * num_time_periods` to detect `treatment_effect`."""
  if treatment_effect == 0:
    raise ValueError("treatment_effect must be non-zero.")
  between, within = validate_variance_inputs(
    mean_obs_per_cell,
    cell_size_cv,
    between_cell_variance_share,
    within_cell_variance_share,
    outcome_sd,
  )
  bracket = _variance_bracket(
    mean_obs_per_cell=mean_obs_per_cell,
    cell_size_cv=cell_size_cv,
    between_cell_variance_share=between,
    within_cell_variance_share=within,
  )
  z = norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power_level)
  return 4.0 * outcome_sd**2 * z**2 * bracket / treatment_effect**2


def summarize_power(
  num_clusters: int,
  num_time_periods: int,
  *,
  mean_obs_per_cell: float,
  cell_size_cv: float,
  between_cell_variance_share: float,
  within_cell_variance_share: float | None = None,
  outcome_sd: float,
  treatment_effect: float | None = None,
  alpha: float = 0.05,
  power_level: float = 0.8,
) -> PowerResult:
  """Variance summary plus MDE; optionally power and required cell count.

  Entry point for the parameter API when you already know the design numbers.
  For data, use `estimate_design_parameters_from_data` then
  `summarize_power_from_design`.
  """
  shared = dict(
    mean_obs_per_cell=mean_obs_per_cell,
    cell_size_cv=cell_size_cv,
    between_cell_variance_share=between_cell_variance_share,
    within_cell_variance_share=within_cell_variance_share,
    outcome_sd=outcome_sd,
  )
  variance_summary = summarize_estimator_variance(
    num_clusters, num_time_periods, **shared
  )
  return PowerResult(
    variance_summary=variance_summary,
    mde=mde(
      num_clusters,
      num_time_periods,
      alpha=alpha,
      power_level=power_level,
      **shared,
    ),
    power=(
      power(
        treatment_effect,
        num_clusters,
        num_time_periods,
        alpha=alpha,
        **shared,
      )
      if treatment_effect is not None
      else None
    ),
    required_num_cells=(
      num_required_randomizations(
        treatment_effect,
        alpha=alpha,
        power_level=power_level,
        **shared,
      )
      if treatment_effect is not None
      else None
    ),
  )


def summarize_power_from_design(
  parameters: DesignParameters,
  *,
  treatment_effect: float | None = None,
  alpha: float = 0.05,
  power_level: float = 0.8,
) -> PowerResult:
  """Like `summarize_power`, using inferred `DesignParameters`."""
  return summarize_power(
    **parameters.summarize_kwargs(),
    treatment_effect=treatment_effect,
    alpha=alpha,
    power_level=power_level,
  )
