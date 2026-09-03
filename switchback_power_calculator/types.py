from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ApproximationValidity = Literal["approximation accurate", "approximation inaccurate"]


def resolve_variance_shares(
  between_cell_variance_share: float,
  within_cell_variance_share: float | None = None,
) -> tuple[float, float]:
  """Return a validated (between, within) pair, defaulting within to the complement."""
  if within_cell_variance_share is None:
    within_cell_variance_share = 1.0 - between_cell_variance_share
  if not 0 <= between_cell_variance_share <= 1 or not 0 <= within_cell_variance_share <= 1:
    raise ValueError("Variance shares must lie in [0, 1].")
  if abs(between_cell_variance_share + within_cell_variance_share - 1.0) > 1e-6:
    raise ValueError(
      "between_cell_variance_share and within_cell_variance_share must sum to 1."
    )
  return between_cell_variance_share, within_cell_variance_share


def validate_variance_inputs(
  mean_obs_per_cell: float,
  cell_size_cv: float,
  between_cell_variance_share: float,
  within_cell_variance_share: float | None,
  outcome_sd: float,
) -> tuple[float, float]:
  """Check the inputs shared by every variance and power calculation."""
  if mean_obs_per_cell <= 0:
    raise ValueError("mean_obs_per_cell must be positive.")
  if cell_size_cv < 0:
    raise ValueError("cell_size_cv must be non-negative.")
  if outcome_sd <= 0:
    raise ValueError("outcome_sd must be positive.")
  return resolve_variance_shares(
    between_cell_variance_share, within_cell_variance_share
  )


@dataclass(frozen=True)
class PowerInputs:
  """Validated inputs for the closed-form switchback variance formula."""

  num_clusters: int
  num_time_periods: int
  mean_obs_per_cell: float
  cell_size_cv: float
  between_cell_variance_share: float
  within_cell_variance_share: float
  outcome_sd: float

  def __post_init__(self) -> None:
    if self.num_clusters < 2:
      raise ValueError(
        "num_clusters must be at least 2; with a single cluster the cluster "
        "shock is not identified across units."
      )
    if self.num_time_periods <= 0:
      raise ValueError("num_time_periods must be positive.")
    between, within = validate_variance_inputs(
      self.mean_obs_per_cell,
      self.cell_size_cv,
      self.between_cell_variance_share,
      self.within_cell_variance_share,
      self.outcome_sd,
    )
    object.__setattr__(self, "between_cell_variance_share", between)
    object.__setattr__(self, "within_cell_variance_share", within)

  def summarize_kwargs(self) -> dict[str, float | int]:
    """Keyword arguments for `summarize_estimator_variance` and `summarize_power`."""
    return {
      "num_clusters": self.num_clusters,
      "num_time_periods": self.num_time_periods,
      "mean_obs_per_cell": self.mean_obs_per_cell,
      "cell_size_cv": self.cell_size_cv,
      "between_cell_variance_share": self.between_cell_variance_share,
      "within_cell_variance_share": self.within_cell_variance_share,
      "outcome_sd": self.outcome_sd,
    }


@dataclass(frozen=True)
class VarianceResult:
  """Summary of estimator variance for one design.

  Attributes:
    variance: Var(tau_hat), in squared outcome units.
    standard_error: Square root of `variance`, in outcome units.
    approximation_validity: Whether the formula is expected to be accurate.
    warnings: Notes about inputs that stress the approximation.
  """

  variance: float
  standard_error: float
  approximation_validity: ApproximationValidity
  warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PowerResult:
  """Power metrics for one switchback design.

  Always includes a variance summary. When `treatment_effect` is supplied to
  `summarize_power`, also includes power and `required_num_cells`.

  Attributes:
    variance_summary: Standard error, variance, and approximation verdict.
    mde: Minimum detectable effect at the requested power level.
    power: Probability of detecting the supplied `treatment_effect`, if any.
    required_num_cells: Cells needed to detect the supplied effect, if any.
  """

  variance_summary: VarianceResult
  mde: float
  power: float | None = None
  required_num_cells: float | None = None

  @property
  def variance(self) -> float:
    return self.variance_summary.variance

  @property
  def standard_error(self) -> float:
    return self.variance_summary.standard_error

  @property
  def approximation_validity(self) -> ApproximationValidity:
    return self.variance_summary.approximation_validity

  @property
  def warnings(self) -> tuple[str, ...]:
    return self.variance_summary.warnings


@dataclass(frozen=True)
class DesignStats:
  """Cluster/time counts and cell-size shape read off the data."""

  num_clusters: int
  num_time_periods: int
  n_cells: int
  mean_obs_per_cell: float
  cell_size_cv: float
  n_observations: int


@dataclass(frozen=True)
class VarianceShares:
  """ANOVA decomposition of outcome variance across design levels."""

  cluster_variance_share: float
  time_period_variance_share: float
  interaction_variance_share: float
  within_cell_variance_share: float
  between_cell_variance_share: float
  outcome_sd: float
  warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DesignParameters:
  """All formula inputs inferred from a pre-period extract.

  Bundles design counts and cell-size stats (`design`) with the variance
  decomposition (`shares`). Pass to `summarize_power_from_design` or unpack
  into `summarize_power`.
  """

  design: DesignStats
  shares: VarianceShares

  @property
  def num_clusters(self) -> int:
    return self.design.num_clusters

  @property
  def num_time_periods(self) -> int:
    return self.design.num_time_periods

  @property
  def mean_obs_per_cell(self) -> float:
    return self.design.mean_obs_per_cell

  @property
  def cell_size_cv(self) -> float:
    return self.design.cell_size_cv

  @property
  def between_cell_variance_share(self) -> float:
    return self.shares.between_cell_variance_share

  @property
  def within_cell_variance_share(self) -> float:
    return self.shares.within_cell_variance_share

  @property
  def outcome_sd(self) -> float:
    return self.shares.outcome_sd

  def summarize_kwargs(self) -> dict[str, float | int]:
    """Keyword arguments for `summarize_estimator_variance` and `summarize_power`."""
    return {
      "num_clusters": self.num_clusters,
      "num_time_periods": self.num_time_periods,
      "mean_obs_per_cell": self.mean_obs_per_cell,
      "cell_size_cv": self.cell_size_cv,
      "between_cell_variance_share": self.between_cell_variance_share,
      "within_cell_variance_share": self.within_cell_variance_share,
      "outcome_sd": self.outcome_sd,
    }


@dataclass(frozen=True)
class DataDiagnostics:
  """What cleaning did to the input frame."""

  rows_input: int
  rows_used: int
  rows_dropped_missing: int
  cells_before_filter: int
  cells_after_filter: int
  min_cell_count: int
  warnings: tuple[str, ...] = field(default_factory=tuple)
