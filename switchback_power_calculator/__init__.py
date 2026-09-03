"""Switchback experiment power calculator."""

from switchback_power_calculator.core import (
    estimator_variance,
    formula_approximation_validity,
    mde,
    num_required_randomizations,
    power,
    standard_error,
    summarize_estimator_variance,
    summarize_estimator_variance_from_design,
    summarize_power,
    summarize_power_from_design,
)
from switchback_power_calculator.from_data import (
    compute_design_stats,
    estimate_design_parameters_from_data,
    estimate_variance_shares,
    power_from_data,
    prepare_data,
)
from switchback_power_calculator.types import (
    ApproximationValidity,
    DataDiagnostics,
    DesignParameters,
    DesignStats,
    PowerInputs,
    PowerResult,
    VarianceResult,
    VarianceShares,
)

__version__ = "1.0.0"

__all__ = [
    "ApproximationValidity",
    "DataDiagnostics",
    "DesignParameters",
    "DesignStats",
    "PowerInputs",
    "PowerResult",
    "VarianceResult",
    "VarianceShares",
    "__version__",
    "compute_design_stats",
    "estimate_design_parameters_from_data",
    "estimate_variance_shares",
    "estimator_variance",
    "formula_approximation_validity",
    "mde",
    "num_required_randomizations",
    "power",
    "power_from_data",
    "prepare_data",
    "standard_error",
    "summarize_estimator_variance",
    "summarize_estimator_variance_from_design",
    "summarize_power",
    "summarize_power_from_design",
]
