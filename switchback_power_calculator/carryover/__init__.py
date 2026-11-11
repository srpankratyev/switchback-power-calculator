"""Carryover-aware planning and analysis for switchback experiments (experimental).

The core package assumes a treated period affects only itself. Here the
effect of a treated period may continue: `l0` in the period itself, `l1` in
the next, and a fading tail after that. The subpackage covers both ends of an
experiment:

Before it runs
  `lag_moments_from_aggregates` reads the noise of the lag estimates off a
  pre-period at each candidate window length; `compare_designs` then reports
  how many weeks each test needs under the effects you plan for.

After it runs
  `experiment_windows` and `estimate_lags` turn the experiment's data into
  per-cluster lag estimates; `run_tests` runs the six tests on them.

The six tests, from the strongest assumption to none (`analysis.TESTS`):
lag0 (1), lag01 (2), two_endpoint (3), moment_bound (4),
moment_bound_floored (4.5) and assumption_free (5).

The top-level package does not import this subpackage.
"""

from switchback_power_calculator.carryover.analysis import (
  ASSUMES,
  TEST_LABELS,
  TEST_NUMBERS,
  TESTS,
  Summary,
  TestResult,
  TestResults,
  assumption_free_z,
  dl_tau2,
  moment_bound_z,
  planning_t1,
  run_tests,
  summarize,
  t1_leave_one_out,
  two_endpoint_z,
)
from switchback_power_calculator.carryover.estimate import estimate_lags, experiment_windows
from switchback_power_calculator.carryover.examples import synthetic_panel
from switchback_power_calculator.carryover.moments import (
  ClusterLagMoments,
  PlannedCovariance,
  ProfileFit,
  aggregate_observations,
  lag_moments_from_aggregates,
  lag_moments_from_parameters,
  leave_one_week_out_expected,
  profile_fingerprint,
  window_totals,
)
from switchback_power_calculator.carryover.plan import DesignComparison, compare_designs
from switchback_power_calculator.carryover.power import (
  MdeResult,
  WeeksResult,
  closed_form_power,
  mde,
  rebound_curve,
  scenario_grid,
  total_effect,
  weeks_to_power,
  window_l1_spread_sd,
  window_means,
)
from switchback_power_calculator.carryover.prereg import PreRegistration, check, preregistration
from switchback_power_calculator.carryover.simulate import (
  SimulatedPower,
  simulated_power,
  simulated_weeks,
  spread_sensitivity,
)
from switchback_power_calculator.carryover.tail import ASSUMPTION, TailBound
from switchback_power_calculator.carryover.types import Direction, LagEstimates, Scenario

__version__ = "0.1.0.dev0"

__all__ = [
  "ASSUMES",
  "ASSUMPTION",
  "TESTS",
  "TEST_LABELS",
  "TEST_NUMBERS",
  "ClusterLagMoments",
  "DesignComparison",
  "Direction",
  "LagEstimates",
  "MdeResult",
  "PlannedCovariance",
  "PreRegistration",
  "ProfileFit",
  "Scenario",
  "SimulatedPower",
  "Summary",
  "TailBound",
  "TestResult",
  "TestResults",
  "WeeksResult",
  "__version__",
  "aggregate_observations",
  "assumption_free_z",
  "check",
  "closed_form_power",
  "compare_designs",
  "dl_tau2",
  "estimate_lags",
  "experiment_windows",
  "lag_moments_from_aggregates",
  "lag_moments_from_parameters",
  "leave_one_week_out_expected",
  "mde",
  "moment_bound_z",
  "planning_t1",
  "preregistration",
  "profile_fingerprint",
  "rebound_curve",
  "run_tests",
  "scenario_grid",
  "simulated_power",
  "simulated_weeks",
  "spread_sensitivity",
  "summarize",
  "synthetic_panel",
  "t1_leave_one_out",
  "total_effect",
  "two_endpoint_z",
  "weeks_to_power",
  "window_l1_spread_sd",
  "window_means",
  "window_totals",
]
