"""One line to write down before an experiment, and a check against it afterwards."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

import switchback_power_calculator
from switchback_power_calculator.carryover import analysis
from switchback_power_calculator.carryover.power import (
  CLOSED_FORM_TESTS,
  closed_form_power,
  total_effect,
  window_l1_spread_sd,
)
from switchback_power_calculator.carryover.simulate import simulated_power
from switchback_power_calculator.carryover.types import (
  Direction,
  LagEstimates,
  normalize_ids,
  validate_window,
)

PREFIX = "carryover-prereg/1"
_ESCAPES = (("%", "%25"), (";", "%3B"), ("=", "%3D"), ("\n", "%0A"))


def _escape(value: str) -> str:
  for raw, coded in _ESCAPES:
    value = value.replace(raw, coded)
  return value


def _unescape(value: str) -> str:
  for raw, coded in reversed(_ESCAPES):
    value = value.replace(coded, raw)
  return value


def cluster_checksum(cluster_ids: Iterable[object]) -> str:
  """First 12 hex digits of sha256 over the sorted cluster ids."""
  lines = sorted(f"{cluster}=1" for cluster in normalize_ids(cluster_ids))
  return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:12]


def window_origin_label(origin: object, window_minutes: float, base_minutes: float) -> str:
  """Where windows start, in a form that survives a shift by whole windows or days.

  "n/a" for windows of one base period. For a datetime origin, its offset
  into the window measured from 1970-01-01 00:00 (UTC for tz-aware origins,
  the same clock for naive ones). A numeric origin is recorded as given.
  """
  if validate_window(window_minutes, base_minutes) == 1:
    return "n/a"
  if origin is None:
    raise ValueError("windows longer than the base period need an origin.")
  if isinstance(origin, (int, float, np.integer, np.floating)):
    return f"{float(origin)!r} (numeric origin)"
  stamp = pd.Timestamp(origin)
  if stamp.tzinfo is not None:
    stamp, tag, epoch = stamp.tz_convert("UTC"), "UTC", pd.Timestamp("1970-01-01", tz="UTC")
  else:
    tag, epoch = "naive", pd.Timestamp("1970-01-01")
  offset = float(((stamp - epoch) / pd.Timedelta(minutes=1)) % window_minutes)
  return f"{offset!r} min after 1970-01-01T00:00 ({tag})"


@dataclass(frozen=True)
class PreRegistration:
  """What an experiment commits to before it runs.

  Attributes:
    metric, unit, direction, alpha: What is tested, and at which one-sided level.
    test_id: The one test chosen, from `analysis.TESTS`.
    analysis_c: "" for tests without a bound, "C" or "stress" for the C the test uses.
    window_minutes, base_minutes: Randomization window and base period.
    window_origin: `window_origin_label` of the window grid.
    weeks: Planned duration in whole weeks.
    clusters: "<n> clusters, ids sha256:<checksum>, rule: <rule>".
    min_windows_per_arm: The analysis' drop rule.
    residual: How outcomes are residualised (the moments' `residual`).
    target: Planned power and the scenario it refers to; informational.
    seed: Seed of the simulation behind `target` (0 for closed form).
    package: Package versions; informational.
    params: Sorted (name, value) pairs the analysis must reuse, e.g. t1 and
      the checksum of C.
  """

  metric: str
  unit: str
  direction: Direction
  alpha: float
  test_id: str
  analysis_c: str
  window_minutes: float
  base_minutes: float
  window_origin: str
  weeks: int
  clusters: str
  min_windows_per_arm: int
  residual: str
  target: str
  seed: int
  package: str
  params: tuple[tuple[str, str], ...] = ()

  def __post_init__(self) -> None:
    if self.test_id not in analysis.TESTS:
      raise ValueError(f"unknown test {self.test_id!r}; choose from {analysis.TESTS}.")
    object.__setattr__(self, "params", tuple(sorted((str(k), str(v)) for k, v in self.params)))

  def to_line(self) -> str:
    """The pre-registration as one line: "carryover-prereg/1; key=value; ..."."""
    parts = [PREFIX]
    for item in fields(self):
      if item.name == "params":
        continue
      value = getattr(self, item.name)
      text = repr(value) if isinstance(value, float) else str(value)
      parts.append(f"{item.name}={_escape(text)}")
    parts += [f"param.{_escape(name)}={_escape(value)}" for name, value in self.params]
    return "; ".join(parts)

  @classmethod
  def from_line(cls, line: str) -> "PreRegistration":
    parts = [part.strip() for part in line.strip().split(";")]
    if not parts or parts[0] != PREFIX:
      raise ValueError(f"not a pre-registration line: it must start with {PREFIX!r}.")
    values: dict[str, object] = {}
    params = []
    for part in parts[1:]:
      if not part:
        continue
      key, _, value = part.partition("=")
      if key.startswith("param."):
        params.append((_unescape(key[len("param."):]), _unescape(value)))
      else:
        values[key] = _unescape(value)
    kinds = {item.name: item.type for item in fields(cls)}
    absent = [name for name in kinds if name != "params" and name not in values]
    if absent:
      raise ValueError(f"pre-registration line lacks {absent}.")
    for name in ("alpha", "window_minutes", "base_minutes"):
      values[name] = float(values[name])
    for name in ("weeks", "min_windows_per_arm", "seed"):
      values[name] = int(values[name])
    return cls(**{name: values[name] for name in kinds if name != "params"}, params=tuple(params))

  @property
  def parameters(self) -> dict[str, str]:
    return dict(self.params)


def preregistration(
  comparison,
  *,
  design: str,
  test: str,
  scenario: str,
  weeks: int,
  cluster_rule: str,
  analysis_c: str | None = None,
  min_windows_per_arm: int = 5,
  residual_note: str = "",
) -> PreRegistration:
  """Pre-registration for one design, test and duration of a `compare_designs` result.

  Args:
    comparison: A `DesignComparison`.
    design: Label of the chosen design.
    test: The one test to pre-register, from `analysis.TESTS`.
    scenario: Label of the scenario the planned power refers to.
    weeks: Whole weeks the experiment will run.
    cluster_rule: How the cluster set was chosen, in words.
    analysis_c: "C" (default) or "stress", for tests that use the bound.
    min_windows_per_arm: The analysis' drop rule.
    residual_note: Anything to add about how outcomes are residualised.
  """
  settings = comparison.settings
  if design not in settings["designs"]:
    raise KeyError(f"no design labelled {design!r}.")
  if test not in settings["tests"]:
    raise ValueError(f"test {test!r} was not part of the comparison.")
  matches = [s for s in settings["scenarios"] if s.label == scenario]
  if not matches:
    raise KeyError(f"no scenario labelled {scenario!r}.")
  if int(weeks) != weeks or weeks < 1:
    raise ValueError("weeks must be a whole number of at least 1.")
  truth = matches[0]
  moments = settings["designs"][design]
  bound = settings["tail_bound"]
  needs_c = test in analysis.TESTS_NEEDING_C
  if needs_c:
    analysis_c = analysis_c or "C"
    if analysis_c not in ("C", "stress"):
      raise ValueError("analysis_c must be 'C' or 'stress'.")
  elif analysis_c:
    raise ValueError(f"test {test!r} does not use a bound; leave analysis_c unset.")
  stress = analysis_c == "stress"

  cov = moments.plan(float(weeks), se_inflation=settings["se_inflation"])
  common = dict(direction=settings["direction"], alpha=settings["alpha"], tail_bound=bound, stress=stress)
  params: dict[str, str] = {}
  seed = 0
  if test in CLOSED_FORM_TESTS:
    power = closed_form_power(cov, test, truth, **common)
  else:
    seed = settings["seed"]
    power = simulated_power(cov, test, truth, n_draws=settings["n_draws"], seed=seed, **common).power
  if needs_c:
    c = bound.window_values(moments.cluster_ids, moments.window_minutes, stress=stress)
    params["C_source"] = bound.stress_source if stress else bound.source
    params["C_fp"] = bound.fingerprint(stress=stress)
    params["window_C_mean"] = repr(float(np.sum(cov.weights * c)))
  if test == "two_endpoint":
    params["tail_sign"] = "same"
  if test in analysis.MOMENT_BOUND_TESTS:
    spread = window_l1_spread_sd(cov, truth)
    params["assumed_window_spread_sd"] = repr(spread)
    params["t1"] = repr(analysis.planning_t1(spread, cov.v11))

  residual = moments.residual + (f"; note: {residual_note}" if residual_note else "")
  target = (
    f"power {power:.2f} at '{truth.label}' (total {total_effect(cov, truth):+.4g}); "
    f"spread {truth.l1_spread_sd}; se_inflation {settings['se_inflation']:g}"
  )
  if test not in CLOSED_FORM_TESTS:
    target += f"; n_draws {settings['n_draws']}; draws keyed by (seed, sha256(cluster id))"
  return PreRegistration(
    metric=settings["metric"],
    unit=settings["unit"],
    direction=settings["direction"],
    alpha=settings["alpha"],
    test_id=test,
    analysis_c=analysis_c or "",
    window_minutes=float(moments.window_minutes),
    base_minutes=float(moments.base_minutes),
    window_origin=window_origin_label(moments.origin, moments.window_minutes, moments.base_minutes),
    weeks=int(weeks),
    clusters=(
      f"{moments.num_clusters} clusters, ids sha256:{cluster_checksum(moments.cluster_ids)}, "
      f"rule: {cluster_rule}"
    ),
    min_windows_per_arm=min_windows_per_arm,
    residual=residual,
    target=target,
    seed=seed,
    package=(
      f"switchback-power-calculator {switchback_power_calculator.__version__}; "
      "carryover 0.1.0.dev0"
    ),
    params=tuple(params.items()),
  )


def _is_number(text: str) -> bool:
  try:
    return math.isfinite(float(text))
  except ValueError:
    return False


def check(line: str | PreRegistration, run: Mapping[str, object]) -> tuple[tuple[str, ...], tuple[str, ...]]:
  """Compare what was run with what was pre-registered.

  `run` describes the analysis as carried out. Keys, all optional except
  `test_id`: `metric`, `direction`, `alpha`, `test_id`, `analysis_c`,
  `window_minutes`, `window_origin`, `weeks_run`, `cluster_ids` (the set
  handed to `estimate_lags`, before its drops), `min_windows_per_arm`,
  `residual`, `params` (name -> value) and `estimates` (the `LagEstimates`).

  Returns `(deviations, notes)`, each a tuple of sentences. A deviation
  means the planned power and the pre-registered test no longer describe
  the analysis. Clusters dropped by the pre-registered drop rule are a note.
  """
  planned = line if isinstance(line, PreRegistration) else PreRegistration.from_line(line)
  if "test_id" not in run:
    raise KeyError("run must name the test_id that was analysed.")
  deviations: list[str] = []
  notes: list[str] = []

  def differs(name: str, ran: object, registered: object) -> None:
    deviations.append(f"{name}: ran {ran}, pre-registered {registered}.")

  for name in ("metric", "direction", "test_id", "analysis_c", "window_origin", "residual"):
    if name in run and str(run[name]) != str(getattr(planned, name)):
      differs(name, run[name], getattr(planned, name))
  for name in ("alpha", "window_minutes"):
    if name in run and not math.isclose(float(run[name]), getattr(planned, name), rel_tol=1e-9):
      differs(name, run[name], getattr(planned, name))
  if "min_windows_per_arm" in run and int(run["min_windows_per_arm"]) != planned.min_windows_per_arm:
    differs("min_windows_per_arm", run["min_windows_per_arm"], planned.min_windows_per_arm)
  if "weeks_run" in run and abs(float(run["weeks_run"]) - planned.weeks) > 1 / 7:
    deviations.append(
      f"weeks: ran {float(run['weeks_run']):g}, pre-registered {planned.weeks}; "
      "the planned power no longer applies."
    )
  if "cluster_ids" in run:
    checksum = cluster_checksum(run["cluster_ids"])
    if f"ids sha256:{checksum}" not in planned.clusters:
      deviations.append(
        f"clusters: the set analysed (sha256:{checksum}) is not the pre-registered one "
        f"({planned.clusters})."
      )
  registered = planned.parameters
  for name, value in dict(run.get("params", {})).items():
    if name not in registered:
      deviations.append(f"param {name}: used {value}, but it was not pre-registered.")
    elif _is_number(str(value)) and _is_number(registered[name]):
      if not math.isclose(float(value), float(registered[name]), rel_tol=1e-9):
        differs(f"param {name}", value, registered[name])
    elif str(value) != registered[name]:
      differs(f"param {name}", value, registered[name])

  estimates = run.get("estimates")
  if isinstance(estimates, LagEstimates) and estimates.n_dropped:
    notes.append(
      f"{estimates.n_dropped} of {estimates.n_input_clusters} clusters dropped by the "
      f"estimator's rules ({estimates.dropped_orders_share:.1%} of orders)."
    )
  return tuple(deviations), tuple(notes)
