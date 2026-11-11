"""Compare candidate designs: noise, detectable effects and weeks needed per test."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from switchback_power_calculator.carryover import analysis
from switchback_power_calculator.carryover.power import (
  CLOSED_FORM_TESTS,
  WeeksResult,
  closed_form_power,
  mde,
  total_effect,
  weeks_to_power,
)
from switchback_power_calculator.carryover.simulate import (
  cluster_normals,
  simulated_power,
  simulated_weeks,
)
from switchback_power_calculator.carryover.tail import ASSUMPTION, TailBound
from switchback_power_calculator.carryover.types import (
  Direction,
  Scenario,
  orientation,
  validate_target_power,
)

FEW_WINDOWS = 50
DEFAULT_CURVE_SHARES = (-0.5, -0.25, 0.0, 0.1, 0.2, 0.3, 0.4)
_NOT_MODELLED = (
  "Not modelled: guardrail and other metrics, cost of a longer test, concurrent experiments, "
  "seasonality beyond the pre-period. Differences under about 0.05 power or one week between "
  "window choices are ties."
)
_MULTIPLICITY = (
  "Each test keeps its false-improvement rate at or below alpha only under its own assumption; "
  "lag-0 and lag-0 + lag-1 report improvements more often than alpha when later carryover "
  "offsets the early effect. Running several tests and reporting the one that passes does not "
  "keep alpha: pre-register one test."
)


def _test_label(test: str, analysis_c: str) -> str:
  label = analysis.TEST_LABELS[test]
  if test == "lag0":
    label += " (same-window effect only)"
  if analysis_c:
    label += " [stress C]" if analysis_c == "stress" else " [C]"
  return label


@dataclass(frozen=True, eq=False)
class DesignComparison:
  """Result of `compare_designs`.

  Attributes:
    basis: One row per design: noise of the lag estimates over one week.
    curve: One row per (design, test, decay, rebound share): the smallest
      detectable same-period effect at `curve_weeks`. Closed-form tests only.
    table: One row per (design, scenario, test): weeks to the target power,
      power at fixed durations, and what the test assumes.
    settings: Every input, for the record.
    notes: Lines printed above the tables.
    warnings: Things to check before relying on the numbers.
  """

  basis: pd.DataFrame
  curve: pd.DataFrame
  table: pd.DataFrame
  settings: Mapping[str, object]
  notes: tuple[str, ...] = ()
  warnings: tuple[str, ...] = ()

  def ranking(self, scenario: str, test: str | None = None) -> pd.DataFrame:
    """Rows of one scenario, fewest weeks first. This is not a recommendation."""
    rows = self.table[self.table["scenario"] == scenario]
    if len(rows) == 0:
      raise KeyError(f"no scenario labelled {scenario!r}.")
    if test is not None:
      rows = rows[rows["test"] == test]
    order = {name: i for i, name in enumerate(analysis.TESTS)}
    keyed = rows.assign(
      _runnable=rows["weeks_runnable"].fillna(math.inf),
      _weeks=rows["weeks"].fillna(math.inf),
      _test=rows["test"].map(order),
    )
    keyed = keyed.sort_values(["_runnable", "_weeks", "_test", "analysis_c", "design"], kind="stable")
    return keyed.drop(columns=["_runnable", "_weeks", "_test"]).reset_index(drop=True)

  def to_text(self, *, scenario: str | None = None) -> str:
    """The comparison as plain text: noise, MDE curve, weeks ranking."""
    return _to_text(self, scenario)


def _weeks_text(result_weeks: float, runnable, note: str, simulated: bool) -> str:
  if note == "not_computed":
    return "not computed"
  if note == "not_reachable":
    return "unreachable"
  if note == "beyond_max_weeks":
    return "beyond max"
  if note == "at_or_below_one_week":
    return "<= 1"
  if simulated:
    return f"{int(runnable)}"
  return f"{result_weeks:.2f} -> run {int(runnable)}"


def compare_designs(
  designs: Mapping[str, object],
  scenarios: Sequence[Scenario],
  *,
  direction: Direction,
  tail_bound: TailBound | None = None,
  tests: Sequence[str] | None = None,
  alpha: float = 0.10,
  power_level: float = 0.80,
  se_inflation: float = 1.0,
  budget_se_inflation: float = 1.2,
  spread_check_factor: float | None = 2.0,
  max_weeks: int = 52,
  report_weeks: Sequence[float] = (1, 2, 4),
  curve_weeks: float = 2.0,
  curve_shares: Sequence[float] = DEFAULT_CURVE_SHARES,
  n_draws: int = 1000,
  seed: int = 0,
  metric: str = "metric",
  unit: str = "",
) -> DesignComparison:
  """Compare designs on how long each test needs under each scenario.

  A design is a window length and cluster set, given as the lag moments read
  off a pre-period at that window (`lag_moments_from_aggregates`). Every
  scenario is stated at the base resolution, so all designs are compared on
  the same true effect.

  Args:
    designs: Label -> `ClusterLagMoments`.
    scenarios: True effects to plan for; labels must be distinct.
    direction: "lower_is_better" or "higher_is_better". No default.
    tail_bound: Bound C on carryover after the next period. Needed by every
      test except lag0 and lag01.
    tests: Test ids from `analysis.TESTS`. Default: lag0 and lag01, plus the
      other four when a `tail_bound` is given. Tests 4, 4.5 and 5 are
      simulated and need the scenario's `l1_spread_sd`.
    alpha: One-sided level.
    power_level: Target power.
    se_inflation: Multiplies every planned SE (1.0 = as noisy as the pre-period).
    budget_se_inflation: SE multiplier of the "if noise is higher" column.
    spread_check_factor: Simulated rows also report power when the true
      spread is this multiple of the stated one; None to skip.
    max_weeks: Longest duration considered.
    report_weeks: Durations at which power is reported.
    curve_weeks: Duration of the MDE curve.
    curve_shares: Rebound shares of the MDE curve.
    n_draws: Simulated experiments per evaluation.
    seed: Seed of the simulations.
    metric: Metric name for the text output.
    unit: Unit name for the text output.
  """
  validate_target_power(power_level, alpha)
  sign = orientation(direction)
  if not designs:
    raise ValueError("at least one design is required.")
  if not scenarios:
    raise ValueError("at least one scenario is required.")
  labels = [scenario.label for scenario in scenarios]
  if len(set(labels)) != len(labels):
    raise ValueError("scenario labels must be distinct.")
  if tests is None:
    tests = analysis.TESTS if tail_bound is not None else ("lag0", "lag01")
  unknown = [name for name in tests if name not in analysis.TESTS]
  if unknown:
    raise ValueError(f"unknown test(s) {unknown}; choose from {analysis.TESTS}.")
  tests = tuple(name for name in analysis.TESTS if name in tests)
  if tail_bound is None and any(name in analysis.TESTS_NEEDING_C for name in tests):
    raise ValueError("a tail_bound is required for every test except lag0 and lag01.")
  if budget_se_inflation < 1 or se_inflation <= 0:
    raise ValueError("se_inflation must be positive and budget_se_inflation at least 1.")
  budget = max(budget_se_inflation, se_inflation)
  has_stress = tail_bound is not None and tail_bound.stress_c is not None

  def variants(test: str) -> tuple[str, ...]:
    if test not in analysis.TESTS_NEEDING_C:
      return ("",)
    return ("C", "stress") if has_stress else ("C",)

  warnings: list[str] = []
  notes: list[str] = []
  if tail_bound is not None:
    banner = f"C is an assumption, not estimable from the pre-period: {tail_bound.label()}"
    if has_stress:
      banner += f"; stress {tail_bound.label(stress=True)}"
    notes.append(banner + f". Window C: geometric, assuming {ASSUMPTION}.")

  basis_rows, curve_rows, table_rows = [], [], []
  simulated_evaluations = 0
  started = time.perf_counter()
  decays = {}
  for scenario in scenarios:
    decays.setdefault(scenario.decay_label, scenario.tail_decay)

  for label, moments in designs.items():
    for warning in moments.warnings:
      warnings.append(f"{label}: {warning}")
    week = moments.plan(1.0, se_inflation=se_inflation)
    row = {
      "design": label,
      "window_minutes": moments.window_minutes,
      "num_clusters": moments.num_clusters,
      "windows_per_cluster_week": week.windows_per_cluster,
      "presence": float(np.median(moments.presence)),
      "residual": moments.residual,
      "se_l0_1wk": week.se_l0,
      "se_l0_plus_l1_1wk": week.se_l0_plus_l1,
      "corr_l0_l1_1wk": week.corr_l0_l1,
    }
    if tail_bound is not None:
      for name, stress in (("", False), ("_stress", True)):
        if stress and not has_stress:
          continue
        c = tail_bound.window_values(moments.cluster_ids, moments.window_minutes, stress=stress)
        row[f"window_c_mean{name}"] = float(np.sum(week.weights * c))
        row[f"se_at_bound{name}_1wk"] = week.contrast_se(1.0 + c)
    basis_rows.append(row)

    curve_cov = moments.plan(curve_weeks, se_inflation=se_inflation)
    for decay_label, decay in decays.items():
      for test in tests:
        if test not in CLOSED_FORM_TESTS:
          continue
        for variant in variants(test):
          for share in curve_shares:
            result = mde(
              curve_cov,
              test,
              rebound_share=share,
              tail_decay=decay,
              direction=direction,
              alpha=alpha,
              power_level=power_level,
              tail_bound=tail_bound,
              stress=variant == "stress",
            )
            curve_rows.append(
              {"design": label, "analysis_c": variant, "weeks": curve_weeks, **result.__dict__}
            )

    normals = None
    for scenario in scenarios:
      total = total_effect(week, scenario)
      c_bar = (
        float(np.sum(week.weights * tail_bound.base_values(moments.cluster_ids)))
        if tail_bound is not None
        else math.nan
      )
      for test in tests:
        closed = test in CLOSED_FORM_TESTS
        for variant in variants(test):
          stress = variant == "stress"
          common = dict(
            direction=direction, alpha=alpha, tail_bound=tail_bound, stress=stress
          )
          power_at = {}
          mc_se = spread_check = math.nan
          note = ""
          if closed:
            result = weeks_to_power(
              moments, test, scenario, power_level=power_level,
              se_inflation=se_inflation, max_weeks=max_weeks, **common,
            )
            budget_result = weeks_to_power(
              moments, test, scenario, power_level=power_level,
              se_inflation=budget, max_weeks=max_weeks, **common,
            )
            for weeks in report_weeks:
              power_at[weeks] = closed_form_power(
                moments.plan(weeks, se_inflation=se_inflation), test, scenario, **common
              )
          elif scenario.l1_spread_sd is None:
            result = budget_result = WeeksResult(math.nan, None, "not_computed", math.nan)
            warnings.append(
              f"{label}: {analysis.TEST_LABELS[test]} not computed for scenario "
              f"{scenario.label!r}: it is simulated and needs l1_spread_sd."
            )
          else:
            if normals is None:
              normals = cluster_normals(seed, moments.cluster_ids, n_draws)
            sim = dict(n_draws=n_draws, seed=seed, normals=normals, **common)
            result = simulated_weeks(
              moments, test, scenario, power_level=power_level,
              se_inflation=se_inflation, max_weeks=max_weeks, **sim,
            )
            simulated_evaluations += 8
            if result.weeks_runnable is None:
              budget_result = result
            else:
              budget_result = simulated_weeks(
                moments, test, scenario, power_level=power_level, se_inflation=budget,
                max_weeks=max_weeks, start_weeks=result.weeks_runnable, **sim,
              )
              runnable_cov = moments.plan(result.weeks_runnable, se_inflation=se_inflation)
              mc_se = simulated_power(runnable_cov, test, scenario, **sim).mc_se
              if spread_check_factor is not None and scenario.l1_spread_sd > 0:
                spread_check = simulated_power(
                  runnable_cov, test, scenario, spread_factor=spread_check_factor, **sim
                ).power
            for weeks in report_weeks:
              power_at[weeks] = simulated_power(
                moments.plan(weeks, se_inflation=se_inflation), test, scenario, **sim
              ).power
            if result.note == "not_reachable":
              note = "no practical duration reaches the target"

          few = bool(
            result.weeks_runnable is not None
            and moments.plan(result.weeks_runnable, se_inflation=se_inflation).windows_per_cluster
            < FEW_WINDOWS
          )
          table_rows.append(
            {
              "design": label,
              "window_minutes": moments.window_minutes,
              "scenario": scenario.label,
              "l0": scenario.l0,
              "l1": scenario.l1,
              "tail_decay": scenario.decay_label,
              "l1_spread_sd": scenario.l1_spread_sd,
              "total_effect": total,
              "total_improves": bool(sign * total < 0),
              "total_range_lo": scenario.l0 + scenario.l1 * (1 + c_bar),
              "total_range_hi": scenario.l0 + scenario.l1 if tail_bound is not None else math.nan,
              "test": test,
              "test_number": analysis.TEST_NUMBERS[test],
              "test_label": _test_label(test, variant),
              "analysis_c": variant,
              "assumes": analysis.ASSUMES[test],
              "method": "closed form" if closed else "simulation",
              "weeks": result.weeks,
              "weeks_runnable": result.weeks_runnable,
              "weeks_note": result.note,
              "power_at_runnable": result.power_at_runnable,
              "weeks_budget": budget_result.weeks,
              "weeks_budget_runnable": budget_result.weeks_runnable,
              "weeks_budget_note": budget_result.note,
              "few_windows": few,
              **{f"power_at_{weeks:g}wk": power_at.get(weeks, math.nan) for weeks in report_weeks},
              "mc_se": mc_se,
              "power_at_spread_check": spread_check,
              "note": note,
            }
          )

  table = pd.DataFrame(table_rows)
  if table["few_windows"].any():
    warnings.append(
      f"some rows run with fewer than {FEW_WINDOWS} windows per cluster: planned power may be "
      "optimistic by up to about 0.05 there."
    )
  settings = {
    "direction": direction,
    "alpha": alpha,
    "power_level": power_level,
    "se_inflation": se_inflation,
    "budget_se_inflation": budget,
    "spread_check_factor": spread_check_factor,
    "max_weeks": max_weeks,
    "report_weeks": tuple(report_weeks),
    "curve_weeks": curve_weeks,
    "curve_shares": tuple(curve_shares),
    "n_draws": n_draws,
    "seed": seed,
    "metric": metric,
    "unit": unit,
    "tests": tests,
    "designs": dict(designs),
    "scenarios": tuple(scenarios),
    "tail_bound": tail_bound,
    "simulated_seconds": time.perf_counter() - started if simulated_evaluations else 0.0,
  }
  return DesignComparison(
    basis=pd.DataFrame(basis_rows),
    curve=pd.DataFrame(curve_rows),
    table=table,
    settings=settings,
    notes=tuple(notes),
    warnings=tuple(dict.fromkeys(warnings)),
  )


def _fmt(value: float, digits: int = 3) -> str:
  return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def _to_text(comparison: DesignComparison, scenario: str | None) -> str:
  s = comparison.settings
  unit = f" {s['unit']}" if s["unit"] else ""
  lines = [
    f"Carryover-aware plan for {s['metric']} ({s['direction'].replace('_', ' ')}); "
    f"one-sided alpha {s['alpha']:g}, target power {s['power_level']:g}",
  ]
  lines += list(comparison.notes)

  lines += ["", f"Q1. Noise at each window length (1 week; SE in{unit or ' metric units'})"]
  basis = comparison.basis
  header = f"  {'design':<12} {'clusters':>8} {'SE(l0)':>9} {'SE(l0+l1)':>10} {'corr':>6}"
  bound_cols = [c for c in ("window_c_mean", "se_at_bound_1wk", "window_c_mean_stress", "se_at_bound_stress_1wk") if c in basis]
  names = {
    "window_c_mean": "window C", "se_at_bound_1wk": "SE(at bound)",
    "window_c_mean_stress": "C (stress)", "se_at_bound_stress_1wk": "SE(stress)",
  }
  header += "".join(f" {names[c]:>12}" for c in bound_cols)
  lines.append(header)
  for _, row in basis.iterrows():
    line = (
      f"  {row['design']:<12} {int(row['num_clusters']):>8} {row['se_l0_1wk']:>9.4f} "
      f"{row['se_l0_plus_l1_1wk']:>10.4f} {row['corr_l0_l1_1wk']:>6.2f}"
    )
    line += "".join(f" {row[c]:>12.4f}" for c in bound_cols)
    lines.append(line)

  curve = comparison.curve
  if len(curve):
    for decay in curve["tail_decay"].unique():
      lines += [
        "",
        f"Q2. Smallest detectable same-period effect by rebound share "
        f"({s['curve_weeks']:g} weeks, decay {decay})",
        "    MDE [total effect it implies under the decay], raw sign",
      ]
      part = curve[curve["tail_decay"] == decay]
      for design in part["design"].unique():
        lines.append(f"  {design}")
        sub = part[part["design"] == design]
        for (test, variant), rows in sub.groupby(["test", "analysis_c"], sort=False):
          cells = []
          for _, row in rows.iterrows():
            if not row["reachable"]:
              cells.append(f"s={row['rebound_share']:+.2f}: unreachable")
              continue
            flag = "" if row["total_improves"] else " (total is not an improvement)"
            cells.append(
              f"s={row['rebound_share']:+.2f}: {row['mde_same_period']:.3f} "
              f"[{row['total_effect_at_mde']:+.3f}]{flag}"
            )
          limit = rows["limit_share"].iloc[0]
          limit_text = "none" if math.isinf(limit) else f"{limit:.2f}"
          lines.append(f"    {_test_label(test, variant)}; limit share {limit_text}")
          lines += [f"      {cell}" for cell in cells]

  table = comparison.table
  chosen = [scenario] if scenario is not None else list(table["scenario"].unique())
  for label in chosen:
    ranked = comparison.ranking(label)
    first = ranked.iloc[0]
    lines += [
      "",
      f"Q3. Weeks to {s['power_level']:.0%} power: {label} -> total {first['total_effect']:+.4f}{unit}"
      + (
        f" [tail at bound {first['total_range_lo']:+.4f}, no tail {first['total_range_hi']:+.4f}]"
        if not math.isnan(first["total_range_hi"])
        else ""
      ),
      "    ranked by weeks to target power; not a recommendation",
    ]
    names = [
      row["test_label"] + (", simulated" if row["method"] == "simulation" else "")
      for _, row in ranked.iterrows()
    ]
    width = max(len(name) for name in names) + 2
    design_width = max(6, *(len(str(design)) for design in ranked["design"])) + 2
    lines.append(
      f"  {'#':<4}{'test':<{width}}{'design':<{design_width}}{'weeks':<18}"
      f"{'if noise +' + format((s['budget_se_inflation'] - 1), '.0%'):<18}"
      + (f"power if spread x{s['spread_check_factor']:g}" if s["spread_check_factor"] else "")
    )
    for name, (_, row) in zip(names, ranked.iterrows()):
      simulated = row["method"] == "simulation"
      weeks = _weeks_text(row["weeks"], row["weeks_runnable"], row["weeks_note"], simulated)
      budget = _weeks_text(
        row["weeks_budget"], row["weeks_budget_runnable"], row["weeks_budget_note"], simulated
      )
      extra = ""
      if simulated and not math.isnan(row["power_at_spread_check"]):
        extra = f"{row['power_at_spread_check']:.2f}"
        if row["power_at_spread_check"] < s["power_level"] - 0.10:
          extra += "  fragile to spread"
      if row["few_windows"]:
        extra += "  (few windows: power may be optimistic by up to 0.05)"
      if row["note"]:
        extra += f"  {row['note']}"
      lines.append(
        f"  {row['test_number']:<4}{name:<{width}}{row['design']:<{design_width}}{weeks:<18}{budget:<18}{extra}".rstrip()
      )
    if not first["total_improves"]:
      lines.append(
        "  The total is not an improvement in this scenario; a test that reaches power here "
        "would still report an improvement."
      )
    elif abs(first["total_effect"]) < abs(first["l0"]) * (1 - 1e-9):
      lines.append(
        f"  lag-0 and lag-0 + lag-1 target the same-window and two-window effects, not the total; "
        f"the total here is {abs(first['total_effect'] / first['l0']):.0%} of the same-period effect."
      )
  lines += ["", _MULTIPLICITY, _NOT_MODELLED]
  if comparison.warnings:
    lines += ["", "Warnings:"] + [f"  - {warning}" for warning in comparison.warnings]
  return "\n".join(lines)
