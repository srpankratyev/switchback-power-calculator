"""Package-level checks for the carryover subpackage, and an end-to-end run."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import switchback_power_calculator
from switchback_power_calculator import carryover

REPO = Path(__file__).resolve().parent.parent


def test_every_exported_name_exists():
  for name in carryover.__all__:
    assert hasattr(carryover, name), name
  assert sorted(carryover.__all__) == sorted(set(carryover.__all__))


def test_top_level_package_is_unchanged_and_does_not_import_the_subpackage():
  assert not any(name.startswith("carryover") for name in switchback_power_calculator.__all__)
  code = (
    "import sys, switchback_power_calculator; "
    "print(any(m.startswith('switchback_power_calculator.carryover') for m in sys.modules))"
  )
  out = subprocess.run(
    [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, check=True
  )
  assert out.stdout.strip() == "False"


def test_plan_then_analyse_an_injected_effect():
  """Plan on a pre-period, then inject a known carryover effect into a later
  experiment on the same panel: the estimates must recover it, and the tests
  must reject about as often as the plan says."""
  from switchback_power_calculator.carryover import (
    TailBound,
    closed_form_power,
    estimate_lags,
    experiment_windows,
    lag_moments_from_aggregates,
    run_tests,
    synthetic_panel,
  )

  bound = TailBound(c=0.43, source="example")
  decay = 0.43 / 1.43
  l0, share = -0.08, 0.3
  l1 = -share * l0
  scenario = bound.scenario(l0, share, label="injected")

  panel = synthetic_panel(150, 4, seed=11)
  weeks = panel["period_start"].min() + pd.to_timedelta([0, 14], unit="D")
  pre = panel[panel["period_start"] < weeks[1]]
  experiment = panel[panel["period_start"] >= weeks[1]].copy()
  moments = lag_moments_from_aggregates(pre, expected_col="expected")
  planned = moments.plan(2)
  planned_power = {
    name: closed_form_power(planned, name, scenario, direction="lower_is_better", tail_bound=bound)
    for name in ("lag0", "lag01", "two_endpoint")
  }

  # One row per cluster-hour; lags are injected hour by hour from the coin flips.
  experiment = experiment.sort_values(["cluster_id", "period_start"]).reset_index(drop=True)
  hour = ((experiment["period_start"] - weeks[1]) / pd.Timedelta(hours=1)).astype(int).to_numpy()
  cluster = pd.factorize(experiment["cluster_id"])[0]
  n_hours = hour.max() + 1
  rng = np.random.default_rng(12)
  rejections = {name: 0 for name in planned_power}
  estimates = []
  n_runs = 150
  for _ in range(n_runs):
    coins = (rng.random((cluster.max() + 1, n_hours)) < 0.5).astype(float)
    effect = l0 * coins
    carry = np.zeros_like(coins)
    for t in range(1, n_hours):
      carry[:, t] = l1 * coins[:, t - 1] + decay * carry[:, t - 1]
    per_unit = (effect + carry)[cluster, hour]
    outcomes = experiment.assign(numerator=experiment["numerator"] + per_unit * experiment["denominator"])
    assignment = experiment[["cluster_id", "period_start"]].assign(treated=coins[cluster, hour])
    est = estimate_lags(
      experiment_windows(outcomes, assignment, window_minutes=60, expected_col="expected"),
      window_minutes=60,
    )
    estimates.append((np.sum(est.weights * est.l0), np.sum(est.weights * est.l1)))
    results = run_tests(
      est, bound.window_values(est.cluster_ids, 60)[0], direction="lower_is_better",
      tests=tuple(planned_power),
    )
    for name in planned_power:
      rejections[name] += results[name].improves

  mean_l0, mean_l1 = np.mean(estimates, axis=0)
  assert mean_l0 == pytest.approx(l0, abs=4 * planned.se_l0 / np.sqrt(n_runs))
  assert mean_l1 == pytest.approx(l1, abs=0.01)
  assert np.std(np.array(estimates)[:, 0], ddof=1) == pytest.approx(planned.se_l0, rel=0.20)
  for name, power in planned_power.items():
    assert 0.2 < power < 0.999
    tolerance = 0.07 + 3 * np.sqrt(power * (1 - power) / n_runs)
    assert rejections[name] / n_runs == pytest.approx(power, abs=tolerance), name
