"""Per-cluster lag-0 and lag-1 effect estimates from a finished experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from switchback_power_calculator.carryover.moments import window_totals
from switchback_power_calculator.carryover.types import LagEstimates, validate_window

DEFAULT_MIN_WINDOWS_PER_ARM = 5


def experiment_windows(
  outcomes: pd.DataFrame,
  assignment: pd.DataFrame,
  *,
  window_minutes: float,
  origin: object = None,
  base_minutes: float = 60,
  cluster_col: str = "cluster_id",
  time_col: str = "period_start",
  numerator_col: str = "numerator",
  denominator_col: str = "denominator",
  expected_col: str | None = None,
  treated_col: str = "treated",
  time_unit_minutes: float | None = None,
) -> pd.DataFrame:
  """Put an experiment's outcomes and assignment on its randomization windows.

  Both frames have one row per cluster and base period. `outcomes` holds the
  sum of the outcome, the number of observations and, optionally, the
  expected sum; `assignment` holds the arm (1 treatment, 0 control). Rows are
  placed on the same window grid the planner uses (`moments.window_totals`),
  so `origin` must be the start of one of the experiment's windows whenever
  the window is longer than the base period.

  A window's arm is the arm of its base periods; if they disagree the call
  raises. A window whose assignment is missing for any base period has an
  unknown arm (never filled in), and `estimate_lags` then leaves it out.

  Returns one row per complete outcome window: `cluster_id`, `window`,
  `residual` (outcome minus expected), `count` and `treated`.
  """
  k = validate_window(window_minutes, base_minutes)
  shared = dict(
    window_minutes=window_minutes,
    base_minutes=base_minutes,
    cluster_col=cluster_col,
    time_col=time_col,
    time_unit_minutes=time_unit_minutes,
  )
  totals = window_totals(
    outcomes,
    origin=origin,
    numerator_col=numerator_col,
    denominator_col=denominator_col,
    expected_col=expected_col,
    **shared,
  )
  for name in (cluster_col, time_col, treated_col):
    if name not in assignment.columns:
      raise KeyError(f"Missing required columns: {[name]}")
  known = assignment[[cluster_col, time_col, treated_col]].dropna(subset=[treated_col])
  arm = pd.to_numeric(known[treated_col], errors="raise")
  if not arm.isin((0, 1)).all():
    raise ValueError(f"{treated_col} must be 1 (treatment), 0 (control) or missing.")
  arms = window_totals(
    known.assign(**{"_arm": arm.astype(float), "_one": 1.0}),
    origin=totals.attrs["origin"],
    numerator_col="_arm",
    denominator_col="_one",
    **shared,
  )
  if not arms["numerator"].isin((0.0, float(k))).all():
    mixed = int((~arms["numerator"].isin((0.0, float(k)))).sum())
    raise ValueError(
      f"{mixed} window(s) hold both arms: check window_minutes and origin against the "
      "experiment's randomization windows."
    )
  arms = arms.assign(treated=arms["numerator"] / k)[["cluster_id", "window", "treated"]]
  out = totals.merge(arms, on=["cluster_id", "window"], how="left")
  out["residual"] = out["numerator"] - out["expected"]
  out = out.rename(columns={"denominator": "count"})
  return out[["cluster_id", "window", "residual", "count", "treated"]]


def estimate_lags(
  windows: pd.DataFrame,
  *,
  window_minutes: float,
  cluster_col: str = "cluster_id",
  window_col: str = "window",
  residual_col: str = "residual",
  count_col: str = "count",
  treated_col: str = "treated",
  min_windows_per_arm: int = DEFAULT_MIN_WINDOWS_PER_ARM,
) -> LagEstimates:
  """Estimate each cluster's same-window and next-window effects.

  `windows` has one row per cluster and randomization window that is present:
  the window's integer index, the sum of the residual outcome over its
  observations (outcome minus expected value), the number of observations,
  and the arm, 1 for treatment and 0 for control. A missing arm means the
  assignment is unknown; such a window is not used and breaks the lag pair
  on either side of it. Windows that are absent do the same.

  Per cluster, the lag-0 effect is the mean residual per observation in
  treated windows minus that in control windows (ratios of sums). The lag-1
  effect is the same comparison for the window that follows, grouped by the
  arm of the window before it. The covariance is design-based: outcomes are
  held fixed and each window's arm is an independent fair coin.

  A cluster is dropped when an arm of either comparison has fewer than
  `min_windows_per_arm` windows or no observations, or when an estimate or
  variance is not finite and positive. Drops are counted in the result.
  """
  if min_windows_per_arm < 1:
    raise ValueError("min_windows_per_arm must be at least 1.")
  columns = [cluster_col, window_col, residual_col, count_col, treated_col]
  absent = [name for name in columns if name not in windows.columns]
  if absent:
    raise KeyError(f"Missing required columns: {absent}")
  if len(windows) == 0:
    raise ValueError("Input frame is empty.")

  cluster = windows[cluster_col].map(str)
  window = pd.to_numeric(windows[window_col], errors="raise")
  if window.isna().any() or np.any(window != np.floor(window)):
    raise ValueError(f"{window_col} must hold whole-number window indices.")
  window = window.astype(np.int64)
  if pd.MultiIndex.from_arrays([cluster, window]).has_duplicates:
    raise ValueError("each cluster may have at most one row per window.")
  residual = pd.to_numeric(windows[residual_col], errors="raise").to_numpy(dtype=float)
  count = pd.to_numeric(windows[count_col], errors="raise").to_numpy(dtype=float)
  if not (np.all(np.isfinite(residual)) and np.all(np.isfinite(count))):
    raise ValueError(f"{residual_col} and {count_col} must be finite.")
  if np.any(count < 0):
    raise ValueError(f"{count_col} must be non-negative.")
  treated = pd.to_numeric(windows[treated_col], errors="raise").to_numpy(dtype=float)
  known = ~np.isnan(treated)
  if not np.all(np.isin(treated[known], (0.0, 1.0))):
    raise ValueError(f"{treated_col} must be 1 (treatment), 0 (control) or missing.")

  row, ids = pd.factorize(cluster, sort=True)
  col = (window - window.min()).to_numpy()
  shape = (len(ids), int(col.max()) + 1)
  R = np.zeros(shape)
  C = np.zeros(shape)
  W = np.zeros(shape)
  U = np.zeros(shape, dtype=bool)
  R[row, col] = residual
  C[row, col] = count
  W[row[known], col[known]] = treated[known]
  U[row[known], col[known]] = True

  # Lag 0: every usable window, grouped by its own arm.
  in_t = W * U
  in_c = (1 - W) * U
  # Lag 1: usable pairs (t, t + 1), outcome of t + 1 grouped by the arm of t.
  pair = U[:, :-1] & U[:, 1:]
  R1, C1 = R[:, 1:], C[:, 1:]
  pair_t = W[:, :-1] * pair
  pair_c = (1 - W[:, :-1]) * pair

  with np.errstate(divide="ignore", invalid="ignore"):
    Bt = (in_t * C).sum(1)
    D = (in_c * C).sum(1)
    B1 = (pair_t * C1).sum(1)
    D1 = (pair_c * C1).sum(1)
    mu_t = (in_t * R).sum(1) / Bt
    mu_c = (in_c * R).sum(1) / D
    mu_t1 = (pair_t * R1).sum(1) / B1
    mu_c1 = (pair_c * R1).sum(1) / D1
    a_t = (R - mu_t[:, None] * C) / Bt[:, None]
    a_c = (R - mu_c[:, None] * C) / D[:, None]
    a_t1 = (R1 - mu_t1[:, None] * C1) / B1[:, None]
    a_c1 = (R1 - mu_c1[:, None] * C1) / D1[:, None]
    psi0 = np.where(U, a_t + a_c, 0.0)
    psi1 = np.where(pair, a_t1 + a_c1, 0.0)
    l0 = mu_t - mu_c
    l1 = mu_t1 - mu_c1
    s00 = 0.25 * (psi0**2).sum(1)
    s11 = 0.25 * (psi1**2).sum(1)
    s01 = 0.25 * (psi0[:, :-1] * psi1).sum(1)

  orders = (U * C).sum(1)
  arm_windows = np.minimum.reduce(
    [in_t.sum(1), in_c.sum(1), pair_t.sum(1), pair_c.sum(1)]
  )
  few = arm_windows < min_windows_per_arm
  empty_arm = ~few & ~((Bt > 0) & (D > 0) & (B1 > 0) & (D1 > 0))
  with np.errstate(invalid="ignore"):
    valid = np.isfinite(l0) & np.isfinite(l1) & (s00 > 0) & (s11 > 0) & np.isfinite(s01)
  degenerate = ~few & ~empty_arm & ~valid
  keep = ~(few | empty_arm | degenerate)
  if not keep.any():
    raise ValueError("no cluster has enough windows in both arms to estimate lag effects.")

  warnings = []
  for mask, reason in (
    (few, f"fewer than {min_windows_per_arm} windows in an arm"),
    (empty_arm, "an arm without observations"),
    (degenerate, "a non-finite estimate or a zero variance"),
  ):
    if mask.any():
      warnings.append(f"{int(mask.sum())} cluster(s) dropped: {reason}.")
  total_orders = orders.sum()
  kept_orders = orders[keep]
  return LagEstimates(
    cluster_ids=tuple(ids[keep]),
    weights=kept_orders / kept_orders.sum(),
    l0=l0[keep],
    l1=l1[keep],
    s00=s00[keep],
    s01=s01[keep],
    s11=s11[keep],
    window_minutes=float(window_minutes),
    orders=kept_orders,
    n_input_clusters=len(ids),
    n_dropped=int((~keep).sum()),
    dropped_orders_share=float(orders[~keep].sum() / total_orders) if total_orders > 0 else 0.0,
    warnings=tuple(warnings),
  )
