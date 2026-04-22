"""
diagnose_tune_xvalues.py

Checks whether the indicator x-values computed during tuning actually
fall within the density x-bin ranges. Also checks what the model
predicts vs actual RMS for a sample of days.

Run from project root:
    python3 diagnose_tune_xvalues.py
"""

import numpy as np
import json
import warnings
from pathlib import Path

from density.density import DensitySet
from density.indicator import GlobalIndicator, Indicator
from data.data_store import DataStore
from data.data_loader import DataLoader
from tools.tickers import Tickers
from data.features import FeatureBuilder
from mc.cpp_adapter import CppAdapter, cpp_available
from mc.transition_model import TransitionModel

# ── load data ──────────────────────────────────────────────────────
print("Loading data...")
ds      = DataStore()
tickers = Tickers().get("sweden_largecap")
raw     = ds.download_full(tickers)
dates, close, high, low, _ = DataLoader.align(raw)
N_DAYS  = close.shape[1]
HIST    = 100
print(f"  {close.shape[0]} tickers, {N_DAYS} days\n")

# ── load most recent tune ──────────────────────────────────────────
tune_paths = sorted(Path("tune/rms_tunes").glob("*/best.json"))
if not tune_paths:
    print("No tunes found.")
    exit(1)

cfg = json.load(open(tune_paths[-1]))
print(f"Tune: {tune_paths[-1].parent.name}")
print(f"  Mode: {cfg.get('sampling_mode')}")
print(f"  Date spec: {cfg.get('tuning_date_spec')}")
print()

# ── load densities ─────────────────────────────────────────────────
def load_dens_list(entries):
    out = []
    for entry in entries:
        meta = entry["density_meta"]
        for folder in sorted(Path("density/densities").glob("*")):
            mf = folder / "meta.json"
            if not mf.exists(): continue
            m = json.load(open(mf))
            if (m["indicator_name"] == meta["indicator_name"] and
                    m["indicator_params"] == meta["indicator_params"]):
                d = DensitySet.load(folder)
                out.append((d, d.indicator))
                break
    return out

R_dens = load_dens_list(cfg["R_densities"])
M_dens = load_dens_list(cfg["M_densities"])
S_dens = load_dens_list(cfg["S_densities"])

print("Loaded densities:")
for comp, dens_list in [("R", R_dens), ("M", M_dens), ("S", S_dens)]:
    for d, ind in dens_list:
        is_glob = d.meta.get("is_global", False)
        print(f"  {comp}: {ind.name} {ind.params} global={is_glob} "
              f"x=[{d.x_bins[0]:.4f}..{d.x_bins[-1]:.4f}]")

# ── sample days from tune window ──────────────────────────────────
from tools.date_range import parse_date_mask
tune_spec = cfg.get("tuning_date_spec")
if tune_spec:
    mask = parse_date_mask(tune_spec, dates)
    valid = np.where(mask)[0]
    valid = valid[valid >= HIST]
else:
    valid = np.arange(HIST, N_DAYS - 1)

print(f"\nTune window: {tune_spec or 'full'}")
print(f"Valid days: {len(valid)}")

np.random.seed(42)
sample_days = np.random.choice(valid[valid < N_DAYS - 1], size=min(200, len(valid)),
                                replace=False)

# ── precompute global series ───────────────────────────────────────
global_series = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for comp, dens_list in [("R", R_dens), ("M", M_dens), ("S", S_dens)]:
        for d, ind in dens_list:
            if d.meta.get("is_global", False):
                key = (f"{comp}_{d.meta['indicator_name']}_"
                       f"{'_'.join(map(str, d.meta['indicator_params']))}")
                if key not in global_series:
                    gi = GlobalIndicator(
                        d.meta["indicator_name"], d.meta["indicator_params"],
                        close, high, low, dates=dates)
                    global_series[key] = gi.compute_series()

# ── check x-value coverage ────────────────────────────────────────
print(f"\n{'='*65}")
print("X-value coverage check (200 random tune-window samples):")
print(f"{'='*65}")

# For each density, collect x-values from sample days
for comp, dens_list in [("R", R_dens), ("M", M_dens), ("S", S_dens)]:
    for d, ind in dens_list:
        is_glob = d.meta.get("is_global", False)
        x_lo, x_hi = d.x_bins[0], d.x_bins[-1]

        x_vals = []
        for day in sample_days:
            if is_glob:
                key = (f"{comp}_{d.meta['indicator_name']}_"
                       f"{'_'.join(map(str, d.meta['indicator_params']))}")
                series = global_series.get(key)
                x = float(series[day - 1]) if series is not None else np.nan
            else:
                # Compute per-ticker indicator for ticker 0
                h = high[0,  day - HIST:day]
                l = low[0,   day - HIST:day]
                c = close[0, day - HIST:day]
                if not np.all(np.isfinite(c)): continue
                try:
                    indicator = Indicator(d.meta["indicator_name"],
                                         d.meta["indicator_params"])
                    x_series = indicator.compute(h, l, c)
                    x = float(x_series[-1]) if len(x_series) > 0 else np.nan
                except Exception:
                    x = np.nan
            if np.isfinite(x):
                x_vals.append(x)

        if not x_vals:
            print(f"  {comp} {ind.name}{ind.params}: NO VALID X-VALUES")
            continue

        x_arr = np.array(x_vals)
        n_below = (x_arr < x_lo).sum()
        n_above = (x_arr > x_hi).sum()
        n_total = len(x_arr)
        pct_clipped = 100 * (n_below + n_above) / n_total

        flag = "✓" if pct_clipped < 5 else ("⚠" if pct_clipped < 15 else "✗")
        print(f"  {flag} {comp} {ind.name}{ind.params}: "
              f"x=[{x_arr.min():+.4f}..{x_arr.max():+.4f}] "
              f"dens=[{x_lo:+.4f}..{x_hi:+.4f}] "
              f"clipped={pct_clipped:.1f}%"
              + (f"  ({n_below} below, {n_above} above)" if pct_clipped > 0 else ""))

# ── predict vs actual for a few samples ───────────────────────────
print(f"\n{'='*65}")
print("Prediction vs actual RMS for 20 sample days (ticker 0):")
print(f"{'='*65}")

if not cpp_available():
    print("  C++ not available")
else:
    weights = cfg["best_weights"]
    cpp = CppAdapter(R_dens, M_dens, S_dens, weights,
                     mixture_mode=(cfg.get("sampling_mode") == "mixture"))

    N_MC = 2000
    check_days = sample_days[:20]

    print(f"  {'day':>6}  {'R_real':>8} {'R_pred':>8}  "
          f"{'M_real':>8} {'M_pred':>8}  "
          f"{'S_real':>8} {'S_pred':>8}")
    print("  " + "-"*65)

    R_errs, M_errs, S_errs = [], [], []

    for day in check_days:
        h = high[0,  day - HIST:day]
        l = low[0,   day - HIST:day]
        c = close[0, day - HIST:day]
        if not np.all(np.isfinite(c)): continue

        P  = c[-1]
        H  = high[0,  day]
        L  = low[0,   day]
        C  = close[0, day]
        if not (np.isfinite(H) and np.isfinite(L) and np.isfinite(C) and H > L):
            continue

        R_real = (H - L) / P
        mid    = (H + L) / 2
        M_real = (mid - P) / P
        half   = (H - L) / 2
        S_real = (C - mid) / half

        # Build global_x
        global_x = {}
        for comp, dens_list in [("R", R_dens), ("M", M_dens), ("S", S_dens)]:
            for d, ind in dens_list:
                if d.meta.get("is_global", False):
                    key = (f"{comp}_{d.meta['indicator_name']}_"
                           f"{'_'.join(map(str, d.meta['indicator_params']))}")
                    series = global_series.get(key)
                    global_x[key] = float(series[day - 1]) if series is not None else np.nan

        err = cpp.evaluate_sample(
            h, l, c, R_real, M_real, S_real, C,
            N_MC, global_x, seed=42)

        # To get predictions, run n_mc steps manually
        from mc.transition_model import TransitionModel
        model = TransitionModel(R_dens, M_dens, S_dens, weights,
                                sampling_mode=cfg.get("sampling_mode", "weighted_mean"))
        x_vals = model.compute_indicator_values(h, l, c, global_x=global_x)
        preds = [model.step_from_x(x_vals, P) for _ in range(N_MC)]
        closes_mc = np.array([p[0] for p in preds])
        highs_mc  = np.array([p[1] for p in preds])
        lows_mc   = np.array([p[2] for p in preds])

        R_pred = np.mean((highs_mc - lows_mc) / P)
        mid_mc = (highs_mc + lows_mc) / 2
        M_pred = np.mean((mid_mc - P) / P)
        half_mc = (highs_mc - lows_mc) / 2
        S_pred = np.mean(np.where(half_mc > 0, (closes_mc - mid_mc) / half_mc, 0))

        R_errs.append((R_pred - R_real)**2)
        M_errs.append((M_pred - M_real)**2)
        S_errs.append((S_pred - S_real)**2)

        date_str = str(dates[day])[:10]
        print(f"  {date_str}  {R_real:>+8.4f} {R_pred:>+8.4f}  "
              f"{M_real:>+8.4f} {M_pred:>+8.4f}  "
              f"{S_real:>+8.4f} {S_pred:>+8.4f}")

    print()
    print(f"  MSE(R)={np.mean(R_errs):.6f}  "
          f"MSE(M)={np.mean(M_errs):.6f}  "
          f"MSE(S)={np.mean(S_errs):.6f}")

    # Compare to naive (predict marginal mean)
    # naive MSE(M) ≈ Var(M), naive MSE(R) ≈ Var(R)
    R_vals_naive, M_vals_naive, S_vals_naive = [], [], []
    for day in check_days:
        P  = close[0, day-1]
        H, L, C = high[0, day], low[0, day], close[0, day]
        if not (np.isfinite(P) and P > 0 and np.isfinite(H)
                and np.isfinite(L) and np.isfinite(C) and H > L):
            continue
        R_vals_naive.append((H - L) / P)
        mid = (H + L) / 2
        M_vals_naive.append((mid - P) / P)
        half = (H - L) / 2
        S_vals_naive.append((C - mid) / half)

    naive_R = float(np.var(R_vals_naive)) if R_vals_naive else 1.0
    naive_M = float(np.var(M_vals_naive)) if M_vals_naive else 1.0
    naive_S = float(np.var(S_vals_naive)) if S_vals_naive else 1.0

    print(f"  Naive Var(R)={naive_R:.6f}  "
          f"Var(M)={naive_M:.6f}  "
          f"Var(S)={naive_S:.6f}")
    print(f"  Model/Naive ratio: "
          f"R={np.mean(R_errs)/naive_R:.3f}  "
          f"M={np.mean(M_errs)/naive_M:.3f}  "
          f"S={np.mean(S_errs)/naive_S:.3f}")
    print()
    print("  Ratio < 1.0 = model beats naive")
    print("  Ratio > 1.0 = model worse than naive (current situation)")

print("\nDone.")
