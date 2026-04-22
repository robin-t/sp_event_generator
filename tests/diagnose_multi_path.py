"""
diagnose_multi_path.py

Runs MultiPathForecaster with tiny N_mc and prints the internal state
day by day so you can visually verify correct behaviour:
  - Global x-values computed from mean predicted prices each day
  - Per-ticker per-path price evolution
  - Target/stop exits as they happen
  - Day-of-week advancing through the window

Run from project root:
    python3 diagnose_multi_path.py
"""

import numpy as np
import json
import datetime
from pathlib import Path

from mc.cpp_adapter import CppAdapter, cpp_available
from density.density import DensitySet
from density.indicator import GlobalIndicator
from data.data_store import DataStore
from data.data_loader import DataLoader
from tools.tickers import Tickers

# ── config ────────────────────────────────────────────────────────
N_MC        = 8       # small but enough to see variance
N_TICKERS   = 10
HOLD_DAYS   = 10
TARGET      = 0.05
STOP        = 0.05
HIST        = 100
START_YEAR  = 2021    # search from here
MIN_PRIOR_RETURN = 0.015  # require prior 5-day mean return > 1.5% across tickers
# ─────────────────────────────────────────────────────────────────

def load_tune(tune_path):
    cfg = json.load(open(tune_path))
    R_dens, M_dens, S_dens = [], [], []
    def load_list(entries, out):
        for entry in entries:
            for folder in sorted(Path("density/densities").glob("*")):
                mf = folder / "meta.json"
                if not mf.exists(): continue
                m = json.load(open(mf))
                if (m["indicator_name"] == entry["density_meta"]["indicator_name"]
                        and m["indicator_params"] == entry["density_meta"]["indicator_params"]):
                    d = DensitySet.load(folder)
                    out.append((d, d.indicator))
                    break
    load_list(cfg["R_densities"], R_dens)
    load_list(cfg["M_densities"], M_dens)
    load_list(cfg["S_densities"], S_dens)
    return R_dens, M_dens, S_dens, cfg["best_weights"], cfg.get("sampling_mode","weighted_mean")

# ── load data ─────────────────────────────────────────────────────
print("Loading data...")
ds           = DataStore()
tickers      = Tickers().get("sweden_largecap")
ticker_names = tickers[:N_TICKERS]
raw          = ds.download_full(ticker_names)
dates, close, high, low, _ = DataLoader.align(raw)
N_DAYS = close.shape[1]
print(f"  {close.shape[0]} tickers, {N_DAYS} days loaded")

if not cpp_available():
    print("rms_cpp not available — rebuild first")
    exit(1)

tune_paths = sorted(Path("tune/rms_tunes").glob("*/best.json"))
if not tune_paths:
    print("No tunes found.")
    exit(1)

R_dens, M_dens, S_dens, weights, mode = load_tune(tune_paths[0])
mixture = (mode == "mixture")
cpp = CppAdapter(R_dens, M_dens, S_dens, weights, mixture_mode=mixture)

# ── pick test day: first Monday after a strong positive prior week ──
t = None
fallback = None
for i in range(HIST + 10, N_DAYS - HOLD_DAYS - 2):
    d = dates[i]
    if hasattr(d, "astype"):
        d = d.astype("datetime64[D]").astype(datetime.date)
    if d.year < START_YEAR:
        continue
    if d.weekday() != 0:  # Monday only
        continue
    if not np.all(np.isfinite(close[:N_TICKERS, i-1])):
        continue

    # Compute mean return over prior 5 trading days across all tickers
    p_now  = close[:N_TICKERS, i-1]
    p_prev = close[:N_TICKERS, i-6]
    valid  = (p_prev > 0) & np.isfinite(p_now) & np.isfinite(p_prev)
    if valid.sum() < N_TICKERS // 2:
        continue
    prior_ret = np.nanmean((p_now[valid] - p_prev[valid]) / p_prev[valid])

    if fallback is None:
        fallback = i

    if prior_ret >= MIN_PRIOR_RETURN:
        t = i
        print(f"  Found: {str(dates[i])[:10]} (prior 5d mean return = {prior_ret:+.2%})")
        break

if t is None:
    t = fallback or (N_DAYS // 2)
    print(f"  No strongly positive Monday found, using {str(dates[t])[:10]}")

date_str = str(dates[t])[:10]
print(f"  Test day: {date_str} (day index {t})\n")

# ── weekdays ──────────────────────────────────────────────────────
DAY_NAMES = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
weekdays = np.full(HOLD_DAYS, np.nan)
for d in range(HOLD_DAYS):
    idx = t + d
    if idx < N_DAYS:
        try:
            day = dates[idx]
            if hasattr(day, "astype"):
                day = day.astype("datetime64[D]").astype(datetime.date)
            weekdays[d] = float(day.weekday())
        except Exception:
            pass

print("Forecast window:")
for d in range(HOLD_DAYS):
    idx = t + d
    dow = weekdays[d]
    name = DAY_NAMES[int(dow)] if np.isfinite(dow) else "?"
    actual_c  = close[0, idx] if idx < N_DAYS else np.nan
    print(f"  Day {d+1:2d}: {str(dates[idx])[:10]} ({name})  "
          f"(actual close ticker0={actual_c:.2f})" if np.isfinite(actual_c)
          else f"  Day {d+1:2d}: {str(dates[idx])[:10]} ({name})")

# ── Python GlobalIndicator reference values ───────────────────────
slice_start = max(0, t - HIST)
print("\nPython GlobalIndicator values at t-1 (reference for day 1):")
global_x_ref = {}
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for comp, dens_list in (("R", R_dens), ("M", M_dens), ("S", S_dens)):
        for d, ind in dens_list:
            if d.meta.get("is_global", False):
                key = (f"{comp}_{d.meta['indicator_name']}_"
                       f"{'_'.join(map(str, d.meta['indicator_params']))}")
                gi = GlobalIndicator(d.meta["indicator_name"], [],
                                     close[:N_TICKERS, slice_start:t],
                                     high[:N_TICKERS, slice_start:t],
                                     low[:N_TICKERS,  slice_start:t])
                val = float(gi.compute_series()[-1])
                global_x_ref[key] = val
                print(f"  {key:<45} = {val:+.6f}")

# ── initial prices and indicators ────────────────────────────────
print(f"\nInitial state (close at t-1 = {str(dates[t-1])[:10]}):")
print(f"  {'Ticker':<14} {'Price':>8}  {'Target':>8}  {'Stop':>8}  "
      f"{'Actual 10d':>10}")
for i in range(N_TICKERS):
    P    = close[i, t-1]
    idx_end = t + HOLD_DAYS - 1
    actual_end = close[i, idx_end] if idx_end < N_DAYS else np.nan
    actual_ret = (actual_end - P) / P if np.isfinite(P) and np.isfinite(actual_end) else np.nan
    name = ticker_names[i] if i < len(ticker_names) else f"t{i}"
    print(f"  [{i}] {name:<12}  {P:>8.2f}  {P*(1+TARGET):>8.2f}  "
          f"{P*(1-STOP):>8.2f}  {actual_ret:>+9.2%}")

# ── run MultiPathForecaster ───────────────────────────────────────
print(f"\n{'='*72}")
print(f"MultiPathForecaster  N_mc={N_MC}  hold={HOLD_DAYS}d  "
      f"target={TARGET:.0%}/stop={STOP:.0%}  mode={mode}")
print(f"{'='*72}")

hc_flat = np.ascontiguousarray(
    np.array([close[i, t-HIST:t] for i in range(N_TICKERS)]).ravel())
hh_flat = np.ascontiguousarray(
    np.array([high[i,  t-HIST:t] for i in range(N_TICKERS)]).ravel())
hl_flat = np.ascontiguousarray(
    np.array([low[i,   t-HIST:t] for i in range(N_TICKERS)]).ravel())

forecaster = cpp.make_multi_path_forecaster()
forecaster.init(hc_flat, hh_flat, hl_flat,
                N_TICKERS, HIST, N_MC,
                TARGET, STOP, weekdays, seed=42)

prev_means = [close[i, t-1] for i in range(N_TICKERS)]

for day in range(HOLD_DAYS):
    dow = weekdays[day]
    dow_str = DAY_NAMES[int(dow)] if np.isfinite(dow) else "?"
    actual_date = str(dates[t + day])[:10] if (t + day) < N_DAYS else "?"
    print(f"\n--- Day {day+1:2d} ({dow_str} {actual_date}) ---")

    forecaster.step()
    ret_so_far = forecaster.get_returns()

    print(f"  {'Ticker':<14} {'mean ret':>9}  {'std':>8}  "
          f"{'impl.price':>10}  {'Δprice':>9}  {'actual ret':>10}")
    for i in range(N_TICKERS):
        mean, std = ret_so_far[i]
        P0   = close[i, t-1]
        impl = P0 * (1 + mean) if np.isfinite(mean) and np.isfinite(P0) else np.nan
        delta = impl - prev_means[i] if np.isfinite(impl) else np.nan
        # actual return on this specific day
        actual_day_idx = t + day
        actual_ret = ((close[i, actual_day_idx] - P0) / P0
                      if actual_day_idx < N_DAYS and np.isfinite(close[i, actual_day_idx])
                      else np.nan)
        name = ticker_names[i] if i < len(ticker_names) else f"t{i}"
        sharpe = mean / std if (np.isfinite(mean) and np.isfinite(std) and std > 0) else np.nan
        print(f"  [{i}] {name:<12}  {mean:>+9.4%}  {std:>8.4%}  "
              f"{impl:>10.2f}  {delta:>+9.3f}  {actual_ret:>+9.2%}")
        prev_means[i] = impl if np.isfinite(impl) else prev_means[i]

# ── final summary ─────────────────────────────────────────────────
print(f"\n{'='*72}")
print("Final results vs actual 10-day returns:")
print(f"{'='*72}")
final = forecaster.get_returns()
print(f"  {'Ticker':<14} {'predicted':>10}  {'std':>8}  "
      f"{'sharpe':>7}  {'actual':>10}  {'error':>9}")
for i in range(N_TICKERS):
    mean, std = final[i]
    P0 = close[i, t-1]
    idx_end = t + HOLD_DAYS - 1
    actual = ((close[i, idx_end] - P0) / P0
              if idx_end < N_DAYS and np.isfinite(P0) and np.isfinite(close[i, idx_end])
              else np.nan)
    error = mean - actual if np.isfinite(mean) and np.isfinite(actual) else np.nan
    sharpe = mean / std if (np.isfinite(mean) and np.isfinite(std) and std > 0) else np.nan
    name = ticker_names[i] if i < len(ticker_names) else f"t{i}"
    print(f"  [{i}] {name:<12}  {mean:>+9.4%}  {std:>8.4%}  "
          f"{sharpe:>+7.3f}  {actual:>+9.2%}  {error:>+9.4%}")

# ── live vs static comparison ─────────────────────────────────────
print(f"\n{'='*72}")
print("Live-global vs static-global forecast_ticker:")
print(f"(seed=42, N_mc={N_MC}, hold={HOLD_DAYS}d)")
print(f"{'='*72}")
print(f"  {'Ticker':<14} {'live':>9}  {'static':>9}  {'Δ':>9}  {'actual':>9}")
for i in range(N_TICKERS):
    hh = high[i,  t-HIST:t].copy()
    hl = low[i,   t-HIST:t].copy()
    hc = close[i, t-HIST:t].copy()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_static, _ = cpp.forecast_ticker(
            hh, hl, hc, N_MC, HOLD_DAYS, TARGET, STOP,
            [global_x_ref] * HOLD_DAYS, seed=42)
    mean_live, _ = final[i]
    P0 = close[i, t-1]
    idx_end = t + HOLD_DAYS - 1
    actual = ((close[i, idx_end] - P0) / P0
              if idx_end < N_DAYS and np.isfinite(P0) and np.isfinite(close[i, idx_end])
              else np.nan)
    name = ticker_names[i] if i < len(ticker_names) else f"t{i}"
    print(f"  [{i}] {name:<12}  {mean_live:>+8.4%}  {mean_static:>+8.4%}  "
          f"{mean_live-mean_static:>+8.4%}  {actual:>+8.2%}")

print("\nDone.")


def load_tune(tune_path):
    cfg = json.load(open(tune_path))
    R_dens, M_dens, S_dens = [], [], []
    def load_list(entries, out):
        for entry in entries:
            for folder in sorted(Path("density/densities").glob("*")):
                mf = folder / "meta.json"
                if not mf.exists(): continue
                m = json.load(open(mf))
                if (m["indicator_name"] == entry["density_meta"]["indicator_name"]
                        and m["indicator_params"] == entry["density_meta"]["indicator_params"]):
                    d = DensitySet.load(folder)
                    out.append((d, d.indicator))
                    break
    load_list(cfg["R_densities"], R_dens)
    load_list(cfg["M_densities"], M_dens)
    load_list(cfg["S_densities"], S_dens)
    return R_dens, M_dens, S_dens, cfg["best_weights"], cfg.get("sampling_mode","weighted_mean")

# ── load data ─────────────────────────────────────────────────────
print("Loading data...")
ds      = DataStore()
tickers = Tickers().get("sweden_largecap")
ticker_names = tickers[:N_TICKERS]
raw     = ds.download_full(ticker_names)
dates, close, high, low, _ = DataLoader.align(raw)
N_DAYS  = close.shape[1]
print(f"  {close.shape[0]} tickers, {N_DAYS} days loaded\n")

if not cpp_available():
    print("rms_cpp not available — rebuild first")
    exit(1)

tune_paths = sorted(Path("tune/rms_tunes").glob("*/best.json"))
if not tune_paths:
    print("No tunes found.")
    exit(1)

R_dens, M_dens, S_dens, weights, mode = load_tune(tune_paths[0])
mixture = (mode == "mixture")
cpp = CppAdapter(R_dens, M_dens, S_dens, weights, mixture_mode=mixture)

# ── pick a test day (a Monday so day-of-week is meaningful) ───────
t = None
for i in range(HIST + 5, N_DAYS - HOLD_DAYS - 1):
    d = dates[i]
    if hasattr(d, "astype"):
        d = d.astype("datetime64[D]").astype(datetime.date)
    if d.weekday() == 0:
        t = i
        break

if t is None:
    t = N_DAYS // 2

date_str = str(dates[t])[:10]
print(f"Test day: {date_str} (day index {t})")
print(f"N_mc={N_MC}, N_tickers={N_TICKERS}, hold={HOLD_DAYS}d, "
      f"target={TARGET:.0%}, stop={STOP:.0%}\n")

# ── build weekdays array ──────────────────────────────────────────
weekdays = np.full(HOLD_DAYS, np.nan)
day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
for d in range(HOLD_DAYS):
    idx = t + d
    if idx < N_DAYS:
        try:
            day = dates[idx]
            if hasattr(day, "astype"):
                day = day.astype("datetime64[D]").astype(datetime.date)
            weekdays[d] = float(day.weekday())
        except Exception:
            pass

print("Forecast window weekdays:")
for d in range(HOLD_DAYS):
    idx = t + d
    dow = weekdays[d]
    name = day_names[int(dow)] if np.isfinite(dow) else "?"
    print(f"  Day {d+1}: {str(dates[idx])[:10]} ({name})")

# ── compute Python global x at t-1 for reference ─────────────────
slice_start = max(0, t - HIST)
print("\nPython GlobalIndicator values at t-1 (reference):")
global_x_ref = {}
for comp, dens_list in (("R", R_dens), ("M", M_dens), ("S", S_dens)):
    for d, ind in dens_list:
        if d.meta.get("is_global", False):
            key = (f"{comp}_{d.meta['indicator_name']}_"
                   f"{'_'.join(map(str, d.meta['indicator_params']))}")
            gi = GlobalIndicator(d.meta["indicator_name"], [],
                                 close[:, slice_start:t],
                                 high[:, slice_start:t],
                                 low[:, slice_start:t])
            val = float(gi.compute_series()[-1])
            global_x_ref[key] = val
            print(f"  {key:<45} = {val:+.5f}")

# ── initial prices ────────────────────────────────────────────────
print("\nInitial prices (close at t-1):")
for i in range(N_TICKERS):
    P = close[i, t-1]
    name = ticker_names[i] if i < len(ticker_names) else f"ticker_{i}"
    print(f"  [{i}] {name:<12} P={P:.2f}  "
          f"target={P*(1+TARGET):.2f}  stop={P*(1-STOP):.2f}")

# ── run MultiPathForecaster with step-by-step printing ───────────
print(f"\n{'='*70}")
print("Running MultiPathForecaster step by step...")
print(f"{'='*70}")

# Pack histories
hc_flat = np.ascontiguousarray(
    np.array([close[i, t-HIST:t] for i in range(N_TICKERS)]).ravel())
hh_flat = np.ascontiguousarray(
    np.array([high[i,  t-HIST:t] for i in range(N_TICKERS)]).ravel())
hl_flat = np.ascontiguousarray(
    np.array([low[i,   t-HIST:t] for i in range(N_TICKERS)]).ravel())

forecaster = cpp.make_multi_path_forecaster()
forecaster.init(hc_flat, hh_flat, hl_flat,
                N_TICKERS, HIST, N_MC,
                TARGET, STOP, weekdays, seed=42)

# Step manually so we can print between steps
for day in range(HOLD_DAYS):
    dow = weekdays[day]
    dow_str = day_names[int(dow)] if np.isfinite(dow) else "?"
    print(f"\n--- Day {day+1} ({dow_str}) ---")
    forecaster.step()
    ret_so_far = forecaster.get_returns()
    print(f"  Returns after day {day+1}:")
    for i in range(N_TICKERS):
        mean, std = ret_so_far[i]
        P0 = close[i, t-1]
        print(f"    [{i}] mean={mean:+.5f}  std={std:.5f}  "
              f"(implied exit≈{P0*(1+mean):.2f})")

print(f"\n{'='*70}")
print("Final results:")
print(f"{'='*70}")
final = forecaster.get_returns()
for i in range(N_TICKERS):
    mean, std = final[i]
    P0 = close[i, t-1]
    name = ticker_names[i] if i < len(ticker_names) else f"ticker_{i}"
    print(f"  [{i}] {name:<12}  mean={mean:+.6f}  std={std:.6f}  "
          f"sharpe={mean/std if std>0 else 0:.3f}")

# ── compare with static-global forecast_ticker ───────────────────
print(f"\n{'='*70}")
print("Comparison: MultiPathForecaster vs static-global forecast_ticker")
print(f"(same seed=42, N_mc={N_MC}, hold={HOLD_DAYS}d)")
print(f"{'='*70}")

for i in range(N_TICKERS):
    hh = high[i,  t-HIST:t].copy()
    hl = low[i,   t-HIST:t].copy()
    hc = close[i, t-HIST:t].copy()
    mean_static, std_static = cpp.forecast_ticker(
        hh, hl, hc, N_MC, HOLD_DAYS, TARGET, STOP,
        [global_x_ref] * HOLD_DAYS, seed=42)
    mean_live, std_live = final[i]
    name = ticker_names[i] if i < len(ticker_names) else f"ticker_{i}"
    print(f"  [{i}] {name:<12}  "
          f"live={mean_live:+.5f}  static={mean_static:+.5f}  "
          f"Δ={mean_live-mean_static:+.5f}")

print("\nDone.")
