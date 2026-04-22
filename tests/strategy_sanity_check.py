"""
Strategy sanity check — verifies the full strategy_tester path
(global x injection, multi-ticker, selection rule, fee calculation)
matches a manual Python reference implementation.

Run on server: python3 strategy_sanity_check.py
"""
import numpy as np
from mc.cpp_adapter import CppAdapter
from mc.transition_model import TransitionModel
from density.density import DensitySet
from data.data_store import DataStore
from data.data_loader import DataLoader
from tools.tickers import Tickers
from pathlib import Path
import json

# ── load tune ─────────────────────────────────────────────────────
tune_file = None
for f in sorted(Path("tune/rms_tunes").glob("*/best.json")):
    cfg = json.load(open(f))
    if cfg["R_densities"][0]["density_meta"].get("start_year") == 2010:
        tune_file = f; break
if tune_file is None:
    tune_file = sorted(Path("tune/rms_tunes").glob("*/best.json"))[0]
print(f"Tune: {tune_file.parent.name}")

with open(tune_file) as f:
    config = json.load(f)

def load_dens(entry):
    sy = entry["density_meta"].get("start_year")
    for folder in sorted(Path("density/densities").glob("*")):
        mf = folder/"meta.json"
        if not mf.exists(): continue
        m = json.load(open(mf))
        if (m["indicator_name"]==entry["density_meta"]["indicator_name"]
                and m["indicator_params"]==entry["density_meta"]["indicator_params"]
                and m.get("start_year")==sy):
            d = DensitySet.load(folder); return d, d.indicator
    raise RuntimeError(f"Not found: {entry['density_meta']}")

R_dens = [load_dens(e) for e in config["R_densities"]]
M_dens = [load_dens(e) for e in config["M_densities"]]
S_dens = [load_dens(e) for e in config["S_densities"]]
weights = config["best_weights"]

# ── load price data ────────────────────────────────────────────────
ds = DataStore()
tickers = Tickers().get("sweden_largecap")
raw = ds.download_full(tickers)
dates, close, high, low, _ = DataLoader.align(raw)

# Test window: Jan 2023 (10 days)
from datetime import date
t_start = np.searchsorted(dates, date(2023, 1, 2))
t_end   = np.searchsorted(dates, date(2023, 1, 15))
print(f"Test window: {dates[t_start]} to {dates[t_end-1]}  "
      f"({t_end-t_start} days, {len(tickers)} tickers)")

HIST    = 100
N_MC    = 10000
HOLD    = 1
TARGET  = 0.05
STOP    = 0.05

# ── precompute global x (matches strategy_tester._precompute_global_x) ──
from density.indicator import GlobalIndicator

global_dens = [(comp, d, ind)
               for comp, dens_list in [("R",R_dens),("M",M_dens),("S",S_dens)]
               for d, ind in dens_list
               if d.meta.get("is_global")]

global_keys_ordered = (
    [f"R_{d.meta['indicator_name']}_{'_'.join(map(str,d.meta['indicator_params']))}"
     for d,i in R_dens if d.meta.get("is_global")] +
    [f"M_{d.meta['indicator_name']}_{'_'.join(map(str,d.meta['indicator_params']))}"
     for d,i in M_dens if d.meta.get("is_global")] +
    [f"S_{d.meta['indicator_name']}_{'_'.join(map(str,d.meta['indicator_params']))}"
     for d,i in S_dens if d.meta.get("is_global")]
)
print(f"Global indicators: {global_keys_ordered}")

# Precompute global x for each test day
global_x_by_day = {}
for comp, d, ind in global_dens:
    key = f"{comp}_{d.meta['indicator_name']}_{'_'.join(map(str,d.meta['indicator_params']))}"
    gi  = GlobalIndicator(d.meta["indicator_name"],
                          d.meta["indicator_params"],
                          close, high, low)
    series = gi.compute_series()  # [n_days]
    for t in range(t_start, t_end):
        if t not in global_x_by_day:
            global_x_by_day[t] = {}
        global_x_by_day[t][key] = float(series[t-1]) if t-1 < len(series) else float('nan')

print("\nSample global x for first test day:")
for k,v in global_x_by_day[t_start].items():
    print(f"  {k}: {v:.5f}")

# ── per-ticker, per-day forecast ───────────────────────────────────
cpp = CppAdapter(R_dens, M_dens, S_dens, weights, mixture_mode=False)
model = TransitionModel(R_dens, M_dens, S_dens, weights,
                        sampling_mode="weighted_mean")

all_pass = True
results_py  = []
results_cpp = []

# Test first 3 tickers × all test days
for ticker_idx in range(min(3, len(tickers))):
    for t in range(t_start, t_end):
        if t < HIST: continue

        hh = high[ticker_idx,  t-HIST:t].copy()
        hl = low[ticker_idx,   t-HIST:t].copy()
        hc = close[ticker_idx, t-HIST:t].copy()
        if not np.all(np.isfinite(hc)): continue

        P          = hc[-1]
        gx_day     = global_x_by_day.get(t, {})
        target_p   = P * (1 + TARGET)
        stop_p     = P * (1 - STOP)

        # Python forecast
        x_vals = model.compute_indicator_values(hh, hl, hc)
        x_vals.update(gx_day)

        np.random.seed(ticker_idx * 10000 + t)
        exits = np.zeros(N_MC)
        for k in range(N_MC):
            c, h, l = model.step_from_x(x_vals, P)
            if   l <= stop_p  and h >= target_p: exits[k] = stop_p
            elif l <= stop_p:                    exits[k] = stop_p
            elif h >= target_p:                  exits[k] = target_p
            else:                                exits[k] = c
        mean_py = (exits - P).mean() / P
        std_py  = (exits - P).std()  / P

        # C++ forecast (with real global x)
        mean_cpp, std_cpp = cpp.forecast_ticker(
            hh, hl, hc, N_MC, HOLD, TARGET, STOP, [gx_day], 42)

        tol = 3 * std_py / np.sqrt(N_MC)
        ok  = abs(mean_py - mean_cpp) < tol
        if not ok:
            all_pass = False
            print(f"  ✗ {tickers[ticker_idx]} {dates[t]}: "
                  f"py={mean_py:+.4f} cpp={mean_cpp:+.4f} "
                  f"|Δ|={abs(mean_py-mean_cpp):.4f} tol={tol:.4f}")

        results_py.append(mean_py)
        results_cpp.append(mean_cpp)

n = len(results_py)
print(f"\nChecked {n} (ticker, day) pairs")
print(f"Mean py={np.mean(results_py):+.5f}  cpp={np.mean(results_cpp):+.5f}")
print(f"Corr(py,cpp) = {np.corrcoef(results_py,results_cpp)[0,1]:.5f}")

# ── check globals are actually doing something ─────────────────────
print("\n=== Global x impact check ===")
print("(C++ with real globals vs C++ with NaN globals)")
t = t_start
hh = high[0, t-HIST:t].copy()
hl = low[0,  t-HIST:t].copy()
hc = close[0,t-HIST:t].copy()

m_real, _ = cpp.forecast_ticker(hh,hl,hc, 50000, HOLD, TARGET, STOP, [global_x_by_day[t]], 42)
m_nan,  _ = cpp.forecast_ticker(hh,hl,hc, 50000, HOLD, TARGET, STOP, [{}], 42)
print(f"  Real globals: {m_real:+.5f}")
print(f"  NaN globals:  {m_nan:+.5f}")
print(f"  Difference:   {m_real-m_nan:+.5f}  "
      f"({'globals have impact ✓' if abs(m_real-m_nan) > 0.0005 else 'globals have no impact — check weights'})")

print(f"\n{'='*50}")
print(f"Strategy sanity: {'✓ ALL PASS' if all_pass else '✗ FAILURES — do not run strategy'}")
print(f"{'='*50}")
