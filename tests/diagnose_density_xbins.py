"""
diagnose_density_xbins.py

Checks whether global indicator values seen during strategy/tuning
fall within the x-bin ranges of the densities they condition.

Out-of-range values get clipped to edge bins in sampling, which can
cause systematic bias — e.g. an extreme market_mean_S value always
sampling from the lowest x-bin regardless of actual market state.

Run from project root:
    python3 diagnose_density_xbins.py
"""

import numpy as np
import json
import warnings
from pathlib import Path

from density.density import DensitySet
from density.indicator import GlobalIndicator
from data.data_store import DataStore
from data.data_loader import DataLoader
from tools.tickers import Tickers

# ── load data ──────────────────────────────────────────────────────
print("Loading data...")
ds      = DataStore()
tickers = Tickers().get("sweden_largecap")
raw     = ds.download_full(tickers)
dates, close, high, low, _ = DataLoader.align(raw)
N_DAYS  = close.shape[1]
print(f"  {close.shape[0]} tickers, {N_DAYS} days\n")

# ── compute actual distribution of each global indicator ───────────
print("Computing global indicator distributions across full dataset...")
print(f"{'='*70}\n")

global_indicators = [
    ("market_mean_R",        []),
    ("market_mean_M",        []),
    ("market_mean_S",        []),
    ("market_vol_dispersion",[]),
]

actual_series = {}
for name, params in global_indicators:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gi = GlobalIndicator(name, params, close, high, low)
        series = gi.compute_series()

    finite = series[np.isfinite(series)]
    actual_series[name] = finite

    print(f"{name}")
    print(f"  n_valid = {len(finite)} / {N_DAYS}")
    print(f"  min     = {finite.min():+.5f}")
    print(f"  p1      = {np.percentile(finite,  1):+.5f}")
    print(f"  p5      = {np.percentile(finite,  5):+.5f}")
    print(f"  p25     = {np.percentile(finite, 25):+.5f}")
    print(f"  median  = {np.percentile(finite, 50):+.5f}")
    print(f"  p75     = {np.percentile(finite, 75):+.5f}")
    print(f"  p95     = {np.percentile(finite, 95):+.5f}")
    print(f"  p99     = {np.percentile(finite, 99):+.5f}")
    print(f"  max     = {finite.max():+.5f}")
    print(f"  mean    = {finite.mean():+.5f}")
    print(f"  std     = {finite.std():+.5f}")
    print()

# ── check density x-bin coverage ──────────────────────────────────
print(f"{'='*70}")
print("Density x-bin coverage vs actual data range:")
print(f"{'='*70}\n")

dens_root = Path("density/densities")
if not dens_root.exists() or not any(dens_root.iterdir()):
    print("No densities found in density/densities/")
else:
    for folder in sorted(dens_root.glob("*")):
        mf = folder / "meta.json"
        if not mf.exists():
            continue
        m = json.load(open(mf))
        name    = m["indicator_name"]
        params  = m["indicator_params"]
        is_glob = m.get("is_global", False)

        d = DensitySet.load(folder)
        xb = d.x_bins
        x_lo, x_hi = float(xb[0]), float(xb[-1])
        n_bins = len(xb) - 1

        print(f"{name}  params={params}  global={is_glob}")
        print(f"  x-bin range: [{x_lo:+.5f} .. {x_hi:+.5f}]  ({n_bins} bins)")

        # Check coverage against actual data
        if name in actual_series:
            series = actual_series[name]
            n_below = np.sum(series < x_lo)
            n_above = np.sum(series > x_hi)
            n_total = len(series)
            pct_below = 100 * n_below / n_total
            pct_above = 100 * n_above / n_total
            pct_covered = 100 - pct_below - pct_above

            status = "✓ OK" if pct_covered > 95 else ("⚠ PARTIAL" if pct_covered > 80 else "✗ POOR")
            print(f"  Data coverage: {pct_covered:.1f}%  {status}")
            if n_below > 0:
                below_vals = series[series < x_lo]
                print(f"  Below x_lo ({x_lo:+.4f}): {n_below} days ({pct_below:.1f}%)  "
                      f"min={below_vals.min():+.5f}")
            if n_above > 0:
                above_vals = series[series > x_hi]
                print(f"  Above x_hi ({x_hi:+.4f}): {n_above} days ({pct_above:.1f}%)  "
                      f"max={above_vals.max():+.5f}")
        print()

# ── check specific problematic dates ──────────────────────────────
print(f"{'='*70}")
print("Global indicator values on specific dates of interest:")
print(f"{'='*70}\n")

import datetime as dt

test_dates_str = [
    "2020-12-30",   # day before the problematic 2021 forecast
    "2021-01-04",
    "2020-01-10",   # day before the 2020 forecast
    "2019-12-31",
]

date_list = []
for i in range(N_DAYS):
    d = dates[i]
    if hasattr(d, "astype"):
        d = d.astype("datetime64[D]").astype(dt.date)
    date_list.append(str(d))

for date_str in test_dates_str:
    if date_str not in date_list:
        print(f"  {date_str}: not in dataset\n")
        continue
    idx = date_list.index(date_str)

    print(f"  {date_str} (day index {idx}):")
    for name, params in global_indicators:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gi = GlobalIndicator(name, params,
                                 close[:, max(0, idx-100):idx+1],
                                 high[:,  max(0, idx-100):idx+1],
                                 low[:,   max(0, idx-100):idx+1])
            val = float(gi.compute_series()[-1])

        # Check if within density range
        coverage = ""
        for folder in sorted(dens_root.glob("*")):
            mf = folder / "meta.json"
            if not mf.exists(): continue
            m = json.load(open(mf))
            if m["indicator_name"] == name:
                d_obj = DensitySet.load(folder)
                xb = d_obj.x_bins
                if val < float(xb[0]):
                    coverage = f"  ← BELOW x_lo={xb[0]:+.4f} (CLIPPED)"
                elif val > float(xb[-1]):
                    coverage = f"  ← ABOVE x_hi={xb[-1]:+.4f} (CLIPPED)"
                else:
                    coverage = f"  ✓ within [{xb[0]:+.4f}..{xb[-1]:+.4f}]"
                break

        print(f"    {name:<30} = {val:+.6f}{coverage}")
    print()

print("Done.")
