"""
General C++ equivalence test.
Compares forecast_ticker (Python step_from_x with target/stop)
vs C++ forecast_ticker across multiple tunes, histories, and modes.

Run on server: python3 equivalence_test.py
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

# ── helpers ───────────────────────────────────────────────────────

def load_dens(entry):
    sy = entry["density_meta"].get("start_year")
    for folder in sorted(Path("density/densities").glob("*")):
        mf = folder / "meta.json"
        if not mf.exists():
            continue
        m = json.load(open(mf))
        if (m["indicator_name"] == entry["density_meta"]["indicator_name"]
                and m["indicator_params"] == entry["density_meta"]["indicator_params"]
                and m.get("start_year") == sy):
            d = DensitySet.load(folder)
            return d, d.indicator
    raise RuntimeError(f"Density not found: {entry['density_meta']}")


def py_forecast(R_dens, M_dens, S_dens, weights, hh, hl, hc, N,
                target, stop, mode):
    """Python equivalent of forecast_ticker with target/stop exits."""
    model = TransitionModel(R_dens, M_dens, S_dens, weights,
                            sampling_mode=mode)
    x_vals = model.compute_indicator_values(hh, hl, hc)
    P = hc[-1]
    target_p = P * (1 + target)
    stop_p   = P * (1 - stop)
    np.random.seed(0)
    exits = np.zeros(N)
    for k in range(N):
        c, h, l = model.step_from_x(x_vals, P)
        if   l <= stop_p  and h >= target_p: exits[k] = stop_p
        elif l <= stop_p:                    exits[k] = stop_p
        elif h >= target_p:                  exits[k] = target_p
        else:                                exits[k] = c
    mean = (exits - P).mean() / P
    std  = (exits - P).std()  / P
    return mean, std


def cpp_forecast(R_dens, M_dens, S_dens, weights, hh, hl, hc, N,
                 target, stop, mode):
    cpp = CppAdapter(R_dens, M_dens, S_dens, weights,
                     mixture_mode=(mode == "mixture"))
    return cpp.forecast_ticker(hh, hl, hc, N, 1, target, stop, [{}], 42)


def check(label, R, M, S, weights, hh, hl, hc, N, target, stop, mode):
    mean_py, std_py   = py_forecast(R, M, S, weights, hh, hl, hc, N,
                                     target, stop, mode)
    mean_cpp, std_cpp = cpp_forecast(R, M, S, weights, hh, hl, hc, N,
                                      target, stop, mode)
    tol = 3 * std_py / np.sqrt(N)
    ok  = abs(mean_py - mean_cpp) < tol
    flag = "✓" if ok else "✗"
    print(f"  {flag} {label}")
    if not ok:
        print(f"      py={mean_py:+.5f} cpp={mean_cpp:+.5f} "
              f"|Δ|={abs(mean_py-mean_cpp):.5f} tol={tol:.5f}")
    return ok


# ── load data ─────────────────────────────────────────────────────

ds = DataStore()
tickers = Tickers().get("sweden_largecap")
raw = ds.download_full(tickers[:1])
dates, close, high, low, _ = DataLoader.align(raw)

test_dates = [4000, 5000, 5500, 6000]
histories  = {}
for t in test_dates:
    histories[t] = (
        high[0,  t-100:t].copy(),
        low[0,   t-100:t].copy(),
        close[0, t-100:t].copy(),
        str(dates[t])[:10],
    )

# ── run tests ─────────────────────────────────────────────────────

N_SLOW = 50000   # per-date Python loop — keep modest for speed
N_FAST = 200000  # C++ only checks

all_pass = True

for tune_path in sorted(Path("tune/rms_tunes").glob("*/best.json")):
    cfg = json.load(open(tune_path))
    density_yr = cfg["R_densities"][0]["density_meta"].get("start_year", "?")
    tune_yr    = cfg.get("tuning_year_range", ["?","?"])
    mode       = cfg.get("sampling_mode", "weighted_mean")

    try:
        R_dens = [load_dens(e) for e in cfg["R_densities"]]
        M_dens = [load_dens(e) for e in cfg["M_densities"]]
        S_dens = [load_dens(e) for e in cfg["S_densities"]]
    except RuntimeError as e:
        print(f"\n{tune_path.parent.name}: SKIP — {e}")
        continue

    weights = cfg["best_weights"]

    print(f"\n{'='*60}")
    print(f"{tune_path.parent.name}  "
          f"(density={density_yr}, tune={tune_yr}, mode={mode})")
    print(f"{'='*60}")

    for t, (hh, hl, hc, date_str) in histories.items():
        print(f"\n  Date {date_str}:")
        for target, stop in [(0.05, 0.05), (0.10, 0.05), (0.99, 0.99)]:
            label = (f"target={target:.0%}/stop={stop:.0%}  "
                     f"weighted_mean")
            ok = check(label, R_dens, M_dens, S_dens, weights,
                       hh, hl, hc, N_SLOW, target, stop,
                       "weighted_mean")
            if not ok:
                all_pass = False

        # mixture mode (one target/stop combo)
        ok = check("target=5%/stop=5%  mixture",
                   R_dens, M_dens, S_dens, weights,
                   hh, hl, hc, N_SLOW, 0.05, 0.05, "mixture")
        if not ok:
            all_pass = False

print(f"\n{'='*60}")
print(f"Overall: {'✓ ALL PASS' if all_pass else '✗ FAILURES DETECTED'}")
print(f"{'='*60}")
