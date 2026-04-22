"""
test_equivalence.py

Tests that:
  1. C++ and Python forecast_ticker agree (mean return within statistical noise)
  2. C++ evaluate_sample returns a 5-tuple {total, R, M, S, C}
  3. C++ component scores are individually finite and sum to total
  4. Python evaluate_sample returns a dict with matching keys
  5. C++ and Python evaluate_sample total scores are statistically consistent

Run from project root:
    python3 test_equivalence.py
"""

import numpy as np
import json
from pathlib import Path

from mc.cpp_adapter import CppAdapter, cpp_available
from mc.transition_model import TransitionModel
from density.density import DensitySet
from data.data_store import DataStore
from data.data_loader import DataLoader
from tools.tickers import Tickers
from tune.tuner import Tuner

# ── helpers ────────────────────────────────────────────────────────

PASS = "✓"
FAIL = "✗"
results = []

def ok(label, passed, detail=""):
    flag = PASS if passed else FAIL
    print(f"  {flag} {label}" + (f"  [{detail}]" if detail else ""))
    results.append(passed)
    return passed


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


# ── load data ──────────────────────────────────────────────────────

print("Loading data...")
ds      = DataStore()
tickers = Tickers().get("sweden_largecap")
raw     = ds.download_full(tickers[:3])
dates, close, high, low, _ = DataLoader.align(raw)

# Pick a few test indices (well inside dataset)
N_DAYS = close.shape[1]
test_indices = [N_DAYS // 4, N_DAYS // 2, N_DAYS * 3 // 4]
HIST = 100

histories = {}
for idx in test_indices:
    if idx < HIST:
        continue
    histories[idx] = (
        high[0,  idx-HIST:idx].copy(),
        low[0,   idx-HIST:idx].copy(),
        close[0, idx-HIST:idx].copy(),
        str(dates[idx])[:10],
    )

print(f"  {len(histories)} test dates, {N_DAYS} total days\n")

# ── check C++ available ────────────────────────────────────────────

if not cpp_available():
    print("✗ rms_cpp C++ module not found — rebuild with: cd cpp && ./build.sh")
    exit(1)
print(f"{PASS} rms_cpp C++ module loaded\n")

# ── iterate over tunes ─────────────────────────────────────────────

tune_paths = sorted(Path("tune/rms_tunes").glob("*/best.json"))
if not tune_paths:
    print("No tunes found in tune/rms_tunes/ — run a tune first.")
    exit(1)

for tune_path in tune_paths:
    cfg  = json.load(open(tune_path))
    mode = cfg.get("sampling_mode", "weighted_mean")
    tune_id = tune_path.parent.name

    try:
        R_dens = [load_dens(e) for e in cfg["R_densities"]]
        M_dens = [load_dens(e) for e in cfg["M_densities"]]
        S_dens = [load_dens(e) for e in cfg["M_densities"]]
        S_dens = [load_dens(e) for e in cfg["S_densities"]]
    except RuntimeError as e:
        print(f"\n{tune_id}: SKIP — {e}\n")
        continue

    weights = cfg["best_weights"]

    cpp = CppAdapter(R_dens, M_dens, S_dens, weights,
                     mixture_mode=(mode == "mixture"))
    model = TransitionModel(R_dens, M_dens, S_dens, weights,
                            sampling_mode=mode)

    print(f"{'='*60}")
    print(f"{tune_id}  (mode={mode})")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. forecast_ticker: C++ vs Python
    # ------------------------------------------------------------------
    print("\n[1] forecast_ticker — C++ vs Python")
    N_FC = 5000

    for t, (hh, hl, hc, date_str) in histories.items():
        for target, stop in [(0.05, 0.05), (0.10, 0.05)]:

            # C++ forecast
            mean_cpp, std_cpp = cpp.forecast_ticker(
                hh, hl, hc, N_FC, 1, target, stop, [{}], seed=42)

            # Python forecast (single hold day, no global)
            np.random.seed(42)
            x_vals = model.compute_indicator_values(hh, hl, hc)
            P = hc[-1]
            exits = np.zeros(N_FC)
            for k in range(N_FC):
                c, h, l = model.step_from_x(x_vals, P)
                if   l <= P*(1-stop) and h >= P*(1+target): exits[k] = P*(1-stop)
                elif l <= P*(1-stop):                        exits[k] = P*(1-stop)
                elif h >= P*(1+target):                      exits[k] = P*(1+target)
                else:                                        exits[k] = c
            mean_py = (exits - P).mean() / P
            std_py  = (exits - P).std()  / P

            tol = 3 * std_py / np.sqrt(N_FC)
            passed = abs(mean_py - mean_cpp) < max(tol, 1e-6)
            ok(f"{date_str} t={target:.0%}/s={stop:.0%}",
               passed,
               f"py={mean_py:+.5f} cpp={mean_cpp:+.5f} |Δ|={abs(mean_py-mean_cpp):.5f}")

    # ------------------------------------------------------------------
    # 2. evaluate_sample: C++ returns 5-component dict
    # ------------------------------------------------------------------
    print("\n[2] evaluate_sample — C++ returns component dict")

    for t, (hh, hl, hc, date_str) in histories.items():
        P = hc[-1]
        # Compute real RMS
        idx = test_indices[list(histories.keys()).index(t)]
        real_R = (high[0, idx] - low[0, idx]) / P
        mid    = (high[0, idx] + low[0, idx]) / 2.0
        real_M = (mid - P) / P
        half   = (high[0, idx] - low[0, idx]) / 2.0
        real_S = (close[0, idx] - mid) / half if half > 0 else 0.0
        real_C = close[0, idx]

        result = cpp.evaluate_sample(
            hh, hl, hc, real_R, real_M, real_S, real_C,
            n_mc=200, global_x={}, seed=42
        )

        # Check it's a dict with correct keys
        ok(f"{date_str} — returns dict with keys",
           isinstance(result, dict) and
           set(result.keys()) == {"total", "R", "M", "S", "C"})

        # Check all finite
        ok(f"{date_str} — all components finite",
           all(np.isfinite(v) for v in result.values()),
           str({k: f"{v:.5f}" for k, v in result.items()}))

        # Check components sum to total
        component_sum = result["R"] + result["M"] + result["S"] + result["C"]
        ok(f"{date_str} — R+M+S+C == total",
           abs(component_sum - result["total"]) < 1e-10,
           f"sum={component_sum:.8f} total={result['total']:.8f}")

    # ------------------------------------------------------------------
    # 3. evaluate_sample: Python path returns dict with same keys
    # ------------------------------------------------------------------
    print("\n[3] evaluate_sample — Python path returns dict")

    tuner = Tuner(R_dens, M_dens, S_dens, [weights],
                  n_mc=200, sampling_mode=mode)
    tuner._global_series = {}  # no globals for this test

    for t, (hh, hl, hc, date_str) in histories.items():
        idx = test_indices[list(histories.keys()).index(t)]
        py_model = TransitionModel(R_dens, M_dens, S_dens, weights,
                                   sampling_mode=mode)
        result_py = tuner._evaluate_sample(
            py_model,
            close[0], high[0], low[0],
            idx, cpp=None
        )

        ok(f"{date_str} — Python returns dict",
           isinstance(result_py, dict) and
           set(result_py.keys()) == {"total", "R", "M", "S", "C"})

        if isinstance(result_py, dict):
            ok(f"{date_str} — Python components finite",
               all(np.isfinite(v) for v in result_py.values()))

            comp_sum = result_py["R"] + result_py["M"] + result_py["S"] + result_py["C"]
            ok(f"{date_str} — Python R+M+S+C == total",
               abs(comp_sum - result_py["total"]) < 1e-10)

    # ------------------------------------------------------------------
    # 4. evaluate_sample: C++ vs Python total score consistency
    # ------------------------------------------------------------------
    print("\n[4] evaluate_sample — C++ vs Python total score")

    N_EVAL = 30
    cpp_totals = []
    py_totals  = []

    n_tickers, n_days = close.shape
    rng = np.random.default_rng(seed=99)

    for _ in range(N_EVAL):
        ticker = int(rng.integers(0, min(3, n_tickers)))
        d = int(rng.integers(HIST, n_days - 2))

        hh_t = high[ticker,  d-HIST:d].copy()
        hl_t = low[ticker,   d-HIST:d].copy()
        hc_t = close[ticker, d-HIST:d].copy()

        P = hc_t[-1]
        if not (np.isfinite(P) and P > 0):
            continue
        if not (np.isfinite(high[ticker, d]) and np.isfinite(low[ticker, d])):
            continue
        H, L, C = high[ticker, d], low[ticker, d], close[ticker, d]
        if H <= L:
            continue

        real_R = (H - L) / P
        mid    = (H + L) / 2.0
        real_M = (mid - P) / P
        half   = (H - L) / 2.0
        real_S = (C - mid) / half
        real_C = C

        r_cpp = cpp.evaluate_sample(hh_t, hl_t, hc_t, real_R, real_M,
                                     real_S, real_C, n_mc=500, global_x={}, seed=d)

        py_model2 = TransitionModel(R_dens, M_dens, S_dens, weights,
                                    sampling_mode=mode)
        tuner._global_series = {}
        r_py = tuner._evaluate_sample(py_model2, close[ticker], high[ticker],
                                       low[ticker], d, cpp=None)

        if r_cpp is not None and r_py is not None:
            cpp_totals.append(r_cpp["total"])
            py_totals.append(r_py["total"])

    if cpp_totals and py_totals:
        mean_cpp_t = np.mean(cpp_totals)
        mean_py_t  = np.mean(py_totals)
        std_py_t   = np.std(py_totals)
        tol        = 3 * std_py_t / np.sqrt(len(py_totals))
        passed     = abs(mean_cpp_t - mean_py_t) < max(tol, 1e-4)
        ok(f"Mean total score consistent (n={len(cpp_totals)})",
           passed,
           f"cpp={mean_cpp_t:.5f} py={mean_py_t:.5f} |Δ|={abs(mean_cpp_t-mean_py_t):.5f} tol={tol:.5f}")
    else:
        ok("Mean total score consistent", False, "no valid samples")

    print()

# ── summary ────────────────────────────────────────────────────────

n_pass = sum(results)
n_fail = len(results) - n_pass
print(f"{'='*60}")
print(f"Results: {n_pass}/{len(results)} passed"
      + (f"  ({n_fail} FAILED)" if n_fail else "  — all OK"))
print(f"{'='*60}")
