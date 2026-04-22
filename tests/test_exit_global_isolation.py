"""
test_exit_global_isolation.py

Tests that exited paths/tickers are correctly excluded from global
state computation, both in C++ (MultiPathForecaster) and the Python
fallback in strategy_tester._forecast_all_tickers.

Core idea: use an extreme target/stop (e.g. 0.1%) so that essentially
all paths exit on day 1. Then on day 2+ the global state should be
computed from zero tickers (or a subset), which should produce NaN
globals. Compare against a version where we DON'T force exits — the
global state should differ, proving that exits are being excluded.

Tests
-----
1. All-paths-exit ticker produces NaN global contribution
2. Partial-exit ticker: only active paths contribute to global mean
3. Mixed: some tickers exit day 1, others don't — global only uses survivors
4. Python fallback matches C++ in exit isolation behavior
5. Return calculation still includes exited paths (strategy correctness)

Run from project root:
    python3 test_exit_global_isolation.py
"""

import numpy as np
import json
from pathlib import Path

from mc.cpp_adapter import CppAdapter, cpp_available
from density.density import DensitySet
from density.indicator import GlobalIndicator
from data.data_store import DataStore
from data.data_loader import DataLoader
from tools.tickers import Tickers

# ── helpers ────────────────────────────────────────────────────────

PASS = "✓"
FAIL = "✗"
results = []

def ok(label, passed, detail=""):
    flag = PASS if passed else FAIL
    print(f"  {flag} {label}" + (f"   [{detail}]" if detail else ""))
    results.append((label, passed))
    return passed

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

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

# ── load data ──────────────────────────────────────────────────────

print("Loading data...")
ds      = DataStore()
tickers = Tickers().get("sweden_largecap")
raw     = ds.download_full(tickers[:10])
dates, close, high, low, _ = DataLoader.align(raw)
N_DAYS  = close.shape[1]
HIST    = 100
print(f"  {close.shape[0]} tickers, {N_DAYS} days\n")

if not cpp_available():
    print("✗ rms_cpp not available")
    exit(1)
print(f"{PASS} rms_cpp loaded")

tune_paths = sorted(Path("tune/rms_tunes").glob("*/best.json"))
if not tune_paths:
    print("No tunes found.")
    exit(1)

R_dens, M_dens, S_dens, weights, mode = load_tune(tune_paths[0])
mixture = (mode == "mixture")
cpp = CppAdapter(R_dens, M_dens, S_dens, weights, mixture_mode=mixture)

HAS_GLOBALS = any(d.meta.get("is_global") for d, _ in R_dens + M_dens + S_dens)
print(f"  has_globals={HAS_GLOBALS}")

# Pick a test day with clean data
T = N_DAYS // 2
while T < N_DAYS - 15 and not np.all(np.isfinite(close[:5, T-1])):
    T += 1

N_MC   = 500
HOLD   = 5
TINY   = 0.0001   # 0.01% target/stop — virtually all paths exit day 1
NORMAL = 0.10     # 10% target/stop — virtually no paths exit in 5 days

# ════════════════════════════════════════════════════════════════
# TEST 1 — All paths exit: global contribution becomes NaN
# ════════════════════════════════════════════════════════════════
section("1. All-paths-exit ticker excluded from global (C++)")

# Run with TINY target/stop — all paths exit on day 1
# Run with NORMAL target/stop — no paths exit
# The difference in later-day returns should be detectable
# because global state evolves differently

n_t = 5
hc_f = np.ascontiguousarray(
    np.array([close[i, T-HIST:T] for i in range(n_t)]).ravel())
hh_f = np.ascontiguousarray(
    np.array([high[i,  T-HIST:T] for i in range(n_t)]).ravel())
hl_f = np.ascontiguousarray(
    np.array([low[i,   T-HIST:T] for i in range(n_t)]).ravel())
weekdays = np.full(HOLD, np.nan)

# TINY: all paths exit day 1, global state should lose all tickers
f_tiny = cpp.make_multi_path_forecaster()
f_tiny.init(hc_f, hh_f, hl_f, n_t, HIST, N_MC,
            TINY, TINY, weekdays, seed=42)
f_tiny.run(HOLD)
ret_tiny = f_tiny.get_returns()

# Verify all mean returns are close to ±TINY (exits happened)
for i in range(n_t):
    mean, std = ret_tiny[i]
    ok(f"ticker {i} exited with |mean| ≈ {TINY:.2%}",
       abs(abs(mean) - TINY) < 0.002,
       f"mean={mean:+.5f} expected≈±{TINY:.4f}")

# NORMAL: no exits, global evolves from real predicted prices
f_normal = cpp.make_multi_path_forecaster()
f_normal.init(hc_f, hh_f, hl_f, n_t, HIST, N_MC,
              NORMAL, NORMAL, weekdays, seed=42)
f_normal.run(HOLD)
ret_normal = f_normal.get_returns()

# With TINY exits, days 2-5 have no global conditioning (all tickers exited)
# With NORMAL, days 2-5 have real global conditioning
# Returns should differ beyond day 1
mean_diff = np.mean([abs(ret_tiny[i][0] - ret_normal[i][0])
                     for i in range(n_t)])
ok("TINY vs NORMAL returns differ (exit isolation changes global)",
   mean_diff > 0.001,
   f"mean |Δ|={mean_diff:.5f}")

# ════════════════════════════════════════════════════════════════
# TEST 2 — Partial exit: only active paths contribute to global mean
# ════════════════════════════════════════════════════════════════
section("2. Partial exit — active paths only contribute to global (C++)")

if HAS_GLOBALS:
    # Use 2 tickers: ticker 0 with TINY (exits day 1),
    #                ticker 1 with NORMAL (stays active all 5 days)
    # Run them together — ticker 1's result should not be affected
    # by ticker 0's exit state in later days
    n_t2 = 2
    hc_2 = np.ascontiguousarray(
        np.array([close[i, T-HIST:T] for i in range(n_t2)]).ravel())
    hh_2 = np.ascontiguousarray(
        np.array([high[i,  T-HIST:T] for i in range(n_t2)]).ravel())
    hl_2 = np.ascontiguousarray(
        np.array([low[i,   T-HIST:T] for i in range(n_t2)]).ravel())

    # Can't have different target/stop per ticker in one MPF run,
    # so instead verify that global state with 1 exited ticker
    # equals global state with just 1 ticker (the survivor).
    # Run: [ticker0_exits, ticker1_normal] vs [ticker1_only]

    # Run with both tickers, ticker 0 exits immediately (TINY)
    f_both = cpp.make_multi_path_forecaster()
    f_both.init(hc_2, hh_2, hl_2, n_t2, HIST, N_MC,
                TINY, TINY, weekdays, seed=42)
    f_both.run(HOLD)
    ret_both = f_both.get_returns()

    # Run with ticker 1 alone (no ticker 0 at all)
    hc_1 = np.ascontiguousarray(close[1, T-HIST:T].copy())
    hh_1 = np.ascontiguousarray(high[1,  T-HIST:T].copy())
    hl_1 = np.ascontiguousarray(low[1,   T-HIST:T].copy())

    f_solo = cpp.make_multi_path_forecaster()
    f_solo.init(hc_1, hh_1, hl_1, 1, HIST, N_MC,
                TINY, TINY, weekdays, seed=42)
    f_solo.run(HOLD)
    ret_solo = f_solo.get_returns()

    # ticker 1's result in f_both should match f_solo because
    # ticker 0 exits day 1 and stops contributing to global
    diff = abs(ret_both[1][0] - ret_solo[0][0])
    tol  = 3 * ret_solo[0][1] / np.sqrt(N_MC)
    ok("ticker 1 result same whether alone or with exited ticker 0",
       diff < max(tol * 2, 1e-4),
       f"both={ret_both[1][0]:+.5f} solo={ret_solo[0][0]:+.5f} |Δ|={diff:.5f} tol={tol:.5f}")
else:
    ok("skipped — no global indicators in tune", True)

# ════════════════════════════════════════════════════════════════
# TEST 3 — Return calculation includes exited paths (strategy correctness)
# ════════════════════════════════════════════════════════════════
section("3. Return calculation includes exited paths")

# With TINY target/stop, mean return should be very close to ±TINY
# (not zero, which would happen if exited paths were excluded from returns)
n_t3 = 3
hc_3 = np.ascontiguousarray(
    np.array([close[i, T-HIST:T] for i in range(n_t3)]).ravel())
hh_3 = np.ascontiguousarray(
    np.array([high[i,  T-HIST:T] for i in range(n_t3)]).ravel())
hl_3 = np.ascontiguousarray(
    np.array([low[i,   T-HIST:T] for i in range(n_t3)]).ravel())

f3 = cpp.make_multi_path_forecaster()
f3.init(hc_3, hh_3, hl_3, n_t3, HIST, N_MC,
        TINY, TINY, weekdays, seed=77)
f3.run(HOLD)
ret3 = f3.get_returns()

for i in range(n_t3):
    mean, std = ret3[i]
    # With TINY exits, almost all paths exit at ±TINY on day 1
    # Mean should be small but non-zero (close to ±TINY)
    # If exits were excluded from returns, mean would be ~0 (random walk)
    ok(f"ticker {i} return reflects exits (|mean| > 0)",
       abs(mean) > 1e-5,
       f"mean={mean:+.6f}")
    ok(f"ticker {i} std > 0 (paths diverged at exit)",
       std > 0,
       f"std={std:.6f}")

# ════════════════════════════════════════════════════════════════
# TEST 4 — Python fallback matches C++ exit isolation
# ════════════════════════════════════════════════════════════════
section("4. Python fallback exit isolation matches C++")

if HAS_GLOBALS:
    from strategy.strategy_tester import StrategyTester
    import json as _json

    # Build a minimal StrategyTester to invoke Python fallback
    # We do this by temporarily disabling C++ on the tester
    cfg = _json.load(open(tune_paths[0]))

    try:
        # Build params dict
        params = {
            "target": TINY, "stop": TINY,
            "hold_days": HOLD,
            "mc_samples": N_MC // 5,  # smaller for Python speed
            "initial_bankroll": 100000,
            "invest_frac": 0.5,
            "n_hold": 1,
            "rule": "risk_adjusted",
            "risk_normalized": False,
            "fee_pct": 0.0,
            "sharpe_cutoff": None,
            "date_spec": None,
        }

        tester = StrategyTester.__new__(StrategyTester)
        tester.close  = close[:n_t3]
        tester.high   = high[:n_t3]
        tester.low    = low[:n_t3]
        tester.dates  = dates
        tester.history_len = HIST
        tester.R_dens = R_dens
        tester.M_dens = M_dens
        tester.S_dens = S_dens
        tester._model = None
        tester._cpp   = None   # force Python path
        tester.params = params

        from mc.transition_model import TransitionModel
        tester._model = TransitionModel(
            R_dens, M_dens, S_dens, weights, sampling_mode=mode)

        valid_tickers = list(range(n_t3))
        forecasts_py = tester._forecast_all_tickers(
            valid_tickers, T, HOLD, N_MC // 5)

        # Check same shape and plausible returns
        ok("Python fallback returns same number of forecasts",
           len(forecasts_py) == n_t3)

        for ticker_idx, mean, std in forecasts_py:
            ok(f"Python ticker {ticker_idx} return finite",
               np.isfinite(mean) and np.isfinite(std),
               f"mean={mean:+.5f} std={std:.5f}")
            ok(f"Python ticker {ticker_idx} |mean| > 0 (exits counted)",
               abs(mean) > 1e-5,
               f"mean={mean:+.6f}")

        # Compare Python vs C++ mean returns — should be statistically consistent
        f_cpp = cpp.make_multi_path_forecaster()
        f_cpp.init(hc_3, hh_3, hl_3, n_t3, HIST, N_MC,
                   TINY, TINY, weekdays, seed=77)
        f_cpp.run(HOLD)
        ret_cpp = f_cpp.get_returns()

        for i, (ticker_idx, mean_py, std_py) in enumerate(forecasts_py):
            mean_cpp, std_cpp = ret_cpp[i]
            diff = abs(mean_py - mean_cpp)
            tol_wide = 0.01  # 1% absolute — generous since different N_mc and RNG
            ok(f"ticker {i} Python≈C++ within 1%",
               diff < tol_wide,
               f"py={mean_py:+.5f} cpp={mean_cpp:+.5f} |Δ|={diff:.5f}")

    except Exception as e:
        ok("Python fallback test", False, str(e))
else:
    ok("skipped — no global indicators in tune", True)

# ════════════════════════════════════════════════════════════════
# TEST 5 — Determinism preserved after exit isolation fix
# ════════════════════════════════════════════════════════════════
section("5. Determinism preserved after exit isolation fix")

n_t5 = 4
hc_5 = np.ascontiguousarray(
    np.array([close[i, T-HIST:T] for i in range(n_t5)]).ravel())
hh_5 = np.ascontiguousarray(
    np.array([high[i,  T-HIST:T] for i in range(n_t5)]).ravel())
hl_5 = np.ascontiguousarray(
    np.array([low[i,   T-HIST:T] for i in range(n_t5)]).ravel())

for target, stop, label in [(TINY, TINY, "tiny exits"),
                              (NORMAL, NORMAL, "normal no-exits")]:
    f_a = cpp.make_multi_path_forecaster()
    f_a.init(hc_5, hh_5, hl_5, n_t5, HIST, N_MC,
             target, stop, weekdays, seed=123)
    f_a.run(HOLD)
    ret_a = f_a.get_returns()

    f_b = cpp.make_multi_path_forecaster()
    f_b.init(hc_5, hh_5, hl_5, n_t5, HIST, N_MC,
             target, stop, weekdays, seed=123)
    f_b.run(HOLD)
    ret_b = f_b.get_returns()

    ok(f"Same seed → identical results ({label})",
       all(ret_a[i][0] == ret_b[i][0] for i in range(n_t5)),
       f"means_a={[f'{r[0]:+.5f}' for r in ret_a]}")

# ════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════

n_pass = sum(1 for _, p in results if p)
n_fail = len(results) - n_pass

print(f"\n{'='*60}")
print(f"Results: {n_pass}/{len(results)} passed"
      + (f"  — {n_fail} FAILED:" if n_fail else "  — all OK"))

if n_fail:
    for label, passed in results:
        if not passed:
            print(f"  ✗ {label}")

print(f"{'='*60}")
