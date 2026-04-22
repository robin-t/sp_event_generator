"""
analyze_density_summaries.py

Load all JSON summaries from density/summaries/ and produce a
comprehensive analysis report: data quality, boundary diagnostics,
conditional mean shapes, bias checks, and warnings.

Run from project root after receiving a summaries tarball:
    tar -xf summaries.tar.xz
    python3 analyze_density_summaries.py

Or point to a specific directory:
    python3 analyze_density_summaries.py --dir /path/to/summaries
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path

# ── CLI ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="density/summaries",
                    help="Directory containing .json summary files")
parser.add_argument("--verbose", action="store_true",
                    help="Print full conditional mean tables")
args = parser.parse_args()

summary_dir = Path(args.dir)
if not summary_dir.exists():
    print(f"Directory not found: {summary_dir}")
    sys.exit(1)

files = sorted(summary_dir.glob("*.json"))
if not files:
    print(f"No .json files found in {summary_dir}")
    sys.exit(1)

print(f"Found {len(files)} density summaries in {summary_dir}\n")

# ── thresholds for warnings ────────────────────────────────────────
EDGE_WARN       = 0.05   # >5% of mass in edge bin = boundary pile-up
SPIKE_WARN      = 5.0    # spike_ratio >5x = likely spike bug
SPARSE_WARN     = 0.30   # >30% sparse cells = poor S conditioning
EMPTY_X_WARN    = 3      # >3 empty x-bins = coverage gap
MIN_SAMPLES     = 1000   # <1000 samples = unreliable density

# ── helpers ────────────────────────────────────────────────────────

PASS = "✓"
WARN = "⚠"
FAIL = "✗"

def flag(val, warn_thresh, fail_thresh=None, low_bad=False):
    if fail_thresh is not None:
        bad = val > fail_thresh if not low_bad else val < fail_thresh
        if bad: return FAIL
    bad = val > warn_thresh if not low_bad else val < warn_thresh
    return WARN if bad else PASS

def fmt_pct(v): return f"{v*100:.2f}%"
def fmt_f(v):   return f"{v:.4f}"

all_warnings = []
all_failures = []

# ════════════════════════════════════════════════════════════════
# PER-DENSITY ANALYSIS
# ════════════════════════════════════════════════════════════════

for fpath in files:
    with open(fpath) as f:
        s = json.load(f)

    name    = s["density_name"]
    meta    = s["meta"]
    ind     = meta.get("indicator_name", "?")
    params  = meta.get("indicator_params", [])
    is_glob = meta.get("is_global", False)
    spec    = meta.get("date_spec", meta.get("start_year","?"))
    tickers = meta.get("ticker_list", "?")

    print(f"{'='*70}")
    print(f"  {ind.upper()}  params={params}  global={is_glob}")
    print(f"  ticker_list={tickers}  date_spec={spec}")
    print(f"  file: {fpath.name}")
    print(f"{'='*70}")

    warnings = []
    failures = []

    # ── 1. Sample counts ──────────────────────────────────────────
    n_R = s["total_samples_R"]
    n_M = s["total_samples_M"]
    n_S = s["total_samples_S"]

    print(f"\n[1] Sample counts")
    for comp, n in [("R", n_R), ("M", n_M), ("S", n_S)]:
        fl = flag(n, MIN_SAMPLES, low_bad=True)
        print(f"  {fl} {comp}: {n:.0f} samples"
              + (" ← LOW" if n < MIN_SAMPLES else ""))
        if n < MIN_SAMPLES:
            warnings.append(f"{comp} low sample count ({n:.0f})")

    # ── 2. X-bin coverage ─────────────────────────────────────────
    n_x_empty  = s["x_empty_bins"]
    n_x_sparse = s["x_sparse_bins"]
    n_x_total  = s["x_total_bins"]
    x_bins     = np.array(s["x_bins"])
    x_centers  = np.array(s["x_centers"])

    print(f"\n[2] X-bin coverage  (n_bins={n_x_total})")
    print(f"  x range: [{x_bins[0]:.4f} .. {x_bins[-1]:.4f}]")
    fl = flag(n_x_empty, EMPTY_X_WARN)
    print(f"  {fl} empty bins: {n_x_empty}/{n_x_total}")
    fl = flag(n_x_sparse, EMPTY_X_WARN * 3)
    print(f"  {fl} sparse bins (<10 samples): {n_x_sparse}/{n_x_total}")
    if n_x_empty > EMPTY_X_WARN:
        warnings.append(f"{n_x_empty} empty x-bins — coverage gap in indicator range")

    # counts per x-bin histogram (compressed)
    x_counts = np.array(s["R_counts_per_x"])
    if len(x_counts) > 0:
        print(f"  counts/bin: min={x_counts.min():.0f}  "
              f"median={np.median(x_counts):.0f}  "
              f"max={x_counts.max():.0f}")

    # ── 3. Boundary diagnostics ───────────────────────────────────
    print(f"\n[3] Boundary pile-up (edge bin fraction)")
    for comp in ["R", "M", "S"]:
        lo = s[f"{comp}_edge_fraction_lo"]
        hi = s[f"{comp}_edge_fraction_hi"]
        fl_lo = flag(lo, EDGE_WARN, EDGE_WARN * 3)
        fl_hi = flag(hi, EDGE_WARN, EDGE_WARN * 3)
        print(f"  {comp}:  lo={fmt_pct(lo)} {fl_lo}   hi={fmt_pct(hi)} {fl_hi}")
        if lo > EDGE_WARN:
            msg = f"{comp} boundary pile-up at lower edge ({fmt_pct(lo)})"
            (failures if lo > EDGE_WARN * 3 else warnings).append(msg)
        if hi > EDGE_WARN:
            msg = f"{comp} boundary pile-up at upper edge ({fmt_pct(hi)})"
            (failures if hi > EDGE_WARN * 3 else warnings).append(msg)

    # ── 4. S spike check ──────────────────────────────────────────
    spike_ratio = s["S_spike_ratio"]
    spike_center = s["S_spike_bin_center"]
    fl = flag(spike_ratio, SPIKE_WARN, SPIKE_WARN * 2)
    print(f"\n[4] S spike check")
    print(f"  {fl} spike_ratio={spike_ratio:.2f}x  "
          f"at S={spike_center:.3f}"
          + (" ← SPIKE BUG" if spike_ratio > SPIKE_WARN * 2 else
             " ← possible boundary pile-up" if spike_ratio > SPIKE_WARN else ""))
    if spike_ratio > SPIKE_WARN:
        msg = f"S spike at bin center {spike_center:.3f} (ratio={spike_ratio:.1f}x)"
        (failures if spike_ratio > SPIKE_WARN * 2 else warnings).append(msg)

    # ── 5. S cell sparsity ────────────────────────────────────────
    sparse_frac = s["S_sparse_fraction"]
    fl = flag(sparse_frac, SPARSE_WARN, SPARSE_WARN * 2)
    print(f"\n[5] S conditioning quality")
    print(f"  {fl} sparse (x,R) cells: "
          f"{s['S_sparse_cells']}/{s['S_total_cells']} = {fmt_pct(sparse_frac)}")
    if sparse_frac > SPARSE_WARN:
        warnings.append(f"S sparse cells {fmt_pct(sparse_frac)} — poor S conditioning")

    # ── 6. Conditional mean shape ─────────────────────────────────
    print(f"\n[6] Conditional mean shape")

    R_mean = np.array(s["R_cond_mean"])
    M_mean = np.array(s["M_cond_mean"])
    S_mean = np.array(s["S_cond_mean"])

    # Check monotonicity / direction of relationship
    # Most indicators should have some monotonic relationship with at least one of R,M,S
    def monotonicity(arr):
        """Returns fraction of consecutive pairs that are increasing."""
        diffs = np.diff(arr)
        if len(diffs) == 0: return 0.5
        return float(np.sum(diffs > 0) / len(diffs))

    R_mono = monotonicity(R_mean)
    M_mono = monotonicity(M_mean)
    S_mono = monotonicity(S_mean)

    print(f"  E[R|x]: range=[{R_mean.min():.4f} .. {R_mean.max():.4f}]  "
          f"monotone={R_mono:.0%}")
    print(f"  E[M|x]: range=[{M_mean.min():.4f} .. {M_mean.max():.4f}]  "
          f"monotone={M_mono:.0%}")
    print(f"  E[S|x]: range=[{S_mean.min():.4f} .. {S_mean.max():.4f}]  "
          f"monotone={S_mono:.0%}")

    # Check for flat (uninformative) conditional means
    R_range = R_mean.max() - R_mean.min()
    M_range = M_mean.max() - M_mean.min()
    S_range = S_mean.max() - S_mean.min()

    R_std_across_x = np.std(R_mean)
    M_std_across_x = np.std(M_mean)
    S_std_across_x = np.std(S_mean)

    print(f"  Variation across x-bins:")
    print(f"    R: std={R_std_across_x:.5f}  range={R_range:.5f}")
    print(f"    M: std={M_std_across_x:.5f}  range={M_range:.5f}")
    print(f"    S: std={S_std_across_x:.5f}  range={S_range:.5f}")

    if M_std_across_x < 0.0005 and not is_glob:
        warnings.append("E[M|x] nearly flat — indicator may not predict M")
    if R_std_across_x < 0.0001 and not is_glob:
        warnings.append("E[R|x] nearly flat — indicator may not predict R")

    # Check for overall bias: is E[R], E[M], E[S] far from zero/expected?
    R_marginal = np.array(s["R_marginal"])
    M_marginal = np.array(s["M_marginal"])
    S_marginal = np.array(s["S_marginal"])
    R_centers_arr = np.array(s["R_centers"])
    M_centers_arr = np.array(s["M_centers"])
    S_centers_arr = np.array(s["S_centers"])

    R_mean_overall = float((R_marginal * R_centers_arr).sum())
    M_mean_overall = float((M_marginal * M_centers_arr).sum())
    S_mean_overall = float((S_marginal * S_centers_arr).sum())

    print(f"\n[7] Overall marginal means (should be near 0 for M and S)")
    fl_M = flag(abs(M_mean_overall), 0.003, 0.008)
    fl_S = flag(abs(S_mean_overall), 0.05,  0.15)
    print(f"  E[R] = {R_mean_overall:+.5f}  (positive by construction)")
    print(f"  {fl_M} E[M] = {M_mean_overall:+.5f}"
          + (" ← downward bias" if M_mean_overall < -0.003 else
             " ← upward bias" if M_mean_overall > 0.003 else ""))
    print(f"  {fl_S} E[S] = {S_mean_overall:+.5f}"
          + (" ← strong negative S bias" if S_mean_overall < -0.05 else
             " ← strong positive S bias" if S_mean_overall > 0.05 else ""))

    if M_mean_overall < -0.003:
        warnings.append(f"Downward M bias: E[M]={M_mean_overall:+.5f}")
    if abs(S_mean_overall) > 0.05:
        warnings.append(f"S bias: E[S]={S_mean_overall:+.5f}")

    if args.verbose:
        print(f"\n  Conditional means per x-bin:")
        for i, (xc, rm, mm, sm) in enumerate(
                zip(x_centers, R_mean, M_mean, S_mean)):
            n = x_counts[i] if i < len(x_counts) else 0
            print(f"    x={xc:+.4f}  E[R]={rm:.4f}  "
                  f"E[M]={mm:+.5f}  E[S]={sm:+.4f}  n={n:.0f}")

    # ── summary for this density ──────────────────────────────────
    print(f"\n  Warnings ({len(warnings)}):")
    for w in warnings:
        print(f"    {WARN} {w}")
    print(f"  Failures ({len(failures)}):")
    for f_ in failures:
        print(f"    {FAIL} {f_}")

    all_warnings.extend([(name, w) for w in warnings])
    all_failures.extend([(name, f_) for f_ in failures])

    print()

# ════════════════════════════════════════════════════════════════
# CROSS-DENSITY SUMMARY
# ════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"OVERALL SUMMARY  ({len(files)} densities)")
print(f"{'='*70}")
print(f"  Total warnings: {len(all_warnings)}")
print(f"  Total failures: {len(all_failures)}")

if all_failures:
    print(f"\n  {FAIL} FAILURES:")
    for name, msg in all_failures:
        print(f"    [{name}] {msg}")

if all_warnings:
    print(f"\n  {WARN} WARNINGS:")
    for name, msg in all_warnings:
        print(f"    [{name}] {msg}")

if not all_warnings and not all_failures:
    print(f"\n  {PASS} All densities look clean.")

print()
