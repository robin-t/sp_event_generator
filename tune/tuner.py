import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from mc.transition_model import TransitionModel
from mc.cpp_adapter import CppAdapter, cpp_available
from tools.mpi_utils import COMM, RANK, SIZE, is_root, gather, mpi_tqdm, bcast, root_print
from tools.date_range import parse_date_mask, describe_mask, spec_from_years


class Tuner:
    """
    Calibrates multi-density RMS MC model.
    """

    # ==========================================================
    # INIT
    # ==========================================================

    def __init__(self, R_densities, M_densities, S_densities,
             weight_grid, n_mc=200, history_len=100,
             dates=None, tune_date_spec=None,
             tune_start_year=None, tune_end_year=None,
             sampling_mode="weighted_mean",
             score_weights=None):
        
        self.R_densities = R_densities
        self.M_densities = M_densities
        self.S_densities = S_densities

        self.weight_grid = weight_grid
        self.n_mc = n_mc
        self.history_len = history_len
        self.sampling_mode = sampling_mode

        # Score component weights: multiplies each normalized error term.
        # Default 1.0 for all (equal weighting). Higher = penalise that component more.
        _default_sw = {"R": 1.0, "M": 1.0, "S": 1.0, "C": 1.0}
        if score_weights is None:
            self.score_weights = _default_sw
        else:
            self.score_weights = {k: float(score_weights.get(k, 1.0))
                                  for k in ("R", "M", "S", "C")}

        # ------------------------------------------
        # Optional tuning date restriction
        # Accepts new tune_date_spec string or old
        # tune_start_year/tune_end_year for backward compat.
        # ------------------------------------------
        self.tune_mask = None
        self.tune_valid_indices = None

        # Resolve spec — prefer new format, fall back to old
        if tune_date_spec is None and tune_start_year is not None and tune_end_year is not None:
            tune_date_spec = spec_from_years(tune_start_year, tune_end_year)

        if dates is not None and tune_date_spec is not None:
            self.tune_mask = parse_date_mask(tune_date_spec, dates)
            # Valid indices must be far enough from start for history window
            all_valid = np.where(self.tune_mask)[0]
            self.tune_valid_indices = all_valid[all_valid >= self.history_len]

            if is_root():
                print("\nDataset summary:")
                print(f"  Total days:      {len(dates)}")
                print(f"  Full range:      {str(dates[0])[:10]} → {str(dates[-1])[:10]}")
                print("\nTuning date range:")
                print(describe_mask(tune_date_spec, self.tune_mask, dates))

        if is_root():
            kernel = "C++ kernel active" if cpp_available() else "Python kernel (C++ not compiled)"
            print(f"  Sampling mode:   {self.sampling_mode}")
            print(f"  Kernel:          {kernel}")
            sw = self.score_weights
            print(f"  Score weights:   R={sw['R']} M={sw['M']} S={sw['S']} C={sw['C']}")
        
    # ==========================================================
    # REAL RMS COMPUTATION
    # ==========================================================

    def _compute_real_RMS(self, close, high, low, idx):

        P = close[idx - 1]
        H = high[idx]
        L = low[idx]
        C = close[idx]

        R = (H - L) / P

        midpoint = (H + L) / 2
        M = (midpoint - P) / P

        half_range = (H - L) / 2
        S = (C - midpoint) / half_range if half_range > 0 else 0.0

        return R, M, S, C

    # ==========================================================
    # SINGLE SAMPLE EVALUATION
    # ==========================================================

    def _evaluate_sample(self, model, close_series,
                        high_series, low_series, idx, cpp=None):

        history_close = close_series[idx - self.history_len:idx]
        history_high = high_series[idx - self.history_len:idx]
        history_low = low_series[idx - self.history_len:idx]

        # --------------------------------------------------
        # Skip if history invalid
        # --------------------------------------------------
        if not np.all(np.isfinite(history_close)):
            return None

        P = history_close[-1]
        if not np.isfinite(P) or P <= 0:
            return None

        # --------------------------------------------------
        # Real candle validity check
        # --------------------------------------------------
        H = high_series[idx]
        L = low_series[idx]
        C = close_series[idx]

        if not (np.isfinite(H) and np.isfinite(L) and np.isfinite(C)):
            return None

        if H <= L:  # zero or negative range → useless candle
            return None

        # --------------------------------------------------
        # Real RMS (needed by both paths)
        # --------------------------------------------------
        R_real, M_real, S_real, C_real = self._compute_real_RMS(
            close_series, high_series, low_series, idx
        )

        if not np.isfinite([R_real, M_real, S_real, C_real]).all():
            return None

        # --------------------------------------------------
        # Precompute global x-values for this sample day
        # --------------------------------------------------
        global_x = {}
        for component, dens_list in (("R", model.R_densities),
                                      ("M", model.M_densities),
                                      ("S", model.S_densities)):
            for density, indicator in dens_list:
                if density.meta.get("is_global", False):
                    key = (f"{component}_{density.meta['indicator_name']}_"
                           f"{'_'.join(map(str, density.meta['indicator_params']))}")
                    series = self._global_series.get(key)
                    # Use idx-1: yesterday's global state conditions today's prediction.
                    # series[idx] would use today's prices — lookahead.
                    global_x[key] = series[idx - 1] if series is not None else np.nan

        # --------------------------------------------------
        # C++ fast path
        # --------------------------------------------------
        if cpp is not None:
            seed = int(np.random.randint(0, 2**31))
            err = cpp.evaluate_sample(
                history_high, history_low, history_close,
                R_real, M_real, S_real, C_real,
                self.n_mc,
                global_x,
                seed,
            )
            if not np.isfinite(err["total"]):
                return None
            return err

        # --------------------------------------------------
        # Python fallback
        # --------------------------------------------------
        closes_mc = np.zeros(self.n_mc)
        highs_mc  = np.zeros(self.n_mc)
        lows_mc   = np.zeros(self.n_mc)

        for k in range(self.n_mc):
            x_vals = model.compute_indicator_values(
                history_high, history_low, history_close,
                global_x=global_x if global_x else None,
            )
            c, h, l = model.step_from_x(x_vals, P)
            closes_mc[k] = c
            highs_mc[k]  = h
            lows_mc[k]   = l

        if closes_mc.size == 0:
            return None

        R_mc = (highs_mc - lows_mc) / P
        mid_mc = (highs_mc + lows_mc) / 2
        M_mc = (mid_mc - P) / P

        half_mc = (highs_mc - lows_mc) / 2
        S_mc = np.where(half_mc > 0,
                        (closes_mc - mid_mc) / half_mc,
                        0)

        R_pred = np.nanmean(R_mc)
        M_pred = np.nanmean(M_mc)
        S_pred = np.nanmean(S_mc)
        C_pred = np.nanmean(closes_mc)

        def _norm(sq, var):
            return sq / var if (np.isfinite(var) and var > 1e-12) else sq

        var_R = getattr(self, "_var_R", 1.0)
        var_M = getattr(self, "_var_M", 1.0)
        var_S = getattr(self, "_var_S", 1.0)
        var_C = getattr(self, "_var_C", 1.0)

        R_err = _norm((R_pred - R_real) ** 2, var_R)
        M_err = _norm((M_pred - M_real) ** 2, var_M)
        S_err = _norm((S_pred - S_real) ** 2, var_S)
        C_err = _norm(((C_pred - C_real) / P) ** 2, var_C)
        err   = R_err + M_err + S_err + C_err

        if not np.isfinite(err):
            return None

        return {"total": err, "R": R_err, "M": M_err, "S": S_err, "C": C_err}


    # ==========================================================
    # RMS VARIANCE COMPUTATION
    # ==========================================================

    def _compute_rms_variances(self, close_matrix, high_matrix, low_matrix,
                                n_samples=500):
        """
        Compute Var(R), Var(M), Var(S), Var(C/P) from training samples.
        Used to normalize the tuning loss so all components contribute equally.
        """
        n_tickers, n_days = close_matrix.shape

        R_vals, M_vals, S_vals, C_vals = [], [], [], []

        rng = np.random.default_rng(seed=0)
        attempts = 0
        while len(R_vals) < n_samples and attempts < n_samples * 10:
            attempts += 1
            t = int(rng.integers(0, n_tickers))
            if self.tune_valid_indices is not None:
                valid = self.tune_valid_indices[self.tune_valid_indices < n_days - 1]
            else:
                valid = np.arange(self.history_len, n_days - 1)
            if len(valid) == 0:
                break
            d = int(rng.choice(valid))

            P = close_matrix[t, d - 1]
            H = high_matrix[t, d]
            L = low_matrix[t, d]
            C = close_matrix[t, d]

            if not (np.isfinite(P) and P > 0 and np.isfinite(H)
                    and np.isfinite(L) and np.isfinite(C) and H > L):
                continue

            R = (H - L) / P
            mid = (H + L) / 2.0
            M = (mid - P) / P
            half = (H - L) / 2.0
            S = (C - mid) / half
            CP = (C - P) / P  # normalized close shift

            if not np.isfinite([R, M, S, CP]).all():
                continue

            R_vals.append(R)
            M_vals.append(M)
            S_vals.append(S)
            C_vals.append(CP)

        def safe_var(vals):
            if len(vals) < 2:
                return 1.0
            v = float(np.var(vals))
            return v if v > 1e-12 else 1.0

        var_R = safe_var(R_vals)
        var_M = safe_var(M_vals)
        var_S = safe_var(S_vals)
        var_C = safe_var(C_vals)

        if is_root():
            print(f"\nRMS component variances (from {len(R_vals)} samples):")
            print(f"  Var(R) = {var_R:.6f}   std(R) = {np.sqrt(var_R):.4f}")
            print(f"  Var(M) = {var_M:.6f}   std(M) = {np.sqrt(var_M):.4f}")
            print(f"  Var(S) = {var_S:.6f}   std(S) = {np.sqrt(var_S):.4f}")
            print(f"  Var(C) = {var_C:.6f}   std(C) = {np.sqrt(var_C):.4f}")

        return var_R, var_M, var_S, var_C

    def _compute_rms_variances_from_samples(self, close_matrix, high_matrix,
                                             low_matrix, samples):
        """
        Compute Var(R), Var(M), Var(S), Var(C/P) from the exact same
        (ticker, day) pairs used for tuning evaluation.
        This guarantees the normalization baseline is consistent with
        the sample distribution being scored.
        """
        R_vals, M_vals, S_vals, C_vals = [], [], [], []

        for t, d in samples:
            P = close_matrix[t, d - 1]
            H = high_matrix[t, d]
            L = low_matrix[t, d]
            C = close_matrix[t, d]

            if not (np.isfinite(P) and P > 0 and np.isfinite(H)
                    and np.isfinite(L) and np.isfinite(C) and H > L):
                continue

            R  = (H - L) / P
            mid = (H + L) / 2.0
            M  = (mid - P) / P
            half = (H - L) / 2.0
            S  = (C - mid) / half
            CP = (C - P) / P

            if not np.isfinite([R, M, S, CP]).all():
                continue

            R_vals.append(R)
            M_vals.append(M)
            S_vals.append(S)
            C_vals.append(CP)

        def safe_var(vals):
            if len(vals) < 2:
                return 1.0
            v = float(np.var(vals))
            return v if v > 1e-12 else 1.0

        var_R = safe_var(R_vals)
        var_M = safe_var(M_vals)
        var_S = safe_var(S_vals)
        var_C = safe_var(C_vals)

        if is_root():
            print(f"\nRMS component variances (from {len(R_vals)} tune samples):")
            print(f"  Var(R) = {var_R:.6f}   std(R) = {np.sqrt(var_R):.4f}")
            print(f"  Var(M) = {var_M:.6f}   std(M) = {np.sqrt(var_M):.4f}")
            print(f"  Var(S) = {var_S:.6f}   std(S) = {np.sqrt(var_S):.4f}")
            print(f"  Var(C) = {var_C:.6f}   std(C) = {np.sqrt(var_C):.4f}")

        return var_R, var_M, var_S, var_C

    # ==========================================================
    # MAIN CALIBRATION LOOP
    # ==========================================================

    def tune(self, close_matrix, high_matrix, low_matrix,
            n_samples=500, dates=None):

        n_tickers, n_days = close_matrix.shape

        # ------------------------------------------
        # Precompute global indicator series once
        # for the full matrix (all days).
        # Stored as {key: series[n_days]} and looked
        # up per sample day in _evaluate_sample.
        # ------------------------------------------
        from density.indicator import GlobalIndicator
        self._global_series = {}
        for component, dens_list in (("R", self.R_densities),
                                      ("M", self.M_densities),
                                      ("S", self.S_densities)):
            for density, indicator in dens_list:
                if density.meta.get("is_global", False):
                    key = (f"{component}_{density.meta['indicator_name']}_"
                           f"{'_'.join(map(str, density.meta['indicator_params']))}")
                    if key not in self._global_series:
                        gi = GlobalIndicator(
                            density.meta["indicator_name"],
                            density.meta["indicator_params"],
                            close_matrix, high_matrix, low_matrix,
                            dates=dates,
                        )
                        self._global_series[key] = gi.compute_series()

        # ------------------------------------------
        # Warn if grid smaller than MPI size
        # ------------------------------------------
        if is_root() and SIZE > 1 and len(self.weight_grid) < SIZE:
            print(f"\nWarning: weight grid size ({len(self.weight_grid)}) "
                f"is smaller than MPI processes ({SIZE}).")

        # ------------------------------------------
        # Create random training samples (root only)
        # ------------------------------------------
        if is_root():
            samples = []

            if self.tune_valid_indices is not None and len(self.tune_valid_indices) == 0:
                raise ValueError("No valid tuning days after applying date spec and history window!")

            while len(samples) < n_samples:
                t = random.randint(0, n_tickers - 1)

                if self.tune_valid_indices is not None:
                    # Sample from masked valid days, exclude last day
                    valid = self.tune_valid_indices[self.tune_valid_indices < n_days - 1]
                    d = int(random.choice(valid))
                else:
                    d = random.randint(self.history_len, n_days - 2)

                samples.append((t, d))
        else:
            samples = None

        # Broadcast identical samples to all ranks
        if COMM:
            samples = COMM.bcast(samples, root=0)

        # ------------------------------------------
        # Compute RMS variances from the SAME samples
        # used for evaluation — guarantees normalization
        # is consistent with the actual score computation.
        # ------------------------------------------
        var_R, var_M, var_S, var_C = self._compute_rms_variances_from_samples(
            close_matrix, high_matrix, low_matrix, samples
        )
        # Broadcast to all ranks
        if COMM:
            var_R, var_M, var_S, var_C = COMM.bcast(
                (var_R, var_M, var_S, var_C), root=0)
        # Incorporate score weights: effective_var = var / weight
        sw = self.score_weights
        self._var_R = var_R / sw["R"] if sw["R"] > 0 else var_R
        self._var_M = var_M / sw["M"] if sw["M"] > 0 else var_M
        self._var_S = var_S / sw["S"] if sw["S"] > 0 else var_S
        self._var_C = var_C / sw["C"] if sw["C"] > 0 else var_C

        # ------------------------------------------
        # Distribute weight grid
        # ------------------------------------------
        local_indices = range(len(self.weight_grid))[RANK::SIZE]

        local_results = []
        skipped_total = 0
        used_total = 0

        iterator = mpi_tqdm(local_indices, desc="Tuning")
        for idx in iterator:

            weights = self.weight_grid[idx]

            model = TransitionModel(
                self.R_densities,
                self.M_densities,
                self.S_densities,
                weights,
                sampling_mode=self.sampling_mode,
            )

            # Build C++ adapter for this weight combo.
            if cpp_available():
                cpp = CppAdapter(
                    self.R_densities, self.M_densities, self.S_densities,
                    weights, mixture_mode=(self.sampling_mode == "mixture")
                )
                cpp.set_variances(var_R, var_M, var_S, var_C)
            else:
                cpp = None

            errors = {"total": [], "R": [], "M": [], "S": [], "C": []}
            skipped = 0
            used = 0

            for t, d in samples:

                err = self._evaluate_sample(
                    model,
                    close_matrix[t],
                    high_matrix[t],
                    low_matrix[t],
                    d,
                    cpp=cpp,
                )

                if err is None:
                    skipped += 1
                    continue
                else:
                    used += 1

                # Both C++ and Python paths now return a dict
                for k in errors:
                    v = err[k]
                    if np.isfinite(v):
                        errors[k].append(v)

            skipped_total += skipped
            used_total += used

            if len(errors["total"]) == 0:
                score = np.inf
                comp_scores = {k: np.inf for k in errors}
            else:
                comp_scores = {k: float(np.mean(errors[k])) if errors[k] else np.inf
                               for k in errors}
                score = comp_scores["total"]

            local_results.append((weights, score, comp_scores))

        # ------------------------------------------
        # Gather results to root
        # ------------------------------------------

        if SIZE > 1:
            gathered = gather({
                "results": local_results,
                "skipped": skipped_total,
                "used": used_total
            })

            if not is_root():
                return None

            results = []
            skipped_total = 0
            used_total = 0

            for g in gathered:
                results += g["results"]
                skipped_total += g["skipped"]
                used_total += g["used"]
        else:
            results = local_results

        # ------------------------------------------
        # Root continues
        # ------------------------------------------
        if SIZE > 1 and not is_root():
            return None

        print(f"\nUsed samples total: {used_total}")
        print(f"\nSkipped samples total: {skipped_total}")

        # ------------------------------------------
        # MPI Weight Grid Sanity Check
        # ------------------------------------------

        if is_root():

            expected = len(self.weight_grid)
            actual = len(results)

            if actual != expected:
                print(f"\nMPI weight check: ERROR "
                    f"({actual}/{expected} weight combinations collected)")
            else:
                print(f"\nMPI weight check: OK "
                    f"({actual} weight combinations collected)")

        best_weights, best_score, best_comp = min(results, key=lambda x: x[1])

        return best_weights, best_score, best_comp, results

    def run_jackknife(self, close, high, low, n_samples, n_jackknife, dates=None):

        root_print(f"Sampling mode:   {self.sampling_mode}")
        jackknife_runs = []

        for i in range(n_jackknife):

            root_print(f"\n=== Jackknife run {i+1}/{n_jackknife} ===")

            result = self.tune(
                close, high, low,
                n_samples=n_samples,
                dates=dates,
            )

            if is_root():
                best_w, best_score, best_comp, results = result

                print("Best weights this run:")
                for k, v in best_w.items():
                    print(f"  {k:<20} {v:.3f}")

                jackknife_runs.append({
                    "best_weights": best_w,
                    "best_score": best_score,
                    "comp_scores": best_comp,
                })

            # Workers do nothing here but must continue loop
            # to preserve MPI symmetry

        # --------------------------------------
        # Only root computes statistics
        # --------------------------------------
        if not is_root():
            return None

        # Compute jackknife statistics
        scores = np.array([r["best_score"] for r in jackknife_runs])
        mean_score = np.mean(scores)
        jackknife_error = np.sqrt((len(scores) - 1) * np.var(scores, ddof=0))

        # --------------------------------------
        # Score-weighted mean weights
        # weight_i = (1/score_i) / sum(1/score_j)
        # --------------------------------------
        inv_scores = 1.0 / scores
        inv_sum    = inv_scores.sum()
        sw_weights  = inv_scores / inv_sum  # shape (n_jackknife,)

        all_keys = list(jackknife_runs[0]["best_weights"].keys())
        mean_weights = {}
        for k in all_keys:
            vals = np.array([r["best_weights"][k] for r in jackknife_runs])
            mean_weights[k] = float(np.dot(sw_weights, vals))

        # Per-component scores come directly from the C++ tuning pass
        comp_keys = ["R", "M", "S", "C", "total"]
        comp_means = {}
        comp_errors = {}
        for ck in comp_keys:
            vals = np.array([r["comp_scores"][ck] for r in jackknife_runs
                             if np.isfinite(r["comp_scores"].get(ck, np.inf))])
            comp_means[ck]  = float(np.mean(vals)) if len(vals) > 0 else np.nan
            comp_errors[ck] = float(np.sqrt((len(vals) - 1) * np.var(vals, ddof=0))) if len(vals) > 1 else np.nan

        # --------------------------------------
        # Find globally best run
        # --------------------------------------
        best_run = min(jackknife_runs, key=lambda r: r["best_score"])

        # --------------------------------------
        # Print summary table
        # --------------------------------------
        print("\n=== Jackknife summary ===")
        print("Scores:", np.round(scores, 3))
        print()
        print(f"  {'Component':<12} {'Mean score':>12} {'Jackknife err':>14}")
        print(f"  {'-'*40}")
        for ck in comp_keys:
            label = "Total" if ck == "total" else ck
            print(f"  {label:<12} {comp_means[ck]:>12.4f} {comp_errors[ck]:>14.4f}")
        print()
        print(f"  Main score (total):   {mean_score:.4f} ± {jackknife_error:.4f}")

        print("\n=== Score-weighted Mean Weights ===")
        for k, v in mean_weights.items():
            print(f"  {k:<30} {v:.4f}")

        print("\n=== Best Single Jackknife Run (for reference) ===")
        print(f"  Score: {best_run['best_score']:.4f}")
        for k, v in best_run["best_weights"].items():
            print(f"  {k:<30} {v:.4f}")

        # --------------------------------------
        # MPI Jackknife Sanity Check
        # --------------------------------------

        expected_runs = n_jackknife
        actual_runs = len(jackknife_runs)

        if actual_runs != expected_runs:
            print("\nMPI jackknife check: ERROR "
                f"({actual_runs}/{expected_runs} runs collected)")
        else:
            # Check no None scores
            none_scores = any(r["best_score"] is None for r in jackknife_runs)

            # Check best_run consistency
            recomputed_best = min(jackknife_runs, key=lambda r: r["best_score"])

            if none_scores:
                print("\nMPI jackknife check: ERROR (None score detected)")
            elif recomputed_best["best_score"] != best_run["best_score"]:
                print("\nMPI jackknife check: ERROR (best_run mismatch)")
            else:
                print("\nMPI jackknife check: OK "
                    f"({actual_runs} runs, best score {best_run['best_score']:.4f})")

        return {
            "runs": jackknife_runs,
            "mean_score": float(mean_score),
            "jackknife_error": float(jackknife_error),
            "best_run": best_run,
            "mean_weights": mean_weights,
            "component_scores": {ck: {"mean": comp_means[ck], "jackknife_error": comp_errors[ck]}
                                  for ck in comp_keys},
        }