from tools.mpi_utils import (
    COMM, RANK, SIZE,
    root_print,
    mpi_tqdm
)

import random
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

from mc.transition_model import TransitionModel
from mc.cpp_adapter import CppAdapter, cpp_available
from data.data_store import DataStore
from tune.tuner import Tuner
from density.density import DensitySet
from density.indicator import Indicator, GlobalIndicator
from tools.date_range import parse_date_mask, describe_mask, validate_spec, spec_from_years


class StrategyTester:

    def __init__(self, tune_config):

        def load_density_from_meta(entry):
            name = entry["indicator"]
            params = entry["params"]
            start_year = entry.get("density_meta", {}).get("start_year")

            # Search density folders — match by name, params, and start_year
            for folder in Path("density/densities").glob("*"):

                meta_file = folder / "meta.json"
                if not meta_file.exists():
                    continue

                with open(meta_file) as f:
                    meta = json.load(f)

                if (
                    meta["indicator_name"] == name
                    and meta["indicator_params"] == params
                    and (start_year is None or meta.get("start_year") == start_year)
                ):
                    density = DensitySet.load(folder)
                    return density, density.indicator

            raise RuntimeError(f"Density not found for {name} {params} start_year={start_year}")

        self.tune = tune_config
        self.data = DataStore()  # consistent with other handlers

        # Load full dataset once
        self.dates = None
        self.close = None
        self.high = None
        self.low = None

        # Load densities & weights from tune
        self.R_dens = [load_density_from_meta(d) for d in tune_config["R_densities"]]
        self.M_dens = [load_density_from_meta(d) for d in tune_config["M_densities"]]
        self.S_dens = [load_density_from_meta(d) for d in tune_config["S_densities"]]
        self.weights = tune_config["best_weights"]

        self.history_len = tune_config.get("history_len", 100)
        self.sampling_mode = tune_config.get("sampling_mode", "weighted_mean")

        # Build C++ adapter once — packs density arrays into C++ memory.
        if cpp_available():
            self._cpp = CppAdapter(self.R_dens, self.M_dens, self.S_dens, self.weights,
                                   mixture_mode=(self.sampling_mode == "mixture"))
            cpp_status = "C++ kernel active"
        else:
            self._cpp = None
            cpp_status = "Python kernel (C++ not compiled)"

        # Python fallback model — built once, reused per ticker call
        self._model = TransitionModel(
            self.R_dens, self.M_dens, self.S_dens,
            self.weights, sampling_mode=self.sampling_mode,
        )

        # Score weights: from tune config, with optional override from params
        _default_sw = {"R": 1.0, "M": 1.0, "S": 1.0, "C": 1.0}
        self.score_weights = tune_config.get("score_weights", _default_sw)

        print(f"  Sampling mode:  {self.sampling_mode}")
        print(f"  Kernel:         {cpp_status}")
        sw = self.score_weights
        print(f"  Score weights:  R={sw['R']} M={sw['M']} S={sw['S']} C={sw['C']}")

    def load_dataset(self, tickers):

        raw = self.data.download_full(tickers)

        from data.data_loader import DataLoader
        self.dates, self.close, self.high, self.low, _ = DataLoader.align(raw)

        self.tickers = tickers

        root_print("Dataset loaded:", len(self.dates), "days")
        
    # ----------------------------------------------------------
    # Run full test suite
    # ----------------------------------------------------------

    def run_tests(
            self,
            valid_indices,
            N_tests,
            N_cycles,
            N_hold_days,
        ):

        # ---------------------------------------
        # MPI setup
        # ---------------------------------------

        if SIZE > 1:
            root_print(f"\nRunning in MPI mode with {SIZE} processes")

        random.seed(1234 + RANK)
        np.random.seed(1234 + RANK)

        all_test_ids = list(range(N_tests))
        local_test_ids = all_test_ids[RANK::SIZE]

        local_results = []
        local_pred = []
        local_sigma = []
        local_score = []
        local_real = []
        local_error = []
        local_trades = []
        local_sharpe_by_bucket = {
            "hit_target":       [],
            "timeout_positive": [],
            "timeout_negative": [],
            "hit_stop":         [],
        }
        local_cycle_fees = []
        local_cycle_trades = []
        local_total_fees_per_test = []
        local_cycle_pred_stats = [[] for _ in range(N_cycles)]

        # ---------------------------------------
        # MAIN LOOP (LOCAL TESTS ONLY)
        # ---------------------------------------


        for test_id in mpi_tqdm(local_test_ids, desc="Strategy tests"):
            stats = {
                "denied_price": 0,
                "selected": 0,
                "eligible": 0,
                "target_hits": 0,
                "stop_hits": 0,
                "timeouts": 0,
                "total_fees": 0.0,
            }

            # Draw start day from valid indices, leaving room for N_cycles ahead
            max_day = self.close.shape[1] - N_cycles * N_hold_days - 1
            eligible = valid_indices[valid_indices <= max_day]
            if len(eligible) == 0:
                continue
            start_day = int(random.choice(eligible))

            bankroll = self.params["initial_bankroll"]
            current_day = start_day

            cycle_bankrolls = [bankroll]

            for cycle in range(N_cycles):

                valid = self._get_valid_tickers(current_day, N_hold_days)

                if len(valid) == 0:
                    current_day += N_hold_days
                    continue

                forecasts = self._forecast_all_tickers(
                    valid, current_day, N_hold_days,
                    self.params["mc_samples"]
                )

                stats["eligible"] += len(valid)

                selected = self._select_candidates(forecasts)

                pred_map = {}
                for entry in selected:
                    idx, mean, std, *rest = entry
                    score = mean / std if std > 0 else 0
                    pred_map[idx] = (mean, std, score)

                positions = self._allocate_capital(
                    selected,
                    current_day,
                    bankroll,
                    stats
                )

                fees_before = stats["total_fees"]

                bankroll, trade_outcomes = self._execute_trades(
                    positions,
                    current_day,
                    bankroll,
                    stats
                )

                cycle_fee = stats["total_fees"] - fees_before
                local_cycle_fees.append(cycle_fee)
                local_cycle_trades.append(len(trade_outcomes))

                cycle_preds = []
                cycle_sigmas = []
                cycle_scores = []
                cycle_reals = []
                cycle_errors = []

                for ticker, real_ret, outcome in trade_outcomes:

                    if ticker not in pred_map:
                        continue

                    mean, std, score = pred_map[ticker]
                    error = real_ret - mean

                    local_pred.append(mean)
                    local_sigma.append(std)
                    local_score.append(score)
                    local_real.append(real_ret)
                    local_error.append(error)

                    local_sharpe_by_bucket[outcome].append(score)

                    local_trades.append({
                        "cycle": cycle,
                        "ticker": int(ticker),
                        "pred_mean": float(mean),
                        "pred_sigma": float(std),
                        "score": float(score),
                        "real_return": float(real_ret),
                        "error": float(error),
                        "outcome": outcome,
                    })

                    cycle_preds.append(mean)
                    cycle_sigmas.append(std)
                    cycle_scores.append(score)
                    cycle_reals.append(real_ret)
                    cycle_errors.append(error)

                if cycle_preds:
                    local_cycle_pred_stats[cycle].append({
                        "mean_pred": float(np.mean(cycle_preds)),
                        "mean_sigma": float(np.mean(cycle_sigmas)),
                        "mean_score": float(np.mean(cycle_scores)),
                        "mean_real": float(np.mean(cycle_reals)),
                        "mean_error": float(np.mean(cycle_errors)),
                        "std_error": float(np.std(cycle_errors)),
                    })

                cycle_bankrolls.append(bankroll)
                current_day += N_hold_days

            local_results.append({
                "final": bankroll,
                "cycles": cycle_bankrolls
            })

            local_total_fees_per_test.append(stats["total_fees"])

        # ---------------------------------------
        # GATHER RESULTS TO RANK 0
        # ---------------------------------------

        if SIZE > 1:
            gathered = COMM.gather({
                "results": local_results,
                "pred": local_pred,
                "sigma": local_sigma,
                "score": local_score,
                "real": local_real,
                "error": local_error,
                "trades": local_trades,
                "cycle_fees": local_cycle_fees,
                "cycle_trades": local_cycle_trades,
                "total_fees": local_total_fees_per_test,
                "cycle_pred_stats": local_cycle_pred_stats,
                "sharpe_by_bucket": local_sharpe_by_bucket,
            }, root=0)

            if RANK != 0:
                return None

            results = []
            all_pred = []
            all_sigma = []
            all_score = []
            all_real = []
            all_error = []
            all_trades = []
            all_cycle_fees = []
            all_cycle_trades = []
            total_fees_per_test = []
            cycle_pred_stats = [[] for _ in range(N_cycles)]
            all_sharpe_by_bucket = {
                "hit_target":       [],
                "timeout_positive": [],
                "timeout_negative": [],
                "hit_stop":         [],
            }

            for g in gathered:
                results += g["results"]
                all_pred += g["pred"]
                all_sigma += g["sigma"]
                all_score += g["score"]
                all_real += g["real"]
                all_error += g["error"]
                all_trades += g["trades"]
                all_cycle_fees += g["cycle_fees"]
                all_cycle_trades += g["cycle_trades"]
                total_fees_per_test += g["total_fees"]

                for i in range(N_cycles):
                    cycle_pred_stats[i] += g["cycle_pred_stats"][i]
                for bucket in all_sharpe_by_bucket:
                    all_sharpe_by_bucket[bucket] += g["sharpe_by_bucket"][bucket]

            # ---------------------------------------
            # MPI Sanity Check (tests)
            # ---------------------------------------

            total_tests_collected = len(results)

            if total_tests_collected == N_tests:
                print(f"\nMPI gather check: OK ({total_tests_collected}/{N_tests} tests collected)")
            else:
                print(f"\nMPI gather check: ERROR ({total_tests_collected}/{N_tests} tests collected)")

        else:
            results = local_results
            all_pred = local_pred
            all_sigma = local_sigma
            all_score = local_score
            all_real = local_real
            all_error = local_error
            all_trades = local_trades
            all_cycle_fees = local_cycle_fees
            all_cycle_trades = local_cycle_trades
            total_fees_per_test = local_total_fees_per_test
            cycle_pred_stats = local_cycle_pred_stats
            all_sharpe_by_bucket = local_sharpe_by_bucket

        # ---------------------------------------
        # ONLY RANK 0 CONTINUES
        # ---------------------------------------

        if SIZE > 1 and RANK != 0:
            return None

        # =========================================
        # CALCULATE SUMMARY
        # =========================================

        finals = np.array([r["final"] for r in results])

        mean_final = finals.mean()
        std_final = finals.std()

        N_cycles = len(results[0]["cycles"]) - 1

        cycle_means = []
        cycle_stds = []

        for i in range(N_cycles):
            returns = []

            for r in results:
                b0 = r["cycles"][i]
                b1 = r["cycles"][i+1]
                returns.append((b1 - b0) / b0)

            returns = np.array(returns)
            cycle_means.append(returns.mean())
            cycle_stds.append(returns.std())

        # =========================================
        # PRINT SUMMARY
        # =========================================

        print("\n=== Strategy Summary ===")
        print(f"Final bankroll: {mean_final:.2f} ± {std_final:.2f}")

        mean_total_fees = np.mean(total_fees_per_test)
        print(f"Mean total fees per test: {mean_total_fees:.2f}")

        if all_cycle_trades:
            mean_trades = np.mean(all_cycle_trades)
            print("\n=== Trading Activity ===")
            print(f"Mean trades per cycle: {mean_trades:.2f}")

        print("\nPer-cycle performance:")

        for i, (m, s) in enumerate(zip(cycle_means, cycle_stds)):
            print(f"Cycle {i+1}: {100*m:.2f}% ± {100*s:.2f}%")

        if all_pred:
            mean_pred = np.mean(all_pred)
            mean_real = np.mean(all_real)
            mean_error = np.mean(all_error)
            std_error = np.std(all_error)
            mean_score = np.mean(all_score)

            print("\n=== Prediction Quality ===")
            print(f"Mean predicted return: {100*mean_pred:.3f}%")
            print(f"Mean realized return:  {100*mean_real:.3f}%")
            print(f"Mean prediction error: {100*mean_error:.3f}%")
            print(f"Error sigma:           {100*std_error:.3f}%")
            print(f"Mean prediction score: {mean_score:.3f}")
        else:
            print("\n=== Prediction Quality ===")
            print("No trades executed — no prediction statistics available.")

        if all_cycle_fees:
            mean_fee = np.mean(all_cycle_fees)
            initial_bankroll = self.params["initial_bankroll"]
            fee_pct = mean_fee / initial_bankroll

            print("\n=== Commission Impact ===")
            print(f"Mean commission cost per cycle: {mean_fee:.2f}")
            print(f"Fees alone ≈ {-100*fee_pct:.3f}% per cycle")

        # =========================================
        # SAVE RESULTS
        # =========================================

        runs_dir = Path("strategy/runs")
        runs_dir.mkdir(parents=True, exist_ok=True)

        existing = [
            int(p.name.split("_")[1])
            for p in runs_dir.glob("run_*")
            if p.name.split("_")[1].isdigit()
        ]

        run_id = max(existing) + 1 if existing else 1

        run_folder = runs_dir / f"run_{run_id:03d}"
        run_folder.mkdir()

        prediction_summary = {}

        if all_pred:
            prediction_summary = {
                "mean_pred": float(np.mean(all_pred)),
                "mean_sigma": float(np.mean(all_sigma)),
                "mean_score": float(np.mean(all_score)),
                "mean_real": float(np.mean(all_real)),
                "mean_error": float(np.mean(all_error)),
                "std_error": float(np.std(all_error)),
                "mean_cycle_fee": float(np.mean(all_cycle_fees))
            }

        output = {
            "params": self.params,
            "tune_config": self.tune,
            "score_weights": self.score_weights,
            "sharpe_by_bucket": {k: v for k, v in all_sharpe_by_bucket.items()},
            "summary": {
                "mean_final": float(mean_final),
                "std_final": float(std_final),
                "cycle_means": cycle_means,
                "cycle_stds": cycle_stds,
                "mean_total_fees": float(np.mean(total_fees_per_test)),
                "mean_cycle_fee": float(np.mean(all_cycle_fees))
            },
            "prediction": {
                "overall": prediction_summary,
                "trades": all_trades
            },
            "tests": results
        }

        with open(run_folder / "results.json", "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {run_folder}")

        return finals
    
    # ----------------------------------------------------------
    # Select candidates to buy
    # ----------------------------------------------------------

    def _select_candidates(self, forecasts):

        rule = self.params["rule"]
        N = self.params["n_hold"]
        sharpe_cutoff = self.params.get("sharpe_cutoff", None)

        candidates = []

        for idx, mean, std in forecasts:

            if rule == "conservative":
                if mean - std > 0:
                    candidates.append((idx, mean, std))

            else:  # risk_adjusted
                score = mean / std if std > 0 else 0.0
                if sharpe_cutoff is not None and score < sharpe_cutoff:
                    continue
                candidates.append((idx, mean, std, score))

        if rule == "risk_adjusted":
            candidates.sort(key=lambda x: x[3], reverse=True)
        else:
            candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:N]
    
    # ----------------------------------------------------------
    # Allocate available capital
    # ----------------------------------------------------------

    def _allocate_capital(self, selected, current_day, bankroll, stats):

        invest_amount = bankroll * self.params["invest_frac"]
        risk_normalized = self.params.get("risk_normalized", False)

        N = len(selected)
        if N == 0:
            return []

        positions = []

        # -----------------------------------
        # Equal weight allocation
        # -----------------------------------
        if not risk_normalized:

            allocation = invest_amount / N

            for idx, mean, std, *rest in selected:

                price = self.close[idx, current_day]

                if price > allocation * 1.5:
                    stats["denied_price"] += 1
                    continue

                shares = int(allocation // price)

                if shares == 0:
                    stats["denied_price"] += 1
                    continue

                positions.append((idx, shares, price))

        # -----------------------------------
        # Risk-normalized allocation
        # -----------------------------------
        else:

            inv_sigmas = []

            for idx, mean, std, *rest in selected:
                inv_sigmas.append(1.0 / std if std > 0 else 0.0)

            total_inv_sigma = sum(inv_sigmas)

            for (entry, inv_sigma) in zip(selected, inv_sigmas):

                idx, mean, std, *rest = entry

                weight = inv_sigma / total_inv_sigma if total_inv_sigma > 0 else 0
                allocation = invest_amount * weight

                price = self.close[idx, current_day]

                if price > allocation * 1.5:
                    stats["denied_price"] += 1
                    continue

                shares = int(allocation // price)

                if shares == 0:
                    stats["denied_price"] += 1
                    continue

                positions.append((idx, shares, price))

        stats["selected"] += len(positions)

        return positions

    # ----------------------------------------------------------
    # Execute trades
    # ----------------------------------------------------------

    def _execute_trades(self, positions, entry_day, bankroll, stats):

        trade_outcomes = []

        fee_fixed = self.params["fee_fixed"]
        fee_pct = self.params["fee_pct"]
        hold_days = self.params["hold_days"]

        target = self.params["target"]
        stop = self.params["stop"]

        total_buy_cost = 0
        total_sell_value = 0

        for ticker, shares, entry_price in positions:

            # -----------------------------------
            # Entry cost
            # -----------------------------------

            buy_value = shares * entry_price
            buy_fee = fee_fixed + fee_pct * buy_value

            total_buy_cost += buy_value + buy_fee

            # -----------------------------------
            # Compute exit thresholds
            # -----------------------------------

            target_price = entry_price * (1 + target)
            stop_price = entry_price * (1 - stop)

            exit_price = None

            # -----------------------------------
            # Walk forward in time
            # -----------------------------------

            for d in range(1, hold_days + 1):

                day = entry_day + d

                high = self.high[ticker, day]
                low = self.low[ticker, day]
                close = self.close[ticker, day]

                # BOTH hit → stop wins
                if low <= stop_price and high >= target_price:
                    exit_price = stop_price
                    stats["stop_hits"] += 1
                    break

                # Stop hit
                if low <= stop_price:
                    exit_price = stop_price
                    stats["stop_hits"] += 1
                    break

                # Target hit
                if high >= target_price:
                    exit_price = target_price
                    stats["target_hits"] += 1
                    break

            # If never exited early
            if exit_price is None:
                exit_price = close
                stats["timeouts"] += 1

            # -----------------------------------
            # Sell value and fees
            # -----------------------------------

            sell_value = shares * exit_price
            sell_fee = fee_fixed + fee_pct * sell_value

            total_sell_value += sell_value - sell_fee

            stats["total_fees"] += buy_fee + sell_fee

            real_return = (exit_price - entry_price) / entry_price

            # Outcome bucket for Sharpe analysis
            if exit_price == target_price:
                outcome = "hit_target"
            elif exit_price == stop_price:
                outcome = "hit_stop"
            elif real_return >= 0:
                outcome = "timeout_positive"
            else:
                outcome = "timeout_negative"

            trade_outcomes.append((ticker, real_return, outcome))

        # ---------------------------------------
        # Update bankroll
        # ---------------------------------------

        bankroll = bankroll - total_buy_cost + total_sell_value

        return bankroll, trade_outcomes
    # ----------------------------------------------------------
    # Convert parameters and run tests
    # ----------------------------------------------------------

    def run_tests_with_params(self, params):

        self.params = params
        self.params["tickers"] = self.tickers

        # Apply score_weights override if provided
        if params.get("score_weights_override") is not None:
            self.score_weights = params["score_weights_override"]

        dates = self.dates

        # Resolve date spec — prefer new format, fall back to start/end year
        date_spec = params.get("date_spec")
        if date_spec is None:
            start_year = params.get("start_year")
            end_year   = params.get("end_year")
            if start_year is not None and end_year is not None:
                date_spec = spec_from_years(start_year, end_year)

        if date_spec is None:
            raise ValueError("No date range specified. Provide 'date_spec' or 'start_year'/'end_year'.")

        mask = parse_date_mask(date_spec, dates)
        # Valid start days: must be far enough from start for history window
        all_valid = np.where(mask)[0]
        valid_indices = all_valid[all_valid >= self.history_len]

        root_print(f"\nDataset summary:")
        root_print(f"  Total days:      {len(dates)}")
        root_print(f"  Full range:      {str(dates[0])[:10]} → {str(dates[-1])[:10]}")
        root_print(f"\nStrategy test period:")
        root_print(describe_mask(date_spec, mask, dates))
        root_print(f"  Sampling mode:   {self.sampling_mode}")

        return self.run_tests(
            valid_indices=valid_indices,
            N_tests=params["n_tests"],
            N_cycles=params["n_cycles"],
            N_hold_days=params["hold_days"],
        )
    
    # ----------------------------------------------------------
    # Find tickers with valid data at a decision time
    # ----------------------------------------------------------

    def _get_valid_tickers(self, t, hold_days):

        valid = []

        n_tickers = self.close.shape[0]

        for i in range(n_tickers):

            # -------------------------------
            # Check history window
            # -------------------------------

            if t - self.history_len < 0:
                continue

            hist = self.close[i, t - self.history_len : t]

            if np.isnan(hist).any():
                continue

            # -------------------------------
            # Check future holding window
            # -------------------------------

            end = t + hold_days

            if end > self.close.shape[1]:
                continue

            future_close = self.close[i, t:end]
            future_high  = self.high[i,  t:end]
            future_low   = self.low[i,   t:end]

            if (
                np.isnan(future_close).any()
                or np.isnan(future_high).any()
                or np.isnan(future_low).any()
            ):
                continue

            valid.append(i)

        return valid


    # ----------------------------------------------------------
    # Global indicator helpers
    # ----------------------------------------------------------

    def _get_global_entries(self):
        """Return list of (key, name, params) for all global indicators."""
        entries = []
        for component, dens_list in (("R", self.R_dens),
                                      ("M", self.M_dens),
                                      ("S", self.S_dens)):
            for density, indicator in dens_list:
                if density.meta.get("is_global", False):
                    key = (f"{component}_{density.meta['indicator_name']}_"
                           f"{'_'.join(map(str, density.meta['indicator_params']))}")
                    entries.append((key, density.meta["indicator_name"],
                                    density.meta["indicator_params"]))
        return entries

    def _compute_global_x_from_prices(self, close_matrix, high_matrix,
                                       low_matrix, dates_arr, global_entries):
        """
        Compute global indicator x-values from a price matrix.
        Returns dict {key: float} using the last value of each series.
        close_matrix etc. should already be sliced to the history window.
        """
        if not global_entries:
            return {}
        day_x = {}
        for key, name, params in global_entries:
            dates_slice = dates_arr if dates_arr is not None else None
            gi = GlobalIndicator(name, params, close_matrix, high_matrix,
                                 low_matrix, dates=dates_slice)
            series = gi.compute_series()
            day_x[key] = float(series[-1]) if len(series) > 0 else np.nan
        return day_x

    # ----------------------------------------------------------
    # Live day-by-day MC forecast for ALL tickers simultaneously
    # ----------------------------------------------------------

    def _forecast_all_tickers(self, valid_tickers, t, hold_days, mc_samples):
        """
        Run MC forecast for all valid tickers with a shared day-by-day outer
        loop. Global indicators are recomputed each day from mean predicted
        prices across all tickers — all inside C++ via MultiPathForecaster.

        Falls back to Python path-by-path if C++ not available.

        Returns list of (ticker_idx, mean_return, std_return).
        """
        n_tickers = len(valid_tickers)
        if n_tickers == 0:
            return []

        target = self.params["target"]
        stop   = self.params["stop"]

        # --------------------------------------------------
        # C++ fast path — MultiPathForecaster
        # --------------------------------------------------
        if self._cpp is not None:
            # Pack histories: flat [n_tickers × hist_len]
            hist_close_flat = np.zeros((n_tickers, self.history_len),
                                        dtype=np.float64)
            hist_high_flat  = np.zeros_like(hist_close_flat)
            hist_low_flat   = np.zeros_like(hist_close_flat)

            for i, ticker_idx in enumerate(valid_tickers):
                hist_close_flat[i] = self.close[ticker_idx,
                                                 t - self.history_len : t]
                hist_high_flat[i]  = self.high[ticker_idx,
                                                t - self.history_len : t]
                hist_low_flat[i]   = self.low[ticker_idx,
                                               t - self.history_len : t]

            # Weekdays for forecast window [t, t+hold_days)
            weekdays = np.full(hold_days, np.nan, dtype=np.float64)
            if self.dates is not None:
                import datetime
                for d in range(hold_days):
                    idx = t + d
                    if idx < len(self.dates):
                        try:
                            dt = self.dates[idx]
                            if hasattr(dt, "astype"):
                                dt = dt.astype("datetime64[D]").astype(
                                    datetime.date)
                            weekdays[d] = float(dt.weekday())
                        except Exception:
                            pass

            seed = int(np.random.randint(0, 2**31))
            forecaster = self._cpp.make_multi_path_forecaster()
            forecaster.init(
                np.ascontiguousarray(hist_close_flat.ravel()),
                np.ascontiguousarray(hist_high_flat.ravel()),
                np.ascontiguousarray(hist_low_flat.ravel()),
                n_tickers, self.history_len, mc_samples,
                target, stop, weekdays, seed,
            )

            forecaster.run(hold_days)

            results = forecaster.get_returns()  # list of (mean, std)
            return [(valid_tickers[i], results[i][0], results[i][1])
                    for i in range(n_tickers)]

        # --------------------------------------------------
        # Python fallback
        # --------------------------------------------------
        global_entries = self._get_global_entries()
        has_globals = len(global_entries) > 0

        hist_close = {}
        hist_high  = {}
        hist_low   = {}
        initial_price = {}
        exit_price    = {}
        active        = {}
        prices        = {}

        for ticker_idx in valid_tickers:
            hc = self.close[ticker_idx, t - self.history_len : t].copy()
            hh = self.high[ticker_idx,  t - self.history_len : t].copy()
            hl = self.low[ticker_idx,   t - self.history_len : t].copy()
            P  = hc[-1]
            hist_close[ticker_idx] = np.tile(hc, (mc_samples, 1))
            hist_high[ticker_idx]  = np.tile(hh, (mc_samples, 1))
            hist_low[ticker_idx]   = np.tile(hl, (mc_samples, 1))
            initial_price[ticker_idx] = P
            exit_price[ticker_idx]    = np.full(mc_samples, np.nan)
            active[ticker_idx]        = np.ones(mc_samples, dtype=bool)
            prices[ticker_idx]        = np.full(mc_samples, P)

        if has_globals:
            slice_start = max(0, t - self.history_len)
            dates_slice = (self.dates[slice_start:t]
                           if self.dates is not None else None)
            global_x = self._compute_global_x_from_prices(
                self.close[:, slice_start:t],
                self.high[:,  slice_start:t],
                self.low[:,   slice_start:t],
                dates_slice, global_entries)
        else:
            global_x = {}

        for d in range(hold_days):
            mean_closes = {}
            for ticker_idx in valid_tickers:
                if not np.any(active[ticker_idx]):
                    continue
                closes_d = np.zeros(mc_samples)
                highs_d  = np.zeros(mc_samples)
                lows_d   = np.zeros(mc_samples)
                for k in range(mc_samples):
                    if not active[ticker_idx][k]:
                        continue
                    x_vals = self._model.compute_indicator_values(
                        hist_high[ticker_idx][k],
                        hist_low[ticker_idx][k],
                        hist_close[ticker_idx][k],
                        global_x=global_x)
                    closes_d[k], highs_d[k], lows_d[k] =                         self._model.step_from_x(
                            x_vals, hist_close[ticker_idx][k, -1])

                P0 = initial_price[ticker_idx]
                tp = P0 * (1 + target)
                sp = P0 * (1 - stop)
                hit_stop   = active[ticker_idx] & (lows_d  <= sp)
                hit_target = active[ticker_idx] & (highs_d >= tp)
                hit_both   = hit_stop & hit_target
                exit_price[ticker_idx][hit_both]               = sp
                active[ticker_idx][hit_both]                   = False
                exit_price[ticker_idx][hit_stop & ~hit_both]   = sp
                active[ticker_idx][hit_stop & ~hit_both]       = False
                exit_price[ticker_idx][hit_target & ~hit_both] = tp
                active[ticker_idx][hit_target & ~hit_both]     = False
                prices[ticker_idx][active[ticker_idx]] =                     closes_d[active[ticker_idx]]

                # Only use active paths for global state — exited paths
                # represent a trading decision, not market evolution.
                active_mask = active[ticker_idx]
                if np.any(active_mask):
                    mean_closes[ticker_idx] = float(np.mean(closes_d[active_mask]))
                # Fully-exited tickers are absent from mean_closes and
                # therefore excluded from the global update below.
                hist_close[ticker_idx] = np.concatenate(
                    [hist_close[ticker_idx][:, 1:],
                     closes_d[:, np.newaxis]], axis=1)
                hist_high[ticker_idx]  = np.concatenate(
                    [hist_high[ticker_idx][:, 1:],
                     highs_d[:,  np.newaxis]], axis=1)
                hist_low[ticker_idx]   = np.concatenate(
                    [hist_low[ticker_idx][:, 1:],
                     lows_d[:,   np.newaxis]], axis=1)

            if has_globals and d < hold_days - 1:
                n_all = self.close.shape[0]
                pred_close = np.full((n_all, self.history_len), np.nan)
                pred_high  = pred_close.copy()
                pred_low   = pred_close.copy()
                for ticker_idx in valid_tickers:
                    if ticker_idx in mean_closes:
                        mc_val = mean_closes[ticker_idx]
                        pred_close[ticker_idx, -1] = mc_val
                        pred_high[ticker_idx,  -1] = mc_val
                        pred_low[ticker_idx,   -1] = mc_val
                next_idx = t + d + 1
                if self.dates is not None and next_idx < len(self.dates):
                    pad = np.full(self.history_len - 1,
                                  self.dates[0], dtype=self.dates.dtype)
                    dates_g = np.concatenate(
                        [pad, np.array([self.dates[next_idx]])])
                else:
                    dates_g = None
                global_x = self._compute_global_x_from_prices(
                    pred_close, pred_high, pred_low,
                    dates_g, global_entries)

        forecasts = []
        for ticker_idx in valid_tickers:
            ep = exit_price[ticker_idx].copy()
            ep[active[ticker_idx]] = prices[ticker_idx][active[ticker_idx]]
            P0 = initial_price[ticker_idx]
            returns = (ep - P0) / P0
            forecasts.append((ticker_idx, float(returns.mean()),
                               float(returns.std())))
        return forecasts