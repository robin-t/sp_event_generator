from tools.mpi_utils import (
    COMM, RANK, SIZE,
    root_print,
    mpi_tqdm
)

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from data.data_store import DataStore
from data.data_loader import DataLoader
from mc.transition_model import TransitionModel
from mc.cpp_adapter import CppAdapter, cpp_available


from density.density import DensitySet
from density.indicator import GlobalIndicator

class LiveStrategyRunner:

    def __init__(self, run_folder):

        self.run_folder = Path("strategy/runs") / run_folder

        with open(self.run_folder / "results.json") as f:
            data = json.load(f)

        self.params = data["params"]
        self.tune = data["tune_config"]

        self.history_len = 100
        self.data_store = DataStore()

        # ----------------------------
        # Load densities properly
        # ----------------------------

        def load_density_from_meta(entry):

            name = entry["indicator"]
            params = entry["params"]

            for folder in Path("density/densities").glob("*"):

                meta_file = folder / "meta.json"
                if not meta_file.exists():
                    continue

                with open(meta_file) as f:
                    meta = json.load(f)

                if (
                    meta["indicator_name"] == name
                    and meta["indicator_params"] == params
                ):
                    density = DensitySet.load(folder)
                    return density, density.indicator

            raise RuntimeError(f"Density not found for {name} {params}")

        self.R_dens = [load_density_from_meta(d) for d in self.tune["R_densities"]]
        self.M_dens = [load_density_from_meta(d) for d in self.tune["M_densities"]]
        self.S_dens = [load_density_from_meta(d) for d in self.tune["S_densities"]]

        self.weights = self.tune["best_weights"]

    # ==========================================================
    # Run Live Strategy
    # ==========================================================

    def run(self):

        if RANK == 0:
            print("\n=== LIVE STRATEGY RUN ===")

            if SIZE > 1:
                print(f"Running live forecast in MPI mode with {SIZE} processes")

            # -------------------------------
            # Bankroll input
            # -------------------------------

            default_bankroll = self.params["initial_bankroll"]

            bankroll_input = input(
                f"Enter bankroll (default {default_bankroll}): "
            ).strip()

            bankroll = float(bankroll_input) if bankroll_input else default_bankroll
            print(f"Using bankroll: {bankroll:.2f}")

            # -------------------------------
            # Invest fraction input
            # -------------------------------

            default_invest_frac = self.params["invest_frac"]

            invest_input = input(
                f"Invest fraction of bankroll (default {default_invest_frac}): "
            ).strip()

            invest_frac = float(invest_input) if invest_input else default_invest_frac
            print(f"Using invest fraction: {invest_frac:.2f}")

            # -------------------------------
            # Optional probability filters
            # -------------------------------

            min_target_input = input(
                "Minimum predicted return (Enter to skip): "
            ).strip()

            min_target = float(min_target_input) if min_target_input else None

            # -------------------------------
            # MC samples override
            # -------------------------------

            mc_input = input(
                f"MC samples (default {self.params['mc_samples']}): "
            ).strip()

            mc_samples = int(mc_input) if mc_input else self.params["mc_samples"]

            print(f"Using MC samples: {mc_samples}")

            payload = {
                "bankroll": bankroll,
                "invest_frac": invest_frac,
                "min_target": min_target,
                "mc_samples": mc_samples
            }

        else:
            payload = None

        # -----------------------------------
        # Broadcast to all ranks
        # -----------------------------------

        if SIZE > 1:
            payload = COMM.bcast(payload, root=0)

        bankroll = payload["bankroll"]
        invest_frac = payload["invest_frac"]
        min_target = payload["min_target"]
        mc_samples = payload["mc_samples"]

        # -------------------------------
        # Load tickers from dataset
        # -------------------------------

        tickers = self.params["tickers"] if "tickers" in self.params else None
        if tickers is None:
            raise RuntimeError("Tickers not saved in run params.")

        if RANK == 0:
            print("Updating data...")
            raw = self.data_store.download_full(tickers)
        else:
            raw = None

        if COMM:
            raw = COMM.bcast(raw, root=0)

        dates, close, high, low, _ = DataLoader.align(raw)

        # -------------------------------
        # Load model (C++ preferred)
        # -------------------------------

        if cpp_available:
            self._cpp = CppAdapter(self.R_dens, self.M_dens, self.S_dens, self.weights)
            model = None
        else:
            self._cpp = None
            model = TransitionModel(self.R_dens, self.M_dens, self.S_dens, self.weights)

        # -------------------------------
        # Forecast all tickers
        # -------------------------------

        current_day = close.shape[1] - 1

        forecasts = []

        root_print("Total days in dataset:", close.shape[1])
        root_print("Required history length:", self.history_len)
            
        

        local_indices = range(len(tickers))[RANK::SIZE]
        local_forecasts = []

        n_skip_nan = 0
        n_skip_range = 0

        for i in mpi_tqdm(local_indices, desc="Live forecasts"):
            ticker = tickers[i]

            if current_day - self.history_len < 0:
                n_skip_range += 1
                continue

            history_close = close[i, current_day - self.history_len: current_day]
            history_high  = high[i, current_day - self.history_len: current_day]
            history_low   = low[i, current_day - self.history_len: current_day]

            if (
                np.isnan(history_close).any() or
                np.isnan(history_high).any() or
                np.isnan(history_low).any()
            ):
                n_skip_nan += 1
                continue

            mean_ret, std_ret = self._forecast_single(
                model,
                history_high,
                history_low,
                history_close,
                close[i, current_day],
                mc_samples,
                current_day,
                close, high, low,
            )

            if min_target is not None and mean_ret < min_target:
                continue

            local_forecasts.append((i, ticker, mean_ret, std_ret))

        print(f"  Skipped (insufficient history): {n_skip_range}")
        print(f"  Skipped (NaN in history):       {n_skip_nan}")
        print(f"  Forecasted:                     {len(local_forecasts)}")

        if SIZE > 1:
            gathered = COMM.gather(local_forecasts, root=0)

            if RANK != 0:
                return  # workers exit here

            forecasts = []
            for g in gathered:
                forecasts.extend(g)
            
            total_from_gather = sum(len(g) for g in gathered)
            if len(forecasts) == total_from_gather:
                print(f"\nMPI gather check: OK ({len(forecasts)} forecasts collected from {SIZE} ranks)")
            else:
                print(f"\nMPI gather check: ERROR (mismatch in gathered forecasts)")
        else:
            forecasts = local_forecasts

        print("Forecast count:", len(forecasts))

        # -------------------------------
        # Apply selection rule
        # -------------------------------

        rule = self.params["rule"]
        N = self.params["n_hold"]

        # ── Diagnostic: show all forecasts before filtering ──
        if forecasts:
            print(f"\n  All forecasts ({rule} rule, need mean-std>0 for conservative):")
            scored = sorted(forecasts, key=lambda x: x[2], reverse=True)
            print(f"  {'Ticker':<10} {'mean%':>7} {'std%':>7} {'mean-std%':>10}  pass?")
            print("  " + "-" * 42)
            for _, ticker, mean, std in scored:
                passes = (mean - std > 0) if rule == "conservative" else True
                flag = "✓" if passes else "✗"
                print(f"  {ticker:<10} {100*mean:>7.2f} {100*std:>7.2f} {100*(mean-std):>10.2f}  {flag}")
        print()

        candidates = []

        for idx, ticker, mean, std in forecasts:

            if rule == "conservative":
                if mean - std > 0:
                    candidates.append((idx, ticker, mean, std))

            else:
                score = mean / std if std > 0 else 0
                candidates.append((idx, ticker, mean, std, score))

        if rule == "risk_adjusted":
            candidates.sort(key=lambda x: x[4], reverse=True)
        else:
            candidates.sort(key=lambda x: x[2], reverse=True)

        selected = candidates[:N]

        print(f"Candidates after filter: {len(candidates)}  →  Selected: {len(selected)} (max {N})")

        # -------------------------------
        # Capital allocation
        # -------------------------------

        invest_amount = bankroll * invest_frac

        risk_normalized = self.params.get("risk_normalized", False)

        positions = []

        if len(selected) > 0:

            if not risk_normalized:

                allocation = invest_amount / len(selected)

                for entry in selected:

                    idx, ticker, mean, std, *rest = entry
                    price = close[idx, current_day]
                    shares = int(allocation // price)

                    if shares <= 0:
                        continue

                    uncertainty = std / np.sqrt(mc_samples)

                    positions.append({
                        "ticker": ticker,
                        "price": price,
                        "shares": shares,
                        "expected_return": mean,
                        "risk": std,
                        "uncertainty": uncertainty
                    })

            else:

                inv_sigmas = [1/std if std > 0 else 0 for _,_,mean,std,*_ in selected]
                total_inv_sigma = sum(inv_sigmas)

                for entry, inv_sigma in zip(selected, inv_sigmas):

                    idx, ticker, mean, std, *rest = entry

                    weight = inv_sigma / total_inv_sigma if total_inv_sigma > 0 else 0
                    allocation = invest_amount * weight

                    price = close[idx, current_day]
                    shares = int(allocation // price)

                    if shares <= 0:
                        continue

                    uncertainty = std / np.sqrt(mc_samples)

                    positions.append({
                        "ticker": ticker,
                        "price": price,
                        "shares": shares,
                        "expected_return": mean,
                        "risk": std,
                        "uncertainty": uncertainty
                    })

        print("Positions count:", len(positions))

        # -------------------------------
        # Output results
        # -------------------------------

        print("\n=== TRADE PLAN ===")

        last_dt = dates[-1]  # datetime64[D], formats directly as "YYYY-MM-DD"

        print("Data date:", str(last_dt))
        print()
        print(f"Hold days:  {self.params['hold_days']}")
        print(f"Target:     {100*self.params['target']:.2f}%")
        print(f"Stop:       {100*self.params['stop']:.2f}%")
        print(f"MC samples: {mc_samples}")
        print(f"Allocation mode: {'Risk-normalized' if self.params.get('risk_normalized') else 'Equal weight'}")
        print()

        print("Ticker | Shares | Price     | Expected return ± risk | Uncertainty")
        print("-------+--------+-----------+------------------------+------------")

        for p in positions:
            ret_risk = f"{100*p['expected_return']:6.2f}% ± {100*p['risk']:6.2f}%"

            print(
                f"{p['ticker']:6s} | "
                f"{p['shares']:6d} | "
                f"{p['price']:9.2f} | "
                f"{ret_risk:>22} | "
                f"{100*p['uncertainty']:10.2f}%"
            )

        total_used = sum(p["shares"] * p["price"] for p in positions)

        deploy_pct = total_used / bankroll if bankroll > 0 else 0
        target_deploy_pct = invest_amount / bankroll if bankroll > 0 else 0

        print()
        print(f"Capital deployed: {total_used:.2f}")
        print(f"Cash remaining:   {bankroll - total_used:.2f}")
        print(f"Target allocation: {100*target_deploy_pct:.2f}% of bankroll")
        print(f"Actual deployed:   {100*deploy_pct:.2f}% of bankroll")

        # -------------------------------
        # Save file
        # -------------------------------

        output_dir = Path("strategy/live_runs")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = output_dir / f"live_{timestamp}.json"

        with open(out_file, "w") as f:
            json.dump({
                "date": str(dates[-1])[:10],
                "bankroll": bankroll,
                "positions": positions
            }, f, indent=2)

        print(f"\nLive run saved to {out_file}")

    # ==========================================================
    # Forecast helper
    # ==========================================================

    def _precompute_global_x(self, t, hold_days, close_mat, high_mat, low_mat):
        """Compute global indicator values for forecast window [t, t+hold_days)."""
        global_entries = []
        for component, dens_list in (("R", self.R_dens),
                                      ("M", self.M_dens),
                                      ("S", self.S_dens)):
            for density, indicator in dens_list:
                if density.meta.get("is_global", False):
                    key = (f"{component}_{density.meta['indicator_name']}_"
                           f"{'_'.join(map(str, density.meta['indicator_params']))}")
                    global_entries.append((key, density.meta["indicator_name"],
                                           density.meta["indicator_params"]))

        if not global_entries:
            return [{} for _ in range(hold_days)]

        slice_start = max(0, t - self.history_len)
        slice_end   = t + hold_days
        close_slice = close_mat[:, slice_start:slice_end]
        high_slice  = high_mat[:,  slice_start:slice_end]
        low_slice   = low_mat[:,   slice_start:slice_end]

        series_cache = {}
        for key, name, params in global_entries:
            if key not in series_cache:
                gi = GlobalIndicator(name, params, close_slice, high_slice, low_slice)
                series_cache[key] = gi.compute_series()

        t_in_slice = t - slice_start
        global_x_per_day = []
        for d in range(hold_days):
            day_x = {}
            for key, series in series_cache.items():
                idx = t_in_slice + d
                day_x[key] = float(series[idx]) if idx < len(series) else np.nan
            global_x_per_day.append(day_x)

        return global_x_per_day

    def _forecast_single(
        self,
        model,
        history_high,
        history_low,
        history_close,
        initial_price,
        mc_samples,
        current_day,
        close_mat, high_mat, low_mat,
    ):
        hold_days = self.params["hold_days"]
        target    = self.params["target"]
        stop      = self.params["stop"]

        global_x_per_day = self._precompute_global_x(
            current_day, hold_days, close_mat, high_mat, low_mat
        )

        # ── C++ fast path ──────────────────────────────────────
        if self._cpp is not None:
            seed = int(np.random.randint(0, 2**31))
            mean_ret, std_ret = self._cpp.forecast_ticker(
                history_high, history_low, history_close,
                mc_samples, hold_days,
                target, stop,
                global_x_per_day,
                seed,
            )
            return mean_ret, std_ret

        # ── Python fallback ─────────────────────────────────────
        initial_price_p = history_close[-1]
        target_price    = initial_price_p * (1 + target)
        stop_price      = initial_price_p * (1 - stop)

        hist_close = np.tile(history_close, (mc_samples, 1))
        hist_high  = np.tile(history_high,  (mc_samples, 1))
        hist_low   = np.tile(history_low,   (mc_samples, 1))

        prices     = np.full(mc_samples, initial_price_p)
        exit_price = np.full(mc_samples, np.nan)
        active     = np.ones(mc_samples, dtype=bool)
        closes_d   = np.zeros(mc_samples)
        highs_d    = np.zeros(mc_samples)
        lows_d     = np.zeros(mc_samples)

        for d in range(hold_days):
            if not np.any(active):
                break
            global_x = global_x_per_day[d]
            for k in range(mc_samples):
                if not active[k]:
                    continue
                x_vals = model.compute_indicator_values(
                    hist_high[k], hist_low[k], hist_close[k], global_x=global_x)
                closes_d[k], highs_d[k], lows_d[k] = model.step_from_x(
                    x_vals, hist_close[k, -1])

            hit_stop   = active & (lows_d  <= stop_price)
            hit_target = active & (highs_d >= target_price)
            hit_both   = hit_stop & hit_target
            exit_price[hit_both]               = stop_price;  active[hit_both]               = False
            exit_price[hit_stop  & ~hit_both]  = stop_price;  active[hit_stop  & ~hit_both]  = False
            exit_price[hit_target & ~hit_both] = target_price; active[hit_target & ~hit_both] = False
            prices[active] = closes_d[active]
            hist_close = np.concatenate([hist_close[:, 1:], closes_d[:, np.newaxis]], axis=1)
            hist_high  = np.concatenate([hist_high[:,  1:], highs_d[:,  np.newaxis]], axis=1)
            hist_low   = np.concatenate([hist_low[:,   1:], lows_d[:,   np.newaxis]], axis=1)

        exit_price[active] = prices[active]
        returns = (exit_price - initial_price_p) / initial_price_p
        return returns.mean(), returns.std()