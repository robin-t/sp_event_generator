from tools.mpi_utils import (
    COMM, RANK, SIZE,
    root_print
)

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from strategy.strategy_tester import StrategyTester
from tools.date_range import validate_spec
from strategy.live_runner import LiveStrategyRunner

from itertools import product
from tqdm import tqdm


class StrategyHandler:

    def __init__(self, tune_handler):
        self.tune_handler = tune_handler

    # ----------------------------------------------------------
    # Menu
    # ----------------------------------------------------------

    def menu(self):

        from mpi4py import MPI
        COMM = MPI.COMM_WORLD
        RANK = COMM.Get_rank()

        while True:

            # ----------------------------
            # Rank 0 prints menu + input
            # ----------------------------
            if RANK == 0:
                print("\n=== Strategy Menu ===")
                print("1. Run strategy")
                print("2. List strategy runs")
                print("3. Compare strategy runs")
                print("4. Run strategy (live)")
                print("0. Back")

                choice = input("Select option: ").strip()
            else:
                choice = None

            # Broadcast choice
            choice = COMM.bcast(choice, root=0)

            # ----------------------------
            # Execute on ALL ranks
            # ----------------------------

            if choice == "1":
                self.run_strategy()

            elif choice == "2":
                if RANK == 0:
                    self.list_strategy_runs()

            elif choice == "3":
                if RANK == 0:
                    self.analyze_strategy_runs()

            elif choice == "4":

                if RANK == 0:
                    selected_run = self._select_run_for_live()
                else:
                    selected_run = None

                selected_run = COMM.bcast(selected_run, root=0)

                if selected_run is not None:
                    runner = LiveStrategyRunner(selected_run)
                    runner.run()

            elif choice == "0":
                break

            else:
                root_print("Invalid choice.")

    # ----------------------------------------------------------
    # Run strategy
    # ----------------------------------------------------------

    def run_strategy(self):

        if RANK == 0:
            print("\n=== Run Strategy ===")
            print("1. Run over grid")
            print("2. Manual input")

            mode = input("Select option: ").strip()
        else:
            mode = None

        mode = COMM.bcast(mode, root=0)

        if mode == "1":
            self._run_strategy_grid()
        elif mode == "2":
            self._run_strategy_manual()

    # ----------------------------------------------------------
    # Test strategy workflow over grid
    # ----------------------------------------------------------

    def _run_strategy_grid(self):

        if RANK == 0:

            print("\n=== Select Tune ===")
            tune_config = self.tune_handler.select_tune()

            if tune_config is None:
                payload = None
            else:
                print("\nSelect ticker universe:")
                lists = self.tune_handler.tickers.list_names()
                for i, name in enumerate(lists):
                    print(f"{i+1}. {name}")

                idx = int(input("Choice: ")) - 1
                tickers = self.tune_handler.tickers.get(lists[idx])

                # --- Select grid file ---
                grid_dir = "strategy/grids"
                os.makedirs(grid_dir, exist_ok=True)

                files = [f for f in os.listdir(grid_dir) if f.endswith(".json")]

                print("\nAvailable grid files:")
                for i, f in enumerate(files):
                    print(f"{i+1}. {f}")
                print("0. Use default grid.json")

                choice = input("Select grid file: ").strip()

                if choice == "0" or not files:
                    grid_file = os.path.join(grid_dir, "grid.json")
                else:
                    grid_file = os.path.join(grid_dir, files[int(choice)-1])

                with open(grid_file) as f:
                    grid = json.load(f)

                payload = {
                    "tune_config": tune_config,
                    "tickers": tickers,
                    "grid": grid
                }

        else:
            payload = None

        payload = COMM.bcast(payload, root=0)

        if payload is None:
            return

        tune_config = payload["tune_config"]
        tickers = payload["tickers"]
        grid = payload["grid"]

        tester = StrategyTester(tune_config)
        tester.load_dataset(tickers)

        # ---------------------------------
        # Expand grid (with paired fields)
        # ---------------------------------

        # Extract paired/special parameters
        target_stop_pairs = grid.get("target_stop", [])
        fee_pairs         = grid.get("fee_structure", [])

        # date_ranges: list of spec strings e.g. ["1990-2007,2010-2019", "2021-2022"]
        # Backward compat: start_end_year: [[2023, 2025]] → "2023-2025"
        raw_date_ranges = grid.get("date_ranges")
        if raw_date_ranges is None:
            year_pairs = grid.get("start_end_year", [])
            raw_date_ranges = [f"{s}-{e}" for s, e in year_pairs]

        # Remove special keys from independent grid
        grid_independent = {
            k: v for k, v in grid.items()
            if k not in ["target_stop", "start_end_year", "fee_structure", "date_ranges"]
        }

        keys = list(grid_independent.keys())
        value_lists = [grid_independent[k] for k in keys]

        combinations = []

        for values in product(*value_lists):
            base = dict(zip(keys, values))

            for date_spec in raw_date_ranges:

                for fee_fixed, fee_pct in fee_pairs:

                    for target, stop in target_stop_pairs:

                        params = base.copy()

                        params["date_spec"] = date_spec

                        params["fee_fixed"] = fee_fixed
                        params["fee_pct"] = fee_pct

                        params["target"] = target
                        params["stop"] = stop

                        params["tickers"] = tickers

                        if "mc_paths_per_day" in params:
                            params["mc_samples"] = (
                                params.pop("mc_paths_per_day") * params["hold_days"]
                            )

                        combinations.append(params)

        # ---------------------------------
        # Create grid run folder (root only)
        # ---------------------------------

        if RANK == 0:

            base_dir = "strategy/grid_runs"
            os.makedirs(base_dir, exist_ok=True)

            existing = [
                int(d.split("_")[1])
                for d in os.listdir(base_dir)
                if d.startswith("grid_") and d.split("_")[1].isdigit()
            ]

            grid_id = max(existing) + 1 if existing else 1

            grid_folder = os.path.join(base_dir, f"grid_{grid_id:03d}")
            os.makedirs(grid_folder)

            # Save grid config
            with open(os.path.join(grid_folder, "grid_config.json"), "w") as f:
                json.dump(grid, f, indent=2)

            combination_map = {}
        else:
            grid_folder = None
            combination_map = None

        grid_folder = COMM.bcast(grid_folder, root=0)

        # ---------------------------------
        # Run combinations
        # ---------------------------------

        iterator = tqdm(range(len(combinations)), desc="Grid combinations") if RANK == 0 else range(len(combinations))

        for i in iterator:

            params = combinations[i]

            finals = tester.run_tests_with_params(params)

            if RANK == 0:

                combo_folder = os.path.join(grid_folder, f"combo_{i+1:03d}")
                os.makedirs(combo_folder)

                # Move last run results into combo folder
                # (since run_tests always saves into strategy/runs/)
                runs_dir = "strategy/runs"
                last_run = sorted(os.listdir(runs_dir))[-1]
                os.rename(
                    os.path.join(runs_dir, last_run),
                    os.path.join(combo_folder)
                )

                combination_map[f"{i+1:03d}"] = params

        if RANK == 0:
            with open(os.path.join(grid_folder, "combination_map.json"), "w") as f:
                json.dump(combination_map, f, indent=2)

            print(f"\nGrid run saved to {grid_folder}")

            # ---------------------------------
            # Build grid summary (root only)
            # ---------------------------------

            print("\nBuilding grid summary...")

            combo_results = []

            for key, params in combination_map.items():

                combo_folder = os.path.join(grid_folder, f"combo_{key}")

                # results.json was moved into this folder
                results_path = None
                for root_dir, dirs, files in os.walk(combo_folder):
                    for f in files:
                        if f == "results.json":
                            results_path = os.path.join(root_dir, f)
                            break

                if results_path is None:
                    continue

                with open(results_path) as f:
                    data = json.load(f)

                mean_final = data["summary"]["mean_final"]
                std_final = data["summary"]["std_final"]

                combo_results.append({
                    "combo_id": key,
                    "mean_final": mean_final,
                    "std_final": std_final,
                    "params": params
                })

            # Rank combinations
            combo_results.sort(
                key=lambda x: (-x["mean_final"], x["std_final"])
            )

            # Save full ranking
            with open(os.path.join(grid_folder, "grid_summary.json"), "w") as f:
                json.dump(combo_results, f, indent=2)

            # Save best only
            best = combo_results[0]

            with open(os.path.join(grid_folder, "best_combination.json"), "w") as f:
                json.dump(best, f, indent=2)

            print("\n=== Grid Best Combination ===")
            print(f"Combo ID:    {best['combo_id']}")
            print(f"Mean final:  {best['mean_final']:.2f}")
            print(f"Std final:   {best['std_final']:.2f}")
            print("\nParameters:")
            for k, v in best["params"].items():
                print(f"  {k:<20} {v}")
            
            # ---------------------------------
            # Compute normalized per-cycle metrics
            # ---------------------------------

            normalized_results = []

            for entry in combo_results:

                params = entry["params"]
                mean_final = entry["mean_final"]
                std_final = entry["std_final"]

                initial_bankroll = params["initial_bankroll"]
                n_cycles = params["n_cycles"]
                hold_days = params["hold_days"]
                fee_pct = params.get("fee_pct", 0.0)
                fee_fixed = params.get("fee_fixed", 0.0)
                invest_frac = params.get("invest_frac", 0.5)
                n_hold = params.get("n_hold", 1)

                # Mean return per cycle (%)
                mean_cycle_return = (
                    (mean_final / initial_bankroll) ** (1 / n_cycles) - 1
                ) * 100

                # Approximate std per cycle (%)
                std_cycle_return = (
                    (std_final / initial_bankroll) / n_cycles
                ) * 100

                sharpe_like = (
                    mean_cycle_return / std_cycle_return
                    if std_cycle_return != 0 else 0
                )

                # Annual net return:
                # cycles_per_year = 252 trading days / hold_days
                # Fee drag per cycle = round-trip fee on deployed capital
                # Assumes ~1 trade per cycle (conservative estimate)
                cycles_per_year = 252.0 / hold_days
                order_size = initial_bankroll * invest_frac / n_hold
                fee_drag_pct = (fee_fixed + fee_pct * order_size) * 2 * n_hold / initial_bankroll * 100
                net_cycle_return = mean_cycle_return - fee_drag_pct
                annual_net_return = (
                    (1 + net_cycle_return / 100) ** cycles_per_year - 1
                ) * 100

                # Annual std: propagate per-cycle std over independent cycles
                # std_annual ≈ std_cycle * sqrt(cycles_per_year)
                # (valid for non-overlapping hold periods)
                annual_std = std_cycle_return * (cycles_per_year ** 0.5)

                annual_gross_return = (
                    (1 + mean_cycle_return / 100) ** cycles_per_year - 1
                ) * 100

                normalized_results.append({
                    "combo_id": entry["combo_id"],
                    "mean_cycle_return": mean_cycle_return,
                    "std_cycle_return": std_cycle_return,
                    "sharpe_like": sharpe_like,
                    "net_cycle_return": net_cycle_return,
                    "annual_net_return": annual_net_return,
                    "annual_gross_return": annual_gross_return,
                    "annual_std": annual_std,
                    "fee_drag_pct": fee_drag_pct,
                    "params": params
                })

            # Sort by mean return descending for plot 1
            normalized_results.sort(
                key=lambda x: -x["mean_cycle_return"]
            )

            # ---------------------------------
            # Plot 1: Mean return ± std per cycle
            # ---------------------------------

            x = np.arange(len(normalized_results))
            means = [r["mean_cycle_return"] for r in normalized_results]
            stds = [r["std_cycle_return"] for r in normalized_results]

            # Identify best
            best = normalized_results[0]
            best_index = 0  # since sorted by return descending

            n_r = len(normalized_results)
            fig, ax = plt.subplots(figsize=(max(10, n_r * 0.55 + 2), 6))
            ax.errorbar(x, means, yerr=stds, fmt='o', capsize=4)
            ax.scatter(best_index, means[best_index], s=150, marker='D', zorder=5)
            ax.annotate(f"Best (ID {best['combo_id']})",
                        (best_index, means[best_index]))
            ax.axhline(0, color="black", alpha=0.75, linewidth=1.0)
            ax.set_xlabel("Strategy combination (sorted by mean return)")
            ax.set_ylabel("Mean return per cycle (%)")
            ax.set_title("Per-cycle Mean Return ± Risk")
            plt.tight_layout()
            plt.savefig(os.path.join(grid_folder, "plot_mean_std_per_cycle.pdf"))
            plt.close()

            # ---------------------------------
            # Plot 2: Risk vs Return scatter
            # ---------------------------------

            risks = [r["std_cycle_return"] for r in normalized_results]
            returns = [r["mean_cycle_return"] for r in normalized_results]

            best_sharpe = max(normalized_results, key=lambda x: x["sharpe_like"])

            best_idx = next(
                i for i, r in enumerate(normalized_results)
                if r["combo_id"] == best_sharpe["combo_id"]
            )
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(risks, returns)
            ax.scatter(risks[best_idx], returns[best_idx], s=150, marker='D', zorder=5)
            ax.annotate(f"Best (ID {best_sharpe['combo_id']})",
                        (risks[best_idx], returns[best_idx]))
            ax.axhline(0, color="black", alpha=0.75, linewidth=1.0)
            ax.set_xlabel("Risk per cycle (%)")
            ax.set_ylabel("Mean return per cycle (%)")
            ax.set_title("Risk vs Return (Per Cycle)")
            plt.tight_layout()
            plt.savefig(os.path.join(grid_folder, "plot_risk_vs_return.pdf"))
            plt.close()

            # ---------------------------------
            # Plot 3: Sharpe-like ranking
            # ---------------------------------

            normalized_results.sort(
                key=lambda x: -x["sharpe_like"]
            )

            x = np.arange(len(normalized_results))
            sharpes = [r["sharpe_like"] for r in normalized_results]

            plt.figure()
            plt.bar(x, sharpes)
            plt.xlabel("Strategy combination (sorted by Sharpe-like)")
            plt.ylabel("Mean / Risk (dimensionless)")
            plt.title("Risk-Adjusted Performance Ranking")
            plt.tight_layout()

            plt.savefig(os.path.join(grid_folder, "plot_sharpe_ranking.pdf"))
            plt.close()

            # ---------------------------------
            # Plot 4: Annual gross vs net return (grouped bars)
            # ---------------------------------

            normalized_results.sort(key=lambda x: -x["annual_net_return"])

            n_combos = len(normalized_results)
            x = np.arange(n_combos)
            w = 0.38

            annual_gross = [r["annual_gross_return"] for r in normalized_results]
            annual_nets  = [r["annual_net_return"]   for r in normalized_results]
            annual_stds  = [r["annual_std"]           for r in normalized_results]

            fig, ax = plt.subplots(figsize=(max(10, n_combos * 0.55 + 2), 6))

            # Gross dots
            ax.errorbar(x - 0.15, annual_gross, fmt='o', color="steelblue",
                        alpha=0.5, markersize=6, label="Gross", linestyle="none")

            # Net dots with ±1σ error bars
            net_colors_pt = ["steelblue" if v >= 0 else "tomato" for v in annual_nets]
            for xi, (yv, ye, col) in enumerate(zip(annual_nets, annual_stds, net_colors_pt)):
                ax.errorbar(xi + 0.15, yv, yerr=ye, fmt='o', color=col,
                            capsize=4, linewidth=1.2, markersize=7,
                            label="Net (after fees)" if xi == 0 else "")

            ax.axhline(0, color="black", alpha=0.75, linewidth=1.0)
            ax.set_xticks(x)
            ax.set_xticklabels([f"ID {r['combo_id']}" for r in normalized_results],
                               rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Estimated annual return (%)")
            ax.set_title("Annual Gross vs Net Return ± 1σ by Strategy Combination")
            ax.legend(fontsize=8)

            best_annual = normalized_results[0]
            ax.annotate(
                f"Best (ID {best_annual['combo_id']})"
                f"  hold={best_annual['params']['hold_days']}d"
                f"  {best_annual['params']['target']:.0%}/{best_annual['params']['stop']:.0%}",
                xy=(0 + 0.15, annual_nets[0]),
                xytext=(max(1, n_combos * 0.1),
                        annual_nets[0] + max(1.5, abs(annual_nets[0]) * 0.25)),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=8,
            )

            plt.tight_layout()
            plt.savefig(os.path.join(grid_folder, "plot_annual_net_return.pdf"))
            plt.close()

            # ---------------------------------
            # Print annual net return summary
            # ---------------------------------

            # ---------------------------------
            # Plot 5: Sharpe score by outcome bucket (box + violin)
            # ---------------------------------

            BUCKETS = ["hit_target", "timeout_positive", "timeout_negative", "hit_stop"]
            BUCKET_LABELS = ["Hit Target", "+ve Timeout", "-ve Timeout", "Hit Stop"]
            BUCKET_COLORS = ["steelblue", "mediumseagreen", "salmon", "tomato"]

            # Collect per-combo bucket data (sorted by annual net return, same order as plot 4)
            combo_bucket_data = []
            for r in normalized_results:  # already sorted by annual_net_return desc
                cid = r["combo_id"]
                combo_folder_path = os.path.join(grid_folder, f"combo_{cid}")
                results_path_b = None
                for root_dir, dirs, files in os.walk(combo_folder_path):
                    for fname in files:
                        if fname == "results.json":
                            results_path_b = os.path.join(root_dir, fname)
                            break
                if results_path_b is None:
                    combo_bucket_data.append({b: [] for b in BUCKETS})
                    continue
                with open(results_path_b) as f_b:
                    data_b = json.load(f_b)
                sbb = data_b.get("sharpe_by_bucket", {b: [] for b in BUCKETS})
                combo_bucket_data.append(sbb)

            n_combos = len(normalized_results)
            x_pos = np.arange(n_combos)
            combo_ids = [r["combo_id"] for r in normalized_results]

            # --- Box plot (full range + clipped) ---
            CLIP = 5.0
            width = 0.18
            offsets = np.linspace(-(1.5 * width), 1.5 * width, 4)

            for clip, suffix, title_suffix in [
                (None,  "full",    ""),
                (CLIP, "clipped", f" [clipped to ±{CLIP}]"),
            ]:
                fig, ax = plt.subplots(figsize=(max(10, n_combos * 0.6 + 2), 6))

                for bi, (bucket, label, color, offset) in enumerate(
                        zip(BUCKETS, BUCKET_LABELS, BUCKET_COLORS, offsets)):
                    data_per_combo = [cbd[bucket] for cbd in combo_bucket_data]

                    if clip is not None:
                        lim = clip
                        data_per_combo = [
                            [v for v in d if -lim <= v <= lim]
                            for d in data_per_combo
                        ]

                    positions = x_pos + offset
                    ax.boxplot(
                        data_per_combo,
                        positions=positions,
                        widths=width * 0.9,
                        patch_artist=True,
                        manage_ticks=False,
                        boxprops=dict(facecolor=color, alpha=0.45),
                        medianprops=dict(color="black", linewidth=1.5, alpha=1.0),
                        whiskerprops=dict(color=color),
                        capprops=dict(color=color),
                        flierprops=dict(marker=".", color=color, alpha=0.5, markersize=3),
                    )
                    ax.plot([], [], color=color, linewidth=6, alpha=0.6, label=label)

                ax.set_xticks(x_pos)
                ax.set_xticklabels([f"ID {{c}}" for c in combo_ids],
                                   rotation=45, ha="right", fontsize=7)
                ax.axhline(0, color="black", alpha=0.75, linewidth=1.0, linestyle="--")
                ax.set_xlabel("Strategy combination (sorted by annual net return)")
                ax.set_ylabel("Predicted Sharpe score at decision time")
                ax.set_title(f"Sharpe Score Distribution by Outcome Bucket (Box){{title_suffix}}")
                ax.legend(loc="upper right", fontsize=8)
                plt.tight_layout()
                plt.savefig(os.path.join(grid_folder,
                            f"plot_sharpe_by_bucket_box_{{suffix}}.pdf"))
                plt.close()

            # --- Violin plot (full range + clipped) ---
            width_v = 0.16

            for clip, suffix, title_suffix in [
                (None,  "full",    ""),
                (CLIP, "clipped", f" [clipped to ±{CLIP}]"),
            ]:
                fig, ax = plt.subplots(figsize=(max(10, n_combos * 0.6 + 2), 6))

                for bi, (bucket, label, color, offset) in enumerate(
                        zip(BUCKETS, BUCKET_LABELS, BUCKET_COLORS, offsets)):
                    for ci, cbd in enumerate(combo_bucket_data):
                        vals = cbd[bucket]
                        if clip is not None:
                            lim = clip
                            vals = [v for v in vals if -lim <= v <= lim]
                        if len(vals) < 3:
                            continue
                        vp = ax.violinplot(
                            vals,
                            positions=[x_pos[ci] + offset],
                            widths=width_v,
                            showmedians=True,
                            showextrema=False,
                        )
                        for body in vp["bodies"]:
                            body.set_facecolor(color)
                            body.set_alpha(0.5)
                        vp["cmedians"].set_color("black")
                        vp["cmedians"].set_linewidth(1.5)
                        vp["cmedians"].set_alpha(1.0)
                    ax.plot([], [], color=color, linewidth=6, alpha=0.5, label=label)

                ax.set_xticks(x_pos)
                ax.set_xticklabels([f"ID {{c}}" for c in combo_ids],
                                   rotation=45, ha="right", fontsize=7)
                ax.axhline(0, color="black", alpha=0.75, linewidth=1.0, linestyle="--")
                ax.set_xlabel("Strategy combination (sorted by annual net return)")
                ax.set_ylabel("Predicted Sharpe score at decision time")
                ax.set_title(f"Sharpe Score Distribution by Outcome Bucket (Violin){{title_suffix}}")
                ax.legend(loc="upper right", fontsize=8)
                plt.tight_layout()
                plt.savefig(os.path.join(grid_folder,
                            f"plot_sharpe_by_bucket_violin_{{suffix}}.pdf"))
                plt.close()

            print("\n=== Annual Net Return Summary ===")
            print(f"{'ID':<6} {'hold':>5} {'n_hold':>6} {'target':>7} {'stop':>5} "
                  f"{'gross/cyc':>10} {'fee/cyc':>8} {'net/cyc':>8} "
                  f"{'annual_net':>11} {'±1σ/yr':>8}")
            print("-" * 85)
            for r in normalized_results:
                p = r["params"]
                print(f"  {r['combo_id']:<4} {p['hold_days']:>5}d {p['n_hold']:>6} "
                      f"{p['target']:>6.0%} {p['stop']:>5.0%} "
                      f"{r['mean_cycle_return']:>+9.3f}% "
                      f"{r['fee_drag_pct']:>7.3f}% "
                      f"{r['net_cycle_return']:>+7.3f}% "
                      f"{r['annual_net_return']:>+10.1f}% "
                      f"{r['annual_std']:>7.1f}%")


    # ----------------------------------------------------------
    # Test strategy workflow
    # ----------------------------------------------------------

    def _run_strategy_manual(self):

        # --------------------------------------------------
        # Rank 0 collects ALL input
        # --------------------------------------------------

        if RANK == 0:

            print("\n=== Select Tune ===")
            tune_config = self.tune_handler.select_tune()

            if tune_config is None:
                print("No tune selected.")
                payload = None
            else:
                print("\nSelect ticker universe:")

                lists = self.tune_handler.tickers.list_names()

                for i, name in enumerate(lists):
                    print(f"{i+1}. {name}")

                idx = int(input("Choice: ")) - 1
                tickers = self.tune_handler.tickers.get(lists[idx])

                params = self._get_strategy_params()

                params["tickers"] = tickers

                payload = {
                    "tune_config": tune_config,
                    "params": params
                }

        else:
            payload = None

        # --------------------------------------------------
        # Broadcast once
        # --------------------------------------------------

        payload = COMM.bcast(payload, root=0)

        if payload is None:
            return

        tune_config = payload["tune_config"]
        params = payload["params"]
        tickers = params["tickers"]

        # --------------------------------------------------
        # ALL ranks compute
        # --------------------------------------------------

        tester = StrategyTester(tune_config)

        tester.load_dataset(tickers)

        tester.run_tests_with_params(params)

    # ----------------------------------------------------------
    # Collect strategy parameters from user
    # ----------------------------------------------------------

    def _get_strategy_params(self):

        print("\n=== Strategy Parameters ===")

        # --- Test range ---
        print("Date range spec (e.g. '2023-2025' or '1990-2007, 2010-2019'):")
        while True:
            date_spec = input("Date range: ").strip()
            if validate_spec(date_spec):
                break
            print("  Invalid format.")

        # --- Backtest structure ---
        n_tests  = int(input("Number of independent tests: "))
        if n_tests < SIZE:
            print(f"Number of tests must be ≥ number of MPI processes ({SIZE}).")
            print("Please enter a larger value.")
            return self._get_strategy_params()
        n_cycles = int(input("Number of cycles per test: "))
        hold_days = int(input("Days to hold each trade: "))

        mc_paths_per_day = int(input("MC paths per day (total = paths_per_day × hold_days): "))
        mc_samples = mc_paths_per_day * hold_days

        # --- Strategy rules ---
        target = float(input("Target gain (%) e.g. 5: ")) / 100
        stop   = float(input("Stop loss (%) e.g. 5: ")) / 100

        bankroll = float(input("Initial bankroll ($): "))

        invest_frac = float(input("Fraction of bankroll to invest per cycle (0–1): "))

        risk_norm_input = input("Risk-normalized allocation? (y/n, default n): ").strip().lower()
        risk_normalized = risk_norm_input == "y"


        n_hold = int(input("Max number of stocks to hold per cycle: "))

        # --- Fees ---
        fee_fixed = float(input("Broker fixed fee per trade ($): "))  
        fee_pct   = float(input("Broker percent fee per trade (%): ")) / 100

        # --- Selection rule ---
        print("\nSelection rule:")
        print("1. Conservative (Only N stocks if : mean − σ > 0)")
        print("2. Risk-adjusted (Always N ordered by: mean / σ ranking)")

        rule = input("Choice: ").strip()
        rule = "conservative" if rule == "1" else "risk_adjusted"

        sharpe_cutoff = None
        if rule == "risk_adjusted":
            cutoff_input = input("Sharpe cutoff (blank = no cutoff, e.g. 0.5): ").strip()
            if cutoff_input:
                sharpe_cutoff = float(cutoff_input)

        return {
            "date_spec": date_spec,
            "n_tests": n_tests,
            "n_cycles": n_cycles,
            "hold_days": hold_days,
            "mc_samples": mc_samples,
            "target": target,
            "stop": stop,
            "invest_frac": invest_frac,
            "initial_bankroll": bankroll,
            "n_hold": n_hold,
            "fee_fixed": fee_fixed,
            "fee_pct": fee_pct,
            "rule": rule,
            "risk_normalized": risk_normalized,
            "sharpe_cutoff": sharpe_cutoff,
        }
    

    def _select_run_for_live(self):
        """
        Interactive selection of a single strategy run for live trading.
        Supports both standalone runs (strategy/runs/) and grid runs (strategy/grid_runs/).
        Returns a run folder path string (relative to strategy/runs/) or None.
        """
        print("\n=== Select Strategy Run ===")
        print("  1. Single runs")
        print("  2. Grid runs")
        print("  0. Cancel")
        mode = input("Choice: ").strip()

        if mode == "0":
            return None

        elif mode == "1":
            # ── Standalone runs ──────────────────────────────────
            runs_dir = "strategy/runs"
            if not os.path.isdir(runs_dir):
                print("No single runs found.")
                return None

            runs = sorted([
                r for r in os.listdir(runs_dir)
                if os.path.isdir(os.path.join(runs_dir, r))
            ])

            if not runs:
                print("No single runs found.")
                return None

            print("\n=== Available Single Runs ===")
            for i, r in enumerate(runs, 1):
                result_path = os.path.join(runs_dir, r, "results.json")
                tag = ""
                if os.path.exists(result_path):
                    with open(result_path) as f:
                        data = json.load(f)
                    p = data["params"]
                    s = data["summary"]
                    ret = (s["mean_final"] / p["initial_bankroll"] - 1) * 100
                    _spec = p.get('date_spec') or f"{p.get('start_year','?')}-{p.get('end_year','?')}"
                    tag = (f"  {_spec}"
                           f"  hold={p.get('hold_days','?')}d"
                           f"  target={100*p.get('target',0):.0f}%"
                           f"  stop={100*p.get('stop',0):.0f}%"
                           f"  ret={ret:+.1f}%")
                print(f"  {i}. {r}{tag}")

            idx = input("\nSelect run number: ").strip()
            if not idx.isdigit() or not (1 <= int(idx) <= len(runs)):
                print("Invalid selection.")
                return None
            return runs[int(idx) - 1]

        elif mode == "2":
            # ── Grid runs ────────────────────────────────────────
            grid_base = "strategy/grid_runs"
            if not os.path.isdir(grid_base):
                print("No grid runs found.")
                return None

            grids = sorted([
                g for g in os.listdir(grid_base)
                if os.path.isdir(os.path.join(grid_base, g))
            ])

            if not grids:
                print("No grid runs found.")
                return None

            print("\n=== Available Grid Runs ===")
            for i, g in enumerate(grids, 1):
                summary_path = os.path.join(grid_base, g, "grid_summary.json")
                tag = ""
                if os.path.exists(summary_path):
                    with open(summary_path) as f:
                        summary = json.load(f)
                    n = len(summary)
                    best = summary[0] if summary else None
                    if best:
                        bp = best["params"]
                        tag = (f"  {n} combos"
                               f"  best: hold={bp.get('hold_days','?')}d"
                               f"  target={100*bp.get('target',0):.0f}%"
                               f"  stop={100*bp.get('stop',0):.0f}%"
                               f"  ret={best['mean_final']/bp.get('initial_bankroll',1)*100-100:+.1f}%")
                print(f"  {i}. {g}{tag}")

            gidx = input("\nSelect grid number: ").strip()
            if not gidx.isdigit() or not (1 <= int(gidx) <= len(grids)):
                print("Invalid selection.")
                return None

            chosen_grid = grids[int(gidx) - 1]
            grid_folder = os.path.join(grid_base, chosen_grid)
            summary_path = os.path.join(grid_folder, "grid_summary.json")

            if not os.path.exists(summary_path):
                print("Grid summary not found.")
                return None

            with open(summary_path) as f:
                summary = json.load(f)  # sorted best-first by mean_final

            print(f"\n=== Runs in {chosen_grid} (best first) ===")
            for rank, entry in enumerate(summary, 1):
                cid = entry["combo_id"]
                p   = entry["params"]
                ret = entry["mean_final"] / p.get("initial_bankroll", 1) * 100 - 100
                std = entry["std_final"]  / p.get("initial_bankroll", 1) * 100
                line = (f"  {rank:>3}. combo_{cid}"
                        f"  hold={p.get('hold_days','?')}d"
                        f"  target={100*p.get('target',0):.0f}%"
                        f"  stop={100*p.get('stop',0):.0f}%"
                        f"  ret={ret:+.1f}%  std={std:.1f}%")
                print(line)

            cidx = input("\nSelect combo number (1 = best): ").strip()
            if not cidx.isdigit() or not (1 <= int(cidx) <= len(summary)):
                print("Invalid selection.")
                return None

            chosen_entry = summary[int(cidx) - 1]
            cid = chosen_entry["combo_id"]

            # The run folder was moved inside grid_runs/grid_NNN/combo_NNN/
            # live_runner expects a path under strategy/runs/ — copy it there first
            combo_folder = os.path.join(grid_folder, f"combo_{cid}")

            # Find results.json inside combo folder
            results_path = None
            for root_dir, dirs, files in os.walk(combo_folder):
                for fname in files:
                    if fname == "results.json":
                        results_path = os.path.join(root_dir, fname)
                        break

            if results_path is None:
                print(f"results.json not found in {combo_folder}.")
                return None

            # Link or copy into strategy/runs so LiveStrategyRunner can find it
            runs_dir = "strategy/runs"
            os.makedirs(runs_dir, exist_ok=True)
            link_name = f"{chosen_grid}_combo_{cid}"
            link_path = os.path.join(runs_dir, link_name)

            if not os.path.exists(link_path):
                run_subdir = os.path.dirname(results_path)
                import shutil
                shutil.copytree(run_subdir, link_path)
                print(f"  Copied run to {link_path}")

            return link_name

        else:
            print("Invalid choice.")
            return None

    def _list_grid_runs(self, grid_base, grid_runs):
        """Print a compact overview of grid runs, then let user drill into one."""

        for i, g in enumerate(grid_runs, 1):
            summary_path = os.path.join(grid_base, g, "grid_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    summary = json.load(f)
                n = len(summary)
                best = summary[0] if summary else None
                if best:
                    bp = best["params"]
                    ret  = best["mean_final"] / bp.get("initial_bankroll", 1) * 100 - 100
                    std  = best["std_final"]  / bp.get("initial_bankroll", 1) * 100
                    tag  = (f"  {n} combos"
                            f"  best: hold={bp.get('hold_days','?')}d"
                            f"  target={100*bp.get('target',0):.0f}%"
                            f"  stop={100*bp.get('stop',0):.0f}%"
                            f"  ret={ret:+.1f}%  std={std:.1f}%")
                else:
                    tag = f"  {n} combos"
            else:
                tag = "  (no summary)"
            print(f"  {i}. {g}{tag}")

        choice = input("\n  Select grid to inspect (or Enter to skip): ").strip()
        if not choice.isdigit() or not (1 <= int(choice) <= len(grid_runs)):
            return

        chosen = grid_runs[int(choice) - 1]
        summary_path = os.path.join(grid_base, chosen, "grid_summary.json")

        if not os.path.exists(summary_path):
            print("  No summary found.")
            return

        with open(summary_path) as f:
            summary = json.load(f)

        print(f"\n=== {chosen} — All Combinations (best first) ===")

        # Build a compact param diff: show only keys that vary across combos
        all_params = [e["params"] for e in summary]
        varying = [k for k in all_params[0]
                   if k not in ("tickers",)
                   and len({str(p.get(k)) for p in all_params}) > 1]
        fixed   = {k: all_params[0][k] for k in all_params[0]
                   if k not in ("tickers",) and k not in varying}

        # Print fixed params once
        print("  Fixed params:")
        for k, v in fixed.items():
            if k in ("start_year","end_year","n_tests","n_cycles",
                     "hold_days","mc_samples","initial_bankroll",
                     "invest_frac","n_hold","rule","risk_normalized"):
                print(f"    {k} = {v}")

        # Column header
        vary_header = "  ".join(f"{k:>12}" for k in varying)
        print(f"\n  {'#':>4}  {'combo':>10}  {'ret%':>6}  {'std%':>6}  {'sharpe':>7}  {vary_header}")
        print("  " + "-" * (44 + 14 * len(varying)))

        for rank, entry in enumerate(summary, 1):
            p    = entry["params"]
            init = p.get("initial_bankroll", 1)
            ret  = entry["mean_final"] / init * 100 - 100
            std  = entry["std_final"]  / init * 100
            sh   = ret / std if std > 0 else 0
            vary_vals = "  ".join(f"{p.get(k,'?'):>12}" for k in varying)
            print(f"  {rank:>4}  combo_{entry['combo_id']:>6}  "
                  f"{ret:>+6.1f}  {std:>6.1f}  {sh:>7.2f}  {vary_vals}")

    def list_strategy_runs(self):

        print("\n=== Available Strategy Runs ===")

        runs_dir = os.path.join("strategy", "runs")
        grid_base = os.path.join("strategy", "grid_runs")

        # ── Collect single runs ──────────────────────────────
        single_runs = []
        if os.path.isdir(runs_dir):
            single_runs = sorted([
                r for r in os.listdir(runs_dir)
                if os.path.isdir(os.path.join(runs_dir, r))
            ])

        # ── Collect grid runs ────────────────────────────────
        grid_runs = []
        if os.path.isdir(grid_base):
            grid_runs = sorted([
                g for g in os.listdir(grid_base)
                if os.path.isdir(os.path.join(grid_base, g))
            ])

        if not single_runs and not grid_runs:
            print("  (none)")
            return

        # ── Show mode selector if both exist ─────────────────
        has_both = bool(single_runs) and bool(grid_runs)
        if has_both:
            print("  1. Single runs")
            print("  2. Grid runs")
            mode = input("  Show: ").strip()
        elif single_runs:
            mode = "1"
        else:
            mode = "2"

        if mode == "2":
            self._list_grid_runs(grid_base, grid_runs)
            return

        run_names = single_runs

        for run in run_names:

            result_path = os.path.join(runs_dir, run, "results.json")

            with open(result_path, "r") as f:
                data = json.load(f)

            params = data["params"]
            summary = data["summary"]
            prediction = data["prediction"]["overall"]

            print(f"\n  {run}")
            print("  " + "-" * len(run))

            # =============================
            # INPUT PARAMETERS
            # =============================

            _spec = params.get('date_spec') or f"{params.get('start_year','?')}-{params.get('end_year','?')}"
            print("Period:", _spec)

            print("Tests:",
                params["n_tests"],
                "| Cycles:",
                params["n_cycles"])

            mc_ppd = params.get("mc_paths_per_day",
                params["mc_samples"] // max(params["hold_days"], 1))
            print("Hold days:",
                params["hold_days"],
                "| MC paths/day:",
                mc_ppd,
                "| MC samples:",
                params["mc_samples"])

            print("Target/Stop:",
                params["target"],
                "/",
                params["stop"])

            print("Invest fraction:",
                params["invest_frac"])
            
            print("Risk normalized:",
                params["risk_normalized"])

            print("Fees:",
                f"{params['fee_fixed']} + {params['fee_pct']*100:.2f}%")

            print("Rule:",
                params["rule"])

            print("Initial bankroll:",
                params["initial_bankroll"])

            # =============================
            # PERFORMANCE RESULTS
            # =============================

            print("\nResults:")

            mean_final = summary["mean_final"]
            std_final = summary["std_final"]

            print(f"  Mean final bankroll: {mean_final:.2f}")
            print(f"  Std final bankroll:  {std_final:.2f}")

            cycle_means = summary["cycle_means"]
            cycle_stds = summary["cycle_stds"]

            print(f"  Avg cycle return:    {np.mean(cycle_means)*100:.3f}%")
            print(f"  Avg cycle volatility:{np.mean(cycle_stds)*100:.3f}%")

            # =============================
            # PREDICTION QUALITY
            # =============================

            print("\nPrediction Quality:")

            print(f"  Mean predicted return: {100*prediction['mean_pred']:.3f}%")
            print(f"  Mean realized return:  {100*prediction['mean_real']:.3f}%")
            print(f"  Mean prediction error: {100*prediction['mean_error']:.3f}%")
            print(f"  Error sigma:           {100*prediction['std_error']:.3f}%")
            print(f"  Mean prediction score: {prediction['mean_score']:.3f}")

            print()

    def analyze_strategy_runs(self):

        runs_dir = os.path.join("strategy", "runs")

        run_names = sorted([
            r for r in os.listdir(runs_dir)
            if os.path.isdir(os.path.join(runs_dir, r))
        ])

        if not run_names:
            print("  (none)")
            return

        analysis = []

        for run in run_names:

            with open(os.path.join(runs_dir, run, "results.json")) as f:
                data = json.load(f)

            params = data["params"]
            summary = data["summary"]
            prediction = data["prediction"]["overall"]

            initial = params["initial_bankroll"]
            mean_final = summary["mean_final"]
            std_final = summary["std_final"]

            mean_return = mean_final / initial - 1
            sharpe_like = (mean_final - initial) / std_final if std_final > 0 else 0

            bias = prediction["mean_error"]
            error_sigma = prediction["std_error"]
            score = prediction["mean_score"]

            mean_total_fees = summary["mean_total_fees"]
            mean_cycle_fee = summary["mean_cycle_fee"]

            trades = data["prediction"]["trades"]
            n_cycles_total = params["n_cycles"] * params["n_tests"]
            trades_per_cycle = len(trades) / n_cycles_total if n_cycles_total > 0 else 0

            analysis.append({
                "run": run,
                "return": mean_return,
                "sharpe": sharpe_like,
                "bias": bias,
                "error_sigma": error_sigma,
                "score": score,
                "fees_test": mean_total_fees,
                "fees_cycle": mean_cycle_fee,
                "trades_cycle": trades_per_cycle
            })

        # -------------------------------------------------
        # Metric Selector
        # -------------------------------------------------

        print("\nSelect ranking metric:")
        print("1. Sharpe-like")
        print("2. Mean return")
        print("3. Lowest bias (abs)")
        print("4. Lowest error sigma")
        print("5. Highest prediction score")
        print("6. Composite score")

        choice = input("Select option [1]: ").strip()
        if choice == "":
            choice = "1"

        # -------------------------------------------------
        # Composite Score (static weights)
        # -------------------------------------------------

        if choice == "6":

            # Normalization helper
            def normalize(vals):
                vmin, vmax = min(vals), max(vals)
                if vmax == vmin:
                    return [0.5] * len(vals)
                return [(v - vmin) / (vmax - vmin) for v in vals]

            returns = normalize([a["return"] for a in analysis])
            sharpes = normalize([a["sharpe"] for a in analysis])
            scores = normalize([a["score"] for a in analysis])
            biases = normalize([abs(a["bias"]) for a in analysis])
            errors = normalize([a["error_sigma"] for a in analysis])
            fees = normalize([a["fees_cycle"] for a in analysis])
            trades = normalize([a["trades_cycle"] for a in analysis])

            # Static weights
            w_return = 0.30
            w_sharpe = 0.25
            w_score = 0.15
            w_bias = 0.10
            w_error = 0.10
            w_fee = 0.05
            w_trade = 0.05

            for i, a in enumerate(analysis):
                composite = (
                    w_return * returns[i] +
                    w_sharpe * sharpes[i] +
                    w_score * scores[i] -
                    w_bias * biases[i] -
                    w_error * errors[i] -
                    w_fee * fees[i] -
                    w_trade * trades[i]
                )
                a["composite"] = composite

            analysis.sort(key=lambda x: x["composite"], reverse=True)

        elif choice == "2":
            analysis.sort(key=lambda x: x["return"], reverse=True)

        elif choice == "3":
            analysis.sort(key=lambda x: abs(x["bias"]))

        elif choice == "4":
            analysis.sort(key=lambda x: x["error_sigma"])

        elif choice == "5":
            analysis.sort(key=lambda x: x["score"], reverse=True)

        else:  # default Sharpe
            analysis.sort(key=lambda x: x["sharpe"], reverse=True)

        # -------------------------------------------------
        # Print Table
        # -------------------------------------------------

        print("\n=== Strategy Run Comparison ===")

        header = (
            f"{'Run':<10}"
            f"{'Ret%':>8}"
            f"{'Sharpe':>8}"
            f"{'Bias%':>8}"
            f"{'Errσ%':>8}"
            f"{'Score':>8}"
            f"{'Fee/cyc':>10}"
            f"{'Trades':>8}"
        )

        print(header)
        print("-" * len(header))

        for a in analysis:
            print(
                f"{a['run']:<10}"
                f"{100*a['return']:>7.2f}"
                f"{a['sharpe']:>8.2f}"
                f"{100*a['bias']:>8.2f}"
                f"{100*a['error_sigma']:>8.2f}"
                f"{a['score']:>8.2f}"
                f"{a['fees_cycle']:>10.2f}"
                f"{a['trades_cycle']:>8.2f}"
            )

        print("\nBest run:", analysis[0]["run"])

        # -------------------------------------------------
        # Plotting Section
        # -------------------------------------------------

        runs = [a["run"] for a in analysis]

        returns = [a["return"] for a in analysis]
        sharpes = [a["sharpe"] for a in analysis]
        biases = [a["bias"] for a in analysis]
        errors = [a["error_sigma"] for a in analysis]
        fees = [a["fees_cycle"] for a in analysis]
        trades = [a["trades_cycle"] for a in analysis]

        # 1. Return vs Sharpe
        plt.figure()
        plt.scatter(sharpes, returns)
        for i, txt in enumerate(runs):
            plt.annotate(txt, (sharpes[i], returns[i]))
        plt.xlabel("Sharpe-like")
        plt.ylabel("Mean Return")
        plt.title("Return vs Sharpe")

        # 2. Bias vs Error
        plt.figure()
        plt.scatter(errors, biases)
        for i, txt in enumerate(runs):
            plt.annotate(txt, (errors[i], biases[i]))
        plt.xlabel("Error Sigma")
        plt.ylabel("Prediction Bias")
        plt.title("Bias vs Error")

        # 3. Fees vs Trades
        plt.figure()
        plt.scatter(trades, fees)
        for i, txt in enumerate(runs):
            plt.annotate(txt, (trades[i], fees[i]))
        plt.xlabel("Trades per Cycle")
        plt.ylabel("Fee per Cycle")
        plt.title("Fees vs Trades")

        # 4. Return vs Trades
        plt.figure()
        plt.scatter(trades, returns)
        for i, txt in enumerate(runs):
            plt.annotate(txt, (trades[i], returns[i]))
        plt.xlabel("Trades per Cycle")
        plt.ylabel("Mean Return")
        plt.title("Return vs Trading Activity")
        plt.show()