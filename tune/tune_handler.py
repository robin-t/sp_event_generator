from pathlib import Path
import json
import numpy as np
import itertools

from density.density import DensitySet
from density.indicator import Indicator
from tune.tuner import Tuner

from data.data_store import DataStore
from data.data_loader import DataLoader
from tools.tickers import Tickers

from tools.mpi_utils import COMM, RANK, SIZE, is_root, bcast
from tools.date_range import validate_spec


class TuneHandler:

    def __init__(self):

        self.density_folder = Path("density/densities")
        self.tune_folder = Path("tune/rms_tunes")

        self.tickers = Tickers()
        self.datastore = DataStore()

    # ==========================================================
    # MENU
    # ==========================================================

    def menu(self):

        while True:

            if is_root():
                print("\n=== Tuning Menu ===")
                print("1. Run Calibration")
                print("2. List available tunes")
                print("0. Back")

                choice = input("Choice: ").strip()
            else:
                choice = None

            choice = bcast(choice)

            if choice == "1":
                self.run_tuning()

            elif choice == "2":
                if is_root():
                    self._list_tunes()

            elif choice == "0":
                break

    # ==========================================================
    # LIST AVAILABLE TUNES (DETAILED)
    # ==========================================================

    def _list_tunes(self):

        folders = sorted(self.tune_folder.glob("*"))

        entries = []

        for folder in folders:
            meta_file = folder / "best.json"
            if not meta_file.exists():
                continue

            with open(meta_file) as f:
                meta = json.load(f)

            entries.append((folder, meta))

        print("\n=== Available Tunes ===")

        if not entries:
            print("  (none)")
            return

        for i, (folder, meta) in enumerate(entries):

            # --------------------------------------------------
            # Basic info
            # --------------------------------------------------

            spec = meta.get("tuning_date_spec") or meta.get("tuning_year_range")
            if spec is None or (isinstance(spec, list) and spec[0] is None):
                year_str = "Full range"
            elif isinstance(spec, list):
                year_str = f"{spec[0]}–{spec[1]}"
            else:
                year_str = spec

            jk = meta.get("jackknife", {})
            mean = jk.get("mean_score", None)
            err = jk.get("jackknife_error", None)

            if mean is not None:
                score_str = f"{mean:.2f} ± {err:.2f}"
            else:
                score_str = "N/A"

            n_runs = len(jk.get("runs", []))

            # best weights
            best_weights = meta.get("best_weights", {})

            n_mc = meta.get("n_mc", "?")
            n_samples = meta.get("n_samples", "?")
            grid_size = meta.get("grid_size", "?")

            sm = meta.get("sampling_mode", "weighted_mean")
            print(
                f"  {i+1}. Tune #{folder.name.split('_')[-1]} | "
                f"{year_str} | "
                f"Nmc={n_mc} | Samples={n_samples} | "
                f"Grid={grid_size} | "
                f"Mode={sm} | "
                f"Score: {score_str}"
            )

            # --------------------------------------------------
            # Helper to print one variable
            # --------------------------------------------------

            def print_component(label, dens_list):

                items = []

                # Collect weights
                weights = []
                for d in dens_list:
                    key = f"{d['indicator']}_{'_'.join(str(p) for p in d['params'])}"
                    w = best_weights.get(key, 0.0)
                    weights.append((d, key, w))

                total = sum(w for _, _, w in weights)
                if total == 0:
                    total = 1.0

                # Normalize within component
                for d, key, w in weights:
                    w_norm = w / total

                    param_str = ",".join(str(p) for p in d["params"])
                    name = f"{d['indicator'].upper()}[{param_str}]"

                    items.append(f"{name} ({w_norm:.2f})")

                print(f"   {label}: " + ", ".join(items))

            # --------------------------------------------------
            # Print R / M / S
            # --------------------------------------------------

            print_component("R", meta.get("R_densities", []))
            print_component("M", meta.get("M_densities", []))
            print_component("S", meta.get("S_densities", []))

            print()
        
        return entries

    # ==========================================================
    # LIST AVAILABLE DENSITIES
    # ==========================================================

    def _list_densities(self):

        folders = sorted(self.density_folder.glob("*"))

        entries = []

        for folder in folders:

            meta_file = folder / "meta.json"
            if not meta_file.exists():
                continue

            with open(meta_file) as f:
                meta = json.load(f)

            entries.append((folder, meta))

        print("\n=== Available Densities ===")

        for i, (_, meta) in enumerate(entries):

            name = meta["indicator_name"]
            params = meta["indicator_params"]
            dataset = meta.get("ticker_list", "?")
            date_spec = meta.get("date_spec")
            date_str = date_spec if date_spec else f"{meta.get('start_year','?')}–{meta.get('end_year','?')}"

            param_str2 = ", ".join(str(p) for p in params)
            p_part = f"  params=[{param_str2}]" if param_str2 else ""
            print(f"  {i+1}. {name.upper()}{p_part}  {dataset}  {date_str}")

        return entries

    # ==========================================================
    # SELECT DENSITIES FOR ONE COMPONENT
    # ==========================================================

    def _select_component(self, label):

        entries = self._list_densities()

        indices = input(f"\nSelect densities for {label} (comma separated): ")
        indices = [int(i.strip()) - 1 for i in indices.split(",")]

        selected = []

        for idx in indices:

            folder, meta = entries[idx]

            density = DensitySet.load(folder)

            indicator = density.indicator

            if density.x_bins is None or density.x_bins.size == 0:
                raise RuntimeError(f"Density at {folder} failed to load bins.")

            param_str = "_".join(str(p) for p in indicator.params)
            key = f"{indicator.name}_{param_str}"
            selected.append((density, indicator, key))


        return selected

    # ==========================================================
    # BUILD WEIGHT GRID
    # ==========================================================

    def _build_weight_grid(self, all_density_names):

        print("\nDefine weight grid:")

        # --------------------------------------------------
        # Allow user to fix one or more weights
        # --------------------------------------------------

        print("\nIndicators to optimise:")
        for i, name in enumerate(all_density_names):
            print(f"  {i+1}. {name}")

        print("\nFix any weights at a constant value?")
        print("Enter comma-separated numbers to fix (e.g. 1,3), or blank to optimise all:")

        fixed_input = input("Fix: ").strip()

        fixed_weights = {}

        if fixed_input:
            fix_indices = [int(x.strip()) - 1 for x in fixed_input.split(",")]

            for idx in fix_indices:
                name = all_density_names[idx]
                val = float(input(f"  Fixed value for {name}: "))
                fixed_weights[name] = val
                print(f"  -> {name} fixed at {val}")

        free_names = [n for n in all_density_names if n not in fixed_weights]

        if free_names:
            print(f"\nOptimising {len(free_names)} free weight(s):")
            for name in free_names:
                print(f"  {name}")

            w_min  = float(input("Min weight: "))
            w_max  = float(input("Max weight: "))
            steps  = int(input("Number of steps: "))
            values = np.linspace(w_min, w_max, steps)

            combos = list(itertools.product(values, repeat=len(free_names)))
        else:
            # Everything is fixed — single combo
            combos = [()]

        grid = []

        for combo in combos:
            weights = dict(zip(free_names, combo))
            weights.update(fixed_weights)   # inject fixed values
            grid.append(weights)

        print(f"\nFixed weights:  {len(fixed_weights)}")
        print(f"Free weights:   {len(free_names)}")
        print(f"Total combinations: {len(grid)}")
        return grid

    # ==========================================================
    # MAIN TUNING
    # ==========================================================

    def run_tuning(self):

        if is_root():
            print("\n=== Run Calibration ===")

            # ------------------------------------------------------
            # Select densities
            # ------------------------------------------------------

            R_sel = self._select_component("R")
            M_sel = self._select_component("M")
            S_sel = self._select_component("S")

            # Unique keys
            all_names = list({
                name for _, _, name in (R_sel + M_sel + S_sel)
            })

            print("\nUnique indicators for weighting:", len(all_names))

            weight_grid = self._build_weight_grid(all_names)

            # MC parameters
            n_mc = int(input("MC paths per step: "))
            n_samples = int(input("Number of calibration samples: "))
            n_jackknife = int(input("Number of jackknife repetitions: "))

            # Sampling mode
            print("\nSampling mode:")
            print("  1. weighted_mean  (default — average over all densities)")
            print("  2. mixture        (randomly select one density per step)")
            sm_choice = input("Choice [1]: ").strip()
            sampling_mode = "mixture" if sm_choice == "2" else "weighted_mean"
            print(f"  -> sampling_mode = {sampling_mode}")

            # Score weights
            print("\nScore component weights:")
            print("  1. Equal (default) — R=1, M=1, S=1, C=1")
            print("  2. Strategy-relevant — R=2, M=2, S=0.5, C=0.5")
            print("  3. Custom — enter weights manually")
            sw_choice = input("Choice [1]: ").strip() or "1"
            if sw_choice == "2":
                score_weights = {"R": 2.0, "M": 2.0, "S": 0.5, "C": 0.5}
                print("  -> R=2.0 M=2.0 S=0.5 C=0.5")
            elif sw_choice == "3":
                score_weights = {}
                for comp in ("R", "M", "S", "C"):
                    val = input(f"  Weight for {comp} [1.0]: ").strip()
                    score_weights[comp] = float(val) if val else 1.0
                print(f"  -> {score_weights}")
            else:
                score_weights = {"R": 1.0, "M": 1.0, "S": 1.0, "C": 1.0}
                print("  -> Equal weights (default)")

            # Date range restriction
            print("\nRestrict tuning to date range:")
            print("  Format: '2021-2022' or '1990-2007, 2010-2019' (blank = no restriction)")
            while True:
                tune_date_spec = input("Date range spec: ").strip()
                if tune_date_spec == "" or validate_spec(tune_date_spec):
                    break
                print("  Invalid format. Use YYYY-YYYY ranges and/or single YYYY years.")
            tune_date_spec = tune_date_spec if tune_date_spec else None

            # Select tickers
            print("\nSelect ticker universe:")

            lists = self.tickers.list_names()
            for i, name in enumerate(lists):
                print(f"{i+1}. {name}")

            idx = int(input("Choice: ")) - 1
            tickers = self.tickers.get(lists[idx])

            print("\nChecking for data updates...")
            raw = self.datastore.download_full(tickers)

            dates, close, high, low, _ = DataLoader.align(raw)

            payload = {
                "R_sel": R_sel,
                "M_sel": M_sel,
                "S_sel": S_sel,
                "weight_grid": weight_grid,
                "n_mc": n_mc,
                "n_samples": n_samples,
                "n_jackknife": n_jackknife,
                "dates": dates,
                "close": close,
                "high": high,
                "low": low,
                "tune_date_spec": tune_date_spec,
                "sampling_mode": sampling_mode,
                "score_weights": score_weights,
            }

        else:
            payload = None

        payload = bcast(payload)

        # ------------------------------------------------------
        # Unpack on all ranks
        # ------------------------------------------------------

        R_sel = payload["R_sel"]
        M_sel = payload["M_sel"]
        S_sel = payload["S_sel"]
        weight_grid = payload["weight_grid"]
        n_mc = payload["n_mc"]
        n_samples = payload["n_samples"]
        n_jackknife = payload["n_jackknife"]
        dates = payload["dates"]
        close = payload["close"]
        high = payload["high"]
        low = payload["low"]
        tune_date_spec = payload.get("tune_date_spec", None)
        sampling_mode = payload.get("sampling_mode", "weighted_mean")
        score_weights = payload.get("score_weights", {"R": 1.0, "M": 1.0, "S": 1.0, "C": 1.0})

        # Prepare density lists
        R_dens = [(d, ind) for d, ind, _ in R_sel]
        M_dens = [(d, ind) for d, ind, _ in M_sel]
        S_dens = [(d, ind) for d, ind, _ in S_sel]

        tuner = Tuner(
            R_dens,
            M_dens,
            S_dens,
            weight_grid,
            n_mc=n_mc,
            dates=dates,
            tune_date_spec=tune_date_spec,
            sampling_mode=sampling_mode,
            score_weights=score_weights,
        )

        jackknife_results = tuner.run_jackknife(
            close, high, low,
            n_samples=n_samples,
            n_jackknife=n_jackknife,
            dates=dates,
        )

        # ------------------------------------------------------
        # Root saves results
        # ------------------------------------------------------

        if not is_root():
            return

        self.tune_folder.mkdir(parents=True, exist_ok=True)

        existing = [
            int(p.name.split("_")[-1])
            for p in self.tune_folder.glob("rms_tune_*")
            if p.name.split("_")[-1].isdigit()
        ]

        run_id = max(existing) + 1 if existing else 1
        folder = self.tune_folder / f"rms_tune_{run_id}"
        folder.mkdir()

        def extract_density_meta(dens_list):

            meta_list = []

            for density, indicator in dens_list:
                meta_list.append({
                    "indicator": indicator.name,
                    "params": indicator.params,
                    "density_meta": density.meta
                })

            return meta_list

        runcard = {
            "R_densities": extract_density_meta(R_dens),
            "M_densities": extract_density_meta(M_dens),
            "S_densities": extract_density_meta(S_dens),
            "tuning_date_spec": tune_date_spec,
            "n_mc": n_mc,
            "n_samples": n_samples,
            "grid_size": len(weight_grid),
            "jackknife": jackknife_results,
            "best_weights": jackknife_results["mean_weights"],
            "best_run_weights": jackknife_results["best_run"]["best_weights"],
            "component_scores": jackknife_results.get("component_scores", {}),
            "history_len": tuner.history_len,
            "sampling_mode": sampling_mode,
            "score_weights": score_weights,
        }

        with open(folder / "best.json", "w") as f:
            json.dump(runcard, f, indent=2)

        print("\nSaved runcard to best.json")
        print("\nCalibration complete.")

    # ----------------------------------------------------------
    # Select and load a tune (for strategy use)
    # ----------------------------------------------------------

    def select_tune(self):
        """
        Select a saved tune or manually enter custom weights.
        Returns a tune_config dict compatible with StrategyTester.
        """
        entries = self._list_tunes()

        if entries:
            print("  0. Enter custom weights manually")
            choice = input("\nSelect tune number (or 0 for custom): ").strip()
        else:
            print("  No saved tunes found.")
            choice = "0"

        if choice == "0":
            return self._build_custom_tune()

        if not choice.isdigit():
            print("Invalid selection.")
            return None

        idx = int(choice) - 1
        if idx < 0 or idx >= len(entries):
            print("Invalid selection.")
            return None

        folder, meta = entries[idx]
        with open(folder / "best.json") as f:
            tune_config = json.load(f)

        return tune_config

    def _build_custom_tune(self):
        """
        Interactively build a tune config from manually entered weights.
        Reuses the existing density selection and weight entry flow.
        """
        print("\n=== Custom Weights ===")
        print("Select densities and enter weights manually.\n")

        R_sel = self._select_component("R")
        M_sel = self._select_component("M")
        S_sel = self._select_component("S")

        all_names = list({name for _, _, name in (R_sel + M_sel + S_sel)})

        print("\nEnter weight for each indicator:")
        weights = {}
        for name in all_names:
            val = float(input(f"  {name}: ").strip())
            weights[name] = val

        def extract_meta(dens_list):
            return [{"indicator": ind.name, "params": ind.params,
                     "density_meta": d.meta}
                    for d, ind, _ in dens_list]

        R_dens = [(d, ind) for d, ind, _ in R_sel]
        M_dens = [(d, ind) for d, ind, _ in M_sel]
        S_dens = [(d, ind) for d, ind, _ in S_sel]

        # Infer history_len from any loaded density
        from tune.tuner import Tuner
        tuner = Tuner(R_dens, M_dens, S_dens, [weights])
        history_len = tuner.history_len

        tune_config = {
            "R_densities": extract_meta(R_sel),
            "M_densities": extract_meta(M_sel),
            "S_densities": extract_meta(S_sel),
            "best_weights": weights,
            "history_len": history_len,
            "tuning_year_range": [None, None],
            "n_mc": None,
            "n_samples": None,
            "grid_size": 1,
            "jackknife": {}
        }

        print("\nCustom tune config built:")
        for k, v in weights.items():
            print(f"  {k:<30} {v}")

        return tune_config


    