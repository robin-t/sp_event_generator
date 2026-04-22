from pathlib import Path
import json

from density.indicator import Indicator, GlobalIndicator
from density.density import DensitySet

from data.data_store import DataStore
from data.data_loader import DataLoader
from data.features import FeatureBuilder

from tools.mpi_utils import is_root, bcast
from tools.date_range import parse_date_mask, describe_mask, validate_spec, _dates_to_years


class DensityHandler:

    def __init__(self, tickers):
        self.tickers = tickers
        self.output_dir = Path("density/densities")

    # ==========================================================
    # MENU
    # ==========================================================

    def menu(self):

        while True:

            if is_root():
                print("\n=== Density Menu ===")
                print("1. Create density")
                print("2. List available densities")
                print("0. Back")

                choice = input("Choice: ").strip()
            else:
                choice = None

            choice = bcast(choice)

            if choice == "1":
                self.create_density()

            elif choice == "2":
                if is_root():
                    self.list_densities()

            elif choice == "0":
                return

    # ==========================================================
    # LIST DENSITIES
    # ==========================================================

    def list_densities(self):

        print("\n=== Available Densities ===")

        if not self.output_dir.exists():
            print("  (none)")
            return

        folders = sorted(self.output_dir.iterdir())
        found = [f for f in folders if (f / "meta.json").exists()]

        if not found:
            print("  (none)")
            return

        for idx, folder in enumerate(found, 1):
            with open(folder / "meta.json") as f:
                meta = json.load(f)

            indicator   = meta.get("indicator_name", "unknown")
            params      = meta.get("indicator_params", [])
            ticker_list = meta.get("ticker_list", "?")
            date_spec   = meta.get("date_spec")
            start_year  = meta.get("start_year", "?")
            end_year    = meta.get("end_year", "?")
            is_global   = meta.get("is_global", False)

            date_str = date_spec if date_spec else f"{start_year}\u2013{end_year}"
            param_str = ", ".join(str(p) for p in params)
            tag       = "GLOBAL" if is_global else "per-ticker"
            p_part    = f"  params=[{param_str}]" if param_str else ""

            print(f"  {idx}. {indicator.upper()}{p_part}  [{tag}]  {ticker_list}  {date_str}")


    # ==========================================================
    # CREATE DENSITY
    # ==========================================================

    def create_density(self):

        if not is_root():
            return

        print("\n=== Create Density ===")


        # ------------------------------------------------------
        # Select ticker universe
        # ------------------------------------------------------

        lists = self.tickers.list_names()

        print("\nAvailable ticker lists:")
        for i, name in enumerate(lists):
            print(f"{i+1}. {name}")

        idx = int(input("Choose list: ")) - 1
        ticker_list_name = lists[idx]
        tickers = self.tickers.get(ticker_list_name)

        # ------------------------------------------------------
        # Select date range
        # ------------------------------------------------------

        print("\nDate range spec (e.g. '1990-2007, 2010-2019, 2021'):")
        while True:
            date_spec = input("Date range: ").strip()
            if validate_spec(date_spec):
                break
            print("  Invalid format. Use YYYY-YYYY ranges and/or single YYYY years, comma-separated.")


        # ------------------------------------------------------
        # Select indicator type
        # ------------------------------------------------------

        print("\nIndicator type:")
        print("1. Per-ticker indicator (uses single stock history)")
        print("2. Global indicator (uses cross-sectional market state)")
        ind_type = input("Choice: ").strip()
        is_global = (ind_type == "2")

        if is_global:
            available = GlobalIndicator.available()
            get_desc  = GlobalIndicator.get_description
        else:
            available = Indicator.available()
            get_desc  = Indicator.get_description

        print("\nAvailable indicators:")
        for i, name in enumerate(available):
            print(f"{i+1}. {name}")

        idx = int(input("Choose indicator: ")) - 1
        indicator_name = available[idx]

        print("\nParameter description:")
        print(get_desc(indicator_name))

        params_raw = input("Enter parameters (comma separated, or Enter if none): ").strip()
        params = [float(p) for p in params_raw.split(",")] if params_raw else []

        # Indicator is constructed after data is loaded (GlobalIndicator needs matrices)

        # ------------------------------------------------------
        # Load and update data
        # ------------------------------------------------------

        datastore = DataStore()

        raw_data = datastore.download_full(tickers)

        dates, close, high, low, tickers = DataLoader.align(raw_data)

        # --------------------------------------
        # Dataset summary
        # --------------------------------------

        # dates are already datetime64[D] from DataStore
        full_start = dates[0]
        full_end   = dates[-1]

        print("\nDataset summary:")
        print(f"  Total days:  {len(dates)}")
        print(f"  Full range:  {str(full_start)} → {str(full_end)}")

        # ------------------------------------------------------
        # Restrict by year range
        # ------------------------------------------------------

        mask = parse_date_mask(date_spec, dates)

        close = close[:, mask]
        high = high[:, mask]
        low = low[:, mask]

        filtered_dates = dates[mask]

        print("\nDate filter applied:")
        print(describe_mask(date_spec, mask, dates))

        if close.shape[1] == 0:
            raise RuntimeError(
                "No data left after year filtering. "
                "Check year range or data availability."
            )

        # Construct indicator now that filtered matrices are available.
        # GlobalIndicator needs the full matrix to compute cross-sectional stats.
        if is_global:
            indicator = GlobalIndicator(indicator_name, params, close, high, low,
                                        dates=filtered_dates)
        else:
            indicator = Indicator(indicator_name, params)

        # ------------------------------------------------------
        # Build RMS features
        # ------------------------------------------------------

        x_vals, R_vals, M_vals, S_vals = FeatureBuilder.build(
            close, high, low, indicator, dates=filtered_dates
        )

        # ------------------------------------------------------
        # Build densities
        # ------------------------------------------------------

        density = DensitySet(indicator)

        # Extract first/last year from spec for display/matching
        _years_in_mask = _dates_to_years(filtered_dates)
        _start_year = int(_years_in_mask[0]) if len(_years_in_mask) > 0 else None
        _end_year   = int(_years_in_mask[-1]) if len(_years_in_mask) > 0 else None

        density.meta.update({
            "ticker_list": ticker_list_name,
            "date_spec":   date_spec,
            "start_year":  _start_year,
            "end_year":    _end_year,
        })

        density.build(x_vals, R_vals, M_vals, S_vals)

        # ------------------------------------------------------
        # Save results
        # ------------------------------------------------------

        param_str = "_".join(str(p) for p in params)

        # Sanitize spec for use in folder name
        _spec_tag = date_spec.replace(", ", "_").replace(",", "_").replace("-", "to").replace(" ", "")
        name = (
            f"{indicator_name}_{param_str}_"
            f"{ticker_list_name}_{_spec_tag}"
        )

        folder = self.output_dir / name

        density.save(folder)
        density.analyze(folder)
        density.summarize(folder)

        print("\nDensity saved to:", folder)
