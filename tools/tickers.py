from pathlib import Path
import csv


class Tickers:
    """
    Registry of predefined ticker lists.

    Static snapshot-based lists are loaded from:
        data/ticker_lists/
    """

    def __init__(self):

        # Built-in small samples (keep for testing)
        self._lists = {
            "sp500_sample": [
                "AAPL", "MSFT", "AMZN", "GOOGL", "META",
                "NVDA", "BRK-B", "JPM", "UNH", "V",
                "PG", "HD", "MA", "XOM", "LLY",
                "PFE", "KO", "PEP", "COST", "MRK",
            ],

            "top_nyse_100_sample": [
                "GE", "IBM", "BA", "CAT", "GS",
                "DIS", "MMM", "AXP", "RTX", "DOW",
                "CVX", "WMT", "JNJ", "T", "F",
            ],

            "rare_earths": [
                "UAMY", "AREC", "MP", "TMRC",
                "REEMF", "LYSCF"
            ],
        }

        # Load static snapshot lists
        self._load_static_list("sp500_full")
        self._load_static_list("russell2000_250")
        self._load_static_list("sweden_largecap")

    # --------------------------------------------------

    def _load_static_list(self, name):
        """
        Load ticker list from data/ticker_lists/<name>.csv
        """

        base_path = Path("tools/data/ticker_lists")
        file_path = base_path / f"{name}.csv"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Ticker list file not found: {file_path}\n"
                f"Create static snapshot CSV before using '{name}'."
            )

        tickers = []

        with open(file_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)

            if "ticker" not in reader.fieldnames:
                raise ValueError(
                    f"{file_path} must contain a 'ticker' column."
                )

            for row in reader:
                t = row["ticker"].strip()

                if not t:
                    continue

                # yfinance uses "-" instead of "." for US share-class
                # dots (e.g. BRK.B -> BRK-B), but exchange suffix dots
                # like .ST, .L, .DE, .HE must be preserved.
                # We use a whitelist of known exchange suffixes.
                EXCHANGE_SUFFIXES = {
                    "ST", "L", "DE", "PA", "MI", "AS", "BR",
                    "HE", "OL", "CO", "LS", "SW", "VI", "PR",
                    "MC", "AT", "IS", "TA", "TW", "HK", "AX",
                    "TO", "V", "SA", "MX", "BO",
                }
                parts = t.split(".")
                if len(parts) >= 2 and parts[-1].upper() in EXCHANGE_SUFFIXES:
                    # Keep exchange suffix dot, replace any remaining dots
                    base = ".".join(parts[:-1]).replace(".", "-")
                    t = base + "." + parts[-1]
                else:
                    t = t.replace(".", "-")

                tickers.append(t)

        if len(tickers) == 0:
            raise ValueError(f"{file_path} contains no tickers.")

        self._lists[name] = tickers

    # --------------------------------------------------

    def list_names(self):
        """Return available ticker list names."""
        return list(self._lists.keys())

    # --------------------------------------------------

    def get(self, name):
        """Return tickers for a given list."""
        if name not in self._lists:
            raise ValueError(f"Unknown ticker list: {name}")
        return self._lists[name]

    # --------------------------------------------------

    def __repr__(self):
        return f"Tickers(lists={self.list_names()})"