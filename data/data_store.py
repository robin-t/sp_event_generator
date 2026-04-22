import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path


class DataStore:
    """
    Handles downloading, caching, and updating OHLC stock data.

    Canonical conventions (applied everywhere):
      - dates stored as datetime64[D]  (day precision, no timezone)
      - prices from yfinance auto_adjust=True, "Close" column
        (split- and dividend-adjusted; consistent for long backtests)
      - cache files written with np.savez_compressed
    """

    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # INTERNAL HELPERS
    # ==========================================================

    def _cache_path(self, ticker):
        return self.cache_dir / f"{ticker}.npz"

    # ----------------------------------------------------------

    def _download_single(self, ticker, start=None):
        """
        Download one ticker from yfinance and return
        (dates[datetime64[D]], close, high, low).
        """
        kwargs = dict(progress=False, auto_adjust=True)

        if start is None:
            kwargs["period"] = "max"
        else:
            kwargs["start"] = start

        df = yf.download(ticker, **kwargs)

        if df.empty:
            raise RuntimeError(f"No data returned for {ticker}")

        # yfinance sometimes returns a MultiIndex (ticker, field) — flatten it
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        dates = df.index.values.astype("datetime64[D]")
        close = df["Close"].values.astype(float).flatten()
        high  = df["High"].values.astype(float).flatten()
        low   = df["Low"].values.astype(float).flatten()

        return dates, close, high, low

    # ----------------------------------------------------------

    def _save(self, ticker, dates, close, high, low):
        """Save to cache. dates must be datetime64[D]."""
        np.savez_compressed(
            self._cache_path(ticker),
            dates=dates,
            close=close,
            high=high,
            low=low,
        )

    # ----------------------------------------------------------

    def _load(self, ticker):
        """Load from cache. Returns (dates[datetime64[D]], close, high, low)."""
        data = np.load(self._cache_path(ticker))

        dates = data["dates"].astype("datetime64[D]")
        close = data["close"].astype(float)
        high  = data["high"].astype(float)
        low   = data["low"].astype(float)

        return dates, close, high, low

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def download_full(self, tickers):
        """
        Ensure all tickers are cached, batch-downloading any that are missing.
        Returns dict: ticker -> (dates[datetime64[D]], close, high, low)
        """
        to_download = [t for t in tickers if not self._cache_path(t).exists()]

        if to_download:
            print(f"Downloading {len(to_download)} missing ticker(s)...")

            failed = []

            for ticker in to_download:
                try:
                    dates, close, high, low = self._download_single(ticker)
                    self._save(ticker, dates, close, high, low)
                    print(f"  {ticker}: {len(dates)} days downloaded")
                except Exception as e:
                    print(f"  {ticker}: download failed — {e}")
                    failed.append(ticker)

            if failed:
                print(f"\n  Warning: {len(failed)} ticker(s) could not be downloaded: {failed}")

        # Load everything from cache
        data = {}
        for ticker in tickers:
            if self._cache_path(ticker).exists():
                data[ticker] = self._load(ticker)
            else:
                print(f"  Warning: cache missing for {ticker}, skipping.")

        return data

    # ==========================================================
    # AUTO UPDATE LOGIC
    # ==========================================================

    def ensure_up_to_date(self, tickers):
        """
        Incrementally update cached data for all tickers up to today.
        """
        print("\nChecking for data updates...")

        for ticker in tickers:

            path = self._cache_path(ticker)

            # No cache → full download
            if not path.exists():
                print(f"  {ticker}: no cache, downloading full history...")
                dates, close, high, low = self._download_single(ticker)
                self._save(ticker, dates, close, high, low)
                continue

            dates, close, high, low = self._load(ticker)

            last_date = dates[-1]  # datetime64[D]
            print(f"  {ticker}: cached to {last_date}, fetching updates...")

            new_dates, new_close, new_high, new_low = self._download_single(
                ticker,
                start=str(last_date),  # datetime64[D] formats as "YYYY-MM-DD"
            )

            # Keep only days strictly after the last cached date
            mask = new_dates > last_date

            if np.any(mask):
                dates = np.concatenate([dates, new_dates[mask]])
                close = np.concatenate([close, new_close[mask]])
                high  = np.concatenate([high,  new_high[mask]])
                low   = np.concatenate([low,   new_low[mask]])

                self._save(ticker, dates, close, high, low)
                print(f"    → added {mask.sum()} new day(s)")
            else:
                print(f"    → already up to date")
