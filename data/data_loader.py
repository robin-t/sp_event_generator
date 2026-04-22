import numpy as np


class DataLoader:

    @staticmethod
    def align(raw_data):
        """
        Align OHLC data using UNION of dates.
        raw_data format:
            {ticker: (dates, close, high, low)}
        """

        # --------------------------------------------------
        # Build union of all dates
        # --------------------------------------------------

        all_dates = set()

        for dates, close, high, low in raw_data.values():
            all_dates.update(dates)

        master_dates = np.array(sorted(all_dates))

        n_tickers = len(raw_data)
        n_days = len(master_dates)

        #print(f"Aligned data shape: ({n_tickers}, {n_days})")

        # --------------------------------------------------
        # Initialize matrices
        # --------------------------------------------------

        close_matrix = np.full((n_tickers, n_days), np.nan)
        high_matrix = np.full((n_tickers, n_days), np.nan)
        low_matrix = np.full((n_tickers, n_days), np.nan)

        tickers = []

        # --------------------------------------------------
        # Fill matrices
        # --------------------------------------------------

        for i, (ticker, data) in enumerate(raw_data.items()):

            dates, close, high, low = data

            tickers.append(ticker)

            # Map to master index
            idx = np.searchsorted(master_dates, dates)

            close_matrix[i, idx] = close
            high_matrix[i, idx] = high
            low_matrix[i, idx] = low

        return master_dates, close_matrix, high_matrix, low_matrix, tickers
