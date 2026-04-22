import numpy as np


class FeatureBuilder:
    """
    Builds RMS features:
        R = range
        M = midpoint drift
        S = skew within range

    relative to previous close.
    """

    @staticmethod
    def build(close_matrix, high_matrix, low_matrix, indicator, dates=None):
        """
        Compute flattened arrays:

            x_values
            R_values
            M_values
            S_values

        dates : optional array of date objects [n_days], passed to indicators
                that require date information (e.g. day_of_week).
        """

        n_tickers, n_days = close_matrix.shape

        x_list = []
        R_list = []
        M_list = []
        S_list = []

        print("Building RMS features...")

        for i in range(n_tickers):

            close = close_matrix[i]
            high = high_matrix[i]
            low = low_matrix[i]

            # --------------------------------------------------
            # Indicator values
            # --------------------------------------------------

            x_ind = indicator.compute(high, low, close, dates=dates)

            # --------------------------------------------------
            # Compute returns relative to previous close
            # --------------------------------------------------

            prev_close = close[:-1]

            rH = (high[1:] - prev_close) / prev_close
            rL = (low[1:] - prev_close) / prev_close
            rC = (close[1:] - prev_close) / prev_close

            # --------------------------------------------------
            # Compute RMS variables
            # --------------------------------------------------

            R = rH - rL
            M = (rH + rL) / 2

            # Avoid division by zero
            valid_range = R > 0

            S = np.zeros_like(R)
            S[valid_range] = (rC[valid_range] - M[valid_range]) / (R[valid_range] / 2)

            # --------------------------------------------------
            # Build valid mask
            # --------------------------------------------------

            mask = (
                valid_range &
                np.isfinite(R) &
                np.isfinite(M) &
                np.isfinite(S) &
                (np.abs(S) <= 1.05)
            )

            # Indicator must also be finite
            mask &= np.isfinite(x_ind[1:])

            # --------------------------------------------------
            # Append filtered samples
            # --------------------------------------------------

            x_list.append(x_ind[1:][mask])
            R_list.append(R[mask])
            M_list.append(M[mask])
            S_list.append(S[mask])

        # ------------------------------------------------------
        # Flatten across tickers
        # ------------------------------------------------------

        x_values = np.concatenate(x_list)
        R_values = np.concatenate(R_list)
        M_values = np.concatenate(M_list)
        S_values = np.concatenate(S_list)

        print("Total RMS samples:", len(x_values))

        return x_values, R_values, M_values, S_values
