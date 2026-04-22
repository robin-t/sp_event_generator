import numpy as np


# ==============================================================
# INDICATOR FUNCTIONS + METADATA
# ==============================================================

INDICATOR_REGISTRY = {
    "macd": {
        "func": None,
        "n_params": 3,
        "description": (
            "MACD indicator parameters:\n"
            "1. Fast EMA window\n"
            "2. Slow EMA window\n"
            "3. Signal EMA window\n"
            "Example: 12,26,9"
        ),
    },
    "rsi": {
        "func": None,
        "n_params": 1,
        "description": (
            "RSI indicator parameters:\n"
            "1. Rolling window length\n"
            "Example: 14"
        ),
    },
    "rsi_velocity": {
        "func": None,
        "n_params": 2,
        "description": (
            "RSI Velocity indicator parameters:\n"
            "1. RSI window\n"
            "2. Velocity smoothing window\n"
            "Example: 14,5\n"
            "\n"
            "Computes smoothed derivative of RSI.\n"
            "Positive = RSI rising, negative = RSI falling."
        ),
    },
    "vol_ratio": {
        "func": None,
        "n_params": 2,
        "description": (
            "Volatility Ratio indicator parameters:\n"
            "1. Short volatility window\n"
            "2. Long volatility window\n"
            "Example: 10,50\n"
            "\n"
            "Computes short-term volatility divided by long-term volatility.\n"
            "Values >1 indicate volatility expansion,\n"
            "values <1 indicate volatility compression."
        ),
    },
    "trend_slope": {
        "func": None,
        "n_params": 1,
        "description": (
            "Trend Slope indicator parameters:\n"
            "1. Rolling window length\n"
            "Example: 20\n"
            "\n"
            "Computes normalized linear regression slope of log-price.\n"
            "Positive = upward trend, negative = downward trend."
        ),
    },
    "range_position": {
        "func": None,
        "n_params": 1,
        "description": (
            "Range Position indicator parameters:\n"
            "1. Rolling window length\n"
            "Example: 5\n"
            "\n"
            "Computes average position of close inside high-low range.\n"
            "1 = closes near highs, 0 = closes near lows."
        ),
    },
    "atr_ratio": {
        "func": None,
        "n_params": 2,
        "description": (
            "ATR Ratio indicator parameters:\n"
            "1. Short ATR window\n"
            "2. Long ATR window\n"
            "Example: 7,30\n"
            "\n"
            "Computes ATR(short) / ATR(long).\n"
            "ATR uses true range so overnight gaps are included,\n"
            "unlike vol_ratio which uses log-returns only.\n"
            "Values >1 = volatility expansion, <1 = compression."
        ),
    },
    "return_nd": {
        "func": None,
        "n_params": 1,
        "description": (
            "N-Day Return indicator parameters:\n"
            "1. Lookback window (days)\n"
            "Example: 5\n"
            "\n"
            "Raw return over the last N days: close/close[N] - 1.\n"
            "Direct momentum signal without RSI normalisation.\n"
            "Positive = recent uptrend, negative = recent downtrend."
        ),
    },
    "high_low_dist": {
        "func": None,
        "n_params": 1,
        "description": (
            "High-Low Distance indicator parameters:\n"
            "1. Rolling window length\n"
            "Example: 20\n"
            "\n"
            "Position of close within N-day high-low range:\n"
            "  (close - N_low) / (N_high - N_low)\n"
            "0 = at N-day low, 1 = at N-day high.\n"
            "Price-based analogue of RSI."
        ),
    },
    "close_open_ratio": {
        "func": None,
        "n_params": 1,
        "description": (
            "Close/Open Ratio indicator parameters:\n"
            "1. Rolling window length\n"
            "Example: 10\n"
            "\n"
            "Mean of (close - midpoint) / (range/2) over the last N days.\n"
            "Positive = stock consistently closes above daily midpoint\n"
            "(strong intraday closes). Useful for predicting S."
        ),
    },
    "mean_S": {
        "func": None,
        "n_params": 1,
        "description": (
            "Mean S indicator parameters:\n"
            "1. Rolling window length\n"
            "Example: 5\n"
            "\n"
            "Rolling mean of the true RMS skew S = (close - midpoint) / half_range\n"
            "where midpoint and half_range are computed relative to PREVIOUS close,\n"
            "matching the S definition used in density building.\n"
            "Captures persistence in where close lands in the daily range.\n"
            "Positive = tends to close near top of range, negative = near bottom."
        ),
    },
    "day_of_week": {
        "func": None,
        "n_params": 0,
        "description": (
            "Day of Week indicator (no parameters)\n"
            "\n"
            "Returns weekday of the current bar: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri.\n"
            "Captures systematic intraday closing bias by day of week\n"
            "(e.g. risk-off into weekend may lower S on Fridays).\n"
            "Requires dates to be passed to compute()."
        ),
    },
}


# --------------------------------------------------------------
# INDICATOR IMPLEMENTATIONS
# --------------------------------------------------------------

def macd(high, low, close, params):
    fast, slow, signal = params
    x = close

    def ema(data, window):
        alpha = 2 / (window + 1)
        out = np.zeros_like(data)
        out[0] = data[0]
        for i in range(1, len(data)):
            out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
        return out

    fast_ema = ema(x, fast)
    slow_ema = ema(x, slow)

    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)

    return macd_line - signal_line


def rsi(high, low, close, params):
    window = int(params[0])
    x = close

    diff = np.diff(x, prepend=x[0])

    gains = np.maximum(diff, 0)
    losses = -np.minimum(diff, 0)

    avg_gain = np.convolve(gains, np.ones(window) / window, mode="same")
    avg_loss = np.convolve(losses, np.ones(window) / window, mode="same")

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))

    return rsi

def rsi_velocity(high, low, close, params):

    rsi_window, vel_window = params

    if rsi_window < 2:
        raise ValueError("rsi_window must be >= 2")

    if vel_window < 1:
        raise ValueError("velocity_window must be >= 1")

    close = np.asarray(close)
    n = len(close)

    # --- Compute RSI first ---
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0.0)

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)

    # Initial average
    avg_gain[rsi_window] = np.mean(gain[1:rsi_window+1])
    avg_loss[rsi_window] = np.mean(loss[1:rsi_window+1])

    for i in range(rsi_window + 1, n):
        avg_gain[i] = (
            (avg_gain[i-1] * (rsi_window - 1) + gain[i]) / rsi_window
        )
        avg_loss[i] = (
            (avg_loss[i-1] * (rsi_window - 1) + loss[i]) / rsi_window
        )

    rs = np.divide(
        avg_gain,
        avg_loss,
        out=np.zeros_like(avg_gain),
        where=avg_loss != 0
    )

    rsi = 100 - (100 / (1 + rs))

    # --- Compute velocity (smoothed derivative) ---
    drsi = np.diff(rsi)
    drsi = np.insert(drsi, 0, 0.0)

    out = np.full(n, np.nan)

    for i in range(rsi_window + vel_window, n):
        window_slice = drsi[i - vel_window + 1 : i + 1]

        if not np.all(np.isfinite(window_slice)):
            continue

        out[i] = np.mean(window_slice)

    return out

def trend_slope(high, low, close, params):

    window = params[0]

    close = np.asarray(close)
    n = len(close)

    out = np.full(n, np.nan)

    if window < 2:
        raise ValueError("trend_slope window must be >= 2")

    x = np.arange(window)

    for i in range(window - 1, n):

        y = np.log(close[i - window + 1 : i + 1])

        if not np.all(np.isfinite(y)):
            continue

        # Linear regression slope
        x_mean = x.mean()
        y_mean = y.mean()

        cov = np.sum((x - x_mean) * (y - y_mean))
        var = np.sum((x - x_mean) ** 2)

        if var == 0:
            continue

        slope = cov / var

        # Optional normalization by volatility of window
        y_std = np.std(y)

        if y_std > 0:
            out[i] = slope / y_std
        else:
            out[i] = slope

    return out

def vol_ratio(high, low, close, params):

    short_w, long_w = params

    if long_w <= short_w:
        raise ValueError("long_window must be greater than short_window")

    close = np.asarray(close)
    n = len(close)

    out = np.full(n, np.nan)

    # Log returns
    logret = np.diff(np.log(close))
    logret = np.insert(logret, 0, 0.0)  # keep same length

    for i in range(long_w, n):

        short_slice = logret[i - short_w + 1 : i + 1]
        long_slice  = logret[i - long_w + 1  : i + 1]

        short_std = np.std(short_slice)
        long_std  = np.std(long_slice)

        if long_std > 0:
            out[i] = short_std / long_std
        else:
            out[i] = np.nan

    return out

def range_position(high, low, close, params):

    window = params[0]

    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)

    n = len(close)
    out = np.full(n, np.nan)

    if window < 1:
        raise ValueError("range_position window must be >= 1")

    for i in range(window - 1, n):

        h_slice = high[i - window + 1 : i + 1]
        l_slice = low[i - window + 1  : i + 1]
        c_slice = close[i - window + 1 : i + 1]

        if not (
            np.all(np.isfinite(h_slice)) and
            np.all(np.isfinite(l_slice)) and
            np.all(np.isfinite(c_slice))
        ):
            continue

        ranges = h_slice - l_slice
        valid = ranges > 0

        if not np.any(valid):
            continue

        pos = (c_slice[valid] - l_slice[valid]) / ranges[valid]

        # Average position over window
        out[i] = np.mean(pos)

    return out


def atr_ratio(high, low, close, params):

    short_w, long_w = int(params[0]), int(params[1])

    if long_w <= short_w:
        raise ValueError("atr_ratio: long_window must be greater than short_window")

    high  = np.asarray(high,  dtype=float)
    low   = np.asarray(low,   dtype=float)
    close = np.asarray(close, dtype=float)

    n = len(close)
    out = np.full(n, np.nan)

    # True range: max of (H-L), |H-prev_C|, |L-prev_C|
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i]  - close[i - 1]),
        )

    # Wilder smoothed ATR
    def wilder_atr(window):
        atr = np.full(n, np.nan)
        if n < window:
            return atr
        atr[window - 1] = np.mean(tr[:window])
        for i in range(window, n):
            atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
        return atr

    atr_short = wilder_atr(short_w)
    atr_long  = wilder_atr(long_w)

    valid = (atr_long > 0) & np.isfinite(atr_short) & np.isfinite(atr_long)
    out[valid] = atr_short[valid] / atr_long[valid]

    return out


def return_nd(high, low, close, params):

    window = int(params[0])

    if window < 1:
        raise ValueError("return_nd: window must be >= 1")

    close = np.asarray(close, dtype=float)
    n     = len(close)
    out   = np.full(n, np.nan)

    for i in range(window, n):
        prev = close[i - window]
        if prev > 0 and np.isfinite(prev) and np.isfinite(close[i]):
            out[i] = close[i] / prev - 1.0

    return out


def high_low_dist(high, low, close, params):

    window = int(params[0])

    if window < 1:
        raise ValueError("high_low_dist: window must be >= 1")

    high  = np.asarray(high,  dtype=float)
    low   = np.asarray(low,   dtype=float)
    close = np.asarray(close, dtype=float)

    n   = len(close)
    out = np.full(n, np.nan)

    for i in range(window - 1, n):
        h_max = np.max(high[i - window + 1 : i + 1])
        l_min = np.min(low[i  - window + 1 : i + 1])
        rng   = h_max - l_min

        if rng > 0 and np.isfinite(close[i]):
            out[i] = (close[i] - l_min) / rng

    return out


def close_open_ratio(high, low, close, params):

    window = int(params[0])

    if window < 1:
        raise ValueError("close_open_ratio: window must be >= 1")

    high  = np.asarray(high,  dtype=float)
    low   = np.asarray(low,   dtype=float)
    close = np.asarray(close, dtype=float)

    n   = len(close)
    out = np.full(n, np.nan)

    # Intraday close strength: where close sits relative to midpoint,
    # normalised to [-1, 1]. This is the daily S value.
    # Positive = closed above midpoint, negative = closed below.
    rng      = high - low
    daily_co = np.where(rng > 0, (close - (high + low) / 2.0) / (rng / 2.0), 0.0)

    for i in range(window - 1, n):
        window_vals = daily_co[i - window + 1 : i + 1]
        if np.all(np.isfinite(window_vals)):
            out[i] = np.mean(window_vals)

    return out

# Register functions
INDICATOR_REGISTRY["macd"]["func"] = macd
INDICATOR_REGISTRY["rsi"]["func"] = rsi
INDICATOR_REGISTRY["rsi_velocity"]["func"] = rsi_velocity
INDICATOR_REGISTRY["vol_ratio"]["func"] = vol_ratio
INDICATOR_REGISTRY["trend_slope"]["func"] = trend_slope
INDICATOR_REGISTRY["range_position"]["func"] = range_position
INDICATOR_REGISTRY["atr_ratio"]["func"] = atr_ratio
INDICATOR_REGISTRY["return_nd"]["func"] = return_nd
INDICATOR_REGISTRY["high_low_dist"]["func"] = high_low_dist
INDICATOR_REGISTRY["close_open_ratio"]["func"] = close_open_ratio


def mean_S(high, low, close, params, dates=None):
    """
    Rolling mean of true RMS skew S over N days.

    S[i] = (rC - rM) / (rR / 2)
    where rH = (high[i]  - close[i-1]) / close[i-1]
          rL = (low[i]   - close[i-1]) / close[i-1]
          rC = (close[i] - close[i-1]) / close[i-1]
          rR = rH - rL  (range relative to prev close)
          rM = (rH + rL) / 2  (midpoint shift)

    This matches the S definition used throughout the density/tuning pipeline,
    unlike close_open_ratio which uses intraday midpoint only.
    """
    window = int(params[0])
    high   = np.asarray(high,  dtype=float)
    low    = np.asarray(low,   dtype=float)
    close  = np.asarray(close, dtype=float)
    n      = len(close)
    out    = np.full(n, np.nan)

    if window < 1:
        raise ValueError("mean_S: window must be >= 1")

    # Compute daily S series
    S_daily = np.full(n, np.nan)
    for i in range(1, n):
        prev = close[i - 1]
        if prev <= 0 or not np.isfinite(prev):
            continue
        rH = (high[i]  - prev) / prev
        rL = (low[i]   - prev) / prev
        rC = (close[i] - prev) / prev
        rR = rH - rL
        rM = (rH + rL) / 2.0
        if rR > 0 and np.isfinite(rH) and np.isfinite(rL) and np.isfinite(rC):
            S_daily[i] = (rC - rM) / (rR / 2.0)

    # Rolling mean over window
    for i in range(window, n):
        window_vals = S_daily[i - window + 1 : i + 1]
        finite_vals = window_vals[np.isfinite(window_vals)]
        if len(finite_vals) > 0:
            out[i] = np.mean(finite_vals)

    return out


def day_of_week(high, low, close, params, dates=None):
    """
    Returns weekday as float: 0=Monday, 1=Tuesday, ..., 4=Friday.
    Requires dates array (numpy datetime64[D] or Python date objects).
    Returns NaN for all bars if dates not provided.
    """
    import datetime
    n   = len(close)
    out = np.full(n, np.nan)

    if dates is None:
        return out

    for i, d in enumerate(dates):
        try:
            if hasattr(d, "astype"):
                # numpy datetime64 → Python date
                dt = d.astype("datetime64[D]").astype(datetime.date)
            else:
                dt = d
            out[i] = float(dt.weekday())  # 0=Mon, 4=Fri
        except Exception:
            pass

    return out


INDICATOR_REGISTRY["mean_S"]["func"] = mean_S
INDICATOR_REGISTRY["day_of_week"]["func"] = day_of_week


# ==============================================================
# INDICATOR CLASS
# ==============================================================

class Indicator:
    """
    Generic indicator wrapper.
    """

    def __init__(self, name: str, params: list):
        if name not in INDICATOR_REGISTRY:
            raise ValueError(f"Unknown indicator: {name}")

        self.name = name
        # JSON always deserialises numbers as float. Normalise params so that
        # whole-number values (e.g. 14.0) are stored as int — this means every
        # indicator function receives the correct type regardless of call site.
        self.params = [int(p) if isinstance(p, float) and p.is_integer() else p
                       for p in params]
        self.func = INDICATOR_REGISTRY[name]["func"]

    # ----------------------------------------------------------

    @staticmethod
    def get_param_count(name):
        return INDICATOR_REGISTRY[name]["n_params"]

    # ----------------------------------------------------------

    def compute(self, high, low, close, dates=None):
        """
        Compute indicator series.

        dates : optional array of date objects, same length as close.
                Required for date-dependent indicators (e.g. day_of_week).
                Silently ignored by all other indicators.
        """
        import inspect
        if "dates" in inspect.signature(self.func).parameters:
            return self.func(high, low, close, self.params, dates=dates)
        return self.func(high, low, close, self.params)

    # ----------------------------------------------------------

    @staticmethod
    def available():
        return list(INDICATOR_REGISTRY.keys())

    # ----------------------------------------------------------

    @staticmethod
    def get_description(name):
        return INDICATOR_REGISTRY[name]["description"]

    # ----------------------------------------------------------

    def __repr__(self):
        return f"Indicator(name={self.name}, params={self.params})"

# ==============================================================
# GLOBAL INDICATOR REGISTRY + IMPLEMENTATIONS
# ==============================================================
#
# Global indicators are computed cross-sectionally — their x-value
# on day d is a function of ALL tickers on that day, not a single
# ticker's history.
#
# Each function signature:
#   f(close_matrix, high_matrix, low_matrix, params)
#       close_matrix : np.ndarray [n_tickers, n_days]
#   Returns : np.ndarray [n_days]  — one value per day

GLOBAL_INDICATOR_REGISTRY = {
    "market_mean_R": {
        "func": None,
        "n_params": 0,
        "description": (
            "Market Mean R (no parameters)\n"
            "\n"
            "Cross-sectional mean of daily range R = (High-Low)/prev_close\n"
            "across all tickers. Captures market-wide volatility regime.\n"
            "High values = elevated market volatility day.\n"
            "Example: (no params needed, just press Enter)"
        ),
    },
    "market_mean_M": {
        "func": None,
        "n_params": 0,
        "description": (
            "Market Mean M (no parameters)\n"
            "\n"
            "Cross-sectional mean of daily midpoint drift M across all tickers.\n"
            "Captures broad market direction (up/down day).\n"
            "Positive = broad up day, negative = broad down day."
        ),
    },
    "market_mean_S": {
        "func": None,
        "n_params": 0,
        "description": (
            "Market Mean S (no parameters)\n"
            "\n"
            "Cross-sectional mean of daily skew S across all tickers.\n"
            "Captures whether stocks broadly close near highs or lows.\n"
            "Positive = broad strong close, negative = broad weak close."
        ),
    },
    "market_breadth": {
        "func": None,
        "n_params": 0,
        "description": (
            "Market Breadth (no parameters)\n"
            "\n"
            "Fraction of tickers with M > 0 on each day.\n"
            "Range [0, 1]. 0.9 = 90% of stocks had positive drift.\n"
            "More informative than mean M when distribution is skewed."
        ),
    },
    "market_vol_dispersion": {
        "func": None,
        "n_params": 0,
        "description": (
            "Market Volatility Dispersion (no parameters)\n"
            "\n"
            "Cross-sectional std of R across tickers on each day.\n"
            "Low = macro shock (everything moves together).\n"
            "High = idiosyncratic day (sector rotation, earnings spread)."
        ),
    },
}


# --------------------------------------------------------------
# GLOBAL INDICATOR IMPLEMENTATIONS
# --------------------------------------------------------------

def _compute_daily_RMS(close_matrix, high_matrix, low_matrix):
    """
    Compute R, M, S matrices [n_tickers, n_days-1] from price matrices.
    Day 0 is dropped since R/M/S require a previous close.
    Returns R, M, S each of shape [n_tickers, n_days-1].
    """
    prev_close = close_matrix[:, :-1]          # [n_tickers, n_days-1]
    H = high_matrix[:, 1:]
    L = low_matrix[:, 1:]
    C = close_matrix[:, 1:]

    R = (H - L) / prev_close
    mid = (H + L) / 2
    M = (mid - prev_close) / prev_close

    half = (H - L) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(half > 0, (C - mid) / half, np.nan)

    # Zero out non-finite values so nanmean works cleanly
    R = np.where(np.isfinite(R) & (R > 0), R, np.nan)
    M = np.where(np.isfinite(M), M, np.nan)
    S = np.where(np.isfinite(S) & (np.abs(S) < 1.5), S, np.nan)

    return R, M, S


def market_mean_R(close_matrix, high_matrix, low_matrix, params):
    R, _, _ = _compute_daily_RMS(close_matrix, high_matrix, low_matrix)
    # Prepend NaN for day 0 to keep length = n_days
    series = np.concatenate([[np.nan], np.nanmean(R, axis=0)])
    return series


def market_mean_M(close_matrix, high_matrix, low_matrix, params):
    _, M, _ = _compute_daily_RMS(close_matrix, high_matrix, low_matrix)
    series = np.concatenate([[np.nan], np.nanmean(M, axis=0)])
    return series


def market_mean_S(close_matrix, high_matrix, low_matrix, params):
    _, _, S = _compute_daily_RMS(close_matrix, high_matrix, low_matrix)
    series = np.concatenate([[np.nan], np.nanmean(S, axis=0)])
    return series


def market_breadth(close_matrix, high_matrix, low_matrix, params):
    _, M, _ = _compute_daily_RMS(close_matrix, high_matrix, low_matrix)
    with np.errstate(invalid="ignore"):
        breadth = np.nanmean(M > 0, axis=0).astype(float)
    series = np.concatenate([[np.nan], breadth])
    return series


def market_vol_dispersion(close_matrix, high_matrix, low_matrix, params):
    R, _, _ = _compute_daily_RMS(close_matrix, high_matrix, low_matrix)
    series = np.concatenate([[np.nan], np.nanstd(R, axis=0)])
    return series


# Register global functions
GLOBAL_INDICATOR_REGISTRY["market_mean_R"]["func"]         = market_mean_R
GLOBAL_INDICATOR_REGISTRY["market_mean_M"]["func"]         = market_mean_M
GLOBAL_INDICATOR_REGISTRY["market_mean_S"]["func"]         = market_mean_S
GLOBAL_INDICATOR_REGISTRY["market_breadth"]["func"]        = market_breadth
GLOBAL_INDICATOR_REGISTRY["market_vol_dispersion"]["func"] = market_vol_dispersion


# ==============================================================
# GLOBAL INDICATOR CLASS
# ==============================================================

class GlobalIndicator:
    """
    Cross-sectional indicator computed from the full ticker matrix.

    Unlike Indicator which operates on a single ticker's history,
    GlobalIndicator computes a single time series from all tickers
    simultaneously. The series is precomputed once at construction
    and stored internally.

    compute() mimics the Indicator.compute() interface — it accepts
    high/low/close arrays (ignored) and returns a slice of the
    precomputed series matching the input length. This means all
    existing call sites (FeatureBuilder, _evaluate_sample) work
    without modification.

    At strategy test time the strategy tester uses
    compute_series() directly to get the full aligned series,
    then injects values into x_vals at the marked injection point
    in _forecast_ticker (Option A design).
    """

    def __init__(self, name: str, params: list,
                 close_matrix, high_matrix, low_matrix):

        if name not in GLOBAL_INDICATOR_REGISTRY:
            raise ValueError(f"Unknown global indicator: {name}")

        self.name   = name
        self.params = params

        func = GLOBAL_INDICATOR_REGISTRY[name]["func"]
        self._series = func(close_matrix, high_matrix, low_matrix, params)

    # ----------------------------------------------------------

    def compute(self, high, low, close):
        """
        Return a slice of the precomputed series matching len(close).
        The last element [-1] is the current-day value, consistent
        with how Indicator.compute() is used everywhere.
        """
        n = len(close)
        if n <= len(self._series):
            return self._series[:n]
        # If requested length exceeds series, pad left with NaN
        pad = np.full(n - len(self._series), np.nan)
        return np.concatenate([pad, self._series])

    # ----------------------------------------------------------

    def compute_series(self):
        """Return the full precomputed series [n_days]."""
        return self._series.copy()

    # ----------------------------------------------------------

    @staticmethod
    def available():
        return list(GLOBAL_INDICATOR_REGISTRY.keys())

    # ----------------------------------------------------------

    @staticmethod
    def get_description(name):
        return GLOBAL_INDICATOR_REGISTRY[name]["description"]

    # ----------------------------------------------------------

    @staticmethod
    def get_param_count(name):
        return GLOBAL_INDICATOR_REGISTRY[name]["n_params"]

    # ----------------------------------------------------------

    def __repr__(self):
        return f"GlobalIndicator(name={self.name}, params={self.params})"

