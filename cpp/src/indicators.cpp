#include "rms_types.h"
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

// ================================================================
// All indicator functions take the same signature:
//
//   compute_X(high, low, close, n, params)
//       high/low/close : raw C pointers into the history array
//       n              : history length
//       params         : indicator parameters
//   Returns : double   (the [-1] value, i.e. current day's value)
//
// This matches indicator.py's compute() which returns the full
// series and the caller takes [-1].
// ================================================================

// ----------------------------------------------------------------
// RSI  — matches Python implementation in indicator.py exactly.
//
// Python uses np.convolve(gains, ones(window)/window, mode="same")
// which is a centred convolution, NOT Wilder smoothing.
// mode="same" at output index i sums gains[i - period//2 + j]
// for j in [0, period), with out-of-bounds terms treated as 0,
// then divides by period (the kernel norm, not the count).
// ----------------------------------------------------------------
double compute_rsi(const double* close, int n, const std::vector<int>& params)
{
    int period = params[0];
    if (n < 2) return std::numeric_limits<double>::quiet_NaN();

    // diffs: diff[0]=0, diff[i] = close[i]-close[i-1]
    // matches: diff = np.diff(x, prepend=x[0])
    std::vector<double> gains(n, 0.0), losses(n, 0.0);
    for (int i = 1; i < n; ++i) {
        double d = close[i] - close[i - 1];
        gains[i]  = d > 0 ?  d : 0.0;
        losses[i] = d < 0 ? -d : 0.0;
    }

    // convolve at last index n-1, mode="same", kernel = ones(period)/period
    // centred kernel: output[i] = sum_j gains[i - period//2 + j] / period
    int half = period / 2;
    double sum_gain = 0.0, sum_loss = 0.0;
    for (int j = 0; j < period; ++j) {
        int idx = (n - 1) - half + j;
        if (idx >= 0 && idx < n) {
            sum_gain  += gains[idx];
            sum_loss  += losses[idx];
        }
    }
    double avg_gain = sum_gain / period;
    double avg_loss = sum_loss / period;

    double rs = avg_gain / (avg_loss + 1e-12);
    return 100.0 - 100.0 / (1.0 + rs);
}

// ----------------------------------------------------------------
// RSI Velocity  (params[0]=rsi_window, params[1]=vel_window)
//
// Matches Python rsi_velocity exactly:
//   1. Compute Wilder-smoothed RSI series (NOT convolve RSI)
//   2. Compute first difference of RSI series
//   3. Return mean of last vel_window differences
//
// Python uses Wilder smoothing:
//   avg_gain[w] = mean(gain[1..w])
//   avg_gain[i] = (avg_gain[i-1]*(w-1) + gain[i]) / w
// ----------------------------------------------------------------
double compute_rsi_velocity(const double* close, int n, const std::vector<int>& params)
{
    int rsi_window = params[0];
    int vel_window = params[1];

    int required = rsi_window + vel_window + 1;
    if (n < required) return std::numeric_limits<double>::quiet_NaN();

    // Step 1: compute Wilder-smoothed RSI series over all n bars
    std::vector<double> delta(n, 0.0);
    for (int i = 1; i < n; ++i)
        delta[i] = close[i] - close[i - 1];

    std::vector<double> gain(n, 0.0), loss(n, 0.0);
    for (int i = 1; i < n; ++i) {
        gain[i] = delta[i] > 0 ?  delta[i] : 0.0;
        loss[i] = delta[i] < 0 ? -delta[i] : 0.0;
    }

    // Wilder seed at index rsi_window
    std::vector<double> rsi(n, std::numeric_limits<double>::quiet_NaN());
    double avg_gain = 0.0, avg_loss = 0.0;
    for (int i = 1; i <= rsi_window; ++i) {
        avg_gain += gain[i];
        avg_loss += loss[i];
    }
    avg_gain /= rsi_window;
    avg_loss /= rsi_window;

    double rs = avg_gain / (avg_loss + 1e-12);
    rsi[rsi_window] = 100.0 - 100.0 / (1.0 + rs);

    for (int i = rsi_window + 1; i < n; ++i) {
        avg_gain = (avg_gain * (rsi_window - 1) + gain[i]) / rsi_window;
        avg_loss = (avg_loss * (rsi_window - 1) + loss[i]) / rsi_window;
        rs = avg_gain / (avg_loss + 1e-12);
        rsi[i] = 100.0 - 100.0 / (1.0 + rs);
    }

    // Step 2: first difference of RSI (prepend 0 to match Python np.insert)
    std::vector<double> drsi(n, 0.0);
    for (int i = 1; i < n; ++i) {
        if (std::isfinite(rsi[i]) && std::isfinite(rsi[i - 1]))
            drsi[i] = rsi[i] - rsi[i - 1];
    }

    // Step 3: mean of last vel_window drsi values
    // Python: out[i] = mean(drsi[i - vel_window + 1 : i + 1])
    // for i = n-1: mean(drsi[n-vel_window .. n-1])
    int start = n - vel_window;
    if (start < rsi_window + 1) return std::numeric_limits<double>::quiet_NaN();

    double sum = 0.0;
    int count = 0;
    for (int i = start; i < n; ++i) {
        if (std::isfinite(drsi[i])) {
            sum += drsi[i];
            ++count;
        }
    }
    if (count == 0) return std::numeric_limits<double>::quiet_NaN();
    return sum / count;
}

// ----------------------------------------------------------------
// Vol Ratio  (params[0]=short_w, params[1]=long_w)
// Returns: std(returns, short_w) / std(returns, long_w)
// ----------------------------------------------------------------
double compute_vol_ratio(const double* close, int n, const std::vector<int>& params)
{
    int short_w = params[0];
    int long_w  = params[1];

    if (n < long_w + 1) return std::numeric_limits<double>::quiet_NaN();

    // Log returns — matches Python which uses np.diff(np.log(close))
    // prepended with 0 to keep same length, then slices logret[i-w+1:i+1]
    std::vector<double> logrets(n, 0.0);
    for (int i = 1; i < n; ++i)
        logrets[i] = std::log(close[i] / close[i - 1]);

    // std of last `len` log returns starting at index `start`
    // Python slices logret[i-w+1:i+1] — for last value i=n-1:
    //   short: logret[n-short_w:n]   = logrets[n-short_w .. n-1]
    //   long:  logret[n-long_w:n]    = logrets[n-long_w  .. n-1]
    auto stddev = [&](int start, int len) -> double {
        if (len <= 1) return 0.0;
        double sum = 0, sum2 = 0;
        for (int i = start; i < start + len; ++i) {
            sum  += logrets[i];
            sum2 += logrets[i] * logrets[i];
        }
        double mean = sum / len;
        double var  = sum2 / len - mean * mean;
        return var > 0 ? std::sqrt(var) : 0.0;
    };

    double vol_short = stddev(n - short_w, short_w);
    double vol_long  = stddev(n - long_w,  long_w);

    if (vol_long < 1e-12)
        return std::numeric_limits<double>::quiet_NaN();
    return vol_short / vol_long;
}

// ----------------------------------------------------------------
// MACD  (params[0]=fast, params[1]=slow, params[2]=signal)
// Returns: macd_line - signal_line (histogram)
// ----------------------------------------------------------------

static double ema_last(const double* data, int n, int period)
{
    if (n < 1) return std::numeric_limits<double>::quiet_NaN();
    double k = 2.0 / (period + 1.0);
    double ema = data[0];
    for (int i = 1; i < n; ++i)
        ema = data[i] * k + ema * (1.0 - k);
    return ema;
}

double compute_macd(const double* close, int n, const std::vector<int>& params)
{
    int fast   = params[0];
    int slow   = params[1];
    int signal = params[2];

    int required = slow + signal;
    if (n < required) return std::numeric_limits<double>::quiet_NaN();

    // Build MACD line over last (signal) points
    std::vector<double> macd_line(signal);
    for (int i = 0; i < signal; ++i) {
        int end = n - signal + i + 1;
        double ema_fast = ema_last(close, end, fast);
        double ema_slow = ema_last(close, end, slow);
        macd_line[i] = ema_fast - ema_slow;
    }

    double signal_line = ema_last(macd_line.data(), signal, signal);
    return macd_line.back() - signal_line;
}

// ----------------------------------------------------------------
// Trend Slope  (params[0]=window)
//
// Matches Python trend_slope exactly:
//   y = log(close[n-window : n])
//   slope = OLS slope of y vs [0..window-1]
//   return slope / std(y)   (if std(y) > 0, else slope)
// ----------------------------------------------------------------
double compute_trend_slope(const double* close, int n, const std::vector<int>& params)
{
    int window = params[0];
    if (n < window) return std::numeric_limits<double>::quiet_NaN();

    // Log prices
    std::vector<double> y(window);
    for (int i = 0; i < window; ++i) {
        if (close[n - window + i] <= 0)
            return std::numeric_limits<double>::quiet_NaN();
        y[i] = std::log(close[n - window + i]);
    }

    // Check all finite
    for (int i = 0; i < window; ++i)
        if (!std::isfinite(y[i]))
            return std::numeric_limits<double>::quiet_NaN();

    // OLS: x = [0, 1, ..., window-1]
    double sx = 0, sy = 0, sxy = 0, sx2 = 0;
    for (int i = 0; i < window; ++i) {
        sx  += i;
        sy  += y[i];
        sxy += i * y[i];
        sx2 += i * i;
    }

    double denom = window * sx2 - sx * sx;
    if (std::abs(denom) < 1e-12)
        return std::numeric_limits<double>::quiet_NaN();

    double slope = (window * sxy - sx * sy) / denom;

    // Normalise by std(y) — matches Python np.std (population std, ddof=0)
    double y_mean = sy / window;
    double var = 0.0;
    for (int i = 0; i < window; ++i)
        var += (y[i] - y_mean) * (y[i] - y_mean);
    var /= window;
    double y_std = std::sqrt(var);

    if (y_std > 0)
        return slope / y_std;
    else
        return slope;
}

// ----------------------------------------------------------------
// Range Position  (params[0]=window)
//
// Matches Python range_position exactly:
//   For each day in the window, compute (close - day_low) / (day_high - day_low)
//   Return mean of these per-day positions (only days where range > 0)
//
// NOT the same as (close[-1] - N_low) / (N_high - N_low)
// ----------------------------------------------------------------
double compute_range_position(const double* high, const double* low,
                               const double* close, int n,
                               const std::vector<int>& params)
{
    int window = params[0];
    if (n < window) return std::numeric_limits<double>::quiet_NaN();

    double sum = 0.0;
    int count = 0;

    for (int i = n - window; i < n; ++i) {
        double h = high[i];
        double l = low[i];
        double c = close[i];
        double rng = h - l;
        if (rng > 0 && std::isfinite(h) && std::isfinite(l) && std::isfinite(c)) {
            sum += (c - l) / rng;
            ++count;
        }
    }

    if (count == 0) return std::numeric_limits<double>::quiet_NaN();
    return sum / count;
}

// ----------------------------------------------------------------
// ATR Ratio  (params[0]=short_w, params[1]=long_w)
// Returns: ATR(short) / ATR(long)  — handles gaps better than vol_ratio
// ----------------------------------------------------------------
double compute_atr_ratio(const double* high, const double* low,
                          const double* close, int n,
                          const std::vector<int>& params)
{
    int short_w = params[0];
    int long_w  = params[1];

    // Need at least long_w + 1 bars (1 for first TR, long_w for Wilder init)
    if (n < long_w + 1) return std::numeric_limits<double>::quiet_NaN();

    // True range: tr[i] uses bar i+1 and close[i] as previous close
    // tr[0] = high[0] - low[0] (no previous close available)
    int ntr = n;
    std::vector<double> tr(ntr);
    tr[0] = high[0] - low[0];
    for (int i = 1; i < ntr; ++i) {
        double h  = high[i];
        double l  = low[i];
        double pc = close[i - 1];
        tr[i] = std::max({ h - l, std::abs(h - pc), std::abs(l - pc) });
    }

    // Wilder smoothed ATR — matches Python wilder_atr():
    //   atr[window-1] = mean(tr[0..window-1])
    //   atr[i]        = (atr[i-1] * (window-1) + tr[i]) / window
    auto wilder_atr = [&](int window) -> double {
        if (ntr < window) return std::numeric_limits<double>::quiet_NaN();
        // Seed with simple mean of first `window` bars
        double atr = 0.0;
        for (int i = 0; i < window; ++i) atr += tr[i];
        atr /= window;
        // Smooth the rest
        for (int i = window; i < ntr; ++i)
            atr = (atr * (window - 1) + tr[i]) / window;
        return atr;
    };

    double atr_short = wilder_atr(short_w);
    double atr_long  = wilder_atr(long_w);

    if (!std::isfinite(atr_long) || atr_long < 1e-12)
        return std::numeric_limits<double>::quiet_NaN();
    return atr_short / atr_long;
}

// ----------------------------------------------------------------
// Return_ND  (params[0]=window)
// Returns: (close[t] - close[t-window]) / close[t-window]
// ----------------------------------------------------------------
double compute_return_nd(const double* close, int n, const std::vector<int>& params)
{
    int window = params[0];
    if (n < window + 1) return std::numeric_limits<double>::quiet_NaN();
    double prev = close[n - 1 - window];
    if (prev <= 0) return std::numeric_limits<double>::quiet_NaN();
    return (close[n - 1] - prev) / prev;
}

// ----------------------------------------------------------------
// High Low Dist  (params[0]=window)
// Returns: (close - N_low) / (N_high - N_low)   same as range_position
// (kept separate to match Python registry — same formula, different name)
// ----------------------------------------------------------------
double compute_high_low_dist(const double* high, const double* low,
                              const double* close, int n,
                              const std::vector<int>& params)
{
    return compute_range_position(high, low, close, n, params);
}

// ----------------------------------------------------------------
// Close Open Ratio  (params[0]=window)
// Rolling mean of daily S = (close - mid) / half_range
// ----------------------------------------------------------------
double compute_close_open_ratio(const double* high, const double* low,
                                 const double* close, int n,
                                 const std::vector<int>& params)
{
    int window = params[0];
    if (n < window) return std::numeric_limits<double>::quiet_NaN();

    double sum = 0.0;
    int count = 0;

    for (int i = n - window; i < n; ++i) {
        double h = high[i];
        double l = low[i];
        double c = close[i];
        double mid  = (h + l) / 2.0;
        double half = (h - l) / 2.0;
        if (half > 0) {
            sum += (c - mid) / half;
            ++count;
        }
    }

    if (count == 0) return std::numeric_limits<double>::quiet_NaN();
    return sum / count;
}

// ----------------------------------------------------------------
// Mean S  (params[0]=window)
//
// Rolling mean of true RMS skew S over the last N days.
// S[i] = (rC - rM) / (rR/2)
//   rH = (high[i]  - close[i-1]) / close[i-1]
//   rL = (low[i]   - close[i-1]) / close[i-1]
//   rC = (close[i] - close[i-1]) / close[i-1]
//   rR = rH - rL,   rM = (rH + rL) / 2
// Matches the S definition used in density building exactly.
// ----------------------------------------------------------------
double compute_mean_S(const double* high, const double* low,
                      const double* close, int n,
                      const std::vector<int>& params)
{
    int window = params[0];
    if (n < window + 1) return std::numeric_limits<double>::quiet_NaN();

    // Compute daily S series for last window+1 bars (need prev close)
    // We only need the last `window` valid S values
    double sum = 0.0;
    int count  = 0;

    // Walk backwards through the last `window` bars (i = n-1 down to n-window)
    // For bar i we need close[i-1] as previous close
    int start = std::max(1, n - window);
    for (int i = start; i < n; ++i) {
        double prev = close[i - 1];
        if (prev <= 0.0 || !std::isfinite(prev)) continue;
        double rH = (high[i]  - prev) / prev;
        double rL = (low[i]   - prev) / prev;
        double rC = (close[i] - prev) / prev;
        double rR = rH - rL;
        double rM = (rH + rL) / 2.0;
        if (rR > 0.0 && std::isfinite(rH) && std::isfinite(rL) && std::isfinite(rC)) {
            sum += (rC - rM) / (rR / 2.0);
            ++count;
        }
    }

    if (count == 0) return std::numeric_limits<double>::quiet_NaN();
    return sum / count;
}

// ================================================================
// DISPATCH
// Calls the right indicator function given an IndicatorSpec
// and history arrays of length n.
// ================================================================
double compute_indicator(const IndicatorSpec& spec,
                         const double* high,
                         const double* low,
                         const double* close,
                         int n)
{
    switch (spec.type) {
        case IndicatorType::RSI:
            return compute_rsi(close, n, spec.params);
        case IndicatorType::RSI_VELOCITY:
            return compute_rsi_velocity(close, n, spec.params);
        case IndicatorType::VOL_RATIO:
            return compute_vol_ratio(close, n, spec.params);
        case IndicatorType::MACD:
            return compute_macd(close, n, spec.params);
        case IndicatorType::TREND_SLOPE:
            return compute_trend_slope(close, n, spec.params);
        case IndicatorType::RANGE_POSITION:
            return compute_range_position(high, low, close, n, spec.params);
        case IndicatorType::ATR_RATIO:
            return compute_atr_ratio(high, low, close, n, spec.params);
        case IndicatorType::RETURN_ND:
            return compute_return_nd(close, n, spec.params);
        case IndicatorType::HIGH_LOW_DIST:
            return compute_high_low_dist(high, low, close, n, spec.params);
        case IndicatorType::CLOSE_OPEN_RATIO:
            return compute_close_open_ratio(high, low, close, n, spec.params);
        case IndicatorType::MEAN_S:
            return compute_mean_S(high, low, close, n, spec.params);
        case IndicatorType::GLOBAL:
            // Should never be called — global x-values are injected externally
            return std::numeric_limits<double>::quiet_NaN();
    }
    return std::numeric_limits<double>::quiet_NaN();
}
