#include "rms_types.h"
#include <array>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <cstring>

// ----------------------------------------------------------------
// Forward declarations from other translation units
// ----------------------------------------------------------------
double compute_indicator(const IndicatorSpec& spec,
                         const double* high, const double* low,
                         const double* close, int n);

double sample_R(double x, const PackedDensity& d, std::mt19937& rng);
double sample_M(double x, double R, const PackedDensity& d, std::mt19937& rng);
double sample_S(double x, double R, const PackedDensity& d, std::mt19937& rng);

// ================================================================
// mixture_select
//
// Randomly select one index from [0, n) proportional to weights[],
// skipping entries where nan_mask[i] is true.
// Returns -1 if all weights are zero or masked.
// ================================================================
static int mixture_select(
    const std::vector<double>& weights,
    const std::vector<double>& x_vals,
    int n,
    std::mt19937& rng)
{
    // Collect valid (non-NaN) weights and compute total
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!std::isnan(x_vals[i])) total += weights[i];
    }
    if (total <= 0.0) return -1;

    // Draw uniform in [0, total) and walk until we exceed it
    std::uniform_real_distribution<double> uni(0.0, total);
    double u = uni(rng);
    double cumsum = 0.0;
    for (int i = 0; i < n; ++i) {
        if (std::isnan(x_vals[i])) continue;
        cumsum += weights[i];
        if (u <= cumsum) return i;
    }
    // Fallback to last valid index (handles floating point edge cases)
    for (int i = n - 1; i >= 0; --i)
        if (!std::isnan(x_vals[i])) return i;
    return -1;
}

// ================================================================
// step_from_x
//
// Sample one day's (close, high, low) given pre-computed indicator
// x-values and the current price P.
// Exact C++ equivalent of TransitionModel.step_from_x().
//
// mixture_mode=false : weighted mean (original behaviour)
// mixture_mode=true  : randomly select one density per component
//                      proportional to weights, sample from that only
// ================================================================
void step_from_x(
    const std::vector<double>& x_R,
    const std::vector<double>& x_M,
    const std::vector<double>& x_S,
    const DensitySet& dens,
    double P,
    std::mt19937& rng,
    double& out_close,
    double& out_high,
    double& out_low,
    bool mixture_mode = false)
{
    double R, M, S;

    if (mixture_mode) {
        // --- R: select one density, sample from it ---
        std::vector<double> w_R(dens.R.size());
        for (int i = 0; i < (int)dens.R.size(); ++i) w_R[i] = dens.R[i].weight;
        int ir = mixture_select(w_R, x_R, (int)dens.R.size(), rng);
        R = (ir >= 0) ? sample_R(x_R[ir], dens.R[ir], rng) : 0.0;

        // --- M: select one density, sample from it ---
        std::vector<double> w_M(dens.M.size());
        for (int i = 0; i < (int)dens.M.size(); ++i) w_M[i] = dens.M[i].weight;
        int im = mixture_select(w_M, x_M, (int)dens.M.size(), rng);
        M = (im >= 0) ? sample_M(x_M[im], R, dens.M[im], rng) : 0.0;

        // --- S: select one density, sample from it ---
        std::vector<double> w_S(dens.S.size());
        for (int i = 0; i < (int)dens.S.size(); ++i) w_S[i] = dens.S[i].weight;
        int is = mixture_select(w_S, x_S, (int)dens.S.size(), rng);
        S = (is >= 0) ? sample_S(x_S[is], R, dens.S[is], rng) : 0.0;

    } else {
        // --- R (weighted mean) ---
        // Skip any density whose x-value is NaN (global data unavailable)
        double R_num = 0.0, R_den = 0.0;
        for (int i = 0; i < (int)dens.R.size(); ++i) {
            if (std::isnan(x_R[i])) continue;
            double r = sample_R(x_R[i], dens.R[i], rng);
            double w = dens.R[i].weight;
            R_num += r * w;
            R_den += w;
        }
        R = R_den > 0 ? R_num / R_den : 0.0;

        // --- M (weighted mean, conditioned on R) ---
        double M_num = 0.0, M_den = 0.0;
        for (int i = 0; i < (int)dens.M.size(); ++i) {
            if (std::isnan(x_M[i])) continue;
            double m = sample_M(x_M[i], R, dens.M[i], rng);
            double w = dens.M[i].weight;
            M_num += m * w;
            M_den += w;
        }
        M = M_den > 0 ? M_num / M_den : 0.0;

        // --- S (weighted mean, conditioned on R) ---
        double S_num = 0.0, S_den = 0.0;
        for (int i = 0; i < (int)dens.S.size(); ++i) {
            if (std::isnan(x_S[i])) continue;
            double s = sample_S(x_S[i], R, dens.S[i], rng);
            double w = dens.S[i].weight;
            S_num += s * w;
            S_den += w;
        }
        S = S_den > 0 ? S_num / S_den : 0.0;
    }

    double midpoint   = P * (1.0 + M);
    double half_range = P * R / 2.0;

    // Clamp S to [-1, 1]: close must stay within [low, high].
    // S bins extend slightly beyond ±1 to avoid boundary pile-up,
    // so sampled S can be marginally outside this range.
    if (S >  1.0) S =  1.0;
    if (S < -1.0) S = -1.0;

    out_high  = midpoint + half_range;
    out_low   = midpoint - half_range;
    out_close = midpoint + S * half_range;
}

// ================================================================
// compute_x_values
//
// Compute all indicator x-values for one path's history.
// Global indicators are skipped — their values come from global_x.
// Fills x_R, x_M, x_S in-place.
// ================================================================
void compute_x_values(
    const double* hist_high,
    const double* hist_low,
    const double* hist_close,
    int hist_len,
    const DensitySet& dens,
    const std::vector<IndicatorSpec>& specs_R,
    const std::vector<IndicatorSpec>& specs_M,
    const std::vector<IndicatorSpec>& specs_S,
    const std::vector<double>& global_x,  // flat: [n_global_R, n_global_M, n_global_S]
    int n_global_R, int n_global_M,       // counts so we can index global_x
    std::vector<double>& x_R,
    std::vector<double>& x_M,
    std::vector<double>& x_S)
{
    int g_r = 0, g_m = 0, g_s = 0;
    const double NaN = std::numeric_limits<double>::quiet_NaN();

    for (int i = 0; i < (int)dens.R.size(); ++i) {
        if (dens.R[i].is_global) {
            int idx = g_r++;
            x_R[i] = idx < (int)global_x.size() ? global_x[idx] : NaN;
        } else {
            x_R[i] = compute_indicator(specs_R[i], hist_high, hist_low, hist_close, hist_len);
        }
    }

    for (int i = 0; i < (int)dens.M.size(); ++i) {
        if (dens.M[i].is_global) {
            int idx = n_global_R + g_m++;
            x_M[i] = idx < (int)global_x.size() ? global_x[idx] : NaN;
        } else {
            x_M[i] = compute_indicator(specs_M[i], hist_high, hist_low, hist_close, hist_len);
        }
    }

    for (int i = 0; i < (int)dens.S.size(); ++i) {
        if (dens.S[i].is_global) {
            int idx = n_global_R + n_global_M + g_s++;
            x_S[i] = idx < (int)global_x.size() ? global_x[idx] : NaN;
        } else {
            x_S[i] = compute_indicator(specs_S[i], hist_high, hist_low, hist_close, hist_len);
        }
    }
}

// ================================================================
// forecast_ticker
//
// Full MC forecast for one ticker — replaces _forecast_ticker body.
//
// Parameters
// ----------
// hist_high/low/close      : [hist_len]  starting price history
// hist_len                 : length of history arrays
// dens                     : packed density set
// specs_R/M/S              : indicator specs parallel to dens.R/M/S
// mc_samples               : number of MC paths
// hold_days                : forecast horizon
// target / stop            : as fractions (e.g. 0.05)
// global_x_per_day         : [hold_days × n_global_total]
//                            n_global_total = n_global_R + n_global_M + n_global_S
// n_global_R/M             : number of global indicators per component
// rng                      : seeded RNG for this rank/test
//
// Returns
// -------
// mean_return, std_return  (written to out_mean, out_std)
// ================================================================
void forecast_ticker(
    const double* hist_high,
    const double* hist_low,
    const double* hist_close,
    int hist_len,
    const DensitySet& dens,
    const std::vector<IndicatorSpec>& specs_R,
    const std::vector<IndicatorSpec>& specs_M,
    const std::vector<IndicatorSpec>& specs_S,
    int mc_samples,
    int hold_days,
    double target,
    double stop,
    const std::vector<double>& global_x_per_day,  // [hold_days × n_global_total]
    int n_global_R,
    int n_global_M,
    int n_global_S,
    std::mt19937& rng,
    double& out_mean,
    double& out_std,
    bool mixture_mode)
{
    int n_global_total = n_global_R + n_global_M + n_global_S;

    double initial_price = hist_close[hist_len - 1];
    double target_price  = initial_price * (1.0 + target);
    double stop_price    = initial_price * (1.0 - stop);

    int nR = (int)dens.R.size();
    int nM = (int)dens.M.size();
    int nS = (int)dens.S.size();

    // Per-path histories: each row is one path's history
    // Layout: [mc_samples × hist_len]
    std::vector<double> ph_close(mc_samples * hist_len);
    std::vector<double> ph_high (mc_samples * hist_len);
    std::vector<double> ph_low  (mc_samples * hist_len);

    for (int k = 0; k < mc_samples; ++k) {
        std::copy(hist_close, hist_close + hist_len, ph_close.data() + k * hist_len);
        std::copy(hist_high,  hist_high  + hist_len, ph_high.data()  + k * hist_len);
        std::copy(hist_low,   hist_low   + hist_len, ph_low.data()   + k * hist_len);
    }

    std::vector<double> prices    (mc_samples, initial_price);
    std::vector<double> exit_price(mc_samples, std::numeric_limits<double>::quiet_NaN());
    std::vector<bool>   active    (mc_samples, true);

    std::vector<double> closes_d(mc_samples);
    std::vector<double> highs_d (mc_samples);
    std::vector<double> lows_d  (mc_samples);

    // Reusable x-value buffers
    std::vector<double> x_R(nR), x_M(nM), x_S(nS);

    for (int d = 0; d < hold_days; ++d) {

        // Global x-values for this day.
        // Guard: if global_x_per_day doesn't have enough data
        // (e.g. empty array passed when globals exist in model),
        // fill gx_vec with NaN so global densities are skipped.
        std::vector<double> gx_vec;
        int expected_gx_size = n_global_total * hold_days;
        if (n_global_total > 0 &&
            (int)global_x_per_day.size() >= expected_gx_size) {
            const double* global_x_day =
                global_x_per_day.data() + d * n_global_total;
            gx_vec.assign(global_x_day, global_x_day + n_global_total);
        } else if (n_global_total > 0) {
            // Insufficient data — fill with NaN so globals are skipped
            gx_vec.assign(n_global_total,
                          std::numeric_limits<double>::quiet_NaN());
        }

        bool any_active = false;
        for (int k = 0; k < mc_samples; ++k) any_active |= active[k];
        if (!any_active) break;

        for (int k = 0; k < mc_samples; ++k) {
            if (!active[k]) continue;

            const double* hc = ph_close.data() + k * hist_len;
            const double* hh = ph_high.data()  + k * hist_len;
            const double* hl = ph_low.data()   + k * hist_len;

            compute_x_values(hh, hl, hc, hist_len,
                             dens, specs_R, specs_M, specs_S,
                             gx_vec, n_global_R, n_global_M,
                             x_R, x_M, x_S);

            step_from_x(x_R, x_M, x_S, dens, hc[hist_len - 1], rng,
                        closes_d[k], highs_d[k], lows_d[k], mixture_mode);
        }

        // Check stop / target
        for (int k = 0; k < mc_samples; ++k) {
            if (!active[k]) continue;

            bool hit_stop   = lows_d[k]  <= stop_price;
            bool hit_target = highs_d[k] >= target_price;

            if (hit_stop || hit_target) {
                exit_price[k] = hit_stop ? stop_price : target_price;
                // Both hit same day: stop wins (conservative)
                if (hit_stop && hit_target) exit_price[k] = stop_price;
                active[k] = false;
            } else {
                prices[k] = closes_d[k];
            }
        }

        // Roll histories forward for all paths
        // (including inactive ones — simpler than branching,
        //  and inactive paths won't be read again)
        for (int k = 0; k < mc_samples; ++k) {
            double* hc = ph_close.data() + k * hist_len;
            double* hh = ph_high.data()  + k * hist_len;
            double* hl = ph_low.data()   + k * hist_len;

            // Left-shift by 1: memmove handles overlapping regions correctly
            std::memmove(hc, hc + 1, (hist_len - 1) * sizeof(double));
            std::memmove(hh, hh + 1, (hist_len - 1) * sizeof(double));
            std::memmove(hl, hl + 1, (hist_len - 1) * sizeof(double));

            hc[hist_len - 1] = closes_d[k];
            hh[hist_len - 1] = highs_d[k];
            hl[hist_len - 1] = lows_d[k];
        }
    }

    // Paths that never exited get their final price
    for (int k = 0; k < mc_samples; ++k)
        if (active[k]) exit_price[k] = prices[k];

    // Compute mean and std of returns
    double sum = 0.0, sum2 = 0.0;
    for (int k = 0; k < mc_samples; ++k) {
        double r = (exit_price[k] - initial_price) / initial_price;
        sum  += r;
        sum2 += r * r;
    }

    double mean = sum / mc_samples;
    double var  = sum2 / mc_samples - mean * mean;
    out_mean = mean;
    out_std  = var > 0 ? std::sqrt(var) : 0.0;
}

// ================================================================
// evaluate_sample
//
// Evaluate one tuning sample — runs n_mc single-step forecasts
// from a fixed history and computes the MSE against real RMS.
// Replaces the MC block in rms_tuner._evaluate_sample().
//
// Returns the error (R, M, S, C squared deviations summed),
// or NaN if the sample should be skipped.
// ================================================================
std::array<double, 5> evaluate_sample(
    const double* hist_high,
    const double* hist_low,
    const double* hist_close,
    int hist_len,
    double real_R,
    double real_M,
    double real_S,
    double real_C,
    const DensitySet& dens,
    const std::vector<IndicatorSpec>& specs_R,
    const std::vector<IndicatorSpec>& specs_M,
    const std::vector<IndicatorSpec>& specs_S,
    int n_mc,
    const std::vector<double>& global_x,  // [n_global_total] for this day
    int n_global_R,
    int n_global_M,
    std::mt19937& rng,
    bool mixture_mode,
    double var_R,   // Var(R) across training samples — for normalization
    double var_M,   // Var(M)
    double var_S,   // Var(S)
    double var_C)   // Var(C/P)
{
    double P = hist_close[hist_len - 1];

    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(P) || P <= 0.0) return {nan, nan, nan, nan, nan};

    int nR = (int)dens.R.size();
    int nM = (int)dens.M.size();
    int nS = (int)dens.S.size();

    std::vector<double> x_R(nR), x_M(nM), x_S(nS);

    // x-values are the same for all paths (fixed history)
    // But we still compute them once here, not inside the path loop
    compute_x_values(hist_high, hist_low, hist_close, hist_len,
                     dens, specs_R, specs_M, specs_S,
                     global_x, n_global_R, n_global_M,
                     x_R, x_M, x_S);

    double sum_R = 0, sum_M = 0, sum_S = 0, sum_C = 0;

    double c_out, h_out, l_out;
    for (int k = 0; k < n_mc; ++k) {
        step_from_x(x_R, x_M, x_S, dens, P, rng, c_out, h_out, l_out, mixture_mode);
        double R_mc = (h_out - l_out) / P;
        double mid  = (h_out + l_out) / 2.0;
        double M_mc = (mid - P) / P;
        double half = (h_out - l_out) / 2.0;
        double S_mc = half > 0 ? (c_out - mid) / half : 0.0;

        sum_R += R_mc;
        sum_M += M_mc;
        sum_S += S_mc;
        sum_C += c_out;
    }

    double R_pred = sum_R / n_mc;
    double M_pred = sum_M / n_mc;
    double S_pred = sum_S / n_mc;
    double C_pred = sum_C / n_mc;

    // Normalize each squared error by component variance so all four
    // components contribute equally to the total score regardless of scale.
    // Safe division: if variance is zero or invalid, use raw squared error.
    auto safe_div = [](double sq_err, double var) -> double {
        return (std::isfinite(var) && var > 1e-12) ? sq_err / var : sq_err;
    };

    double R_err = safe_div((R_pred - real_R) * (R_pred - real_R), var_R);
    double M_err = safe_div((M_pred - real_M) * (M_pred - real_M), var_M);
    double S_err = safe_div((S_pred - real_S) * (S_pred - real_S), var_S);
    double C_err = safe_div(((C_pred - real_C) / P) * ((C_pred - real_C) / P), var_C);
    double total = R_err + M_err + S_err + C_err;

    return {
        std::isfinite(total) ? total : nan,
        std::isfinite(R_err) ? R_err : nan,
        std::isfinite(M_err) ? M_err : nan,
        std::isfinite(S_err) ? S_err : nan,
        std::isfinite(C_err) ? C_err : nan,
    };
}

// ================================================================
// step_ticker_one_day
//
// Run n_mc single-step MC samples for one ticker from a fixed history.
// Returns predicted (close, high, low) for all paths as flat vectors.
//
// Used by the Python day-by-day outer loop that recomputes global
// indicators from cross-ticker predicted prices after each step.
//
// Parameters
// ----------
// hist_high/low/close  : [hist_len]  current price history
// hist_len             : length of history arrays
// global_x             : [n_global_total] for today
// n_global_R/M         : counts for indexing global_x
// n_mc                 : number of paths to sample
// rng                  : seeded RNG
// out_close/high/low   : output vectors [n_mc] — written in place
// ================================================================
void step_ticker_one_day(
    const double* hist_high,
    const double* hist_low,
    const double* hist_close,
    int hist_len,
    const DensitySet& dens,
    const std::vector<IndicatorSpec>& specs_R,
    const std::vector<IndicatorSpec>& specs_M,
    const std::vector<IndicatorSpec>& specs_S,
    const std::vector<double>& global_x,
    int n_global_R,
    int n_global_M,
    int n_mc,
    std::mt19937& rng,
    bool mixture_mode,
    std::vector<double>& out_close,
    std::vector<double>& out_high,
    std::vector<double>& out_low)
{
    int nR = (int)dens.R.size();
    int nM = (int)dens.M.size();
    int nS = (int)dens.S.size();

    std::vector<double> x_R(nR), x_M(nM), x_S(nS);

    // x-values are fixed for all paths (same history)
    compute_x_values(hist_high, hist_low, hist_close, hist_len,
                     dens, specs_R, specs_M, specs_S,
                     global_x, n_global_R, n_global_M,
                     x_R, x_M, x_S);

    out_close.resize(n_mc);
    out_high.resize(n_mc);
    out_low.resize(n_mc);

    for (int k = 0; k < n_mc; ++k) {
        step_from_x(x_R, x_M, x_S, dens,
                    hist_close[hist_len - 1], rng,
                    out_close[k], out_high[k], out_low[k],
                    mixture_mode);
    }
}
