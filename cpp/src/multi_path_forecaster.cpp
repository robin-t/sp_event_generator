#include "rms_types.h"
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <stdexcept>

// ----------------------------------------------------------------
// Forward declarations from other translation units
// ----------------------------------------------------------------
double compute_indicator(const IndicatorSpec& spec,
                         const double* high, const double* low,
                         const double* close, int n);

double sample_R(double x, const PackedDensity& d, std::mt19937& rng);
double sample_M(double x, double R, const PackedDensity& d, std::mt19937& rng);
double sample_S(double x, double R, const PackedDensity& d, std::mt19937& rng);

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
    bool mixture_mode);

void compute_x_values(
    const double* hist_high,
    const double* hist_low,
    const double* hist_close,
    int hist_len,
    const DensitySet& dens,
    const std::vector<IndicatorSpec>& specs_R,
    const std::vector<IndicatorSpec>& specs_M,
    const std::vector<IndicatorSpec>& specs_S,
    const std::vector<double>& global_x,
    int n_global_R, int n_global_M,
    std::vector<double>& x_R,
    std::vector<double>& x_M,
    std::vector<double>& x_S);

// ================================================================
// GlobalState
//
// Cross-sectional statistics computed from mean predicted prices
// across all tickers after each MC step. These become the global
// indicator x-values for the next day's forecast.
//
// Computed indicators:
//   market_mean_R      = mean(R_i)  across tickers
//   market_mean_M      = mean(M_i)
//   market_mean_S      = mean(S_i)
//   market_vol_disp    = std(R_i)
//   day_of_week        = 0..4 (Mon..Fri), from weekdays array
// ================================================================
struct GlobalState {
    double market_mean_R    = std::numeric_limits<double>::quiet_NaN();
    double market_mean_M    = std::numeric_limits<double>::quiet_NaN();
    double market_mean_S    = std::numeric_limits<double>::quiet_NaN();
    double market_vol_disp  = std::numeric_limits<double>::quiet_NaN();
    double day_of_week      = std::numeric_limits<double>::quiet_NaN();
};

// ================================================================
// compute_global_state
//
// Given mean (close, high, low, prev_close) per ticker,
// compute cross-sectional RMS statistics for use as global x-values.
// ================================================================
static GlobalState compute_global_state(
    const std::vector<double>& mean_close,   // [n_tickers]
    const std::vector<double>& mean_high,
    const std::vector<double>& mean_low,
    const std::vector<double>& prev_close,   // close from day before
    double weekday)                          // 0=Mon..4=Fri, NaN if unknown
{
    int n = (int)mean_close.size();
    GlobalState gs;
    gs.day_of_week = weekday;

    if (n == 0) return gs;

    std::vector<double> R_vals, M_vals, S_vals;
    R_vals.reserve(n); M_vals.reserve(n); S_vals.reserve(n);

    for (int i = 0; i < n; ++i) {
        double P = prev_close[i];
        double H = mean_high[i];
        double L = mean_low[i];
        double C = mean_close[i];

        if (!std::isfinite(P) || P <= 0.0) continue;
        if (!std::isfinite(H) || !std::isfinite(L) || !std::isfinite(C)) continue;
        if (H <= L) continue;

        double R = (H - L) / P;
        double mid = (H + L) / 2.0;
        double M = (mid - P) / P;
        double half = (H - L) / 2.0;
        double S = (C - mid) / half;

        if (!std::isfinite(R) || !std::isfinite(M) || !std::isfinite(S)) continue;
        if (std::abs(S) > 1.05) continue;

        R_vals.push_back(R);
        M_vals.push_back(M);
        S_vals.push_back(S);
    }

    if (R_vals.empty()) return gs;

    // mean R, M, S
    auto mean_vec = [](const std::vector<double>& v) {
        double s = 0;
        for (double x : v) s += x;
        return s / v.size();
    };

    gs.market_mean_R = mean_vec(R_vals);
    gs.market_mean_M = mean_vec(M_vals);
    gs.market_mean_S = mean_vec(S_vals);

    // std of R
    double mr = gs.market_mean_R;
    double var = 0;
    for (double r : R_vals) var += (r - mr) * (r - mr);
    gs.market_vol_disp = std::sqrt(var / R_vals.size());

    return gs;
}

// ================================================================
// build_global_x
//
// Convert a GlobalState into a flat global_x vector in the order
// expected by compute_x_values: [global_R values, global_M values,
// global_S values], matching the order densities appear in the model.
//
// indicator_names_R/M/S: the indicator name for each global density
// in order — used to map GlobalState fields to positions.
// ================================================================
static std::vector<double> build_global_x(
    const GlobalState& gs,
    const std::vector<std::string>& global_names_R,
    const std::vector<std::string>& global_names_M,
    const std::vector<std::string>& global_names_S)
{
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> gx;

    auto push_named = [&](const std::vector<std::string>& names) {
        for (const auto& name : names) {
            if      (name == "market_mean_R")        gx.push_back(gs.market_mean_R);
            else if (name == "market_mean_M")        gx.push_back(gs.market_mean_M);
            else if (name == "market_mean_S")        gx.push_back(gs.market_mean_S);
            else if (name == "market_vol_dispersion")gx.push_back(gs.market_vol_disp);
            else if (name == "day_of_week")          gx.push_back(gs.day_of_week);
            else                                     gx.push_back(NaN);
        }
    };

    push_named(global_names_R);
    push_named(global_names_M);
    push_named(global_names_S);

    return gx;
}

// ================================================================
// MultiPathForecaster
//
// Holds N_mc independent price paths for multiple tickers.
// Runs day-by-day MC simulation where global indicators are
// recomputed each day from mean predicted prices across all tickers.
//
// Usage
// -----
//   forecaster.init(histories, weekdays, target, stop)
//   for d in range(hold_days):
//       forecaster.step()
//   returns = forecaster.get_returns()
// ================================================================
struct MultiPathForecaster {

    // Model
    const DensitySet*                dens    = nullptr;
    const std::vector<IndicatorSpec>* specs_R = nullptr;
    const std::vector<IndicatorSpec>* specs_M = nullptr;
    const std::vector<IndicatorSpec>* specs_S = nullptr;
    bool mixture_mode = false;
    int  n_global_R = 0;
    int  n_global_M = 0;
    int  n_global_S = 0;

    // Global indicator name lists (for build_global_x)
    std::vector<std::string> global_names_R;
    std::vector<std::string> global_names_M;
    std::vector<std::string> global_names_S;

    // Per-ticker state
    int n_tickers = 0;
    int n_mc      = 0;
    int hist_len  = 0;

    // Path histories: [n_tickers × n_mc × hist_len] flat
    std::vector<double> ph_close;
    std::vector<double> ph_high;
    std::vector<double> ph_low;

    // Path prices & exit tracking
    std::vector<double> prices;      // [n_tickers × n_mc] current close
    std::vector<double> exit_price;  // [n_tickers × n_mc]
    std::vector<char>    active;  // 0=inactive 1=active (avoid vector<bool> bit-packing)      // [n_tickers × n_mc]

    double target_price_factor = 1.05;
    double stop_price_factor   = 0.95;

    // Initial prices per ticker
    std::vector<double> initial_price;   // [n_tickers]
    std::vector<double> target_price_v;  // [n_tickers]
    std::vector<double> stop_price_v;    // [n_tickers]

    // Weekdays for forecast window [hold_days]
    std::vector<double> weekdays;

    // RNG
    std::mt19937 rng;

    // Current day index
    int current_day = 0;

    // Previous close per ticker (for global computation)
    std::vector<double> prev_mean_close;  // [n_tickers]

    // -----------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------

    // Index into [n_tickers × n_mc × hist_len]
    double* hist_ptr(std::vector<double>& buf, int ticker, int path) {
        return buf.data() + (ticker * n_mc + path) * hist_len;
    }
    const double* hist_ptr(const std::vector<double>& buf,
                            int ticker, int path) const {
        return buf.data() + (ticker * n_mc + path) * hist_len;
    }

    // Index into [n_tickers × n_mc]
    double& price_ref(std::vector<double>& buf, int ticker, int path) {
        return buf[ticker * n_mc + path];
    }
    char& active_ref(int ticker, int path) {
        return active[ticker * n_mc + path];
    }

    // -----------------------------------------------------------
    // init
    //
    // histories: flat [n_tickers × hist_len] for close/high/low
    // weekdays_:  [hold_days] weekday integers (0=Mon..4=Fri),
    //             NaN if not available
    // -----------------------------------------------------------
    void init(
        const std::vector<double>& hist_close_flat,
        const std::vector<double>& hist_high_flat,
        const std::vector<double>& hist_low_flat,
        int n_tickers_, int hist_len_, int n_mc_,
        double target, double stop,
        const std::vector<double>& weekdays_,
        int seed)
    {
        n_tickers = n_tickers_;
        hist_len  = hist_len_;
        n_mc      = n_mc_;
        rng.seed(seed);
        weekdays  = weekdays_;
        current_day = 0;

        target_price_factor = 1.0 + target;
        stop_price_factor   = 1.0 - stop;

        int total = n_tickers * n_mc * hist_len;
        ph_close.resize(total);
        ph_high.resize(total);
        ph_low.resize(total);

        prices.resize(n_tickers * n_mc);
        exit_price.assign(n_tickers * n_mc,
                          std::numeric_limits<double>::quiet_NaN());
        active.assign(n_tickers * n_mc, 1);

        initial_price.resize(n_tickers);
        target_price_v.resize(n_tickers);
        stop_price_v.resize(n_tickers);
        prev_mean_close.resize(n_tickers);

        for (int t = 0; t < n_tickers; ++t) {
            const double* src_c = hist_close_flat.data() + t * hist_len;
            const double* src_h = hist_high_flat.data()  + t * hist_len;
            const double* src_l = hist_low_flat.data()   + t * hist_len;

            double P = src_c[hist_len - 1];
            initial_price[t]  = P;
            target_price_v[t] = P * target_price_factor;
            stop_price_v[t]   = P * stop_price_factor;
            prev_mean_close[t] = P;

            // Tile history across all mc paths
            for (int k = 0; k < n_mc; ++k) {
                std::copy(src_c, src_c + hist_len, hist_ptr(ph_close, t, k));
                std::copy(src_h, src_h + hist_len, hist_ptr(ph_high,  t, k));
                std::copy(src_l, src_l + hist_len, hist_ptr(ph_low,   t, k));
                price_ref(prices, t, k) = P;
            }
        }
    }

    // -----------------------------------------------------------
    // compute_current_global_state
    //
    // Uses prev_mean_close as the "previous close" to compute
    // R/M/S relative to the day before.
    // -----------------------------------------------------------
    GlobalState compute_current_global_state() {
        std::vector<double> mean_c(n_tickers, 0.0);
        std::vector<double> mean_h(n_tickers, 0.0);
        std::vector<double> mean_l(n_tickers, 0.0);

        // Only average over ACTIVE paths — exited paths represent a trading
        // decision, not market evolution. Frozen paths should not influence
        // the global state that conditions still-active tickers.
        for (int t = 0; t < n_tickers; ++t) {
            double sc = 0, sh = 0, sl = 0;
            int cnt = 0;
            for (int k = 0; k < n_mc; ++k) {
                if (!active_ref(t, k)) continue;   // skip exited paths
                const double* hc = hist_ptr(ph_close, t, k);
                const double* hh = hist_ptr(ph_high,  t, k);
                const double* hl = hist_ptr(ph_low,   t, k);
                sc += hc[hist_len - 1];
                sh += hh[hist_len - 1];
                sl += hl[hist_len - 1];
                ++cnt;
            }
            if (cnt > 0) {
                mean_c[t] = sc / cnt;
                mean_h[t] = sh / cnt;
                mean_l[t] = sl / cnt;
            } else {
                // All paths for this ticker have exited — use NaN so
                // compute_global_state skips this ticker entirely.
                mean_c[t] = std::numeric_limits<double>::quiet_NaN();
                mean_h[t] = std::numeric_limits<double>::quiet_NaN();
                mean_l[t] = std::numeric_limits<double>::quiet_NaN();
            }
        }

        double dow = (current_day < (int)weekdays.size())
                     ? weekdays[current_day]
                     : std::numeric_limits<double>::quiet_NaN();

        return compute_global_state(mean_c, mean_h, mean_l,
                                    prev_mean_close, dow);
    }

    // -----------------------------------------------------------
    // step
    //
    // Advance all paths by one day.
    // Computes global state from current mean prices, then steps
    // each path forward using per-path history.
    // -----------------------------------------------------------
    void step() {
        // Build global x from current state
        GlobalState gs = compute_current_global_state();
        std::vector<double> gx = build_global_x(
            gs, global_names_R, global_names_M, global_names_S);

        int nR = (int)dens->R.size();
        int nM = (int)dens->M.size();
        int nS = (int)dens->S.size();

        std::vector<double> x_R(nR), x_M(nM), x_S(nS);

        for (int t = 0; t < n_tickers; ++t) {
            double tp = target_price_v[t];
            double sp = stop_price_v[t];

            double sum_close = 0.0;
            int    n_active  = 0;

            for (int k = 0; k < n_mc; ++k) {
                if (!active_ref(t, k)) continue;

                const double* hc = hist_ptr(ph_close, t, k);
                const double* hh = hist_ptr(ph_high,  t, k);
                const double* hl = hist_ptr(ph_low,   t, k);
                double P = hc[hist_len - 1];

                // Compute x-values from this path's history
                compute_x_values(hh, hl, hc, hist_len,
                                  *dens, *specs_R, *specs_M, *specs_S,
                                  gx, n_global_R, n_global_M,
                                  x_R, x_M, x_S);

                double c_out, h_out, l_out;
                step_from_x(x_R, x_M, x_S, *dens, P, rng,
                             c_out, h_out, l_out, mixture_mode);

                // Check exits
                bool hit_stop   = (l_out <= sp);
                bool hit_target = (h_out >= tp);

                if (hit_stop || hit_target) {
                    exit_price[t * n_mc + k] =
                        (hit_stop && hit_target) ? sp :
                        hit_stop                 ? sp : tp;
                    active_ref(t, k) = 0;
                } else {
                    price_ref(prices, t, k) = c_out;
                    sum_close += c_out;
                    ++n_active;
                }

                // Roll history forward
                double* wc = hist_ptr(ph_close, t, k);
                double* wh = hist_ptr(ph_high,  t, k);
                double* wl = hist_ptr(ph_low,   t, k);
                std::memmove(wc, wc + 1, (hist_len - 1) * sizeof(double));
                std::memmove(wh, wh + 1, (hist_len - 1) * sizeof(double));
                std::memmove(wl, wl + 1, (hist_len - 1) * sizeof(double));
                wc[hist_len - 1] = c_out;
                wh[hist_len - 1] = h_out;
                wl[hist_len - 1] = l_out;
            }

            // Update prev_mean_close for next step's global computation.
            // If all paths exited, use mean exit price (not initial_price)
            // so the global state reflects where this ticker actually went.
            if (n_active > 0) {
                prev_mean_close[t] = sum_close / n_active;
            } else {
                double sum_exit = 0.0;
                int n_exit = 0;
                for (int k = 0; k < n_mc; ++k) {
                    double ep = exit_price[t * n_mc + k];
                    if (std::isfinite(ep)) { sum_exit += ep; ++n_exit; }
                }
                prev_mean_close[t] = (n_exit > 0)
                    ? sum_exit / n_exit
                    : initial_price[t];
            }
        }

        ++current_day;
    }

    // -----------------------------------------------------------
    // run
    //
    // Execute the full forecast in one call: step hold_days times.
    // Python calls this once instead of looping over step().
    // -----------------------------------------------------------
    void run(int hold_days) {
        for (int d = 0; d < hold_days; ++d)
            step();
    }

    // -----------------------------------------------------------
    // get_returns
    //
    // Returns mean and std of returns for each ticker.
    // flat vector: [n_tickers × 2] = [mean0, std0, mean1, std1, ...]
    // -----------------------------------------------------------
    std::vector<double> get_returns() {
        std::vector<double> result(n_tickers * 2);

        for (int t = 0; t < n_tickers; ++t) {
            double P0 = initial_price[t];
            double sum = 0.0, sum2 = 0.0;

            for (int k = 0; k < n_mc; ++k) {
                double ep = active_ref(t, k)
                    ? price_ref(prices, t, k)
                    : exit_price[t * n_mc + k];
                double r = (ep - P0) / P0;
                sum  += r;
                sum2 += r * r;
            }

            double mean = sum / n_mc;
            double var  = sum2 / n_mc - mean * mean;
            result[t * 2 + 0] = mean;
            result[t * 2 + 1] = var > 0 ? std::sqrt(var) : 0.0;
        }

        return result;
    }
};
