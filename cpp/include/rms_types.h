#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <random>

// ================================================================
// PackedDensity
//
// A density set for one indicator, packed into flat C++ vectors
// so the hot MC loop has no Python object access overhead.
// Built once from the Python RMSDensitySet at strategy/tune setup,
// then passed by const-ref into forecast_ticker / evaluate_sample.
//
// Mirrors RMSDensitySet's numpy arrays:
//   x_bins   [n_x+1]
//   R_bins   [n_R+1]
//   M_bins   [n_M+1]
//   S_bins   [n_S+1]   (fixed -1..1 grid, 100 bins)
//   R_prob   [n_x, n_R]   row-major, rows sum to 1
//   M_prob   [n_x, n_M]
//   MR_prob  [n_x, n_R, n_M]  optional — empty if not built
//   S_prob   [n_x, n_R, n_S]
// ================================================================

struct PackedDensity {
    // Bin edges
    std::vector<double> x_bins;
    std::vector<double> R_bins;
    std::vector<double> M_bins;
    std::vector<double> S_bins;

    // Probability tables (row-major)
    std::vector<double> R_prob;   // [n_x × n_R]
    std::vector<double> M_prob;   // [n_x × n_M]
    std::vector<double> MR_prob;  // [n_x × n_R × n_M]  may be empty
    std::vector<double> S_prob;   // [n_x × n_R × n_S]

    // Dimensions (number of bins, not edges)
    int n_x = 0;
    int n_R = 0;
    int n_M = 0;
    int n_S = 0;

    // Weight for this density in the mixture
    double weight = 1.0;

    // Is this a global (cross-sectional) indicator?
    // If true, x-value is injected externally — not computed from history.
    bool is_global = false;
};

// ================================================================
// DensitySet
//
// The three component density lists that the model uses.
// Each vector holds one PackedDensity per indicator in the mixture.
// ================================================================

struct DensitySet {
    std::vector<PackedDensity> R;
    std::vector<PackedDensity> M;
    std::vector<PackedDensity> S;
};

// ================================================================
// IndicatorType
//
// Identifies which indicator function to call.
// Matches INDICATOR_REGISTRY names in indicator.py.
// ================================================================

enum class IndicatorType {
    RSI,
    RSI_VELOCITY,
    VOL_RATIO,
    MACD,
    TREND_SLOPE,
    RANGE_POSITION,
    ATR_RATIO,
    RETURN_ND,
    HIGH_LOW_DIST,
    CLOSE_OPEN_RATIO,
    MEAN_S,
    GLOBAL,   // x-value injected externally, no compute needed
};

// ================================================================
// IndicatorSpec
//
// One indicator attached to a density — carries the type and
// integer params so the C++ compute functions can be called
// without any Python object access.
// ================================================================

struct IndicatorSpec {
    IndicatorType type = IndicatorType::GLOBAL;
    std::vector<int> params;   // e.g. {14} for RSI(14)
};
