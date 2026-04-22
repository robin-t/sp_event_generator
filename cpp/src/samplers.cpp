#include "rms_types.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ================================================================
// All samplers take a thread-local mt19937 RNG passed by reference
// so the caller controls seeding and each MPI rank / thread has
// its own independent stream.
// ================================================================

// ----------------------------------------------------------------
// Categorical sample from a probability row.
// Returns the chosen bin index, or -1 if all weights are zero.
// ----------------------------------------------------------------
static int categorical(const double* probs, int n, std::mt19937& rng)
{
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += probs[i];
    if (sum < 1e-12) return -1;

    std::uniform_real_distribution<double> u(0.0, sum);
    double r = u(rng);
    double cum = 0.0;
    for (int i = 0; i < n; ++i) {
        cum += probs[i];
        if (r <= cum) return i;
    }
    return n - 1;  // rounding fallback
}

// ----------------------------------------------------------------
// Bin lookup: find index such that bins[idx] <= x < bins[idx+1]
// Clipped to [0, n_bins-2].
// ----------------------------------------------------------------
static int find_bin(double x, const double* bins, int n_edges)
{
    // Binary search for efficiency on large bin arrays
    int lo = 0, hi = n_edges - 2;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (bins[mid] <= x) lo = mid;
        else                hi = mid - 1;
    }
    return lo;
}

// ================================================================
// sample_2d
// Samples y from f(y | x) using the 2D probability table.
// Used for R given indicator x.
// ================================================================
double sample_2d(double x,
                 const PackedDensity& d,
                 const double* y_bins, int n_y_edges,
                 const double* prob,   // [n_x × n_y]
                 int n_y,
                 std::mt19937& rng)
{
    int ix = find_bin(x, d.x_bins.data(), (int)d.x_bins.size());

    const double* row = prob + ix * n_y;
    int iy = categorical(row, n_y, rng);
    if (iy < 0) return 0.0;

    std::uniform_real_distribution<double> u(y_bins[iy], y_bins[iy + 1]);
    return u(rng);
}

// ================================================================
// sample_R
// f(R | x)
// ================================================================
double sample_R(double x, const PackedDensity& d, std::mt19937& rng)
{
    return sample_2d(x,
                     d,
                     d.R_bins.data(), (int)d.R_bins.size(),
                     d.R_prob.data(), d.n_R,
                     rng);
}

// ================================================================
// sample_M
// f(M | x, R)  — uses joint MR density if available,
//                falls back to f(M | x) if not.
// ================================================================
double sample_M(double x, double R, const PackedDensity& d, std::mt19937& rng)
{
    if (d.MR_prob.empty()) {
        // Fallback: f(M | x) independent of R
        return sample_2d(x,
                         d,
                         d.M_bins.data(), (int)d.M_bins.size(),
                         d.M_prob.data(), d.n_M,
                         rng);
    }

    int ix = find_bin(x, d.x_bins.data(), (int)d.x_bins.size());
    int ir = find_bin(R, d.R_bins.data(), (int)d.R_bins.size());

    // MR_prob layout: [n_x × n_R × n_M]
    const double* row = d.MR_prob.data() + (ix * d.n_R + ir) * d.n_M;

    int im = categorical(row, d.n_M, rng);
    if (im < 0) {
        // Empty bin fallback to marginal f(M | x)
        return sample_2d(x,
                         d,
                         d.M_bins.data(), (int)d.M_bins.size(),
                         d.M_prob.data(), d.n_M,
                         rng);
    }

    std::uniform_real_distribution<double> u(d.M_bins[im], d.M_bins[im + 1]);
    return u(rng);
}

// ================================================================
// sample_S
// f(S | x, R)  — always uses joint density
// ================================================================
double sample_S(double x, double R, const PackedDensity& d, std::mt19937& rng)
{
    int ix = find_bin(x, d.x_bins.data(), (int)d.x_bins.size());
    int ir = find_bin(R, d.R_bins.data(), (int)d.R_bins.size());

    // S_prob layout: [n_x × n_R × n_S]
    const double* row = d.S_prob.data() + (ix * d.n_R + ir) * d.n_S;

    int is = categorical(row, d.n_S, rng);
    if (is < 0) return 0.0;

    std::uniform_real_distribution<double> u(d.S_bins[is], d.S_bins[is + 1]);
    return u(rng);
}
