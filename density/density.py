import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from density.indicator import Indicator


class _GlobalIndicatorPlaceholder:
    """
    Lightweight stand-in for GlobalIndicator when loading a saved global density.

    GlobalIndicator requires the full price matrix at construction time, which
    is not available during density load. This placeholder carries the name and
    params so that meta round-trips correctly. The strategy tester uses
    _precompute_global_x() to recompute the actual series from the test window
    data at runtime, so the placeholder's compute() is never called in the MC path.
    """

    def __init__(self, name, params):
        self.name   = name
        self.params = params

    def compute(self, high, low, close):
        raise RuntimeError(
            f"GlobalIndicator '{self.name}' placeholder cannot compute — "
            "global x-values must be injected via _precompute_global_x()."
        )

    def __repr__(self):
        return f"_GlobalIndicatorPlaceholder(name={self.name}, params={self.params})"


class DensitySet:
    """
    Builds and stores conditional densities:

        R density: f(R | x)
        M density: f(M | x)
        S density: f(S | x, R)

    NEW:
        Joint M-R density: f(M | x, R)
    """

    # ==========================================================
    # BIN DEFINITIONS
    # ==========================================================

    @staticmethod
    def quantile_bins(values, n_bins):
        values = np.asarray(values, dtype=float).ravel()
        # Clip at 1st/99th percentile before computing bin edges.
        # Prevents extreme events (e.g. 2008 crash) from stretching the
        # bin grid asymmetrically. Extreme samples still contribute to
        # the edge bins — only the grid geometry is protected.
        lo, hi = np.nanpercentile(values, [1, 99])
        values = np.clip(values, lo, hi)
        q = np.linspace(0, 1, n_bins)
        bins = np.quantile(values, q)
        return np.unique(bins)

    @staticmethod
    def skew_bins(n_bins=100):
        # Extend slightly beyond ±1 so that exact values of S=±1.0
        # (close at high or low exactly) fall in interior bins rather
        # than piling up at the boundary edges.
        return np.linspace(-1.05, 1.05, n_bins)

    # ==========================================================
    # INIT
    # ==========================================================

    def __init__(self, indicator=None):

        self.indicator = indicator

        self.x_bins = None
        self.R_bins = None
        self.M_bins = None
        self.S_bins = self.skew_bins()

        self.R_prob = None
        self.M_prob = None
        self.S_prob = None

        self.R_counts = None
        self.M_counts = None
        self.S_counts = None

        self.MR_prob = None
        self.MR_counts = None

        if indicator is not None:
            from density.indicator import GlobalIndicator
            self.meta = {
                "indicator_name": indicator.name,
                "indicator_params": indicator.params,
                "is_global": isinstance(indicator, GlobalIndicator),
            }
        else:
            self.meta = {}

    # ==========================================================
    # BUILD DENSITIES
    # ==========================================================

    def build(self, x_values, R_values, M_values, S_values,
            n_x_bins=60, n_R_bins=60, n_M_bins=60):

        print("\nBuilding RMS densities...")

        # ------------------------------------------------------
        # Binning
        # ------------------------------------------------------

        self.x_bins = self.quantile_bins(x_values, n_x_bins)
        self.R_bins = self.quantile_bins(R_values, n_R_bins)
        self.M_bins = self.quantile_bins(M_values, n_M_bins)

        # ------------------------------------------------------
        # Compute ALL indices once (critical)
        # ------------------------------------------------------

        x_idx = np.digitize(x_values, self.x_bins) - 1
        R_idx = np.digitize(R_values, self.R_bins) - 1
        M_idx = np.digitize(M_values, self.M_bins) - 1
        S_idx = np.digitize(S_values, self.S_bins) - 1

        valid = (
            (x_idx >= 0) & (x_idx < len(self.x_bins)-1) &
            (R_idx >= 0) & (R_idx < len(self.R_bins)-1)
        )

        # ------------------------------------------------------
        # R density
        # ------------------------------------------------------

        R_counts, _, _ = np.histogram2d(
            x_values, R_values,
            bins=[self.x_bins, self.R_bins]
        )

        self.R_counts = R_counts
        row_sums = R_counts.sum(axis=1, keepdims=True)
        self.R_prob = np.divide(R_counts, row_sums, where=row_sums > 0)

        # ------------------------------------------------------
        # M density (independent version)
        # ------------------------------------------------------

        M_counts, _, _ = np.histogram2d(
            x_values, M_values,
            bins=[self.x_bins, self.M_bins]
        )

        self.M_counts = M_counts
        row_sums = M_counts.sum(axis=1, keepdims=True)
        self.M_prob = np.divide(M_counts, row_sums, where=row_sums > 0)

        # ------------------------------------------------------
        # NEW: Joint M | (x, R)
        # ------------------------------------------------------

        self.MR_counts = np.zeros(
            (len(self.x_bins)-1,
            len(self.R_bins)-1,
            len(self.M_bins)-1)
        )

        valid_MR = (
            valid &
            (M_idx >= 0) & (M_idx < len(self.M_bins)-1)
        )

        for xi, ri, mi in zip(
            x_idx[valid_MR],
            R_idx[valid_MR],
            M_idx[valid_MR]
        ):
            self.MR_counts[xi, ri, mi] += 1

        self.MR_prob = np.zeros_like(self.MR_counts)

        for i in range(self.MR_counts.shape[0]):
            for j in range(self.MR_counts.shape[1]):
                total = self.MR_counts[i, j].sum()
                if total > 0:
                    self.MR_prob[i, j] = self.MR_counts[i, j] / total

        # ------------------------------------------------------
        # S density (existing logic)
        # ------------------------------------------------------

        self.S_counts = np.zeros(
            (len(self.x_bins)-1,
            len(self.R_bins)-1,
            len(self.S_bins)-1)
        )

        valid_S = (
            valid &
            (S_idx >= 0) & (S_idx < len(self.S_bins)-1)
        )

        for xi, ri, si in zip(
            x_idx[valid_S],
            R_idx[valid_S],
            S_idx[valid_S]
        ):
            self.S_counts[xi, ri, si] += 1

        self.S_prob = np.zeros_like(self.S_counts)

        for i in range(self.S_counts.shape[0]):
            for j in range(self.S_counts.shape[1]):
                total = self.S_counts[i, j].sum()
                if total > 0:
                    self.S_prob[i, j] = self.S_counts[i, j] / total

        print("RMS densities built successfully.")

    # ==========================================================
    # SAVE / LOAD
    # ==========================================================

    def save(self, folder):

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        np.savez(
            folder / "density.npz",
            x_bins=self.x_bins,
            R_bins=self.R_bins,
            M_bins=self.M_bins,
            S_bins=self.S_bins,
            R_prob=self.R_prob,
            M_prob=self.M_prob,
            S_prob=self.S_prob,
            R_counts=self.R_counts,
            M_counts=self.M_counts,
            S_counts=self.S_counts,
            MR_prob=self.MR_prob,
            MR_counts=self.MR_counts,
        )

        with open(folder / "meta.json", "w") as f:
            json.dump(self.meta, f, indent=2)

    @classmethod
    def load(cls, folder):

        folder = Path(folder)

        with open(folder / "meta.json") as f:
            meta = json.load(f)

        if meta.get("is_global", False):
            from density.indicator import GlobalIndicator
            # GlobalIndicator needs the price matrix to compute its series,
            # but at load time we don't have it. Create a placeholder that
            # carries the name/params — the strategy tester will recompute
            # the series fresh for each test window via _precompute_global_x.
            indicator = _GlobalIndicatorPlaceholder(
                meta["indicator_name"],
                meta["indicator_params"]
            )
        else:
            indicator = Indicator(
                meta["indicator_name"],
                meta["indicator_params"]
            )

        density = cls(indicator)

        data = np.load(folder / "density.npz", allow_pickle=False)

        for name in ["x_bins", "R_bins", "M_bins", "S_bins",
                     "R_prob", "M_prob", "S_prob",
                     "R_counts", "M_counts", "S_counts"]:
            if name in data:
                setattr(density, name, np.array(data[name], dtype=float))

        if "MR_prob" in data:
            density.MR_prob = np.array(data["MR_prob"], dtype=float)

        if "MR_counts" in data:
            density.MR_counts = np.array(data["MR_counts"], dtype=float)

        density.meta = meta

        return density
    
    # ==========================================================
    # ANALYSIS PLOTS
    # ==========================================================

    def analyze(self, folder_path):
        """
        Generate diagnostic plots for R, M, S densities.
        """

        indicator_label = f"{self.indicator.name.upper()} value"


        plot_dir = Path(folder_path) / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Compute marginal skew density
        S_marginal = np.zeros((self.S_prob.shape[0], self.S_prob.shape[2]))

        for i in range(self.S_prob.shape[0]):
            for j in range(self.S_prob.shape[1]):
                S_marginal[i] += self.R_prob[i, j] * self.S_prob[i, j]

        S_counts_marginal = np.sum(self.S_counts, axis=1)

        variables = {
            "R": (self.R_bins, self.R_prob, self.R_counts, "Range R (High - Low) [%]"),
            "M": (self.M_bins, self.M_prob, self.M_counts, "Midpoint Shift M [%]"),
            "S": (self.S_bins, S_marginal, S_counts_marginal, "Skew S (Close position)"),
        }


        x_centers = 0.5 * (self.x_bins[:-1] + self.x_bins[1:])

        for name, (y_bins, prob_grid, counts_grid, ylabel) in variables.items():

            is_percent = name in ["R", "M"]

            y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])

            if is_percent:
                y_centers *= 100

            mean = np.sum(prob_grid * y_centers, axis=1)

            var = np.sum(prob_grid * (y_centers**2), axis=1) - mean**2
            std = np.sqrt(np.maximum(var, 0))

            counts = counts_grid.sum(axis=1)
            uncertainty = std / np.sqrt(np.maximum(counts, 1))


            # --------------------------------------------------
            # Mean plot
            # --------------------------------------------------

            plt.figure(figsize=(9, 6))

            # Spread band
            plt.fill_between(
                x_centers,
                mean - std,
                mean + std,
                alpha=0.6,
                label="Spread (σ)"
            )

            # Statistical uncertainty
            plt.fill_between(
                x_centers,
                mean - uncertainty,
                mean + uncertainty,
                alpha=0.8,
                label="Statistical uncertainty (σ/√N)"
            )

            # Mean line
            plt.plot(x_centers, mean, linewidth=2, label="Conditional mean")

            # Zero reference line
            plt.axhline(0, color="black", linestyle="-", linewidth=1)

            plt.xlabel(indicator_label)
            plt.ylabel(ylabel)
            plt.title(f"{name} Conditional Mean vs Indicator")
            plt.legend()
            plt.grid(True)

            plt.savefig(plot_dir / f"{name}.png", dpi=150)
            plt.close()

            # --------------------------------------------------
            # Heatmap
            # --------------------------------------------------

            plt.figure(figsize=(8, 6))
            ymin = 100*y_bins[0] if is_percent else y_bins[0]
            ymax = 100*y_bins[-1] if is_percent else y_bins[-1]


            plt.imshow(
                prob_grid.T,
                origin="lower",
                aspect="auto",
                extent=[
                    self.x_bins[0],
                    self.x_bins[-1],
                    ymin,
                    ymax,
                ],
            )

            plt.colorbar(label="Probability density")

            plt.xlabel(indicator_label)
            plt.ylabel(ylabel)
            plt.title(f"{name} Conditional Density")

            plt.savefig(plot_dir / f"{name}_heatmap.png", dpi=150)
            plt.close()

        print("[Density] Analysis plots saved")

        # =====================================================
        # ADDITIONAL DIAGNOSTIC:
        #   S vs R (marginalized over indicator)
        # =====================================================

        print("[Density] Building S vs R diagnostics...")

        # --- Marginalize over indicator bins ---
        # shape: (R_bins, S_bins)
        SR_counts = self.S_counts.sum(axis=0)

        # Normalize to conditional probability f(S | R)
        SR_prob = np.zeros_like(SR_counts)

        row_sums = SR_counts.sum(axis=1, keepdims=True)
        valid = row_sums[:, 0] > 0

        SR_prob[valid] = SR_counts[valid] / row_sums[valid]

        # --- Bin centers ---
        R_centers = 100 * (0.5 * (self.R_bins[:-1] + self.R_bins[1:]))  # percent
        S_centers = 0.5 * (self.S_bins[:-1] + self.S_bins[1:])

        # =====================================================
        # Mean skew vs range
        # =====================================================

        mean = np.sum(SR_prob * S_centers, axis=1)

        var = np.sum(SR_prob * (S_centers**2), axis=1) - mean**2
        std = np.sqrt(np.maximum(var, 0))

        counts = SR_counts.sum(axis=1)
        uncertainty = std / np.sqrt(np.maximum(counts, 1))

        plt.figure(figsize=(9, 6))

        # Spread band
        plt.fill_between(
            R_centers,
            mean - std,
            mean + std,
            alpha=0.6,
            label="Spread (σ)"
        )

        # Statistical uncertainty band
        plt.fill_between(
            R_centers,
            mean - uncertainty,
            mean + uncertainty,
            alpha=0.8,
            label="Statistical uncertainty (σ/√N)"
        )

        # Mean line
        plt.plot(R_centers, mean, linewidth=2, label="Mean skew")

        # Zero reference
        plt.axhline(0, color="black", linewidth=1)

        plt.xlabel("Range R (%)")
        plt.ylabel("Skew S")
        plt.title("Skew vs Range (Indicator Averaged)")
        plt.legend()
        plt.grid(True)

        plt.savefig(plot_dir / "S_vs_R.png", dpi=150)
        plt.close()

        # -----------------------------------------------------
        # Zoomed version: 0–5% range
        # -----------------------------------------------------

        plt.figure(figsize=(9, 6))

        plt.fill_between(
            R_centers,
            mean - std,
            mean + std,
            alpha=0.6,
            label="Spread (σ)"
        )

        plt.fill_between(
            R_centers,
            mean - uncertainty,
            mean + uncertainty,
            alpha=0.8,
            label="Statistical uncertainty (σ/√N)"
        )

        plt.plot(R_centers, mean, linewidth=2, label="Mean skew")
        plt.axhline(0, color="black", linewidth=1)

        plt.xlim(0, 5)

        plt.xlabel("Range R (%)")
        plt.ylabel("Skew S")
        plt.title("Skew vs Range (Zoom 0–5%)")
        plt.legend()
        plt.grid(True)

        plt.savefig(plot_dir / "S_vs_R_zoom.png", dpi=150)
        plt.close()

        # =====================================================
        # Heatmap: f(S | R)
        # =====================================================

        plt.figure(figsize=(8, 6))

        plt.imshow(
            SR_prob.T,
            origin="lower",
            aspect="auto",
            extent=[
                R_centers[0],
                R_centers[-1],
                S_centers[0],
                S_centers[-1],
            ],
        )

        plt.colorbar(label="Probability density")

        plt.xlabel("Range R (%)")
        plt.ylabel("Skew S")
        plt.title("Conditional Density f(S | R)")

        plt.savefig(plot_dir / "S_vs_R_heatmap.png", dpi=150)
        plt.close()

        print("[Density] S vs R diagnostics saved")


        # =====================================================
        # NEW DIAGNOSTIC:
        #   M vs R (marginalized over indicator)
        # =====================================================

        print("[Density] Building M vs R diagnostics...")

        # --- Marginalize over indicator bins ---
        # shape: (R_bins, M_bins)
        MR_counts = self.MR_counts.sum(axis=0)

        # Normalize to conditional probability f(M | R)
        MR_prob = np.zeros_like(MR_counts)

        row_sums = MR_counts.sum(axis=1, keepdims=True)
        valid = row_sums[:, 0] > 0

        MR_prob[valid] = MR_counts[valid] / row_sums[valid]

        # --- Bin centers ---
        R_centers = 100 * (0.5 * (self.R_bins[:-1] + self.R_bins[1:]))  # percent
        M_centers = 100 * (0.5 * (self.M_bins[:-1] + self.M_bins[1:]))

        # =====================================================
        # Mean midpoint vs range
        # =====================================================

        mean = np.sum(MR_prob * M_centers, axis=1)

        var = np.sum(MR_prob * (M_centers**2), axis=1) - mean**2
        std = np.sqrt(np.maximum(var, 0))

        counts = MR_counts.sum(axis=1)
        uncertainty = std / np.sqrt(np.maximum(counts, 1))

        plt.figure(figsize=(9, 6))

        plt.fill_between(
            R_centers,
            mean - std,
            mean + std,
            alpha=0.6,
            label="Spread (σ)"
        )

        plt.fill_between(
            R_centers,
            mean - uncertainty,
            mean + uncertainty,
            alpha=0.8,
            label="Statistical uncertainty (σ/√N)"
        )

        plt.plot(R_centers, mean, linewidth=2, label="Mean midpoint")

        plt.axhline(0, color="black", linewidth=1)

        plt.xlabel("Range R (%)")
        plt.ylabel("Midpoint M (%)")
        plt.title("Midpoint vs Range (Indicator Averaged)")
        plt.legend()
        plt.grid(True)

        plt.savefig(plot_dir / "M_vs_R.png", dpi=150)
        plt.close()

        # -----------------------------------------------------
        # Zoomed version: 0–5% range
        # -----------------------------------------------------

        plt.figure(figsize=(9, 6))

        plt.fill_between(
            R_centers,
            mean - std,
            mean + std,
            alpha=0.6,
            label="Spread (σ)"
        )

        plt.fill_between(
            R_centers,
            mean - uncertainty,
            mean + uncertainty,
            alpha=0.8,
            label="Statistical uncertainty (σ/√N)"
        )

        plt.plot(R_centers, mean, linewidth=2, label="Mean midpoint")
        plt.axhline(0, color="black", linewidth=1)

        plt.xlim(0, 5)

        plt.xlabel("Range R (%)")
        plt.ylabel("Midpoint M (%)")
        plt.title("Midpoint vs Range (Zoom 0–5%)")
        plt.legend()
        plt.grid(True)

        plt.savefig(plot_dir / "M_vs_R_zoom.png", dpi=150)
        plt.close()

        # =====================================================
        # Heatmap: f(M | R)
        # =====================================================

        plt.figure(figsize=(8, 6))

        plt.imshow(
            MR_prob.T,
            origin="lower",
            aspect="auto",
            extent=[
                R_centers[0],
                R_centers[-1],
                M_centers[0],
                M_centers[-1],
            ],
        )

        plt.colorbar(label="Probability density")

        plt.xlabel("Range R (%)")
        plt.ylabel("Midpoint M (%)")
        plt.title("Conditional Density f(M | R)")

        plt.savefig(plot_dir / "M_vs_R_heatmap.png", dpi=150)
        plt.close()

        print("[Density] M vs R diagnostics saved")
    # ==========================================================
    # SUMMARY FOR ANALYSIS
    # ==========================================================

    def summarize(self, density_folder, summary_root="density/summaries"):
        """
        Write a compact JSON summary of this density to density/summaries/.
        Contains all statistics needed for external analysis — bin edges,
        marginal distributions, conditional means/stds, count tables,
        boundary diagnostics, and data quality metrics.

        The summary is self-contained: no need for the full density.npz.
        """
        import json as _json
        from pathlib import Path as _Path

        summary_dir = _Path(summary_root)
        summary_dir.mkdir(parents=True, exist_ok=True)

        density_name = _Path(density_folder).name
        out_path = summary_dir / f"{density_name}.json"

        x_centers = 0.5 * (self.x_bins[:-1] + self.x_bins[1:])
        R_centers = 0.5 * (self.R_bins[:-1] + self.R_bins[1:])
        M_centers = 0.5 * (self.M_bins[:-1] + self.M_bins[1:])
        S_centers = 0.5 * (self.S_bins[:-1] + self.S_bins[1:])

        # ── marginal counts per x-bin ──────────────────────────
        R_counts_marg = self.R_counts.sum(axis=1)   # [n_x]
        M_counts_marg = self.M_counts.sum(axis=1)   # [n_x]
        S_counts_marg = self.S_counts.sum(axis=(1, 2))  # [n_x] (marginalized over R too)

        total_R = float(self.R_counts.sum())
        total_M = float(self.M_counts.sum())
        total_S = float(self.S_counts.sum())

        # ── conditional means and stds ─────────────────────────
        # E[R | x], std[R | x]
        def cond_mean_std(prob, centers):
            # prob: [n_x, n_y], centers: [n_y]
            mean = (prob * centers[np.newaxis, :]).sum(axis=1)
            var  = (prob * (centers[np.newaxis, :] - mean[:, np.newaxis])**2).sum(axis=1)
            return mean.tolist(), np.sqrt(np.maximum(var, 0)).tolist()

        R_cond_mean, R_cond_std = cond_mean_std(self.R_prob, R_centers)
        M_cond_mean, M_cond_std = cond_mean_std(self.M_prob, M_centers)

        # S marginal over R: E[S | x] marginalizing over R
        S_marg_prob = np.zeros((self.S_prob.shape[0], self.S_prob.shape[2]))
        for i in range(self.S_prob.shape[0]):
            for j in range(self.S_prob.shape[1]):
                S_marg_prob[i] += self.R_prob[i, j] * self.S_prob[i, j]
        # renormalize rows
        row_sums = S_marg_prob.sum(axis=1, keepdims=True)
        S_marg_prob_norm = np.where(row_sums > 0, S_marg_prob / row_sums, 0.0)
        S_cond_mean, S_cond_std = cond_mean_std(S_marg_prob_norm, S_centers)

        # ── boundary diagnostics ───────────────────────────────
        # Fraction of counts in edge bins (first + last) vs interior
        def edge_fraction(counts_1d):
            total = counts_1d.sum()
            if total == 0: return 0.0, 0.0
            return float(counts_1d[0] / total), float(counts_1d[-1] / total)

        R_marg_over_x = self.R_counts.sum(axis=0)   # [n_R]
        M_marg_over_x = self.M_counts.sum(axis=0)   # [n_M]
        S_marg_over_xR = self.S_counts.sum(axis=(0, 1))  # [n_S]

        R_edge_lo, R_edge_hi = edge_fraction(R_marg_over_x)
        M_edge_lo, M_edge_hi = edge_fraction(M_marg_over_x)
        S_edge_lo, S_edge_hi = edge_fraction(S_marg_over_xR)

        # ── x coverage ────────────────────────────────────────
        x_counts_per_bin = R_counts_marg.tolist()
        empty_x_bins = int((np.array(x_counts_per_bin) == 0).sum())
        sparse_x_bins = int((np.array(x_counts_per_bin) < 10).sum())

        # ── S spike check: std of S bin-mass across all S bins ─
        # A spike will show as one bin having >> average mass
        s_bin_mass = S_marg_over_xR
        s_total = s_bin_mass.sum()
        s_expected = s_total / len(s_bin_mass) if len(s_bin_mass) > 0 else 1
        s_max_bin = float(s_bin_mass.max())
        s_spike_ratio = float(s_max_bin / s_expected) if s_expected > 0 else 0.0
        s_spike_bin_idx = int(s_bin_mass.argmax())

        # ── overall data quality ───────────────────────────────
        # Sparsity: fraction of (x,R) cells with < 5 S samples
        S_cell_counts = self.S_counts.sum(axis=2)  # [n_x, n_R]
        sparse_cells = int((S_cell_counts < 5).sum())
        total_cells  = int(S_cell_counts.size)

        summary = {
            # ── identity ──────────────────────────────────────
            "density_name":    density_name,
            "meta":            self.meta,

            # ── bin edges ─────────────────────────────────────
            "x_bins":  self.x_bins.tolist(),
            "R_bins":  self.R_bins.tolist(),
            "M_bins":  self.M_bins.tolist(),
            "S_bins":  self.S_bins.tolist(),

            # ── bin centers ───────────────────────────────────
            "x_centers": x_centers.tolist(),
            "R_centers": R_centers.tolist(),
            "M_centers": M_centers.tolist(),
            "S_centers": S_centers.tolist(),

            # ── total sample counts ───────────────────────────
            "total_samples_R": total_R,
            "total_samples_M": total_M,
            "total_samples_S": total_S,

            # ── counts per x-bin (tells us data density) ──────
            "R_counts_per_x":  R_counts_marg.tolist(),
            "M_counts_per_x":  M_counts_marg.tolist(),
            "S_counts_per_x":  S_counts_marg.tolist(),

            # ── conditional means and stds ────────────────────
            "R_cond_mean":  R_cond_mean,
            "R_cond_std":   R_cond_std,
            "M_cond_mean":  M_cond_mean,
            "M_cond_std":   M_cond_std,
            "S_cond_mean":  S_cond_mean,   # E[S|x] marginalized over R
            "S_cond_std":   S_cond_std,

            # ── marginal distributions (prob mass per bin) ─────
            "R_marginal":  (R_marg_over_x / R_marg_over_x.sum()).tolist()
                            if R_marg_over_x.sum() > 0 else [],
            "M_marginal":  (M_marg_over_x / M_marg_over_x.sum()).tolist()
                            if M_marg_over_x.sum() > 0 else [],
            "S_marginal":  (S_marg_over_xR / S_marg_over_xR.sum()).tolist()
                            if S_marg_over_xR.sum() > 0 else [],

            # ── boundary diagnostics ──────────────────────────
            # High edge fractions indicate boundary pile-up / clipping
            "R_edge_fraction_lo": R_edge_lo,
            "R_edge_fraction_hi": R_edge_hi,
            "M_edge_fraction_lo": M_edge_lo,
            "M_edge_fraction_hi": M_edge_hi,
            "S_edge_fraction_lo": S_edge_lo,
            "S_edge_fraction_hi": S_edge_hi,

            # ── S spike diagnostics ───────────────────────────
            # spike_ratio >> 1 means one bin dominates (boundary pile-up)
            "S_spike_ratio":    s_spike_ratio,
            "S_spike_bin_idx":  s_spike_bin_idx,
            "S_spike_bin_center": float(S_centers[s_spike_bin_idx])
                                  if s_spike_bin_idx < len(S_centers) else None,

            # ── x-bin coverage ────────────────────────────────
            "x_empty_bins":   empty_x_bins,
            "x_sparse_bins":  sparse_x_bins,
            "x_total_bins":   len(x_counts_per_bin),

            # ── cell sparsity (S conditioning quality) ────────
            "S_sparse_cells":     sparse_cells,
            "S_total_cells":      total_cells,
            "S_sparse_fraction":  sparse_cells / total_cells if total_cells > 0 else 1.0,

            # ── full S probability table (x, S) marginalised ──
            # Shape [n_x, n_S] — allows full reconstruction of
            # E[S|x], heatmaps, spike analysis etc.
            "S_prob_marginal_xS":  S_marg_prob_norm.tolist(),

            # ── full R prob table [n_x, n_R] ──────────────────
            "R_prob_table":  self.R_prob.tolist(),

            # ── full M prob table [n_x, n_M] ──────────────────
            "M_prob_table":  self.M_prob.tolist(),
        }

        with open(out_path, "w") as f:
            _json.dump(summary, f, separators=(",", ":"))

        print(f"[Density] Summary written to {out_path}")
        return out_path

