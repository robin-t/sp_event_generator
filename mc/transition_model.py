import numpy as np


class TransitionModel:
    """
    Multi-density RMS transition model.

    Supports mixture of densities for:
        R (range)
        M (midpoint shift)
        S (skew)

    Primary call path:
        compute_indicator_values()  →  step_from_x()

    simulate_many() is a convenience wrapper used by the tuner's
    density sanity checks — it runs n_paths single-step forecasts
    from the same starting history.
    """

    # ==========================================================
    # INIT
    # ==========================================================

    SAMPLING_MODES = ("weighted_mean", "mixture")

    def __init__(self, R_densities, M_densities, S_densities, weights,
                 sampling_mode="weighted_mean"):
        """
        Parameters
        ----------
        sampling_mode : str
            "weighted_mean" (default, backward compatible) — sample from every
            density and return the weight-averaged value. Smooths the mixture.

            "mixture" — normalize weights to probabilities, randomly select one
            density per component per step, sample purely from that density.
            Preserves the full shape of each individual density and produces
            heavier-tailed, less smeared forecasts. Statistically the correct
            way to sample from a mixture model.
        """
        self.R_densities = R_densities
        self.M_densities = M_densities
        self.S_densities = S_densities
        self.weights = weights
        if sampling_mode not in self.SAMPLING_MODES:
            raise ValueError(
                f"sampling_mode must be one of {self.SAMPLING_MODES}, "
                f"got '{sampling_mode}'"
            )
        self.sampling_mode = sampling_mode

    # ==========================================================
    # GENERIC DENSITY SAMPLERS
    # ==========================================================

    def _sample_2d(self, x, x_bins, prob, y_bins):
        ix = np.digitize(x, x_bins) - 1
        ix = np.clip(ix, 0, len(x_bins) - 2)

        row = prob[ix]
        if row.sum() == 0:
            return 0.0

        iy = np.random.choice(len(row), p=row)
        return np.random.uniform(y_bins[iy], y_bins[iy + 1])

    def _sample_3d(self, x, R, density):
        d = density

        ix = np.digitize(x, d.x_bins) - 1
        ir = np.digitize(R, d.R_bins) - 1
        ix = np.clip(ix, 0, len(d.x_bins) - 2)
        ir = np.clip(ir, 0, len(d.R_bins) - 2)

        row = d.S_prob[ix, ir]
        if row.sum() == 0:
            return 0.0

        is_bin = np.random.choice(len(row), p=row)
        return np.random.uniform(d.S_bins[is_bin], d.S_bins[is_bin + 1])

    def _sample_3d_M(self, x, R, density):
        d = density

        if getattr(d, "MR_prob", None) is None:
            return self._sample_2d(x, d.x_bins, d.M_prob, d.M_bins)

        ix = np.digitize(x, d.x_bins) - 1
        ir = np.digitize(R, d.R_bins) - 1
        ix = np.clip(ix, 0, len(d.x_bins) - 2)
        ir = np.clip(ir, 0, len(d.R_bins) - 2)

        row = d.MR_prob[ix, ir]
        if row.sum() == 0:
            return self._sample_2d(x, d.x_bins, d.M_prob, d.M_bins)

        im = np.random.choice(len(row), p=row)
        return np.random.uniform(d.M_bins[im], d.M_bins[im + 1])

    # ==========================================================
    # WEIGHT LOOKUP
    # ==========================================================

    def _weight_lookup(self, key):
        """
        Tolerates both prefixed (R_rsi_14) and unprefixed (rsi_14) key formats.
        Prefixed is canonical; unprefixed is a fallback for weight dicts built
        by older versions of tune_handler.
        """
        if key in self.weights:
            return self.weights[key]
        unprefixed = "_".join(key.split("_")[1:])
        if unprefixed in self.weights:
            return self.weights[unprefixed]
        raise KeyError(
            f"Weight missing for key '{key}' (also tried '{unprefixed}'). "
            f"Available weights: {list(self.weights.keys())}"
        )

    # ==========================================================
    # INDICATOR COMPUTATION
    # ==========================================================

    def compute_indicator_values(self, high_history, low_history, close_history,
                                 global_x=None, dates=None):
        """
        Compute per-ticker indicator x-values from one path's price history.
        Global indicators are skipped — their values come from global_x,
        precomputed once per day by the caller.

        Parameters
        ----------
        global_x : dict or None
            {key: float} for global indicators. Merged after local computation.
        dates : array or None
            Date objects matching the history arrays, for date-dependent indicators.
        """
        x_vals = {}

        for density, indicator in self.R_densities:
            key = f"R_{density.meta['indicator_name']}_{'_'.join(map(str, density.meta['indicator_params']))}"
            if not density.meta.get("is_global", False):
                x_vals[key] = indicator.compute(high_history, low_history, close_history, dates=dates)[-1]

        for density, indicator in self.M_densities:
            key = f"M_{density.meta['indicator_name']}_{'_'.join(map(str, density.meta['indicator_params']))}"
            if not density.meta.get("is_global", False):
                x_vals[key] = indicator.compute(high_history, low_history, close_history, dates=dates)[-1]

        for density, indicator in self.S_densities:
            key = f"S_{density.meta['indicator_name']}_{'_'.join(map(str, density.meta['indicator_params']))}"
            if not density.meta.get("is_global", False):
                x_vals[key] = indicator.compute(high_history, low_history, close_history, dates=dates)[-1]

        if global_x:
            x_vals.update(global_x)

        return x_vals

    # ==========================================================
    # SAMPLING KERNEL  ←  primary C++ migration target
    # ==========================================================

    def _collect_weighted(self, densities, component, x_vals):
        """
        Collect (density, x, weight) tuples for a component, skipping NaN x.
        Returns list of (density, x, weight) and sum of weights.
        """
        entries = []
        w_total = 0.0
        for density, indicator in densities:
            key = (f"{component}_{density.meta['indicator_name']}_"
                   f"{'_'.join(map(str, density.meta['indicator_params']))}")
            x = x_vals.get(key, np.nan)  # missing global keys → NaN → skipped
            if np.isnan(x):
                continue
            w = self._weight_lookup(key)
            entries.append((density, x, w))
            w_total += w
        return entries, w_total

    def step_from_x(self, x_vals, P):
        """
        Sample one day's close/high/low given pre-computed indicator values.

        This is the hot inner kernel and the primary target for C++ migration.
        Indicator computation is separated out so that:
          - Each MC path uses its own evolved history for non-global indicators
          - Global indicator values are injected once per day for all paths
          - The C++ port only needs to implement this function + the samplers

        Parameters
        ----------
        x_vals : dict  {key: float}
        P : float  last close price
        """
        if self.sampling_mode == "mixture":
            R = self._step_mixture_R(x_vals)
            M = self._step_mixture_M(x_vals, R)
            S = self._step_mixture_S(x_vals, R)
        else:
            R = self._step_weighted_mean_R(x_vals)
            M = self._step_weighted_mean_M(x_vals, R)
            S = self._step_weighted_mean_S(x_vals, R)

        midpoint   = P * (1 + M)
        half_range = P * R / 2

        # Clamp S to [-1, 1]: close must stay within [low, high].
        # S bins extend slightly beyond ±1 to avoid boundary pile-up.
        S = max(-1.0, min(1.0, S))

        return (
            midpoint + S * half_range,   # close
            midpoint + half_range,        # high
            midpoint - half_range,        # low
        )

    # ----------------------------------------------------------
    # weighted_mean mode helpers
    # ----------------------------------------------------------

    def _step_weighted_mean_R(self, x_vals):
        entries, w_total = self._collect_weighted(self.R_densities, "R", x_vals)
        if not entries:
            return 0.0
        return sum(
            self._sample_2d(x, d.x_bins, d.R_prob, d.R_bins) * w
            for d, x, w in entries
        ) / w_total

    def _step_weighted_mean_M(self, x_vals, R):
        entries, w_total = self._collect_weighted(self.M_densities, "M", x_vals)
        if not entries:
            return 0.0
        return sum(
            self._sample_3d_M(x, R, d) * w
            for d, x, w in entries
        ) / w_total

    def _step_weighted_mean_S(self, x_vals, R):
        entries, w_total = self._collect_weighted(self.S_densities, "S", x_vals)
        if not entries:
            return 0.0
        return sum(
            self._sample_3d(x, R, d) * w
            for d, x, w in entries
        ) / w_total

    # ----------------------------------------------------------
    # mixture mode helpers
    # ----------------------------------------------------------

    def _mixture_select(self, entries, w_total):
        """
        Randomly select one entry proportional to its weight.
        Returns (density, x).
        """
        probs = np.array([w for _, _, w in entries]) / w_total
        idx = np.random.choice(len(entries), p=probs)
        density, x, _ = entries[idx]
        return density, x

    def _step_mixture_R(self, x_vals):
        entries, w_total = self._collect_weighted(self.R_densities, "R", x_vals)
        if not entries:
            return 0.0
        d, x = self._mixture_select(entries, w_total)
        return self._sample_2d(x, d.x_bins, d.R_prob, d.R_bins)

    def _step_mixture_M(self, x_vals, R):
        entries, w_total = self._collect_weighted(self.M_densities, "M", x_vals)
        if not entries:
            return 0.0
        d, x = self._mixture_select(entries, w_total)
        return self._sample_3d_M(x, R, d)

    def _step_mixture_S(self, x_vals, R):
        entries, w_total = self._collect_weighted(self.S_densities, "S", x_vals)
        if not entries:
            return 0.0
        d, x = self._mixture_select(entries, w_total)
        return self._sample_3d(x, R, d)

    # ==========================================================
    # CONVENIENCE WRAPPER
    # ==========================================================

    def simulate_many(self, high_history, low_history, close_history, n_paths):
        """
        Run n_paths single-step forecasts from the same starting history.
        Used by the tuner. Routes through the canonical path.
        """
        x_vals = self.compute_indicator_values(high_history, low_history, close_history)
        P = close_history[-1]

        closes = np.zeros(n_paths)
        highs  = np.zeros(n_paths)
        lows   = np.zeros(n_paths)

        for k in range(n_paths):
            closes[k], highs[k], lows[k] = self.step_from_x(x_vals, P)

        return closes, highs, lows
