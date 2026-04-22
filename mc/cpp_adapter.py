"""
mc/cpp_adapter.py

Thin Python layer between the existing strategy/tuner code and the
rms_cpp C++ module.  Responsible for:

  1. Building a PackedModel once from the Python density lists and weights
  2. Preparing global_x arrays in the shape the C++ kernel expects
  3. Providing drop-in replacements for _forecast_ticker and _evaluate_sample
     that the strategy tester and tuner can call with minimal changes

Usage
-----
from mc.cpp_adapter import CppAdapter

# Build once per model config (not per ticker/sample)
adapter = CppAdapter(R_dens, M_dens, S_dens, weights)

# In strategy tester (replaces _forecast_ticker body):
mean, std = adapter.forecast_ticker(
    hist_high, hist_low, hist_close,
    mc_samples, hold_days,
    target, stop,
    global_x_per_day,   # list of dicts from _precompute_global_x
    seed,
)

# In tuner (replaces MC block in _evaluate_sample):
err = adapter.evaluate_sample(
    hist_high, hist_low, hist_close,
    real_R, real_M, real_S, real_C,
    n_mc,
    global_x,           # dict from _global_series lookup
    seed,
)
"""

import numpy as np

try:
    import rms_cpp
    _CPP_AVAILABLE = True
    print("✓ rms_cpp C++ module loaded")
except ImportError:
    _CPP_AVAILABLE = False
    print("⚠ rms_cpp C++ module not found — falling back to Python kernel")


def cpp_available() -> bool:
    return _CPP_AVAILABLE


class CppAdapter:
    """
    Wraps a rms_cpp.PackedModel and provides forecast_ticker /
    evaluate_sample with the same signatures used by the Python callers.

    Parameters
    ----------
    R_dens, M_dens, S_dens : list of (DensitySet, Indicator)
        Same format as TransitionModel / StrategyTester attributes.
    weights : dict
        Same format as TransitionModel.weights.
    """

    def __init__(self, R_dens, M_dens, S_dens, weights, mixture_mode=False):
        if not _CPP_AVAILABLE:
            raise RuntimeError(
                "rms_cpp C++ module not found. "
                "Build it with:  cd cpp && ./build.sh"
            )

        # Build the list format expected by PackedModel:
        #   [(DensitySet, name, params_as_ints, is_global), ...]
        def make_list(dens_list):
            out = []
            for density, indicator in dens_list:
                name      = density.meta["indicator_name"]
                params    = [int(p) for p in density.meta["indicator_params"]]
                is_global = bool(density.meta.get("is_global", False))
                out.append((density, name, params, is_global))
            return out

        self._model = rms_cpp.PackedModel(
            make_list(R_dens),
            make_list(M_dens),
            make_list(S_dens),
            weights,
        )

        # Store the key ordering for global indicators so we can
        # convert dict-of-floats → flat numpy array consistently.
        self._global_keys_R = [
            _make_key("R", d, ind)
            for d, ind in R_dens
            if d.meta.get("is_global", False)
        ]
        self._global_keys_M = [
            _make_key("M", d, ind)
            for d, ind in M_dens
            if d.meta.get("is_global", False)
        ]
        self._global_keys_S = [
            _make_key("S", d, ind)
            for d, ind in S_dens
            if d.meta.get("is_global", False)
        ]
        self._global_keys = (
            self._global_keys_R + self._global_keys_M + self._global_keys_S
        )

        # Ordered global indicator names per component (for MultiPathForecaster)
        self._global_names_R = [
            d.meta["indicator_name"]
            for d, ind in R_dens
            if d.meta.get("is_global", False)
        ]
        self._global_names_M = [
            d.meta["indicator_name"]
            for d, ind in M_dens
            if d.meta.get("is_global", False)
        ]
        self._global_names_S = [
            d.meta["indicator_name"]
            for d, ind in S_dens
            if d.meta.get("is_global", False)
        ]
        self._n_global = len(self._global_keys)
        self._mixture_mode = mixture_mode
        self._var_R = 1.0
        self._var_M = 1.0
        self._var_S = 1.0
        self._var_C = 1.0

    def set_variances(self, var_R: float, var_M: float,
                      var_S: float, var_C: float):
        """
        Set per-component variances for normalized evaluate_sample errors.
        Call once after computing variances from training samples.
        Defaults to 1.0 (no normalization) until set.
        """
        self._var_R = var_R
        self._var_M = var_M
        self._var_S = var_S
        self._var_C = var_C

    # ------------------------------------------------------------------
    # forecast_ticker
    # ------------------------------------------------------------------

    def forecast_ticker(
        self,
        hist_high: np.ndarray,
        hist_low: np.ndarray,
        hist_close: np.ndarray,
        mc_samples: int,
        hold_days: int,
        target: float,
        stop: float,
        global_x_per_day: list,   # list of dicts, length hold_days
        seed: int,
    ):
        """
        Returns (mean_return, std_return).

        global_x_per_day is the output of _precompute_global_x — a list of
        dicts {key: float}, one per forecast day.  Converted to a contiguous
        float64 array [hold_days × n_global] before passing to C++.
        """
        gx_arr = self._pack_global_x_per_day(global_x_per_day, hold_days)

        return rms_cpp.forecast_ticker(
            np.ascontiguousarray(hist_high,  dtype=np.float64),
            np.ascontiguousarray(hist_low,   dtype=np.float64),
            np.ascontiguousarray(hist_close, dtype=np.float64),
            self._model,
            mc_samples,
            hold_days,
            target,
            stop,
            gx_arr,
            seed,
            self._mixture_mode,
        )

    # ------------------------------------------------------------------
    # evaluate_sample
    # ------------------------------------------------------------------

    def evaluate_sample(
        self,
        hist_high: np.ndarray,
        hist_low: np.ndarray,
        hist_close: np.ndarray,
        real_R: float,
        real_M: float,
        real_S: float,
        real_C: float,
        n_mc: int,
        global_x: dict,   # {key: float} for this sample day
        seed: int,
    ) -> float:
        """
        Returns dict {total, R, M, S, C} of squared errors (nan if sample should be skipped).
        """
        gx_arr = self._pack_global_x(global_x)

        total, R_err, M_err, S_err, C_err = rms_cpp.evaluate_sample(
            np.ascontiguousarray(hist_high,  dtype=np.float64),
            np.ascontiguousarray(hist_low,   dtype=np.float64),
            np.ascontiguousarray(hist_close, dtype=np.float64),
            real_R, real_M, real_S, real_C,
            self._model,
            n_mc,
            gx_arr,
            seed,
            self._mixture_mode,
            self._var_R, self._var_M, self._var_S, self._var_C,
        )
        return {"total": total, "R": R_err, "M": M_err, "S": S_err, "C": C_err}

    # ------------------------------------------------------------------
    # make_multi_path_forecaster
    # ------------------------------------------------------------------

    def make_multi_path_forecaster(self) -> "rms_cpp.MultiPathForecaster":
        """
        Create and configure a MultiPathForecaster bound to this model.
        The forecaster holds all N_mc path histories internally and
        recomputes global indicators from mean predicted prices each day.

        Call forecaster.init(...) then forecaster.step() hold_days times,
        then forecaster.get_returns() to collect results.
        """
        f = rms_cpp.MultiPathForecaster()
        f.set_model(self._model, self._mixture_mode)

        # Build ordered lists of global indicator names for each component
        def global_names(dens_list):
            names = []
            for d, ind in dens_list:
                if d.meta.get("is_global", False):
                    names.append(d.meta["indicator_name"])
            return names

        # These are stored on the adapter from __init__
        f.set_global_names(
            self._global_names_R,
            self._global_names_M,
            self._global_names_S,
        )
        return f

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pack_global_x_per_day(self, global_x_per_day: list, hold_days: int) -> np.ndarray:
        """
        Convert list of dicts → float64 array [hold_days, n_global].
        Returns empty array if no global indicators.
        """
        if self._n_global == 0:
            return np.empty((hold_days, 0), dtype=np.float64)

        arr = np.full((hold_days, self._n_global), np.nan, dtype=np.float64)
        for d, gx in enumerate(global_x_per_day):
            for j, key in enumerate(self._global_keys):
                arr[d, j] = gx.get(key, np.nan)
        return np.ascontiguousarray(arr)

    def _pack_global_x(self, global_x: dict) -> np.ndarray:
        """
        Convert single-day dict → float64 array [n_global].
        Returns empty array if no global indicators.
        """
        if self._n_global == 0:
            return np.empty(0, dtype=np.float64)

        arr = np.array(
            [global_x.get(k, np.nan) for k in self._global_keys],
            dtype=np.float64,
        )
        return arr


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _make_key(component: str, density, indicator) -> str:
    params = density.meta["indicator_params"]
    params_str = "_".join(map(str, params))
    return f"{component}_{density.meta['indicator_name']}_{params_str}"
