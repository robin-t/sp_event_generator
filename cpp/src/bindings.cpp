#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "rms_types.h"
#include <array>

namespace py = pybind11;

// ----------------------------------------------------------------
// Forward declarations from other translation units
// ----------------------------------------------------------------
void forecast_ticker(
    const double* hist_high, const double* hist_low, const double* hist_close,
    int hist_len,
    const DensitySet& dens,
    const std::vector<IndicatorSpec>& specs_R,
    const std::vector<IndicatorSpec>& specs_M,
    const std::vector<IndicatorSpec>& specs_S,
    int mc_samples, int hold_days,
    double target, double stop,
    const std::vector<double>& global_x_per_day,
    int n_global_R, int n_global_M, int n_global_S,
    std::mt19937& rng,
    double& out_mean, double& out_std,
    bool mixture_mode);

// MultiPathForecaster defined in multi_path_forecaster.cpp
// Include header-style to get the struct definition for pybind11 binding.
#include "multi_path_forecaster.cpp"

void step_ticker_one_day(
    const double* hist_high, const double* hist_low, const double* hist_close,
    int hist_len,
    const DensitySet& dens,
    const std::vector<IndicatorSpec>& specs_R,
    const std::vector<IndicatorSpec>& specs_M,
    const std::vector<IndicatorSpec>& specs_S,
    const std::vector<double>& global_x,
    int n_global_R, int n_global_M,
    int n_mc,
    std::mt19937& rng,
    bool mixture_mode,
    std::vector<double>& out_close,
    std::vector<double>& out_high,
    std::vector<double>& out_low);

std::array<double, 5> evaluate_sample(
    const double* hist_high, const double* hist_low, const double* hist_close,
    int hist_len,
    double real_R, double real_M, double real_S, double real_C,
    const DensitySet& dens,
    const std::vector<IndicatorSpec>& specs_R,
    const std::vector<IndicatorSpec>& specs_M,
    const std::vector<IndicatorSpec>& specs_S,
    int n_mc,
    const std::vector<double>& global_x,
    int n_global_R, int n_global_M,
    std::mt19937& rng,
    bool mixture_mode,
    double var_R, double var_M, double var_S, double var_C);

// ================================================================
// Python → C++ helpers
// ================================================================

// ----------------------------------------------------------------
// Build a PackedDensity from a Python RMSDensitySet object.
// Called once per density at setup time, not in the hot loop.
// ----------------------------------------------------------------
static PackedDensity pack_density(py::object py_density, double weight, bool is_global)
{
    PackedDensity d;
    d.weight    = weight;
    d.is_global = is_global;

    auto copy_array = [&](const char* attr) -> std::vector<double> {
        py::array_t<double> arr = py_density.attr(attr).cast<py::array_t<double>>();
        auto buf = arr.request();
        const double* ptr = static_cast<const double*>(buf.ptr);
        return std::vector<double>(ptr, ptr + buf.size);
    };

    d.x_bins = copy_array("x_bins");
    d.R_bins = copy_array("R_bins");
    d.M_bins = copy_array("M_bins");
    d.S_bins = copy_array("S_bins");
    d.R_prob = copy_array("R_prob");
    d.M_prob = copy_array("M_prob");
    d.S_prob = copy_array("S_prob");

    d.n_x = (int)d.x_bins.size() - 1;
    d.n_R = (int)d.R_bins.size() - 1;
    d.n_M = (int)d.M_bins.size() - 1;
    d.n_S = (int)d.S_bins.size() - 1;

    // MR_prob is optional
    if (py::hasattr(py_density, "MR_prob") &&
        !py_density.attr("MR_prob").is_none())
    {
        py::array_t<double> arr = py_density.attr("MR_prob")
                                            .cast<py::array_t<double>>();
        auto buf = arr.request();
        if (buf.size > 0)
            d.MR_prob = std::vector<double>(
                static_cast<const double*>(buf.ptr),
                static_cast<const double*>(buf.ptr) + buf.size);
    }

    return d;
}

// ----------------------------------------------------------------
// Build an IndicatorSpec from indicator name string + params.
// ----------------------------------------------------------------
static IndicatorSpec make_spec(const std::string& name,
                                const std::vector<int>& params,
                                bool is_global)
{
    IndicatorSpec s;
    s.params = params;

    if (is_global) { s.type = IndicatorType::GLOBAL; return s; }

    if      (name == "rsi")              s.type = IndicatorType::RSI;
    else if (name == "rsi_velocity")     s.type = IndicatorType::RSI_VELOCITY;
    else if (name == "vol_ratio")        s.type = IndicatorType::VOL_RATIO;
    else if (name == "macd")             s.type = IndicatorType::MACD;
    else if (name == "trend_slope")      s.type = IndicatorType::TREND_SLOPE;
    else if (name == "range_position")   s.type = IndicatorType::RANGE_POSITION;
    else if (name == "atr_ratio")        s.type = IndicatorType::ATR_RATIO;
    else if (name == "return_nd")        s.type = IndicatorType::RETURN_ND;
    else if (name == "high_low_dist")    s.type = IndicatorType::HIGH_LOW_DIST;
    else if (name == "close_open_ratio") s.type = IndicatorType::CLOSE_OPEN_RATIO;
    else if (name == "mean_S")           s.type = IndicatorType::MEAN_S;
    else if (name == "day_of_week")      s.type = IndicatorType::GLOBAL;  // injected as global
    else
        throw std::runtime_error("Unknown indicator name: " + name);

    return s;
}

// ================================================================
// PackedModel
//
// Python-visible class that holds the packed densities + specs
// for one model configuration. Built once from Python objects,
// then passed to forecast_ticker / evaluate_sample.
// Avoids repacking on every call.
// ================================================================
struct PackedModel {
    DensitySet dens;
    std::vector<IndicatorSpec> specs_R, specs_M, specs_S;
    int n_global_R = 0, n_global_M = 0, n_global_S = 0;

    // Build from Python density lists and weight dict.
    // density_lists: (R_list, M_list, S_list)
    //   each list is [(RMSDensitySet, indicator_name, params, is_global), ...]
    // weights: dict {key: float}  — same format as RMSTransitionModel.weights
    PackedModel(py::list R_list, py::list M_list, py::list S_list,
                py::dict weights)
    {
        auto build = [&](py::list& lst,
                         std::vector<PackedDensity>& dens_vec,
                         std::vector<IndicatorSpec>& spec_vec,
                         int& n_global,
                         const std::string& component)
        {
            for (auto item : lst) {
                py::tuple t = item.cast<py::tuple>();
                py::object density  = t[0];
                std::string name    = t[1].cast<std::string>();
                std::vector<int> params = t[2].cast<std::vector<int>>();
                bool is_global      = t[3].cast<bool>();

                // Build weight key — same format as _weight_lookup
                std::string params_str;
                for (auto& p : params) {
                    if (!params_str.empty()) params_str += "_";
                    params_str += std::to_string(p);
                }
                std::string key = component + "_" + name + "_" + params_str;

                // Try prefixed key first, then unprefixed (legacy)
                double w = 1.0;
                if (weights.contains(key)) {
                    w = weights[key.c_str()].cast<double>();
                } else {
                    std::string unprefixed = name + "_" + params_str;
                    if (weights.contains(unprefixed)) {
                        w = weights[unprefixed.c_str()].cast<double>();
                    } else {
                        throw std::runtime_error(
                            "Weight missing for key '" + key + "'");
                    }
                }

                dens_vec.push_back(pack_density(density, w, is_global));
                spec_vec.push_back(make_spec(name, params, is_global));

                if (is_global) ++n_global;
            }
        };

        build(R_list, dens.R, specs_R, n_global_R, "R");
        build(M_list, dens.M, specs_M, n_global_M, "M");
        build(S_list, dens.S, specs_S, n_global_S, "S");
    }
};

// ================================================================
// Python-callable forecast_ticker wrapper
// ================================================================
static py::tuple py_forecast_ticker(
    py::array_t<double, py::array::c_style | py::array::forcecast> hist_high,
    py::array_t<double, py::array::c_style | py::array::forcecast> hist_low,
    py::array_t<double, py::array::c_style | py::array::forcecast> hist_close,
    const PackedModel& model,
    int mc_samples,
    int hold_days,
    double target,
    double stop,
    py::array_t<double, py::array::c_style | py::array::forcecast> global_x_per_day,
    int seed,
    bool mixture_mode)
{
    auto hh = hist_high.request();
    auto hl = hist_low.request();
    auto hc = hist_close.request();
    int hist_len = (int)hc.size;

    auto gx = global_x_per_day.request();
    std::vector<double> gx_vec(
        static_cast<const double*>(gx.ptr),
        static_cast<const double*>(gx.ptr) + gx.size);

    std::mt19937 rng(seed);

    double mean, std;
    forecast_ticker(
        static_cast<const double*>(hh.ptr),
        static_cast<const double*>(hl.ptr),
        static_cast<const double*>(hc.ptr),
        hist_len,
        model.dens,
        model.specs_R, model.specs_M, model.specs_S,
        mc_samples, hold_days, target, stop,
        gx_vec,
        model.n_global_R, model.n_global_M, model.n_global_S,
        rng,
        mean, std,
        mixture_mode);

    return py::make_tuple(mean, std);
}

// ================================================================
// Python-callable evaluate_sample wrapper
// ================================================================
static py::tuple py_evaluate_sample(
    py::array_t<double, py::array::c_style | py::array::forcecast> hist_high,
    py::array_t<double, py::array::c_style | py::array::forcecast> hist_low,
    py::array_t<double, py::array::c_style | py::array::forcecast> hist_close,
    double real_R, double real_M, double real_S, double real_C,
    const PackedModel& model,
    int n_mc,
    py::array_t<double, py::array::c_style | py::array::forcecast> global_x,
    int seed,
    bool mixture_mode,
    double var_R, double var_M, double var_S, double var_C)
{
    auto hh = hist_high.request();
    auto hl = hist_low.request();
    auto hc = hist_close.request();
    int hist_len = (int)hc.size;

    auto gx = global_x.request();
    std::vector<double> gx_vec(
        static_cast<const double*>(gx.ptr),
        static_cast<const double*>(gx.ptr) + gx.size);

    std::mt19937 rng(seed);

    auto result = evaluate_sample(
        static_cast<const double*>(hh.ptr),
        static_cast<const double*>(hl.ptr),
        static_cast<const double*>(hc.ptr),
        hist_len,
        real_R, real_M, real_S, real_C,
        model.dens,
        model.specs_R, model.specs_M, model.specs_S,
        n_mc,
        gx_vec,
        model.n_global_R, model.n_global_M,
        rng,
        mixture_mode,
        var_R, var_M, var_S, var_C);

    // result = {total, R, M, S, C}  (variance-normalized)
    return py::make_tuple(result[0], result[1], result[2], result[3], result[4]);
}

// ================================================================
// Python-callable step_ticker_one_day wrapper
// Returns (closes, highs, lows) as numpy arrays [n_mc]
// ================================================================
static py::tuple py_step_ticker_one_day(
    py::array_t<double, py::array::c_style | py::array::forcecast> hist_high,
    py::array_t<double, py::array::c_style | py::array::forcecast> hist_low,
    py::array_t<double, py::array::c_style | py::array::forcecast> hist_close,
    const PackedModel& model,
    py::array_t<double, py::array::c_style | py::array::forcecast> global_x,
    int n_mc,
    int seed,
    bool mixture_mode)
{
    auto hh = hist_high.request();
    auto hl = hist_low.request();
    auto hc = hist_close.request();
    int hist_len = (int)hc.size;

    auto gx = global_x.request();
    std::vector<double> gx_vec(
        static_cast<const double*>(gx.ptr),
        static_cast<const double*>(gx.ptr) + gx.size);

    std::mt19937 rng(seed);

    std::vector<double> out_c, out_h, out_l;
    step_ticker_one_day(
        static_cast<const double*>(hh.ptr),
        static_cast<const double*>(hl.ptr),
        static_cast<const double*>(hc.ptr),
        hist_len,
        model.dens,
        model.specs_R, model.specs_M, model.specs_S,
        gx_vec,
        model.n_global_R, model.n_global_M,
        n_mc, rng, mixture_mode,
        out_c, out_h, out_l);

    auto to_array = [&](std::vector<double>& v) {
        auto arr = py::array_t<double>(v.size());
        std::copy(v.begin(), v.end(),
                  static_cast<double*>(arr.request().ptr));
        return arr;
    };

    return py::make_tuple(to_array(out_c), to_array(out_h), to_array(out_l));
}

// ================================================================
// Module definition
// ================================================================
PYBIND11_MODULE(rms_cpp, m)
{
    py::class_<MultiPathForecaster>(m, "MultiPathForecaster")
        .def(py::init<>())
        .def("set_model", [](MultiPathForecaster& self,
                              const PackedModel& model,
                              bool mixture_mode) {
            self.dens      = &model.dens;
            self.specs_R   = &model.specs_R;
            self.specs_M   = &model.specs_M;
            self.specs_S   = &model.specs_S;
            self.mixture_mode = mixture_mode;
            self.n_global_R   = model.n_global_R;
            self.n_global_M   = model.n_global_M;
            self.n_global_S   = model.n_global_S;

            // Build global name lists from PackedModel specs + densities
            auto extract_names = [](
                    const std::vector<PackedDensity>& dens_vec,
                    const std::vector<IndicatorSpec>& specs) {
                std::vector<std::string> names;
                for (int i = 0; i < (int)dens_vec.size(); ++i) {
                    if (dens_vec[i].is_global) {
                        // Indicator name is encoded in specs[i].params for
                        // GLOBAL type — we pass it as a string via Python.
                        // For now use a placeholder; Python sets names below.
                        names.push_back("__global__");
                    }
                }
                return names;
            };
            // Names are set separately via set_global_names
        },
        py::arg("model"), py::arg("mixture_mode") = false)

        .def("set_global_names", [](MultiPathForecaster& self,
                                     py::list names_R,
                                     py::list names_M,
                                     py::list names_S) {
            self.global_names_R.clear();
            self.global_names_M.clear();
            self.global_names_S.clear();
            for (auto n : names_R) self.global_names_R.push_back(n.cast<std::string>());
            for (auto n : names_M) self.global_names_M.push_back(n.cast<std::string>());
            for (auto n : names_S) self.global_names_S.push_back(n.cast<std::string>());
        },
        py::arg("names_R"), py::arg("names_M"), py::arg("names_S"))

        .def("init", [](MultiPathForecaster& self,
                         py::array_t<double, py::array::c_style | py::array::forcecast> hist_close,
                         py::array_t<double, py::array::c_style | py::array::forcecast> hist_high,
                         py::array_t<double, py::array::c_style | py::array::forcecast> hist_low,
                         int n_tickers, int hist_len, int n_mc,
                         double target, double stop,
                         py::array_t<double, py::array::c_style | py::array::forcecast> weekdays,
                         int seed) {
            auto c = hist_close.request();
            auto h = hist_high.request();
            auto l = hist_low.request();
            auto w = weekdays.request();

            std::vector<double> vc(static_cast<const double*>(c.ptr),
                                   static_cast<const double*>(c.ptr) + c.size);
            std::vector<double> vh(static_cast<const double*>(h.ptr),
                                   static_cast<const double*>(h.ptr) + h.size);
            std::vector<double> vl(static_cast<const double*>(l.ptr),
                                   static_cast<const double*>(l.ptr) + l.size);
            std::vector<double> vw(static_cast<const double*>(w.ptr),
                                   static_cast<const double*>(w.ptr) + w.size);

            self.init(vc, vh, vl, n_tickers, hist_len, n_mc,
                      target, stop, vw, seed);
        },
        py::arg("hist_close"), py::arg("hist_high"), py::arg("hist_low"),
        py::arg("n_tickers"), py::arg("hist_len"), py::arg("n_mc"),
        py::arg("target"), py::arg("stop"),
        py::arg("weekdays"), py::arg("seed"))

        .def("step", &MultiPathForecaster::step)
        .def("run",  &MultiPathForecaster::run, py::arg("hold_days"),
             "Run the full forecast: step hold_days times in one C++ call.")

        .def("get_returns", [](MultiPathForecaster& self) {
            auto raw = self.get_returns();
            // Returns list of (mean, std) tuples, one per ticker
            py::list result;
            for (int t = 0; t < self.n_tickers; ++t) {
                result.append(py::make_tuple(raw[t * 2], raw[t * 2 + 1]));
            }
            return result;
        });


    m.doc() = "RMS Monte Carlo C++ kernel";

    py::class_<PackedModel>(m, "PackedModel")
        .def(py::init<py::list, py::list, py::list, py::dict>(),
             py::arg("R_list"), py::arg("M_list"), py::arg("S_list"),
             py::arg("weights"),
             R"doc(
Pack density arrays and weights into C++ memory.

Each list entry is a tuple:
    (RMSDensitySet, indicator_name: str, params: list[int], is_global: bool)

weights is the same dict as RMSTransitionModel.weights.
Build this once per model configuration, not per ticker/sample.
)doc");

    m.def("forecast_ticker", &py_forecast_ticker,
          py::arg("hist_high"), py::arg("hist_low"), py::arg("hist_close"),
          py::arg("model"),
          py::arg("mc_samples"), py::arg("hold_days"),
          py::arg("target"), py::arg("stop"),
          py::arg("global_x_per_day"),
          py::arg("seed"),
          py::arg("mixture_mode") = false,
          R"doc(
Run MC forecast for one ticker. Returns (mean_return, std_return).

global_x_per_day : 2D array [hold_days, n_global_total]
                   Pass np.empty((hold_days, 0)) if no global indicators.
seed             : RNG seed — use RANK-dependent value for MPI reproducibility.
)doc");

    m.def("step_ticker_one_day", &py_step_ticker_one_day,
          py::arg("hist_high"), py::arg("hist_low"), py::arg("hist_close"),
          py::arg("model"),
          py::arg("global_x"),
          py::arg("n_mc"),
          py::arg("seed"),
          py::arg("mixture_mode") = false,
          R"doc(
Single-step MC forecast for one ticker. Returns (closes, highs, lows) arrays [n_mc].
Used by the day-by-day outer loop for live global indicator updates.
)doc");

    m.def("evaluate_sample", &py_evaluate_sample,
          py::arg("hist_high"), py::arg("hist_low"), py::arg("hist_close"),
          py::arg("real_R"), py::arg("real_M"), py::arg("real_S"), py::arg("real_C"),
          py::arg("model"),
          py::arg("n_mc"),
          py::arg("global_x"),
          py::arg("seed"),
          py::arg("mixture_mode") = false,
          py::arg("var_R") = 1.0,
          py::arg("var_M") = 1.0,
          py::arg("var_S") = 1.0,
          py::arg("var_C") = 1.0,
          R"doc(
Evaluate one tuning sample.
Returns tuple (total, R, M, S, C) of squared errors.
Any component is nan if the sample should be skipped.

global_x : 1D array [n_global_total] for this sample day.
           Pass np.empty(0) if no global indicators.
)doc");
}
