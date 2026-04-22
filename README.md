# Monte Carlo Stock Strategy Generator

A physics-inspired Monte Carlo event generator for equity price prediction and systematic trading strategy development, built as a personal project to explore quantitative finance through hands-on implementation.

## Overview

The core idea is borrowed from particle physics event generators: instead of modelling a stock price directly, each trading day is treated as an *event* characterised by three quantities derived from the daily OHLC candle:

| Symbol | Meaning | Definition |
|--------|---------|------------|
| **R** | Range (volatility) | `(High − Low) / Close_prev` |
| **M** | Midpoint (direction) | `((High + Low)/2 − Close_prev) / Close_prev` |
| **S** | Skew (close position) | `(Close − Midpoint) / (Range/2)` ∈ [−1, 1] |

Joint probability densities `f(R, M, S | x)` are estimated from historical data, conditioned on technical indicator values `x`. At forecast time, MC paths are sampled from these densities to produce a distribution of future prices, from which expected returns and strategy signals are derived.

## Architecture

```
sp_event_generator/
├── cpp/                  # C++ simulation kernel
│   ├── src/              # Samplers, indicators, forecast, bindings
│   └── include/          # Types and headers
├── density/              # Density estimation and indicator definitions
│   ├── density.py        # DensitySet: build, save, load, analyse
│   ├── indicator.py      # Per-ticker and global indicator registry
│   └── features.py       # OHLC feature extraction
├── mc/                   # Monte Carlo sampling layer
│   ├── cpp_adapter.py    # Python ↔ C++ bridge (pybind11)
│   └── transition_model.py  # Python fallback sampler
├── tune/                 # Jackknife calibration
│   ├── tuner.py          # Grid search + jackknife scoring
│   └── tune_handler.py   # Interactive tuning UI
├── strategy/             # Backtesting and strategy evaluation
│   ├── strategy_tester.py
│   └── strategy_handler.py
└── data/                 # Data loading and alignment
```

## Key Features

- **C++ kernel** — performance-critical Monte Carlo sampling implemented in C++ with pybind11 bindings, achieving significant speedup over the pure Python path
- **MPI parallelisation** — jackknife calibration runs are distributed across MPI processes for fast hyperparameter grid search
- **MultiPathForecaster** — C++ class that evolves N independent MC paths simultaneously, computing live global market indicators (cross-sectional mean R/M/S) from predicted prices at each step, so the market state updates as paths evolve
- **Global indicators** — cross-sectional market indicators (`market_mean_R/M/S`, `market_breadth`, `market_vol_dispersion`) condition predictions on broad market state, not just per-ticker history
- **Jackknife calibration** — variance-normalised MSE scoring with jackknife resampling for robust hyperparameter selection and uncertainty estimation
- **Target/stop exit modelling** — MC paths exit when predicted price hits a target or stop level; exited paths are excluded from global state computation but included in return calculation
- **Density summaries** — each built density exports a compact JSON summary for offline analysis and quality control

## Technical Stack

- **C++17** with pybind11 for the hot sampling kernel
- **Python 3** for the pipeline, analysis, and interactive UI
- **MPI** (mpi4py) for parallel jackknife calibration
- **NumPy / Matplotlib** for data handling and visualisation
- **yfinance** for historical OHLC data

## Indicators Supported

Per-ticker: `rsi`, `rsi_velocity`, `atr_ratio`, `return_nd`, `mean_S`, `high_low_dist`, `range_position`, `close_open_ratio`, `vol_ratio`, `trend_slope`, `macd`

Global (cross-sectional): `market_mean_R`, `market_mean_M`, `market_mean_S`, `market_vol_dispersion`, `market_breadth`, `day_of_week`


## Installation & Build

### Prerequisites
- Python 3.9+
- C++17 compiler (g++ or clang++)
- CMake 3.15+
- MPI implementation (e.g. OpenMPI): `sudo apt install libopenmpi-dev` or `brew install open-mpi`

### Python dependencies
```bash
pip install numpy matplotlib yfinance mpi4py pybind11
```

### Build the C++ kernel
```bash
bash cpp/build.sh
```

### Run
```bash
python main.py
# Or with MPI for parallel jackknife calibration:
mpirun -n 8 python main.py
```

> **Note:** If the C++ build fails or the `.so` is not found, the system automatically falls back to the pure Python sampler in `mc/transition_model.py`. Performance will be slower but all functionality remains available.


## Status

The pipeline is fully functional end-to-end — density building, jackknife calibration, and strategy backtesting all run. Active development is ongoing.
