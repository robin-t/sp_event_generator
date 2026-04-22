"""
Verify fixed C++ indicator implementations match Python.
Uses Python reimplementations of the FIXED C++ code.
Run on server: python3 indicator_check.py
Then rebuild C++ and run equivalence_test.py
"""
import numpy as np
from data.data_store import DataStore
from data.data_loader import DataLoader
from tools.tickers import Tickers
from density.indicator import Indicator

ds = DataStore()
tickers = Tickers().get("sweden_largecap")
raw = ds.download_full(tickers[:1])
dates, close, high, low, _ = DataLoader.align(raw)

test_dates = [3000, 4000, 5000, 5500, 6000]

# ---- Fixed C++ manual implementations ----

def cpp_rsi(close, period):
    n = len(close)
    gains = np.zeros(n); losses = np.zeros(n)
    for i in range(1, n):
        d = close[i] - close[i-1]
        gains[i]  = max(d, 0); losses[i] = max(-d, 0)
    half = period // 2
    sg = sl = 0.0
    for j in range(period):
        idx = n-1-half+j
        if 0 <= idx < n:
            sg += gains[idx]; sl += losses[idx]
    rs = (sg/period) / (sl/period + 1e-12)
    return 100.0 - 100.0/(1.0+rs)

def cpp_rsi_velocity_fixed(close, rsi_window, vel_window):
    """Fixed: Wilder RSI + mean of last vel_window differences"""
    n = len(close)
    if n < rsi_window + vel_window + 1:
        return float('nan')
    delta = np.zeros(n)
    delta[1:] = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    # Wilder seed
    ag = gain[1:rsi_window+1].mean()
    al = loss[1:rsi_window+1].mean()
    rsi = np.full(n, np.nan)
    rsi[rsi_window] = 100 - 100/(1 + ag/(al+1e-12))
    for i in range(rsi_window+1, n):
        ag = (ag*(rsi_window-1) + gain[i]) / rsi_window
        al = (al*(rsi_window-1) + loss[i]) / rsi_window
        rsi[i] = 100 - 100/(1 + ag/(al+1e-12))
    # drsi
    drsi = np.zeros(n)
    for i in range(1, n):
        if np.isfinite(rsi[i]) and np.isfinite(rsi[i-1]):
            drsi[i] = rsi[i] - rsi[i-1]
    # mean of last vel_window drsi
    start = n - vel_window
    if start < rsi_window + 1:
        return float('nan')
    vals = [drsi[i] for i in range(start, n) if np.isfinite(drsi[i])]
    return np.mean(vals) if vals else float('nan')

def cpp_vol_ratio_fixed(close, short_w, long_w):
    n = len(close)
    logrets = np.zeros(n)
    logrets[1:] = np.diff(np.log(close))
    def std(start, length):
        sl = logrets[start:start+length]
        m = sl.mean(); v = ((sl-m)**2).mean()
        return np.sqrt(v) if v > 0 else 0.0
    vs = std(n-short_w, short_w); vl = std(n-long_w, long_w)
    return vs/vl if vl > 1e-12 else 1.0

def cpp_atr_ratio_fixed(high, low, close, short_w, long_w):
    n = len(close)
    tr = np.zeros(n); tr[0] = high[0]-low[0]
    for i in range(1,n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    def wilder(w):
        a = tr[:w].mean()
        for i in range(w,n): a = (a*(w-1)+tr[i])/w
        return a
    s,l = wilder(short_w), wilder(long_w)
    return s/l if l>1e-12 else 1.0

def cpp_trend_slope_fixed(close, window):
    """Fixed: OLS on log prices / std(log prices)"""
    n = len(close)
    y = np.log(close[n-window:n])
    if not np.all(np.isfinite(y)): return float('nan')
    x = np.arange(window, dtype=float)
    sx=x.sum(); sy=y.sum(); sxy=(x*y).sum(); sx2=(x**2).sum()
    denom = window*sx2 - sx**2
    if abs(denom) < 1e-12: return float('nan')
    slope = (window*sxy - sx*sy) / denom
    y_std = np.std(y)  # population std
    return slope/y_std if y_std > 0 else slope

def cpp_return_nd(close, window):
    n = len(close); prev = close[n-1-window]
    return (close[-1]-prev)/prev if prev > 0 else float('nan')

def cpp_range_position_fixed(high, low, close, window):
    """Fixed: mean of per-day (close-low)/(high-low)"""
    n = len(close)
    vals = []
    for i in range(n-window, n):
        rng = high[i]-low[i]
        if rng > 0 and np.isfinite(high[i]) and np.isfinite(low[i]) and np.isfinite(close[i]):
            vals.append((close[i]-low[i])/rng)
    return np.mean(vals) if vals else float('nan')

def cpp_close_open_ratio(high, low, close, window):
    n = len(close)
    vals = []
    for i in range(n-window, n):
        h,l,c = high[i],low[i],close[i]
        half=(h-l)/2
        if half>0: vals.append((c-(h+l)/2)/half)
    return np.mean(vals) if vals else float('nan')

# ---- Run comparison ----
print(f"{'Indicator':<25} {'Date':<12} {'Python':>12} {'C++ fixed':>12} {'Diff':>12} {'Match'}")
print("-"*80)

indicators = [
    ("rsi[14]",              lambda h,l,c: Indicator("rsi",[14]).compute(h,l,c)[-1],
                             lambda h,l,c: cpp_rsi(c, 14)),
    ("rsi_velocity[14,5]",   lambda h,l,c: Indicator("rsi_velocity",[14,5]).compute(h,l,c)[-1],
                             lambda h,l,c: cpp_rsi_velocity_fixed(c, 14, 5)),
    ("vol_ratio[5,30]",      lambda h,l,c: Indicator("vol_ratio",[5,30]).compute(h,l,c)[-1],
                             lambda h,l,c: cpp_vol_ratio_fixed(c, 5, 30)),
    ("atr_ratio[5,30]",      lambda h,l,c: Indicator("atr_ratio",[5,30]).compute(h,l,c)[-1],
                             lambda h,l,c: cpp_atr_ratio_fixed(h,l,c, 5, 30)),
    ("trend_slope[20]",      lambda h,l,c: Indicator("trend_slope",[20]).compute(h,l,c)[-1],
                             lambda h,l,c: cpp_trend_slope_fixed(c, 20)),
    ("return_nd[5]",         lambda h,l,c: Indicator("return_nd",[5]).compute(h,l,c)[-1],
                             lambda h,l,c: cpp_return_nd(c, 5)),
    ("range_position[10]",   lambda h,l,c: Indicator("range_position",[10]).compute(h,l,c)[-1],
                             lambda h,l,c: cpp_range_position_fixed(h,l,c, 10)),
    ("close_open_ratio[10]", lambda h,l,c: Indicator("close_open_ratio",[10]).compute(h,l,c)[-1],
                             lambda h,l,c: cpp_close_open_ratio(h,l,c, 10)),
]

all_pass = True
for t in test_dates:
    hc = close[0, t-100:t].copy()
    hh = high[0,  t-100:t].copy()
    hl = low[0,   t-100:t].copy()
    for name, py_fn, cpp_fn in indicators:
        pv = py_fn(hh, hl, hc); cv = cpp_fn(hh, hl, hc)
        diff = abs(pv-cv) if (np.isfinite(pv) and np.isfinite(cv)) else float('nan')
        ok = np.isfinite(diff) and diff < 1e-6
        if not ok: all_pass = False
        print(f"{name:<25} {str(dates[t])[:10]:<12} {pv:>12.6f} {cv:>12.6f} {diff:>12.2e} {'✓' if ok else '✗'}")
    print()

print("="*50)
print(f"All indicators match: {'✓ YES' if all_pass else '✗ NO — fix before rebuilding'}")
