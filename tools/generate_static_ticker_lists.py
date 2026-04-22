import os
import random
import requests
import pandas as pd
from pathlib import Path

# ----------------------------
# Setup
# ----------------------------

OUTPUT_DIR = Path("data/ticker_lists")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)  # reproducible sampling

# ----------------------------
# 1️⃣ S&P 500
# ----------------------------

print("Fetching S&P 500...")

headers = {
    "User-Agent": "Mozilla/5.0"
}

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
response = requests.get(url, headers=headers)
response.raise_for_status()

tables = pd.read_html(response.text)

sp500 = tables[0]["Symbol"].tolist()
sp500 = [s.replace(".", "-") for s in sp500]

sp500 = [s.replace(".", "-") for s in sp500]

pd.DataFrame({"ticker": sp500}).to_csv(
    OUTPUT_DIR / "sp500_full.csv",
    index=False
)

print(f"Saved {len(sp500)} S&P 500 tickers.")


# ----------------------------
# 2️⃣ Russell 2000 (via IWM ETF holdings)
# ----------------------------

print("Fetching Russell 2000 via IWM holdings...")

iwm_url = (
    "https://www.ishares.com/us/products/239710/"
    "ishares-russell-2000-etf/1467271812596.ajax?"
    "fileType=csv&fileName=IWM_holdings&dataType=fund"
)

r = requests.get(iwm_url)
r.raise_for_status()

with open("iwm_holdings_temp.csv", "wb") as f:
    f.write(r.content)

iwm_df = pd.read_csv("iwm_holdings_temp.csv", skiprows=9)

if "Ticker" not in iwm_df.columns:
    raise RuntimeError("Could not parse IWM holdings.")

russell_full = iwm_df["Ticker"].dropna().tolist()

# Clean formatting
russell_full = [t.replace(".", "-") for t in russell_full]

# Sample 250
russell_sample = random.sample(russell_full, 250)

pd.DataFrame({"ticker": russell_sample}).to_csv(
    OUTPUT_DIR / "russell2000_250.csv",
    index=False
)

print(f"Saved 250 Russell 2000 tickers.")


# ----------------------------
# 3️⃣ Sweden Large Cap
# ----------------------------

print("Fetching Swedish Large Cap (OMXS30)...")

headers = {
    "User-Agent": "Mozilla/5.0"
}

# ---- OMXS30 ----
omx_url = "https://en.wikipedia.org/wiki/OMX_Stockholm_30"
response = requests.get(omx_url, headers=headers)
response.raise_for_status()

tables = pd.read_html(response.text)

# The table containing tickers is usually the second table
omxs30_table = tables[1]

if "Ticker" not in omxs30_table.columns:
    raise RuntimeError("Could not find Ticker column in OMXS30 table.")

omxs30 = omxs30_table["Ticker"].dropna().tolist()

# Add .ST suffix for yfinance
omxs30 = [
    t.strip() + ".ST" if not t.strip().endswith(".ST") else t.strip()
    for t in omxs30
]

# Save
pd.DataFrame({"ticker": sorted(set(omxs30))}).to_csv(
    OUTPUT_DIR / "sweden_largecap.csv",
    index=False
)

print(f"Saved {len(omxs30)} Swedish large cap tickers.")

# ----------------------------
# Cleanup
# ----------------------------

if os.path.exists("iwm_holdings_temp.csv"):
    os.remove("iwm_holdings_temp.csv")

print("\nAll ticker lists generated successfully.")