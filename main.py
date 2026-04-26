"""
main.py
-------
Madstrat 2.0 Backtester — Streamlit entry point.

Run with:
    streamlit run main.py
"""

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Madstrat 2.0 Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar nav info ──────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("Madstrat 2.0")
    st.caption("EMA-Based Confluence Backtester")
    st.divider()
    st.markdown("""
    **Workflow**
    1. 📥 **Load Data** — fetch OHLCV from yfinance or TradingView
    2. ⚙️ **Process Data** — add EMAs, SMAs and price levels
    3. 🚀 **Run Backtest** — coming soon
    """)
    st.divider()
    st.caption("Supports: EURUSD · GBPUSD · EURGBP · XAUUSD")

# ── Home page ─────────────────────────────────────────────────────────────────
st.title("📈 Madstrat 2.0 Backtester")
st.markdown(
    "A systematic backtesting engine for the **Madstrat 2.0** "
    "EMA-confluence trading strategy."
)

st.divider()

# Pipeline status cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📥 Step 1 — Load Data")
    st.markdown(
        "Fetch raw OHLCV data from **yfinance** or **TradingView** "
        "for any supported instrument and date range."
    )
    st.info("Navigate to **Load Data** in the sidebar to get started.")

with col2:
    st.markdown("### ⚙️ Step 2 — Process Data")
    st.markdown(
        "Select a raw CSV and enrich it with **EMAs, SMAs** and "
        "**daily/weekly price levels** (PDH, PDL, PD-EQ, PWH, PWL)."
    )
    st.info("Navigate to **Process Data** in the sidebar.")

with col3:
    st.markdown("### 🚀 Step 3 — Run Backtest")
    st.markdown(
        "Walk through processed data bar-by-bar, fire signals based on "
        "Madstrat confluence rules and log all trades."
    )
    st.warning("Coming soon.")

st.divider()

# Quick pipeline status
st.markdown("### Pipeline Status")

from pathlib import Path
RAW_DIR       = Path(__file__).parent / "madstrat_backtest" / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent / "madstrat_backtest" / "data" / "processed"

raw_files       = list(RAW_DIR.glob("*.csv"))
processed_files = list(PROCESSED_DIR.glob("*.csv"))

c1, c2, c3 = st.columns(3)
c1.metric("Raw files",       len(raw_files),       help=str(RAW_DIR))
c2.metric("Processed files", len(processed_files), help=str(PROCESSED_DIR))
c3.metric("Ready to backtest",
          len(processed_files),
          help="Files in data/processed/ that can be used for backtesting")

if not raw_files:
    st.warning("No raw data found. Go to **Load Data** to fetch your first dataset.")
elif not processed_files:
    st.warning("Raw data found but not yet processed. Go to **Process Data** next.")
else:
    st.success(
        f"✅ {len(processed_files)} processed file(s) ready. "
        f"Backtest engine coming in the next step."
    )
