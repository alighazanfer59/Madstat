"""
pages/1_Load_Data.py
--------------------
Streamlit page for fetching raw OHLCV data via DataFetcher.
Supports both yfinance and TradingView sources.
"""

import sys
import logging
from pathlib import Path
from datetime import date, timedelta

import streamlit as st
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for p in [str(ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data.fetcher import DataFetcher, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Load Data — Madstrat",
    page_icon="📥",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("Madstrat 2.0")
    st.caption("Step 1 of 3")
    st.divider()
    st.markdown("""
    **Workflow**
    1. 📥 **Load Data** ← you are here
    2. ⚙️ Process Data
    3. 🚀 Run Backtest
    """)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📥 Load Data")
st.markdown("Fetch raw OHLCV data and save it to `data/raw/`.")
st.divider()

# ── Config form ───────────────────────────────────────────────────────────────
with st.form("fetch_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Instrument & Timeframes")

        instrument = st.selectbox(
            "Instrument",
            ["EURUSD", "GBPUSD", "EURGBP", "XAUUSD"],
            index=0,
            help="Madstrat 2.0 is optimised for these pairs only."
        )

        timeframes = st.multiselect(
            "Timeframes to fetch",
            ["15m", "30m", "1h", "2h", "4h", "1d"],
            default=["15m", "1h", "4h"],
            help="Select all timeframes you need. 4H is auto-resampled from 1H for yfinance."
        )

        save_raw = st.checkbox("Save to data/raw/ as CSV", value=True)

    with col2:
        st.markdown("#### Date Range")

        default_end   = date.today()
        default_start = default_end - timedelta(days=60)

        start_date = st.date_input("Start date", value=default_start)
        end_date   = st.date_input("End date",   value=default_end)

        st.markdown("#### Data Source")
        source = st.radio(
            "Source",
            ["yfinance", "tradingview"],
            horizontal=True,
            help="yfinance: free, no login. TradingView: deeper history, native 4H."
        )

    # TradingView credentials (only shown when TV selected)
    tv_user = tv_pass = ""
    if source == "tradingview":
        st.divider()
        st.markdown("#### TradingView Credentials *(optional)*")
        st.caption(
            "Leave blank to connect without login. "
            "Some symbols require an account for full history."
        )
        cred_col1, cred_col2 = st.columns(2)
        with cred_col1:
            tv_user = st.text_input("Username", type="default")
        with cred_col2:
            tv_pass = st.text_input("Password", type="password")

    submitted = st.form_submit_button("🚀 Fetch Data", use_container_width=True)

# ── Validation ────────────────────────────────────────────────────────────────
if submitted:
    errors = []
    if not timeframes:
        errors.append("Select at least one timeframe.")
    if start_date >= end_date:
        errors.append("Start date must be before end date.")
    if source == "yfinance":
        delta_days = (end_date - start_date).days
        if delta_days > 60 and any(tf in timeframes for tf in ["15m", "30m"]):
            st.warning(
                "⚠️ yfinance limits 15m/30m history to ~60 days. "
                "Your range is larger — data may be truncated. "
                "Consider switching to TradingView for longer ranges."
            )

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # ── Run fetch ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Fetching...")

    fetcher = DataFetcher(
        instrument=instrument,
        start=str(start_date),
        end=str(end_date),
        source=source,
        tv_username=tv_user or None,
        tv_password=tv_pass or None,
        save_raw=save_raw,
    )

    fetch_results = {}
    progress = st.progress(0, text="Starting...")
    status   = st.empty()

    for i, tf in enumerate(timeframes):
        status.info(f"Fetching [{tf}]...")
        try:
            df = fetcher.fetch(tf)
            fetch_results[tf] = {"df": df, "error": None}
        except Exception as e:
            fetch_results[tf] = {"df": None, "error": str(e)}
        progress.progress((i + 1) / len(timeframes), text=f"Done: {tf}")

    status.empty()
    progress.empty()

    # ── Results summary ───────────────────────────────────────────────────────
    st.markdown("### Results")

    all_ok = all(v["error"] is None for v in fetch_results.values())

    summary_rows = []
    for tf, result in fetch_results.items():
        if result["df"] is not None:
            df = result["df"]
            summary_rows.append({
                "Timeframe":  tf,
                "Bars":       len(df),
                "Start":      str(df.index[0]),
                "End":        str(df.index[-1]),
                "Status":     "✅ OK",
            })
        else:
            summary_rows.append({
                "Timeframe":  tf,
                "Bars":       0,
                "Start":      "—",
                "End":        "—",
                "Status":     f"❌ {result['error']}",
            })

    st.dataframe(
        pd.DataFrame(summary_rows).set_index("Timeframe"),
        use_container_width=True,
    )

    if all_ok:
        st.success(
            f"✅ All {len(timeframes)} timeframe(s) fetched successfully. "
            f"Files saved to `data/raw/`."
        )
    else:
        st.warning("Some timeframes failed — check errors above.")

    # ── Data preview tabs ─────────────────────────────────────────────────────
    successful = {tf: v["df"] for tf, v in fetch_results.items() if v["df"] is not None}
    if successful:
        st.divider()
        st.markdown("### Data Preview")
        tabs = st.tabs(list(successful.keys()))
        for tab, (tf, df) in zip(tabs, successful.items()):
            with tab:
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.dataframe(df.head(20), use_container_width=True)
                with col_b:
                    st.markdown(f"**Shape:** `{df.shape}`")
                    st.markdown(f"**Columns:** `{list(df.columns)}`")
                    st.markdown(f"**Index tz:** `{df.index.tz}`")
                    st.markdown("**Descriptive stats:**")
                    st.dataframe(
                        df[["open","high","low","close"]].describe().round(6),
                        use_container_width=True,
                    )

# ── Existing raw files ────────────────────────────────────────────────────────
st.divider()
st.markdown("### Existing Files in `data/raw/`")

raw_files = sorted(RAW_DATA_DIR.glob("*.csv"))
if not raw_files:
    st.info("No raw files yet. Use the form above to fetch your first dataset.")
else:
    file_rows = []
    for f in raw_files:
        size_kb = f.stat().st_size / 1024
        file_rows.append({"File": f.name, "Size (KB)": round(size_kb, 1)})

    st.dataframe(
        pd.DataFrame(file_rows).set_index("File"),
        use_container_width=True,
    )
    st.caption(f"{len(raw_files)} file(s) — {RAW_DATA_DIR}")
