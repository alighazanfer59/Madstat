"""
pages/2_Process_Data.py
-----------------------
Streamlit page for processing raw CSVs — adds EMAs, SMAs and price levels.
"""

import sys
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for p in [str(ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data.process_data import DataProcessor, RAW_DATA_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Process Data — Madstrat",
    page_icon="⚙️",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("Madstrat 2.0")
    st.caption("Step 2 of 3")
    st.divider()
    st.markdown("""
    **Workflow**
    1. 📥 Load Data
    2. ⚙️ **Process Data** ← you are here
    3. 🚀 Run Backtest
    """)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚙️ Process Data")
st.markdown(
    "Select a raw CSV from `data/raw/`, configure indicator settings, "
    "and save the enriched file to `data/processed/`."
)
st.divider()

# ── Check raw files exist ─────────────────────────────────────────────────────
raw_files = sorted(RAW_DATA_DIR.glob("*.csv"))

# Always show the resolved path so path issues are immediately visible
st.caption(f"📁 Looking in: `{RAW_DATA_DIR}`")

if not raw_files:
    st.warning(
        "No raw CSV files found in `data/raw/`. "
        "Go to **📥 Load Data** first to fetch some data."
    )
    st.info(f"Expected path: `{RAW_DATA_DIR}` — verify this matches where your CSV files are.")
    st.stop()

# ── File selection + settings ─────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("#### 1. Select Raw File")

    file_names    = [f.name for f in raw_files]
    selected_file = st.selectbox(
        "Raw file",
        file_names,
        help="Files in data/raw/ — fetched via the Load Data page."
    )
    selected_path = RAW_DATA_DIR / selected_file

    # Quick preview of selected file
    if selected_path.exists():
        preview_df = pd.read_csv(selected_path, index_col=0, parse_dates=True, nrows=5)
        st.caption(f"Preview (first 5 rows of {selected_file}):")
        st.dataframe(preview_df, use_container_width=True)

with col_right:
    st.markdown("#### 2. Indicators Applied")
    st.info(
        "The following are applied automatically:\n\n"
        "- **EMA 9, 18, 50** on close price\n"
        "- **SMA 50** on close price\n"
        "- **PDH, PDL, PD_EQ** — previous day levels\n"
        "- **PWH, PWL, PW_EQ** — previous week levels\n"
        "- **WH, WL** — current week running high/low"
    )

    st.markdown("#### 3. Output File")
    custom_output = st.text_input(
        "Output filename (optional)",
        value="",
        placeholder="Leave blank to auto-name as <input>_processed.csv",
    )

st.divider()

# ── Process button ────────────────────────────────────────────────────────────
if st.button("⚙️ Process Selected File", use_container_width=True, type="primary"):

    with st.spinner(f"Processing {selected_file}..."):
        try:
            processor = DataProcessor()

            output_name = custom_output.strip() or None
            df = processor.process_file(
                selected_path,
                output_filename=output_name,
            )

            st.session_state["processed_df"]   = df
            st.session_state["processed_file"] = selected_file
            st.success(
                f"✅ Done! `{selected_file}` processed — "
                f"{len(df)} rows × {len(df.columns)} columns saved to `data/processed/`."
            )

        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
            st.exception(e)
            st.stop()

# ── Results (shown after processing) ─────────────────────────────────────────
if "processed_df" in st.session_state:
    df = st.session_state["processed_df"]

    st.divider()
    st.markdown("### Results")

    # ── Column summary ────────────────────────────────────────────────────────
    ohlcv_cols     = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    ema_cols       = [c for c in df.columns if c.startswith("EMA_")]
    sma_cols       = [c for c in df.columns if c.startswith("SMA_")]
    level_cols     = [c for c in df.columns if c in ["PDH","PDL","PD_EQ","PWH","PWL","PW_EQ","WH","WL"]]
    day_cols       = [c for c in df.columns if c in ["GD", "RD", "GSD", "RSD"]]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total rows",    len(df))
    m2.metric("EMAs added",    len(ema_cols))
    m3.metric("SMAs added",    len(sma_cols))
    m4.metric("Price levels",  len(level_cols))
    m5.metric("Day classif.",  len(day_cols))

    # ── Tabs: data / indicators / levels / day classification / chart ─────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Raw Data", "📊 EMAs & SMAs", "🎯 Price Levels", "📅 Day Classification", "📈 Chart"])

    with tab1:
        st.dataframe(df[ohlcv_cols].tail(50), use_container_width=True)

    with tab2:
        if ema_cols or sma_cols:
            st.dataframe(df[ema_cols + sma_cols].tail(50), use_container_width=True)
            st.markdown("**NaN counts (warm-up period):**")
            nan_df = df[ema_cols + sma_cols].isnull().sum().rename("NaN count").to_frame()
            nan_df["NaN %"] = (nan_df["NaN count"] / len(df) * 100).round(2)
            st.dataframe(nan_df, use_container_width=True)
        else:
            st.info("No EMA/SMA columns found.")

    with tab3:
        if level_cols:
            st.dataframe(df[level_cols].tail(50), use_container_width=True)

            # NaN breakdown for levels
            level_nan = df[level_cols].isnull().sum().rename("NaN count").to_frame()
            level_nan["NaN %"] = (level_nan["NaN count"] / len(df) * 100).round(2)
            st.markdown("**NaN counts:**")
            st.dataframe(level_nan, use_container_width=True)
        else:
            st.info("No price level columns found.")

    with tab4:
        if day_cols:
            # ── Summary counts ────────────────────────────────────────────────
            if "day_type" in df.columns:
                day_type_counts = (
                    df.drop_duplicates(subset=[df.index.normalize().name if df.index.name else df.index.to_series().dt.normalize().name])
                    if False  # keep all rows, count by unique days below
                    else df.groupby(df.index.normalize())["day_type"].last()
                    .value_counts()
                    .reset_index()
                    .rename(columns={"day_type": "Day Type", "count": "Count"})
                )
                st.markdown("**Day type distribution (unique trading days):**")
                st.dataframe(day_type_counts, use_container_width=True)

            st.markdown("**Raw classification columns (first 50 rows):**")
            st.dataframe(df[day_cols].tail(50), use_container_width=True)

            # NaN summary
            nan_day = df[day_cols].isnull().sum().rename("NaN count").to_frame()
            nan_day["NaN %"] = (nan_day["NaN count"] / len(df) * 100).round(2)
            if nan_day["NaN count"].sum() > 0:
                st.markdown("**NaN counts:**")
                st.dataframe(nan_day[nan_day["NaN count"] > 0], use_container_width=True)
        else:
            st.info("No day classification columns found.")

    with tab5:
        # Limit sample to keep chart fast
        sample_size = min(len(df), 500)
        sample = df.iloc[:sample_size]

        fig, axes = plt.subplots(
            2, 1, figsize=(14, 8),
            gridspec_kw={"height_ratios": [3, 1]}
        )

        # Close + EMAs/SMAs
        ax = axes[0]
        ax.plot(sample.index, sample["close"],
                color="#263238", linewidth=1.0, label="close", zorder=5)

        colors = ["#FF5722", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
        for col, color in zip(ema_cols, colors):
            ax.plot(sample.index, sample[col],
                    color=color, linewidth=0.85, alpha=0.9, label=col)
        for col, color in zip(sma_cols, colors[len(ema_cols):]):
            ax.plot(sample.index, sample[col],
                    color=color, linewidth=0.85, linestyle="--", alpha=0.9, label=col)

        ax.set_title(f"EMAs & SMAs — {st.session_state['processed_file']} (first {sample_size} bars)",
                     fontsize=11)
        ax.legend(fontsize=8, ncol=6)
        ax.grid(True, alpha=0.25)

        # Price levels
        ax2 = axes[1]
        ax2.plot(sample.index, sample["close"],
                 color="#263238", linewidth=0.8, alpha=0.4)

        level_styles = {
            "PDH":   ("#E53935", "-"),
            "PDL":   ("#43A047", "-"),
            "PD_EQ": ("#FB8C00", "--"),
            "PWH":   ("#D81B60", ":"),
            "PWL":   ("#00897B", ":"),
            "PW_EQ": ("#7B1FA2", ":"),
            "WH":    ("#FF6F00", "-"),
            "WL":    ("#1565C0", "-"),
        }
        for col, (color, ls) in level_styles.items():
            if col in sample.columns:
                ax2.plot(sample.index, sample[col],
                         color=color, linewidth=1.0, linestyle=ls,
                         alpha=0.9, label=col)

        ax2.set_title("Daily & Weekly Price Levels", fontsize=11)
        ax2.legend(fontsize=8, ncol=5)
        ax2.grid(True, alpha=0.25)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ── Existing processed files ──────────────────────────────────────────────────
st.divider()
st.markdown("### All Processed Files in `data/processed/`")

processed_files = sorted(PROCESSED_DIR.glob("*.csv"))
if not processed_files:
    st.info("No processed files yet.")
else:
    rows = []
    for f in processed_files:
        size_kb = f.stat().st_size / 1024
        # Peek at column count
        try:
            peek = pd.read_csv(f, nrows=1)
            cols = len(peek.columns)
        except Exception:
            cols = "?"
        rows.append({
            "File":       f.name,
            "Size (KB)":  round(size_kb, 1),
            "Columns":    cols,
        })

    st.dataframe(
        pd.DataFrame(rows).set_index("File"),
        use_container_width=True,
    )
    st.caption(f"{len(processed_files)} file(s) — {PROCESSED_DIR}")
