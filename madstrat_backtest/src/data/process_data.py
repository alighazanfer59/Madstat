"""
src/data/process_data.py
------------------------
Standalone data processor for Madstrat 2.0.

Loads a raw OHLCV CSV from data/raw/, adds all required indicators
and price levels, then saves the enriched file to data/processed/.

No external indicator imports — all calculations are self-contained.

How daily data is sourced
--------------------------
All day-level classifications (GSD, RSD, GD, RD, inside_day, FBR,
clean_BO) require a proper daily candle — open, high, low, close for
the full session.

When working from a 15m (or any intraday) file, resampling to daily
is imperfect because:
  - The daily open  = first 15m bar of the session (may miss pre-market)
  - The daily close = last  15m bar (may differ from the broker's daily close)

To get accurate daily candles, pass a separate daily CSV via the
`daily_filename` parameter.  If omitted, the processor falls back to
resampling the intraday data — still useful for quick tests.

Indicators added
----------------
EMAs  : EMA_9, EMA_18, EMA_50   (on close price of the intraday file)
SMA   : SMA_50                   (on close price of the intraday file)

Price levels added
------------------
PDH    Previous Day High
PDL    Previous Day Low
PD_EQ  Previous Day Equilibrium  = (PDH + PDL) / 2
PWH    Previous Week High
PWL    Previous Week Low
PW_EQ  Previous Week Equilibrium = (PWH + PWL) / 2
WH     Current Week High         (running high since Monday open)
WL     Current Week Low          (running low since Monday open)

Day classification columns  (computed on daily candles, ffill to intraday)
--------------------------------------------------------------------------
GD            bool   Green Day (bullish, not a GSD)
RD            bool   Red Day   (bearish, not an RSD)
GSD           bool   Green Setup Day - green after 2+ consecutive red days
RSD           bool   Red Setup Day   - red   after 2+ consecutive green days
inside_day    bool   Daily range fully inside prior day's range
FBR           bool   False Break Reversal (bull or bear)
FBR_bull      bool   High broke above PWH, close back inside
FBR_bear      bool   Low  broke below PWL, close back inside
clean_BO      bool   Clean Breakout (bull or bear)
clean_BO_bull bool   Daily close above PWH
clean_BO_bear bool   Daily close below PWL
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# ── Directory constants ───────────────────────────────────────────────────────
# process_data.py -> src/data/ -> src/ -> madstrat_backtest/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR  = _PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  DataProcessor
# ╚══════════════════════════════════════════════════════════════════════════════

class DataProcessor:
    """
    Loads a raw OHLCV CSV and enriches it with EMAs, SMAs,
    daily/weekly price levels and day classifications.

    Usage — intraday file only (daily data resampled internally)
    ------------------------------------------------------------
    processor = DataProcessor()
    df = processor.process_file("EURUSD_15m_2024-01-01_2024-06-30_yfinance.csv")

    Usage — intraday + separate daily file (recommended for accuracy)
    -----------------------------------------------------------------
    df = processor.process_file(
        filename       = "EURUSD_15m_2024-01-01_2024-06-30_yfinance.csv",
        daily_filename = "EURUSD_1d_2024-01-01_2024-06-30_yfinance.csv",
    )
    """

    def __init__(self):
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def process_file(
        self,
        filename: str,
        daily_filename: str = None,
        output_filename: str = None,
    ) -> pd.DataFrame:
        """
        Load one raw CSV, add all indicators and levels, save to data/processed/.

        Parameters
        ----------
        filename : str
            Intraday CSV filename in data/raw/ or a full path.
        daily_filename : str, optional
            Separate daily CSV filename in data/raw/ or a full path.
            When provided, all day-level calculations use real daily candles
            instead of resampling from the intraday file.
        output_filename : str, optional
            Output filename in data/processed/.
            Defaults to <intraday_stem>_processed.csv

        Returns
        -------
        pd.DataFrame  fully enriched intraday DataFrame
        """
        input_path  = self._resolve_input(filename)
        output_path = self._resolve_output(input_path, output_filename)

        log.info(f"Processing : {input_path.name}")

        # ── Load intraday data ────────────────────────────────────────────────
        df = self._load(input_path)

        # ── Load or resample daily data ───────────────────────────────────────
        if daily_filename:
            daily_path = self._resolve_input(daily_filename)
            log.info(f"Daily file : {daily_path.name}")
            daily = self._load(daily_path)
        else:
            log.warning(
                "No daily_filename provided — resampling intraday data to "
                "build daily candles. For best accuracy supply a separate "
                "daily CSV fetched from the same source."
            )
            daily = self._resample_to_daily(df)

        # ── Pipeline ──────────────────────────────────────────────────────────
        df = self._add_ema_sma(df)
        df = self._add_daily_levels(df, daily)
        df = self._add_weekly_levels(df, daily)
        df = self._add_day_classification(df, daily)
        df = self._add_fbr(df, daily)
        df = self._add_clean_breakout(df, daily)

        df.to_csv(output_path)
        log.info(f"Saved -> {output_path.name}  ({len(df)} rows x {len(df.columns)} columns)")

        return df

    def process_all(self) -> dict:
        """Process every CSV in data/raw/ and save each to data/processed/."""
        raw_files = sorted(RAW_DATA_DIR.glob("*.csv"))
        if not raw_files:
            log.warning(f"No CSV files found in {RAW_DATA_DIR}")
            return {}

        results = {}
        for path in raw_files:
            try:
                results[path.name] = self.process_file(path)
            except Exception as exc:
                log.error(f"Failed: {path.name} - {exc}", exc_info=True)
        return results

    # ── Step 1: Load ──────────────────────────────────────────────────────────

    def _load(self, path: Path) -> pd.DataFrame:
        """Read CSV, set DatetimeIndex, enforce lowercase column names.

        Handles three common CSV layouts:
          1. yfinance   — datetime is the index column (index_col=0)
          2. TradingView — datetime is a regular column named 'datetime'
          3. Other       — datetime column named 'date', 'time', or 'timestamp'

        All indexes are normalised to tz-naive UTC so reindex() comparisons
        never raise a dtype mismatch between aware and naive timestamps.
        """
        # ── Read without forcing an index first ───────────────────────────────
        df = pd.read_csv(path)

        # Lowercase all column names immediately
        df.columns = [c.lower() for c in df.columns]

        # ── Locate and set the datetime index ─────────────────────────────────
        # Priority order of candidate column names
        dt_candidates = ["datetime", "date", "time", "timestamp"]

        dt_col = None
        for candidate in dt_candidates:
            if candidate in df.columns:
                dt_col = candidate
                break

        if dt_col:
            # Datetime is a regular column — parse and set as index
            df[dt_col] = pd.to_datetime(df[dt_col])
            df = df.set_index(dt_col)
            df.index.name = "datetime"
        else:
            # Assume first column is the datetime index (yfinance layout)
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.columns = [c.lower() for c in df.columns]

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"{path.name}: could not find a datetime index or column. "
                f"Columns found: {list(df.columns)}"
            )

        # ── Strip timezone — normalise to tz-naive UTC ────────────────────────
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        # ── Confirm required OHLC columns are present ─────────────────────────
        required = ["open", "high", "low", "close"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path.name}: missing columns {missing}. "
                f"Got: {list(df.columns)}"
            )

        df = df.sort_index()
        log.info(f"Loaded {len(df)} rows | {df.index[0]} -> {df.index[-1]}")
        return df

    # ── Step 1b: Resample fallback ────────────────────────────────────────────

    def _resample_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a daily OHLCV frame from intraday data.
        Used only when no daily_filename is supplied.
        """
        daily = pd.DataFrame({
            "open":  df["open"].resample("1D").first(),
            "high":  df["high"].resample("1D").max(),
            "low":   df["low"].resample("1D").min(),
            "close": df["close"].resample("1D").last(),
        }).dropna()
        log.info(f"Resampled to {len(daily)} daily bars (fallback mode)")
        return daily

    # ── Step 2: EMAs and SMA ──────────────────────────────────────────────────

    def _add_ema_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA_9, EMA_18, EMA_50 and SMA_50 on the intraday close price."""
        close = df["close"]
        df["EMA_9"]  = close.ewm(span=9,  adjust=False).mean()
        df["EMA_18"] = close.ewm(span=18, adjust=False).mean()
        df["EMA_50"] = close.ewm(span=50, adjust=False).mean()
        df["SMA_50"] = close.rolling(window=50).mean()
        log.info("Added: EMA_9, EMA_18, EMA_50, SMA_50")
        return df

    # ── Step 3: Daily levels ──────────────────────────────────────────────────

    def _add_daily_levels(self, df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add PDH, PDL, PD_EQ using the daily frame.
        Shifts by 1 row so each bar carries the PREVIOUS day's levels,
        then forward-fills onto every intraday bar.
        """
        d = daily[["high", "low"]].copy()
        d["PDH"]   = d["high"].shift(1)
        d["PDL"]   = d["low"].shift(1)
        d["PD_EQ"] = (d["PDH"] + d["PDL"]) / 2

        aligned = d[["PDH", "PDL", "PD_EQ"]].reindex(df.index, method="ffill")
        df["PDH"]   = aligned["PDH"]
        df["PDL"]   = aligned["PDL"]
        df["PD_EQ"] = aligned["PD_EQ"]

        log.info("Added: PDH, PDL, PD_EQ")
        return df

    # ── Step 4: Weekly levels ─────────────────────────────────────────────────

    def _add_weekly_levels(self, df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add PWH, PWL, PW_EQ from the daily frame (previous week),
        and WH, WL from the intraday frame (current week running).
        """
        # Previous week levels built from daily candles
        weekly = (
            daily["high"].resample("W-MON", label="left", closed="left").max()
            .to_frame("weekly_high")
            .join(
                daily["low"].resample("W-MON", label="left", closed="left")
                .min()
                .rename("weekly_low")
            )
            .dropna()
        )

        weekly["PWH"]   = weekly["weekly_high"].shift(1)
        weekly["PWL"]   = weekly["weekly_low"].shift(1)
        weekly["PW_EQ"] = (weekly["PWH"] + weekly["PWL"]) / 2

        weekly_aligned = weekly[["PWH", "PWL", "PW_EQ"]].reindex(df.index, method="ffill")
        df["PWH"]   = weekly_aligned["PWH"]
        df["PWL"]   = weekly_aligned["PWL"]
        df["PW_EQ"] = weekly_aligned["PW_EQ"]

        log.info("Added: PWH, PWL, PW_EQ")

        # Current week running high/low on intraday bars
        iso_week   = df.index.to_series().dt.isocalendar().week
        iso_year   = df.index.to_series().dt.isocalendar().year
        week_group = iso_year.astype(str) + "_" + iso_week.astype(str)

        df["WH"] = df.groupby(week_group)["high"].cummax()
        df["WL"] = df.groupby(week_group)["low"].cummin()

        log.info("Added: WH, WL")
        return df

    # ── Step 5: Day classification ────────────────────────────────────────────

    def _add_day_classification(self, df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each daily candle then forward-fill onto intraday bars.

        GSD : today GREEN  AND prior 2 days both RED
        RSD : today RED    AND prior 2 days both GREEN
        GD  : today GREEN, not a GSD
        RD  : today RED,   not an RSD
        inside_day : high <= prior high AND low >= prior low
        """
        d = daily[["open", "high", "low", "close"]].copy()

        # Basic direction
        d["GD"] = d["close"] > d["open"]
        d["RD"] = d["close"] < d["open"]

        # GSD / RSD vectorised
        d["GSD"] = np.where(
            d["GD"] & d["RD"].shift(1) & d["RD"].shift(2),
            True, False
        )
        d["RSD"] = np.where(
            d["RD"] & d["GD"].shift(1) & d["GD"].shift(2),
            True, False
        )

        # Clear GD/RD on setup days so each day has one label only
        d.loc[d["GSD"], "GD"] = False
        d.loc[d["RSD"], "RD"] = False

        # Inside day
        d["inside_day"] = (
            (d["high"] <= d["high"].shift(1)) &
            (d["low"]  >= d["low"].shift(1))
        )

        # Forward-fill onto intraday bars
        cols = ["GD", "RD", "GSD", "RSD", "inside_day"]
        aligned = d[cols].reindex(df.index, method="ffill")
        for col in cols:
            df[col] = aligned[col]

        counts = {col: int(d[col].sum()) for col in cols}
        log.info(f"Day classification: {counts}")
        return df

    # ── Step 6: False Break Reversal ─────────────────────────────────────────

    def _add_fbr(self, df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
        """
        False Break Reversal (FBR):
            Bull FBR : daily high > PWH  AND  daily close <= PWH
            Bear FBR : daily low  < PWL  AND  daily close >= PWL

        PWH/PWL come from the intraday df (step 4), sampled at the
        last bar of each day so they align with the daily candle.

        Columns: FBR, FBR_bull, FBR_bear
        """
        d = daily[["high", "low", "close"]].copy()

        # Pull PWH/PWL from intraday (last value per day = value in force that day)
        d["PWH"] = df["PWH"].resample("1D").last()
        d["PWL"] = df["PWL"].resample("1D").last()
        d = d.dropna(subset=["PWH", "PWL"])

        d["FBR_bull"] = (d["high"]  > d["PWH"]) & (d["close"] <= d["PWH"])
        d["FBR_bear"] = (d["low"]   < d["PWL"]) & (d["close"] >= d["PWL"])
        d["FBR"]      = d["FBR_bull"] | d["FBR_bear"]

        for col in ["FBR", "FBR_bull", "FBR_bear"]:
            df[col] = d[col].reindex(df.index, method="ffill")

        counts = {col: int(d[col].sum()) for col in ["FBR", "FBR_bull", "FBR_bear"]}
        log.info(f"Added FBR: {counts}")
        return df

    # ── Step 7: Clean Breakout ────────────────────────────────────────────────

    def _add_clean_breakout(self, df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
        """
        Clean Breakout: daily close holds outside the weekly range.
            Bull : daily close > PWH
            Bear : daily close < PWL

        Columns: clean_BO, clean_BO_bull, clean_BO_bear
        """
        d = daily[["close"]].copy()
        d["PWH"] = df["PWH"].resample("1D").last()
        d["PWL"] = df["PWL"].resample("1D").last()
        d = d.dropna(subset=["PWH", "PWL"])

        d["clean_BO_bull"] = d["close"] > d["PWH"]
        d["clean_BO_bear"] = d["close"] < d["PWL"]
        d["clean_BO"]      = d["clean_BO_bull"] | d["clean_BO_bear"]

        for col in ["clean_BO", "clean_BO_bull", "clean_BO_bear"]:
            df[col] = d[col].reindex(df.index, method="ffill")

        counts = {col: int(d[col].sum()) for col in ["clean_BO", "clean_BO_bull", "clean_BO_bear"]}
        log.info(f"Added clean_BO: {counts}")
        return df

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_input(filename) -> Path:
        path = Path(filename)
        if path.is_absolute() or path.exists():
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            return path
        full = RAW_DATA_DIR / path.name
        if not full.exists():
            available = "\n".join(f"  {f.name}" for f in RAW_DATA_DIR.glob("*.csv"))
            raise FileNotFoundError(
                f"'{path.name}' not found in data/raw/.\n"
                f"Available files:\n{available or '  (none)'}"
            )
        return full

    @staticmethod
    def _resolve_output(input_path: Path, output_filename) -> Path:
        if output_filename:
            p = Path(output_filename)
            return p if p.is_absolute() else PROCESSED_DIR / p.name
        return PROCESSED_DIR / f"{input_path.stem}_processed.csv"


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  CLI  -  python process_data.py [filename] [--daily daily_filename]
# ╚══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Madstrat 2.0 - process a raw CSV with EMAs and price levels."
    )
    parser.add_argument(
        "filename", nargs="?", default=None,
        help="Intraday CSV in data/raw/ or full path. Omit to process ALL files."
    )
    parser.add_argument(
        "--daily", default=None, dest="daily_filename",
        help="Separate daily CSV in data/raw/ for accurate day classification."
    )
    args = parser.parse_args()

    processor = DataProcessor()

    if args.filename:
        df = processor.process_file(
            args.filename,
            daily_filename=args.daily_filename,
        )
        print(f"\nDone - {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
    else:
        results = processor.process_all()
        print(f"\nDone - processed {len(results)} file(s)")


if __name__ == "__main__":
    main()
