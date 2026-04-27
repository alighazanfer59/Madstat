"""
src/data/process_data.py
------------------------
Standalone data processor for Madstrat 2.0.

Loads a raw OHLCV CSV from data/raw/, adds all required indicators
and price levels, then saves the enriched file to data/processed/.

No external indicator imports — all calculations are self-contained.

Indicators added
----------------
EMAs  : EMA_9, EMA_18, EMA_50   (on close price)
SMA   : SMA_50                   (on close price)

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
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# ── Directory constants ───────────────────────────────────────────────────────
# process_data.py → src/data/ → src/ → madstrat_backtest/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR  = _PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  DataProcessor
# ╚══════════════════════════════════════════════════════════════════════════════

class DataProcessor:
    """
    Loads a raw OHLCV CSV and enriches it with EMAs, SMAs and
    daily/weekly price levels required by the Madstrat strategy.

    All calculations are self-contained — no external indicator modules needed.

    Usage
    -----
    processor = DataProcessor()
    df = processor.process_file("EURUSD_15m_2024-01-01_2024-06-30_yfinance.csv")
    """

    def __init__(self):
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def process_file(
        self,
        filename: str,
        output_filename: str = None,
    ) -> pd.DataFrame:
        """
        Load one raw CSV, add all indicators and levels, save to data/processed/.

        Parameters
        ----------
        filename : str
            Filename in data/raw/ (e.g. EURUSD_15m_2024-01-01_2024-06-30_yfinance.csv)
            or a full absolute path.
        output_filename : str, optional
            Output filename in data/processed/.
            Defaults to <stem>_processed.csv

        Returns
        -------
        pd.DataFrame  fully enriched DataFrame
        """
        input_path  = self._resolve_input(filename)
        output_path = self._resolve_output(input_path, output_filename)

        log.info(f"Processing: {input_path.name}")

        df = self._load(input_path)
        df = self._add_ema_sma(df)
        df = self._add_daily_levels(df)
        df = self._add_weekly_levels(df)

        df.to_csv(output_path)
        log.info(f"Saved → {output_path.name}  ({len(df)} rows × {len(df.columns)} columns)")

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
                log.error(f"Failed: {path.name} — {exc}", exc_info=True)
        return results

    # ── Step 1: Load ──────────────────────────────────────────────────────────

    def _load(self, path: Path) -> pd.DataFrame:
        """Read CSV, set DatetimeIndex, enforce lowercase column names."""
        df = pd.read_csv(path, index_col=0, parse_dates=True)

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"{path.name}: could not parse index as DatetimeIndex. "
                f"Got {type(df.index).__name__}."
            )

        # Normalise column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Confirm required columns are present
        required = ["open", "high", "low", "close"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}. Got: {list(df.columns)}")

        df = df.sort_index()
        log.info(f"Loaded {len(df)} rows | {df.index[0]} → {df.index[-1]}")
        return df

    # ── Step 2: EMAs and SMA ──────────────────────────────────────────────────

    def _add_ema_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add EMA_9, EMA_18, EMA_50 and SMA_50 columns calculated on close price.
        Uses pandas ewm() for EMAs and rolling() for SMA.
        """
        close = df["close"]

        df["EMA_9"]  = close.ewm(span=9,  adjust=False).mean()
        df["EMA_18"] = close.ewm(span=18, adjust=False).mean()
        df["EMA_50"] = close.ewm(span=50, adjust=False).mean()
        df["SMA_50"] = close.rolling(window=50).mean()

        log.info("Added: EMA_9, EMA_18, EMA_50, SMA_50")
        return df

    # ── Step 3: Daily levels ──────────────────────────────────────────────────

    def _add_daily_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add PDH, PDL, PD_EQ — all based on the PREVIOUS calendar day's range.

        Method
        ------
        1. Build a daily OHLC summary from the base data (works for any timeframe).
        2. Shift it by 1 day to get "previous day" values.
        3. Forward-fill onto every intraday bar within that day.
        """
        # ── Daily OHLC from the raw bars ──────────────────────────────────────
        daily = (
            df["high"].resample("1D").max()
            .to_frame("daily_high")
            .join(df["low"].resample("1D").min().rename("daily_low"))
            .dropna()
        )

        # Previous day values (shift by 1 day)
        daily["PDH"]   = daily["daily_high"].shift(1)
        daily["PDL"]   = daily["daily_low"].shift(1)
        daily["PD_EQ"] = (daily["PDH"] + daily["PDL"]) / 2

        # Keep only the level columns
        daily = daily[["PDH", "PDL", "PD_EQ"]]

        # Forward-fill onto every intraday bar: reindex to the original index
        # then ffill so all bars within a day carry the same prior-day levels
        daily_reindexed = daily.reindex(df.index, method="ffill")

        df["PDH"]   = daily_reindexed["PDH"]
        df["PDL"]   = daily_reindexed["PDL"]
        df["PD_EQ"] = daily_reindexed["PD_EQ"]

        log.info("Added: PDH, PDL, PD_EQ")
        return df

    # ── Step 4: Weekly levels ─────────────────────────────────────────────────

    def _add_weekly_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add PWH, PWL, PW_EQ, WH, WL.

        PWH / PWL / PW_EQ — PREVIOUS week's high, low and midpoint.
            Built from weekly OHLC resampled to week-start (Monday),
            then shifted by 1 week and forward-filled onto intraday bars.

        WH / WL — CURRENT (running) week high and low.
            At each bar: the highest high / lowest low since Monday open
            of the current week.
        """
        # ── Previous week levels ──────────────────────────────────────────────
        weekly = (
            df["high"].resample("W-MON", label="left", closed="left").max()
            .to_frame("weekly_high")
            .join(
                df["low"].resample("W-MON", label="left", closed="left")
                .min()
                .rename("weekly_low")
            )
            .dropna()
        )

        weekly["PWH"]   = weekly["weekly_high"].shift(1)
        weekly["PWL"]   = weekly["weekly_low"].shift(1)
        weekly["PW_EQ"] = (weekly["PWH"] + weekly["PWL"]) / 2

        weekly = weekly[["PWH", "PWL", "PW_EQ"]]

        weekly_reindexed = weekly.reindex(df.index, method="ffill")
        df["PWH"]   = weekly_reindexed["PWH"]
        df["PWL"]   = weekly_reindexed["PWL"]
        df["PW_EQ"] = weekly_reindexed["PW_EQ"]

        log.info("Added: PWH, PWL, PW_EQ")

        # ── Current week running high / low ───────────────────────────────────
        # dayofweek: Monday=0, Sunday=6
        # We group by ISO year+week so the window resets every Monday.
        iso_week = df.index.to_series().dt.isocalendar().week
        iso_year = df.index.to_series().dt.isocalendar().year
        week_group = iso_year.astype(str) + "_" + iso_week.astype(str)

        df["WH"] = df.groupby(week_group)["high"].cummax()
        df["WL"] = df.groupby(week_group)["low"].cummin()

        log.info("Added: WH, WL")
        return df

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_input(filename) -> Path:
        path = Path(filename)
        # If it's already an absolute or existing path, use it directly
        if path.is_absolute() or path.exists():
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            return path
        # Otherwise look in data/raw/
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
# ║  CLI  —  python process_data.py [filename]
# ╚══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Madstrat 2.0 — process a raw CSV with EMAs and price levels."
    )
    parser.add_argument(
        "filename", nargs="?", default=None,
        help="Filename in data/raw/ or full path. Omit to process ALL files."
    )
    args = parser.parse_args()

    processor = DataProcessor()

    if args.filename:
        df = processor.process_file(args.filename)
        print(f"\nDone — {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
    else:
        results = processor.process_all()
        print(f"\nDone — processed {len(results)} file(s)")


if __name__ == "__main__":
    main()
