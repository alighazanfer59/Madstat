"""
src/data/process_data.py
------------------------
Loads a raw OHLCV CSV from data/raw/, calculates all EMAs, SMAs and
daily/weekly price levels, then saves the enriched DataFrame to data/processed/.

Column contract
---------------
Our fetcher saves lowercase columns: open, high, low, close, volume.
The EMA_SMA_Calculator and PriceLevelsCalculator in the indicators folder
use Title-case columns (Open, High, Low, Close).  This module normalises
to Title-case before calling those helpers and converts back to lowercase
before saving — so the processed files stay consistent with the rest of
the pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

# ── Make src/ importable regardless of working directory ─────────────────────
sys.path.append(str(Path(__file__).resolve().parents[1]))   # adds .../src/

from indicators.ema    import EMA_SMA_Calculator
from indicators.levels import PriceLevelsCalculator

log = logging.getLogger(__name__)

# ── Directory constants ───────────────────────────────────────────────────────
_PROJECT_ROOT  = Path(__file__).resolve().parents[2]  # madstrat_backtest/
RAW_DATA_DIR   = _PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR  = _PROJECT_ROOT / "data" / "processed"


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  DataProcessor
# ╚══════════════════════════════════════════════════════════════════════════════

class DataProcessor:
    """
    Enriches a raw OHLCV DataFrame with EMAs, SMAs and key price levels.

    Default indicator set (mirrors Madstrat manual requirements):
        EMAs : 9, 18, 50
        SMAs : 50
        Levels : PDH, PDL, PD-EQ, PWH, PWL (rolling 5-day window)

    Usage
    -----
    processor = DataProcessor()

    # Process one file and save
    df = processor.process_file("EURUSD_15m_2024-01-01_2024-06-30_yfinance.csv")

    # Process all raw files at once
    results = processor.process_all()
    """

    def __init__(
        self,
        ema_periods: List[int] = None,
        sma_periods: List[int] = None,
        pwh_window:  int       = 5,
    ):
        self.ema_periods = ema_periods or [9, 18, 50]
        self.sma_periods = sma_periods or [50]
        self.pwh_window  = pwh_window

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def process_file(
        self,
        filename: Union[str, Path],
        output_filename: Union[str, Path] = None,
    ) -> pd.DataFrame:
        """
        Load one raw CSV, add all indicators, save to data/processed/.

        Parameters
        ----------
        filename : str | Path
            Filename only (looked up in data/raw/) OR a full path.
        output_filename : str | Path, optional
            Custom output filename in data/processed/.
            Defaults to <original_stem>_processed.csv

        Returns
        -------
        pd.DataFrame  — fully enriched DataFrame
        """
        input_path  = self._resolve_input(filename)
        output_path = self._resolve_output(input_path, output_filename)

        log.info(f"Processing: {input_path.name}")

        df = self._load(input_path)
        df = self._add_emas_smas(df)
        df = self._add_price_levels(df)

        df.to_csv(output_path)
        log.info(
            f"Saved → {output_path.name}  "
            f"({len(df)} rows × {len(df.columns)} columns)"
        )

        return df

    def process_all(self) -> Dict[str, pd.DataFrame]:
        """
        Process every CSV in data/raw/ and save each to data/processed/.

        Returns
        -------
        dict  {filename: processed_df}
        """
        raw_files = sorted(RAW_DATA_DIR.glob("*.csv"))
        if not raw_files:
            log.warning(f"No CSV files found in {RAW_DATA_DIR}")
            return {}

        log.info(f"Found {len(raw_files)} file(s) to process.")
        results = {}

        for path in raw_files:
            try:
                results[path.name] = self.process_file(path)
            except Exception as exc:
                log.error(f"Failed to process {path.name}: {exc}", exc_info=True)

        return results

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self, path: Path) -> pd.DataFrame:
        """Read CSV, set DatetimeIndex, normalise columns to Title-case."""
        df = pd.read_csv(path, index_col=0, parse_dates=True)

        # Ensure the index is a proper DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"{path.name}: index is not a DatetimeIndex after parsing. "
                f"Got {type(df.index).__name__}. Check the raw file."
            )

        # Normalise column names to Title-case so the indicator helpers
        # (which expect 'High', 'Low', 'Close', etc.) work without changes.
        df = self._to_titlecase(df)

        log.info(f"Loaded {len(df)} rows | columns: {list(df.columns)}")
        return df

    # ── Indicators ────────────────────────────────────────────────────────────

    def _add_emas_smas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA_{period} and SMA_{period} columns via EMA_SMA_Calculator."""

        # EMAs
        df = EMA_SMA_Calculator.calculate_multiple_emas(
            df, price_col="Close", periods=self.ema_periods
        )
        log.info(f"Added EMAs: {['EMA_' + str(p) for p in self.ema_periods]}")

        # SMAs
        df = EMA_SMA_Calculator.calculate_multiple_smas(
            df, price_col="Close", periods=self.sma_periods
        )
        log.info(f"Added SMAs: {['SMA_' + str(p) for p in self.sma_periods]}")

        return df

    def _add_price_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add PDH, PDL, PD-EQ, PWH, PWL columns via PriceLevelsCalculator.

        levels.py has two code paths:
          - timeframe='1D'   → uses df['High'].shift(1) directly (works fine)
          - intraday         → originally used df['Date'] column, but our data
                               has a DatetimeIndex.  We inject a 'Date' column
                               temporarily so the existing helper works unchanged.
        """
        # Detect if this is daily data (only one bar per calendar date)
        is_daily = df.index.normalize().nunique() == len(df)

        if is_daily:
            df = PriceLevelsCalculator.calculate_all_levels(
                df, pwh_window=self.pwh_window
            )
        else:
            # Intraday: inject a 'Date' column the helper expects, then clean up
            df = df.copy()
            df["Date"] = df.index
            df = PriceLevelsCalculator.calculate_all_levels(
                df, pwh_window=self.pwh_window
            )
            if "Date" in df.columns:
                df.drop(columns=["Date"], inplace=True)

        log.info("Added price levels: PDH, PDL, PD-EQ, PWH, PWL")
        return df

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_titlecase(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename lowercase OHLCV columns to Title-case.
        e.g. open → Open, high → High, close → Close, volume → Volume
        Leaves any already-correct or unrecognised columns untouched.
        """
        rename_map = {
            "open":   "Open",
            "high":   "High",
            "low":    "Low",
            "close":  "Close",
            "volume": "Volume",
        }
        return df.rename(columns={k: v for k, v in rename_map.items()
                                   if k in df.columns})

    @staticmethod
    def _resolve_input(filename: Union[str, Path]) -> Path:
        path = Path(filename)
        if path.is_absolute() or path.exists():
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            return path
        # Treat as filename relative to data/raw/
        full = RAW_DATA_DIR / path.name
        if not full.exists():
            raise FileNotFoundError(
                f"'{path.name}' not found in data/raw/ ({RAW_DATA_DIR}). "
                f"Available files:\n"
                + "\n".join(f"  {f.name}" for f in RAW_DATA_DIR.glob("*.csv"))
            )
        return full

    @staticmethod
    def _resolve_output(
        input_path: Path,
        output_filename: Union[str, Path, None],
    ) -> Path:
        if output_filename:
            p = Path(output_filename)
            return p if p.is_absolute() else PROCESSED_DIR / p.name
        return PROCESSED_DIR / f"{input_path.stem}_processed.csv"


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  CLI entry point  — python process_data.py [filename]
# ╚══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Process a raw Madstrat CSV — adds EMAs, SMAs and price levels."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default=None,
        help=(
            "CSV filename in data/raw/ (e.g. EURUSD_15m_2024-01-01_2024-06-30_yfinance.csv) "
            "or a full path. Omit to process ALL files in data/raw/."
        ),
    )
    parser.add_argument(
        "--ema", nargs="+", type=int, default=[9, 18, 50],
        help="EMA periods (default: 9 18 50)"
    )
    parser.add_argument(
        "--sma", nargs="+", type=int, default=[50],
        help="SMA periods (default: 50)"
    )
    parser.add_argument(
        "--pwh-window", type=int, default=5,
        help="Rolling window for PWH/PWL in days (default: 5)"
    )

    args = parser.parse_args()

    processor = DataProcessor(
        ema_periods=args.ema,
        sma_periods=args.sma,
        pwh_window=args.pwh_window,
    )

    if args.filename:
        df = processor.process_file(args.filename)
        print(f"\nDone — {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
    else:
        results = processor.process_all()
        print(f"\nDone — processed {len(results)} file(s)")


if __name__ == "__main__":
    main()
