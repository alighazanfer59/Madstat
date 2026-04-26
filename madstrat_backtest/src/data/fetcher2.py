"""
src/data/fetcher.py
-------------------
Fetches raw OHLCV data for Madstrat 2.0 backtesting.

Primary source : yfinance  (no API key, easy setup)
Fallback note  : For deeper forex history or tighter spreads,
                 swap the backend to MetaTrader5 — the public
                 interface (fetch / save / load) stays identical.

Supported instruments : EURUSD=X, GBPUSD=X, EURGBP=X, XAUUSD=X
Supported timeframes  : 1m, 5m, 15m, 30m, 1h, 2h, 4h, 1d
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# ── yfinance ticker symbols for each instrument ──────────────────────────────
TICKER_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "EURGBP": "EURGBP=X",
    "XAUUSD": "GC=F",          # Gold futures — closest yfinance proxy
}

# ── yfinance interval strings ─────────────────────────────────────────────────
INTERVAL_MAP = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "2h":  "2h",
    "4h":  "4h",           # NOTE: yfinance does not support 4h natively;
    "1d":  "1d",           #       we resample from 1h automatically.
}

# ── yfinance history limits per interval ─────────────────────────────────────
# yfinance restricts how far back intraday data goes.
MAX_LOOKBACK_DAYS = {
    "1m":  7,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "1h":  730,
    "2h":  730,
    "4h":  730,
    "1d":  3650,
}

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


class DataFetcher:
    """
    Downloads and caches OHLCV data for a single instrument.

    Usage
    -----
    fetcher = DataFetcher(instrument="EURUSD", start="2024-01-01", end="2024-06-30")
    df_15m  = fetcher.fetch("15m")
    df_1h   = fetcher.fetch("1h")
    df_4h   = fetcher.fetch("4h")   # auto-resampled from 1h
    """

    def __init__(
        self,
        instrument: str,
        start: str,
        end: str,
        save_raw: bool = True,
    ):
        """
        Parameters
        ----------
        instrument : str
            One of EURUSD, GBPUSD, EURGBP, XAUUSD
        start : str
            Start date in YYYY-MM-DD format
        end : str
            End date in YYYY-MM-DD format
        save_raw : bool
            If True, saves a CSV to data/raw/ after each fetch
        """
        instrument = instrument.upper()
        if instrument not in TICKER_MAP:
            raise ValueError(
                f"Unsupported instrument '{instrument}'. "
                f"Choose from: {list(TICKER_MAP.keys())}"
            )

        self.instrument = instrument
        self.ticker     = TICKER_MAP[instrument]
        self.start      = start
        self.end        = end
        self.save_raw   = save_raw

        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        log.info(
            f"DataFetcher ready — {instrument} | {start} → {end}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(self, timeframe: str) -> pd.DataFrame:
        """
        Fetch OHLCV data for the given timeframe.

        Returns a DataFrame with columns:
            open, high, low, close, volume
        Index: DatetimeIndex (UTC-aware)

        4h data is automatically resampled from 1h.
        """
        if timeframe not in INTERVAL_MAP:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Choose from: {list(INTERVAL_MAP.keys())}"
            )

        # Try loading from cache first
        cached = self._load_cache(timeframe)
        if cached is not None:
            log.info(f"[{self.instrument}][{timeframe}] Loaded from cache "
                     f"({len(cached)} bars)")
            return cached

        # 4h: resample from 1h (yfinance has no native 4h)
        if timeframe == "4h":
            return self._fetch_4h()

        return self._fetch_from_yfinance(timeframe)

    def fetch_all(self, timeframes: list = None) -> dict:
        """
        Convenience method — fetches multiple timeframes at once.

        Returns dict keyed by timeframe string:
            { "15m": df, "1h": df, "4h": df }
        """
        if timeframes is None:
            timeframes = ["15m", "1h", "4h"]

        result = {}
        for tf in timeframes:
            log.info(f"Fetching {self.instrument} [{tf}]...")
            result[tf] = self.fetch(tf)

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fetch_from_yfinance(self, timeframe: str) -> pd.DataFrame:
        """Download from yfinance, clean, optionally cache."""
        self._check_lookback_limit(timeframe)

        interval = INTERVAL_MAP[timeframe]
        log.info(
            f"[{self.instrument}][{timeframe}] Downloading from yfinance "
            f"(ticker={self.ticker}, interval={interval})..."
        )

        raw = yf.download(
            tickers=self.ticker,
            start=self.start,
            end=self.end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            raise RuntimeError(
                f"yfinance returned empty data for {self.ticker} [{interval}]. "
                f"Check date range or try a shorter lookback."
            )

        df = self._clean(raw)

        log.info(
            f"[{self.instrument}][{timeframe}] Downloaded {len(df)} bars "
            f"| {df.index[0]} → {df.index[-1]}"
        )

        if self.save_raw:
            self._save_cache(df, timeframe)

        return df

    def _fetch_4h(self) -> pd.DataFrame:
        """Resample 1h data into 4h candles."""
        log.info(
            f"[{self.instrument}][4h] Resampling from 1h data..."
        )
        df_1h = self.fetch("1h")
        df_4h = self._resample_to_4h(df_1h)

        log.info(
            f"[{self.instrument}][4h] Resampled to {len(df_4h)} bars"
        )

        if self.save_raw:
            self._save_cache(df_4h, "4h")

        return df_4h

    def _resample_to_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 1h OHLCV into 4h bars."""
        resampled = df.resample("4h").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna()

        return resampled

    def _clean(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Standardise column names, drop NaNs, sort index."""
        # yfinance returns MultiIndex columns when downloading single ticker
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw.copy()
        df.columns = [c.lower() for c in df.columns]

        # Keep only OHLCV
        df = df[["open", "high", "low", "close", "volume"]]

        # Drop rows with any NaN in OHLC
        before = len(df)
        df = df.dropna(subset=["open", "high", "low", "close"])
        dropped = before - len(df)
        if dropped > 0:
            log.warning(f"Dropped {dropped} rows with NaN OHLC values.")

        # Remove zero-price rows (bad ticks)
        df = df[(df["open"] > 0) & (df["close"] > 0)]

        # Ensure chronological order
        df = df.sort_index()

        return df

    def _check_lookback_limit(self, timeframe: str):
        """Warn if the requested date range exceeds yfinance limits."""
        max_days = MAX_LOOKBACK_DAYS.get(timeframe, 9999)
        start_dt = datetime.strptime(self.start, "%Y-%m-%d")
        days_requested = (datetime.now() - start_dt).days

        if days_requested > max_days:
            log.warning(
                f"[{timeframe}] yfinance only supports ~{max_days} days of "
                f"history for this interval, but {days_requested} days were "
                f"requested. Data may be truncated. "
                f"Consider MetaTrader5 for deeper history."
            )

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_path(self, timeframe: str) -> Path:
        fname = f"{self.instrument}_{timeframe}_{self.start}_{self.end}.csv"
        return RAW_DATA_DIR / fname

    def _save_cache(self, df: pd.DataFrame, timeframe: str):
        path = self._cache_path(timeframe)
        df.to_csv(path)
        log.info(f"Cached → {path.name}")

    def _load_cache(self, timeframe: str) -> pd.DataFrame | None:
        path = self._cache_path(timeframe)
        if not path.exists():
            return None

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
