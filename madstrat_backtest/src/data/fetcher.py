"""
src/data/fetcher.py
-------------------
Fetches raw OHLCV data for Madstrat 2.0 backtesting.

Two data sources are supported — select via the `source` parameter:

  "yfinance"     — Free, no login required. Good for quick testing.
                   No native 4H interval; 4H is resampled from 1H.
                   Intraday history is limited (~60 days for 15m/30m).

  "tradingview"  — Uses the tvdatafeed library (rongardF/tvdatafeed).
                   Supports native 4H. Up to 5000 bars per request.
                   Optional login for broader symbol access.
                   Install: pip install git+https://github.com/rongardF/tvdatafeed.git

Supported instruments : EURUSD, GBPUSD, EURGBP, XAUUSD
Supported timeframes  : 1m, 5m, 15m, 30m, 1h, 2h, 4h, 1d
"""

import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

log = logging.getLogger(__name__)

# ── Raw data output directory ─────────────────────────────────────────────────
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  YFINANCE CONFIGURATION
# ╚══════════════════════════════════════════════════════════════════════════════

YF_TICKER_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "EURGBP": "EURGBP=X",
    "XAUUSD": "GC=F",       # Gold futures — closest yfinance proxy
}

# yfinance interval strings (no native 4H — resampled from 1H instead)
YF_INTERVAL_MAP = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "2h":  "2h",
    "1d":  "1d",
}

# Maximum lookback in days that yfinance supports per interval
YF_MAX_LOOKBACK_DAYS = {
    "1m":  7,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "1h":  730,
    "2h":  730,
    "1d":  3650,
}


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  TRADINGVIEW (tvdatafeed) CONFIGURATION
# ╚══════════════════════════════════════════════════════════════════════════════

# TradingView symbol + exchange for each Madstrat instrument
TV_SYMBOL_MAP = {
    "EURUSD": ("EURUSD", "FX"),
    "GBPUSD": ("GBPUSD", "FX"),
    "EURGBP": ("EURGBP", "FX"),
    "XAUUSD": ("XAUUSD", "OANDA"),   # Spot gold on OANDA feed
}

# Maps our timeframe strings to tvdatafeed Interval enum attribute names
# tvdatafeed natively supports 4H — no resampling needed
TV_INTERVAL_MAP = {
    "1m":  "in_1_minute",
    "3m":  "in_3_minute",
    "5m":  "in_5_minute",
    "15m": "in_15_minute",
    "30m": "in_30_minute",
    "45m": "in_45_minute",
    "1h":  "in_1_hour",
    "2h":  "in_2_hour",
    "3h":  "in_3_hour",
    "4h":  "in_4_hour",      # Native 4H — no resampling required
    "1d":  "in_daily",
    "1w":  "in_weekly",
}

# tvdatafeed hard limit on bars per request
TV_MAX_BARS = 5000


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  DataFetcher
# ╚══════════════════════════════════════════════════════════════════════════════

class DataFetcher:
    """
    Downloads and caches OHLCV data for a single Madstrat instrument.

    Parameters
    ----------
    instrument : str
        One of: EURUSD, GBPUSD, EURGBP, XAUUSD
    start : str
        Start date — YYYY-MM-DD format
    end : str
        End date   — YYYY-MM-DD format
    source : str
        "yfinance" (default) or "tradingview"
    tv_username : str, optional
        TradingView username (only used when source="tradingview")
    tv_password : str, optional
        TradingView password (only used when source="tradingview")
    save_raw : bool
        If True, writes a CSV to data/raw/ after each download

    Usage
    -----
    # yfinance (quick start, no login needed)
    fetcher = DataFetcher("EURUSD", "2024-01-01", "2024-06-30")
    df_15m  = fetcher.fetch("15m")
    df_1h   = fetcher.fetch("1h")
    df_4h   = fetcher.fetch("4h")   # auto-resampled from 1H

    # TradingView (deeper history, native 4H)
    fetcher = DataFetcher(
        "EURUSD", "2024-01-01", "2024-06-30",
        source="tradingview",
        tv_username="your_user",
        tv_password="your_pass",
    )
    df_4h = fetcher.fetch("4h")     # native 4H — no resampling
    """

    SUPPORTED_INSTRUMENTS = list(YF_TICKER_MAP.keys())

    def __init__(
        self,
        instrument: str,
        start: str,
        end: str,
        source: str = "yfinance",
        tv_username: str = None,
        tv_password: str = None,
        save_raw: bool = True,
    ):
        instrument = instrument.upper()
        if instrument not in self.SUPPORTED_INSTRUMENTS:
            raise ValueError(
                f"Unsupported instrument '{instrument}'. "
                f"Choose from: {self.SUPPORTED_INSTRUMENTS}"
            )

        source = source.lower()
        if source not in ("yfinance", "tradingview"):
            raise ValueError(
                f"Unsupported source '{source}'. "
                f"Choose 'yfinance' or 'tradingview'."
            )

        self.instrument   = instrument
        self.start        = start
        self.end          = end
        self.source       = source
        self.save_raw     = save_raw

        # Lazy-initialised TradingView client (created on first TV fetch)
        self._tv          = None
        self._tv_username = tv_username
        self._tv_password = tv_password

        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        log.info(
            f"DataFetcher ready — {instrument} | {start} → {end} | "
            f"source={source}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(self, timeframe: str) -> pd.DataFrame:
        """
        Fetch OHLCV data for the given timeframe.

        Returns a clean DataFrame with columns:
            open, high, low, close, volume
        Index: DatetimeIndex

        Notes
        -----
        - yfinance source    : 4H is automatically resampled from 1H.
        - tradingview source : 4H is fetched natively (no resampling).
        """
        # ── Cache hit ─────────────────────────────────────────────────────────
        cached = self._load_cache(timeframe)
        if cached is not None:
            log.info(
                f"[{self.instrument}][{timeframe}] Loaded from cache "
                f"({len(cached)} bars)"
            )
            return cached

        # ── Route to the correct backend ──────────────────────────────────────
        if self.source == "yfinance":
            return self._yf_fetch(timeframe)
        else:
            return self._tv_fetch(timeframe)

    def fetch_all(self, timeframes: list = None) -> dict:
        """
        Fetch multiple timeframes in one call.

        Returns
        -------
        dict  keyed by timeframe string, e.g. {"15m": df, "1h": df, "4h": df}
        """
        if timeframes is None:
            timeframes = ["15m", "1h", "4h"]

        result = {}
        for tf in timeframes:
            log.info(
                f"[{self.instrument}] Fetching [{tf}] via {self.source}..."
            )
            result[tf] = self.fetch(tf)
        return result

    # ╔══════════════════════════════════════════════════════════════════════════
    # ║  YFINANCE BACKEND
    # ╚══════════════════════════════════════════════════════════════════════════

    def _yf_fetch(self, timeframe: str) -> pd.DataFrame:
        """Route yfinance requests — 4H is always resampled from 1H."""
        if timeframe == "4h":
            return self._yf_fetch_4h()

        if timeframe not in YF_INTERVAL_MAP:
            raise ValueError(
                f"[yfinance] Unsupported timeframe '{timeframe}'. "
                f"Available: {list(YF_INTERVAL_MAP.keys())} + '4h' (resampled)"
            )

        return self._yf_download(timeframe)

    def _yf_download(self, timeframe: str) -> pd.DataFrame:
        """Download a single timeframe directly from yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is not installed. Run: pip install yfinance"
            )

        self._yf_check_lookback(timeframe)

        ticker   = YF_TICKER_MAP[self.instrument]
        interval = YF_INTERVAL_MAP[timeframe]

        log.info(
            f"[{self.instrument}][{timeframe}] Downloading from yfinance "
            f"(ticker={ticker}, interval={interval})..."
        )

        raw = yf.download(
            tickers=ticker,
            start=self.start,
            end=self.end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,   # Flat columns — avoids MultiIndex overhead
        )

        if raw.empty:
            raise RuntimeError(
                f"yfinance returned empty data for {ticker} [{interval}]. "
                f"Check date range — intraday limit for {timeframe} is "
                f"~{YF_MAX_LOOKBACK_DAYS.get(timeframe, '?')} days."
            )

        df = self._clean_yf(raw)

        log.info(
            f"[{self.instrument}][{timeframe}] {len(df)} bars downloaded "
            f"| {df.index[0]} → {df.index[-1]}"
        )

        if self.save_raw:
            self._save_cache(df, timeframe)

        return df

    def _yf_fetch_4h(self) -> pd.DataFrame:
        """
        Build 4H candles by resampling 1H data.
        yfinance has no native 4H interval so this path is always taken.
        """
        log.info(
            f"[{self.instrument}][4h] yfinance has no native 4H interval — "
            f"fetching 1H and resampling..."
        )

        df_1h = self._yf_download("1h")
        df_4h = self._resample_to_4h(df_1h)

        log.info(
            f"[{self.instrument}][4h] Resampled {len(df_1h)} x 1H bars "
            f"→ {len(df_4h)} x 4H bars"
        )

        if self.save_raw:
            self._save_cache(df_4h, "4h")

        return df_4h

    def _yf_check_lookback(self, timeframe: str):
        """Warn if the date range exceeds yfinance's intraday history limit."""
        max_days       = YF_MAX_LOOKBACK_DAYS.get(timeframe, 9999)
        start_dt       = datetime.strptime(self.start, "%Y-%m-%d")
        days_requested = (datetime.now() - start_dt).days

        if days_requested > max_days:
            log.warning(
                f"[yfinance][{timeframe}] History limit is ~{max_days} days "
                f"but {days_requested} days were requested. "
                f"Data may be truncated. "
                f"Use source='tradingview' for deeper intraday history."
            )

    def _clean_yf(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise yfinance output to our schema.

        With multi_level_index=False the columns are already flat,
        so no MultiIndex handling is needed — just lowercase and filter.
        """
        df = raw.copy()

        # Normalise column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Keep only OHLCV
        df = df[["open", "high", "low", "close", "volume"]]

        # Drop NaN OHLC rows
        before  = len(df)
        df      = df.dropna(subset=["open", "high", "low", "close"])
        dropped = before - len(df)
        if dropped:
            log.warning(
                f"[yfinance] Dropped {dropped} rows with NaN OHLC values."
            )

        # Remove zero-price rows (bad ticks occasionally appear in forex feeds)
        df = df[(df["open"] > 0) & (df["close"] > 0)]

        # Ensure chronological order
        df = df.sort_index()

        return df

    # ╔══════════════════════════════════════════════════════════════════════════
    # ║  TRADINGVIEW BACKEND  (tvdatafeed — rongardF/tvdatafeed)
    # ╚══════════════════════════════════════════════════════════════════════════

    def _tv_fetch(self, timeframe: str) -> pd.DataFrame:
        """Fetch data from TradingView using tvdatafeed."""
        if timeframe not in TV_INTERVAL_MAP:
            raise ValueError(
                f"[tradingview] Unsupported timeframe '{timeframe}'. "
                f"Available: {list(TV_INTERVAL_MAP.keys())}"
            )

        client           = self._get_tv_client()
        symbol, exchange = TV_SYMBOL_MAP[self.instrument]
        interval         = self._tv_interval(timeframe)
        n_bars           = self._tv_n_bars(timeframe)

        log.info(
            f"[{self.instrument}][{timeframe}] Downloading from TradingView "
            f"(symbol={symbol}, exchange={exchange}, n_bars={n_bars})..."
        )

        raw = client.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            n_bars=n_bars,
        )

        if raw is None or raw.empty:
            raise RuntimeError(
                f"TradingView returned no data for {symbol}/{exchange} "
                f"[{timeframe}]. Try logging in or reducing the date range."
            )

        df = self._clean_tv(raw)
        df = self._trim_to_window(df)

        log.info(
            f"[{self.instrument}][{timeframe}] {len(df)} bars downloaded "
            f"| {df.index[0]} → {df.index[-1]}"
        )

        if self.save_raw:
            self._save_cache(df, timeframe)

        return df

    def _get_tv_client(self):
        """
        Lazily initialise the TvDatafeed client.
        Reuses the same instance across multiple fetch() calls.
        """
        if self._tv is not None:
            return self._tv

        try:
            from tvDatafeed import TvDatafeed
        except ImportError:
            raise ImportError(
                "tvdatafeed is not installed.\n"
                "Run: pip install git+https://github.com/rongardF/tvdatafeed.git"
            )

        if self._tv_username and self._tv_password:
            log.info("Connecting to TradingView with login credentials...")
            self._tv = TvDatafeed(self._tv_username, self._tv_password)
        else:
            log.warning(
                "Connecting to TradingView without login — some symbols may "
                "be unavailable. Pass tv_username and tv_password for full access."
            )
            self._tv = TvDatafeed()

        return self._tv

    def _tv_interval(self, timeframe: str):
        """Return the tvdatafeed Interval enum value for the given timeframe."""
        from tvDatafeed import Interval
        interval_name = TV_INTERVAL_MAP[timeframe]
        return getattr(Interval, interval_name)

    def _tv_n_bars(self, timeframe: str) -> int:
        """
        Estimate how many bars are needed to cover start → end.
        Caps at TV_MAX_BARS (5000) — the tvdatafeed hard limit.
        """
        start_dt   = datetime.strptime(self.start, "%Y-%m-%d")
        end_dt     = datetime.strptime(self.end,   "%Y-%m-%d")
        total_days = (end_dt - start_dt).days

        # Approximate trading bars per day for each timeframe
        bars_per_day = {
            "1m": 1440, "3m": 480,  "5m": 288, "15m": 96,
            "30m": 48,  "45m": 32,  "1h": 24,  "2h": 12,
            "3h": 8,    "4h": 6,    "1d": 1,   "1w": 0.2,
        }

        multiplier   = bars_per_day.get(timeframe, 24)
        trading_days = total_days * (5 / 7)            # ~5 trading days/week
        estimated    = int(trading_days * multiplier * 1.1)  # 10% buffer

        n_bars = min(estimated, TV_MAX_BARS)
        log.debug(
            f"[{timeframe}] Estimated {estimated} bars for {total_days} "
            f"calendar days → requesting {n_bars}"
        )
        return n_bars

    def _clean_tv(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise tvdatafeed output to our standard schema.

        tvdatafeed returns: symbol, open, high, low, close, volume
        Index is a DatetimeIndex named 'datetime'.
        """
        df = raw.copy()

        # Drop the 'symbol' column if present
        if "symbol" in df.columns:
            df = df.drop(columns=["symbol"])

        # Standardise column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Validate required columns are present
        required = ["open", "high", "low", "close", "volume"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"TradingView response is missing columns: {missing}. "
                f"Got: {list(df.columns)}"
            )

        df = df[required]

        # Drop NaN OHLC rows
        before  = len(df)
        df      = df.dropna(subset=["open", "high", "low", "close"])
        dropped = before - len(df)
        if dropped:
            log.warning(f"[TV] Dropped {dropped} rows with NaN OHLC values.")

        # Remove zero-price rows
        df = df[(df["open"] > 0) & (df["close"] > 0)]

        df = df.sort_index()
        return df

    def _trim_to_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Slice the DataFrame to the requested start/end window.

        tvdatafeed fetches n_bars going backwards from now, so the result
        typically extends beyond the requested range — we trim it here.
        """
        start = pd.Timestamp(self.start)
        end   = pd.Timestamp(self.end)

        # Match timezone awareness of the index
        if df.index.tz is not None:
            start = start.tz_localize(df.index.tz)
            end   = end.tz_localize(df.index.tz)

        return df.loc[start:end]

    # ╔══════════════════════════════════════════════════════════════════════════
    # ║  SHARED HELPERS
    # ╚══════════════════════════════════════════════════════════════════════════

    def _resample_to_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 1H OHLCV bars into 4H candles.

        Anchored to start_day so candle boundaries align with TradingView's
        4H candles (midnight UTC). Incomplete edge bars are dropped.
        """
        resampled = (
            df.resample("4h", origin="start_day")
            .agg({
                "open":   "first",
                "high":   "max",
                "low":    "min",
                "close":  "last",
                "volume": "sum",
            })
            .dropna(subset=["open", "high", "low", "close"])
        )

        # Drop incomplete 4H bars (fewer than 4 constituent 1H bars)
        bar_counts = df.resample("4h", origin="start_day").size()
        complete   = bar_counts[bar_counts >= 4].index
        resampled  = resampled.loc[resampled.index.isin(complete)]

        return resampled

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_path(self, timeframe: str) -> Path:
        """Unique cache filename per instrument / timeframe / date range / source."""
        fname = (
            f"{self.instrument}_{timeframe}_"
            f"{self.start}_{self.end}_{self.source}.csv"
        )
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
        log.debug(f"Cache hit: {path.name}")
        return df
