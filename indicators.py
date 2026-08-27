"""Technical indicators (RSI, MACD, Bollinger Bands, ATR, EMA/SMA).

Two backends are supported and selected with the ``INDICATOR_BACKEND`` env var:

``pandas``   (default)
    Self-contained implementations written directly on top of pandas. No C
    extensions, no extra wheels, works on every Python version the rest of the
    stack supports.

``pandas_ta``
    Uses the ``pandas-ta`` package when it is installed. Note that pandas-ta
    only publishes wheels for Python >= 3.12 these days, which is why it is an
    optional extra (see ``requirements-optional.txt``) rather than a hard
    dependency.

``auto``
    Use ``pandas_ta`` if it imports cleanly, otherwise fall back to ``pandas``.

Both backends implement Wilder's smoothing (RMA) for RSI and ATR, an
exponential moving average for MACD, and a population standard deviation for
the Bollinger Bands, so their outputs agree to within floating point noise.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

_PANDAS_TA = None  # cached module handle, resolved lazily


def _try_import_pandas_ta():
    """Import pandas-ta once, caching both success and failure."""
    global _PANDAS_TA
    if _PANDAS_TA is None:
        try:
            import pandas_ta  # type: ignore

            _PANDAS_TA = pandas_ta
        except Exception as exc:  # pragma: no cover - depends on environment
            LOG.debug("pandas-ta unavailable (%s); using the pandas backend", exc)
            _PANDAS_TA = False
    return _PANDAS_TA or None


def resolve_backend(preference: str = "auto") -> str:
    """Return the indicator backend that will actually be used."""
    preference = (preference or "auto").lower()
    if preference == "pandas":
        return "pandas"
    if preference in {"pandas_ta", "pandas-ta", "auto"}:
        if _try_import_pandas_ta() is not None:
            return "pandas_ta"
        if preference != "auto":
            LOG.warning(
                "INDICATOR_BACKEND=%s requested but pandas-ta is not importable; "
                "using the built-in pandas backend instead.",
                preference,
            )
        return "pandas"
    LOG.warning("Unknown INDICATOR_BACKEND=%r; using the pandas backend", preference)
    return "pandas"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential moving average (``adjust=False``, the TA convention)."""
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothed moving average, used by RSI and ATR."""
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    Returns values in ``[0, 100]``; ``NaN`` for the warm-up window.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    # avg_loss == 0 means an unbroken run of up candles -> RSI 100.
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(~((avg_loss == 0.0) & (avg_gain == 0.0)), 50.0)
    return out.where(avg_gain.notna() & avg_loss.notna())


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_line - signal_line,
        }
    )


def bollinger_bands(
    close: pd.Series, length: int = 20, std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands (population standard deviation, ``ddof=0``)."""
    mid = sma(close, length)
    dev = close.rolling(window=length, min_periods=length).std(ddof=0)
    return pd.DataFrame(
        {
            "bb_lower": mid - std * dev,
            "bb_mid": mid,
            "bb_upper": mid + std * dev,
        }
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's True Range."""
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.Series:
    """Average True Range (Wilder's smoothing)."""
    return rma(true_range(high, low, close), length)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def calculate_indicators(df: pd.DataFrame, settings) -> pd.DataFrame:
    """Attach every indicator column the strategies need to an OHLCV frame.

    Args:
        df: OHLCV frame with ``open/high/low/close/volume`` columns indexed (or
            columned) by candle open time. It is not mutated.
        settings: A :class:`config.Settings` instance supplying the periods.

    Returns:
        A copy of ``df`` with the indicator columns appended. Rows where the
        indicators are still warming up keep ``NaN`` values; the strategy layer
        checks for those explicitly.
    """
    if df is None or df.empty:
        raise ValueError("calculate_indicators() received an empty frame")

    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in df]
    if missing:
        raise ValueError(f"OHLCV frame is missing columns: {missing}")

    out = df.copy()
    close, high, low = out["close"], out["high"], out["low"]

    backend = resolve_backend(getattr(settings, "indicator_backend", "auto"))
    ta = _try_import_pandas_ta() if backend == "pandas_ta" else None

    if ta is not None:  # pragma: no cover - optional dependency path
        out["rsi"] = ta.rsi(close, length=settings.rsi_period)
        macd_df = ta.macd(
            close,
            fast=settings.macd_fast,
            slow=settings.macd_slow,
            signal=settings.macd_signal,
        )
        bb_df = ta.bbands(close, length=settings.bb_period, std=settings.bb_std)
        out["atr"] = ta.atr(high, low, close, length=settings.atr_period)
        # pandas-ta names columns like MACD_12_26_9 / BBL_20_2.0; map positionally
        # so we stay independent of its naming across versions.
        out["macd"], out["macd_hist"], out["macd_signal"] = (
            macd_df.iloc[:, 0],
            macd_df.iloc[:, 1],
            macd_df.iloc[:, 2],
        )
        out["bb_lower"], out["bb_mid"], out["bb_upper"] = (
            bb_df.iloc[:, 0],
            bb_df.iloc[:, 1],
            bb_df.iloc[:, 2],
        )
    else:
        out["rsi"] = rsi(close, settings.rsi_period)
        out = out.join(
            macd(close, settings.macd_fast, settings.macd_slow, settings.macd_signal)
        )
        out = out.join(bollinger_bands(close, settings.bb_period, settings.bb_std))
        out["atr"] = atr(high, low, close, settings.atr_period)

    # Shared confirmation inputs (identical for both backends).
    out["volume_sma"] = sma(out["volume"], settings.volume_sma_period)
    trend_len = _fit_trend_length(len(out), settings.trend_ema_period)
    out["trend_ema"] = ema(close, trend_len) if trend_len else np.nan
    out.attrs["trend_ema_length"] = trend_len
    out.attrs["indicator_backend"] = backend
    return out


def _fit_trend_length(rows: int, preferred: int) -> Optional[int]:
    """Pick the longest trend EMA the available history can actually support.

    A 200-EMA needs 200 candles; when we only pulled 300 candles that is fine,
    but a thinly traded pair may return fewer. Rather than emitting an all-NaN
    column (which would silently kill the trend confirmation) we step down to
    100 or 50, and give up below that.
    """
    for length in (preferred, 100, 50):
        if length and rows >= length + 5:
            return length
    return None
