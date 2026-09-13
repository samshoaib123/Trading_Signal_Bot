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
from typing import Optional, Sequence

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


def stoch_rsi(
    close: pd.Series, length: int = 14, stoch_length: int = 14,
    smooth_k: int = 3, smooth_d: int = 3,
) -> pd.DataFrame:
    """Stochastic RSI: where RSI sits inside its own recent range.

    RSI tells you momentum; Stoch-RSI tells you whether that momentum is
    stretched *relative to how stretched it has recently been*, which is why it
    turns earlier than RSI and why it is paired with a slower filter.

    A flat RSI window has no range to normalise against. Rather than divide by
    zero we emit 50 (dead centre), which reads as "no information" to every
    caller instead of as an extreme.
    """
    base = rsi(close, length)
    low = base.rolling(window=stoch_length, min_periods=stoch_length).min()
    high = base.rolling(window=stoch_length, min_periods=stoch_length).max()
    span = high - low
    raw = 100.0 * (base - low) / span.replace(0.0, np.nan)
    raw = raw.where(span != 0.0, 50.0).where(base.notna() & low.notna())
    # The value is a percentage position inside a range, so [0, 100] is exact by
    # definition. Rounding can still land a ulp outside it, and a caller
    # comparing against an 80 threshold should never have to wonder.
    raw = raw.clip(0.0, 100.0)
    k = raw.rolling(window=smooth_k, min_periods=smooth_k).mean()
    return pd.DataFrame({"stochrsi_k": k,
                         "stochrsi_d": k.rolling(window=smooth_d,
                                                 min_periods=smooth_d).mean()})


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.DataFrame:
    """Wilder's ADX with its +DI / -DI components.

    ADX measures how *strongly* price is trending, not which way: a reading
    climbing through 25 says the move has conviction, and a reading under it
    says the same setup is far likelier to chop. +DI / -DI carry the direction.
    """
    up = high.diff()
    down = -low.diff()
    # A bar only counts towards one direction: the larger move wins, and a bar
    # that expanded on neither side counts for neither.
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=low.index)

    atr_ = rma(true_range(high, low, close), length)
    safe_atr = atr_.replace(0.0, np.nan)
    plus_di = 100.0 * rma(plus_dm, length) / safe_atr
    minus_di = 100.0 * rma(minus_dm, length) / safe_atr

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return pd.DataFrame({"adx": rma(dx, length),
                         "plus_di": plus_di, "minus_di": minus_di})


def keltner_channels(
    high: pd.Series, low: pd.Series, close: pd.Series,
    length: int = 20, multiplier: float = 1.5,
) -> pd.DataFrame:
    """Keltner Channels: an EMA envelope scaled by ATR."""
    mid = ema(close, length)
    band = multiplier * atr(high, low, close, length)
    return pd.DataFrame({"kc_lower": mid - band, "kc_mid": mid, "kc_upper": mid + band})


def squeeze(
    high: pd.Series, low: pd.Series, close: pd.Series,
    bb_length: int = 20, bb_std: float = 2.0,
    kc_length: int = 20, kc_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Bollinger-in-Keltner squeeze: volatility coiled and about to release.

    When the Bollinger Bands contract *inside* the Keltner Channels, realised
    volatility has fallen below its ATR-implied baseline. That is the classic
    squeeze: it says nothing about direction, only that the range is unusually
    tight, so a breakout from here has room to run.

    ``squeeze_off`` marks the first bar after a squeeze ends — the fire signal,
    as opposed to the wait. ``bb_bandwidth`` is the raw tightness measure, kept
    because it is comparable across instruments in a way band prices are not.
    """
    bb = bollinger_bands(close, bb_length, bb_std)
    kc = keltner_channels(high, low, close, kc_length, kc_multiplier)
    on = (bb["bb_lower"] > kc["kc_lower"]) & (bb["bb_upper"] < kc["kc_upper"])
    known = bb["bb_lower"].notna() & kc["kc_lower"].notna()
    on = on.where(known)
    bandwidth = (bb["bb_upper"] - bb["bb_lower"]) / bb["bb_mid"].replace(0.0, np.nan)
    return pd.DataFrame({
        "squeeze_on": on,
        # .shift(1) is the previous bar's state, so "was on, is now off".
        "squeeze_off": (on == False) & (on.shift(1) == True),  # noqa: E712
        "bb_bandwidth": bandwidth,
    })


def ema_ribbon(
    close: pd.Series, lengths: Sequence[int], min_width: float = 0.0
) -> pd.DataFrame:
    """A fan of EMAs, plus whether it is cleanly stacked *and* meaningfully open.

    One EMA tells you the trend; a ribbon tells you how *orderly* it is. Fully
    stacked (every fast EMA above every slow one) is a trend with agreement
    across horizons. Tangled is the same trend without it, which is where
    trend-following setups go to die — so the strategy layer gates on the stack,
    not on a single crossover.

    Ordering on its own is not enough. In a flat, noisy market the EMAs sit
    almost on top of each other and their order flips essentially at random, so
    a pure stacking test calls chop a trend roughly half the time. ``min_width``
    is the floor the fan must be open by, as a fraction of price — measured
    relative to price so one threshold works from a $0.50 altcoin to $4,000
    gold. It defaults to 0 (ordering only); the strategy layer passes a real
    value.
    """
    ordered = sorted(lengths)
    frame = pd.DataFrame(index=close.index)
    lines = []
    for length in ordered:
        col = f"ribbon_{length}"
        frame[col] = ema(close, length)
        lines.append(frame[col])

    stacked = pd.concat(lines, axis=1)
    known = stacked.notna().all(axis=1)
    widest, narrowest = stacked.max(axis=1), stacked.min(axis=1)
    width = (widest - narrowest) / close.replace(0.0, np.nan)
    frame["ribbon_width"] = width

    # Fast-to-slow order: strictly descending values = bullish stack.
    diffs = stacked.diff(axis=1).iloc[:, 1:]
    open_enough = width >= min_width if min_width > 0 else pd.Series(True, close.index)
    frame["ribbon_bull"] = (
        ((diffs < 0).all(axis=1) & open_enough & known).where(known)
    )
    frame["ribbon_bear"] = (
        ((diffs > 0).all(axis=1) & open_enough & known).where(known)
    )
    return frame


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

    # Trend Sniper inputs. Always computed here rather than in the pandas-ta
    # branch above: pandas-ta's squeeze and ribbon differ between versions, and
    # a setup that silently changes shape with an optional dependency is worse
    # than one that is a touch slower.
    out = out.join(stoch_rsi(close, settings.rsi_period, settings.stoch_rsi_period,
                             settings.stoch_rsi_k, settings.stoch_rsi_d))
    out = out.join(adx(high, low, close, settings.adx_period))
    out = out.join(squeeze(high, low, close, settings.bb_period, settings.bb_std,
                           settings.kc_period, settings.kc_multiplier))
    out = out.join(ema_ribbon(close, settings.ribbon_lengths,
                          settings.ribbon_min_width))

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
