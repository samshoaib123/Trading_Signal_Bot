"""Setup detection, ATR-based risk levels and confidence scoring.

Three setups are implemented, all evaluated on the *last closed* candle only:

``rsi_reversal``
    RSI(14) dipped below 30 and closed back above it -> BUY.
    RSI(14) poked above 70 and closed back below it -> SELL.

``macd_crossover``
    MACD(12,26,9) line crosses above its signal line -> BUY, below -> SELL.

``bb_breakout``
    Close breaks above the upper Bollinger Band(20, 2) -> BUY (momentum
    breakout); below the lower band -> SELL.

Every setup requires a *fresh* cross: the condition must be false on the
previous candle and true on the current one. That keeps a price riding the
upper band (or an RSI parked above 30) from re-firing every 15 minutes, and it
is what makes deduplication cheap.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
import re
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

LOG = logging.getLogger(__name__)

BUY = "BUY"
SELL = "SELL"

SETUP_LABELS = {
    "rsi_reversal": "RSI Reversal",
    "macd_crossover": "MACD Crossover",
    "bb_breakout": "Bollinger Bands Breakout",
}

MAX_CONFIDENCE = 3

# Columns that must be present and non-NaN before any setup can be evaluated.
REQUIRED_COLUMNS = (
    "close",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_lower",
    "atr",
)


def timeframe_minutes(timeframe: str) -> int:
    """Convert a ccxt timeframe string (``"15m"``, ``"4h"``, ``"1d"``) to minutes.

    Returns ``0`` for anything unrecognised, which callers treat as "unknown".
    """
    match = re.fullmatch(r"(\d+)([mhdwM])", (timeframe or "").strip())
    if not match:
        return 0
    amount, unit = int(match.group(1)), match.group(2)
    return amount * {"m": 1, "h": 60, "d": 1440, "w": 10080, "M": 43200}[unit]


@dataclass
class Signal:
    """A single actionable alert for one pair / setup / direction."""

    symbol: str
    setup: str
    side: str
    entry: float
    stop_loss: float
    take_profit: float
    atr: float
    timeframe: str
    candle_time: datetime
    confidence: int = 1
    confirmations: List[str] = field(default_factory=list)
    position_size: Optional[float] = None
    position_notional: Optional[float] = None
    risk_amount: Optional[float] = None

    @property
    def setup_label(self) -> str:
        return SETUP_LABELS.get(self.setup, self.setup)

    @property
    def base_asset(self) -> str:
        return self.symbol.split("/")[0]

    @property
    def candle_close_time(self) -> datetime:
        """When the signal candle actually closed.

        ``candle_time`` is the candle's *open* time (that is what exchanges
        return), so the close is one timeframe later.
        """
        minutes = timeframe_minutes(self.timeframe)
        return self.candle_time + timedelta(minutes=minutes)

    @property
    def quote_asset(self) -> str:
        parts = self.symbol.split("/")
        return parts[1].split(":")[0] if len(parts) > 1 else "USDT"

    @property
    def risk_reward(self) -> float:
        """Reward-to-risk ratio implied by the ATR multipliers."""
        risk = abs(self.entry - self.stop_loss)
        if risk <= 0:
            return 0.0
        return abs(self.take_profit - self.entry) / risk

    @property
    def dedupe_key(self) -> str:
        """State key used to suppress repeats of the same setup."""
        return f"{self.symbol}|{self.setup}|{self.side}"

    def to_dict(self) -> Dict[str, object]:
        """Plain-dict view, handy for logging and tests."""
        return {
            "symbol": self.symbol,
            "setup": self.setup,
            "side": self.side,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "atr": self.atr,
            "confidence": self.confidence,
            "confirmations": list(self.confirmations),
            "candle_time": self.candle_time.isoformat(),
            "timeframe": self.timeframe,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_number(value) -> bool:
    """True for a real, finite float (guards against NaN warm-up rows)."""
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _crossed_above(curr: float, prev: float, curr_ref: float, prev_ref: float) -> bool:
    """Series crossed from at-or-below ``ref`` to above ``ref``."""
    return prev <= prev_ref and curr > curr_ref


def _crossed_below(curr: float, prev: float, curr_ref: float, prev_ref: float) -> bool:
    """Series crossed from at-or-above ``ref`` to below ``ref``."""
    return prev >= prev_ref and curr < curr_ref


def _candle_time(row: pd.Series, index_value) -> datetime:
    """Best-effort UTC timestamp for a candle row."""
    value = row.get("timestamp", index_value)
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return datetime.now(timezone.utc)
    return ts.to_pydatetime()


# ---------------------------------------------------------------------------
# Setup detectors: each returns BUY, SELL or None
# ---------------------------------------------------------------------------
def _detect_rsi_reversal(curr: pd.Series, prev: pd.Series, settings) -> Optional[str]:
    """RSI exits oversold -> BUY; exits overbought -> SELL."""
    if _crossed_above(
        curr["rsi"], prev["rsi"], settings.rsi_oversold, settings.rsi_oversold
    ):
        return BUY
    if _crossed_below(
        curr["rsi"], prev["rsi"], settings.rsi_overbought, settings.rsi_overbought
    ):
        return SELL
    return None


def _detect_macd_crossover(curr: pd.Series, prev: pd.Series, settings) -> Optional[str]:
    """MACD line crossing its signal line."""
    if _crossed_above(
        curr["macd"], prev["macd"], curr["macd_signal"], prev["macd_signal"]
    ):
        return BUY
    if _crossed_below(
        curr["macd"], prev["macd"], curr["macd_signal"], prev["macd_signal"]
    ):
        return SELL
    return None


def _detect_bb_breakout(curr: pd.Series, prev: pd.Series, settings) -> Optional[str]:
    """Close breaking out of the Bollinger envelope."""
    if _crossed_above(
        curr["close"], prev["close"], curr["bb_upper"], prev["bb_upper"]
    ):
        return BUY
    if _crossed_below(
        curr["close"], prev["close"], curr["bb_lower"], prev["bb_lower"]
    ):
        return SELL
    return None


DETECTORS: Dict[str, Callable[[pd.Series, pd.Series, object], Optional[str]]] = {
    "rsi_reversal": _detect_rsi_reversal,
    "macd_crossover": _detect_macd_crossover,
    "bb_breakout": _detect_bb_breakout,
}


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------
def _volume_spike(curr: pd.Series, settings) -> Optional[bool]:
    if not _is_number(curr.get("volume_sma")) or curr["volume_sma"] <= 0:
        return None
    return bool(curr["volume"] >= settings.volume_spike_multiplier * curr["volume_sma"])


def _trend_aligned(curr: pd.Series, side: str) -> Optional[bool]:
    if not _is_number(curr.get("trend_ema")):
        return None
    return bool(
        curr["close"] > curr["trend_ema"]
        if side == BUY
        else curr["close"] < curr["trend_ema"]
    )


def _momentum_aligned(curr: pd.Series, prev: pd.Series, side: str) -> Optional[bool]:
    if not (_is_number(curr.get("macd_hist")) and _is_number(prev.get("macd_hist"))):
        return None
    return bool(
        curr["macd_hist"] > prev["macd_hist"]
        if side == BUY
        else curr["macd_hist"] < prev["macd_hist"]
    )


def _rsi_has_room(curr: pd.Series, side: str, settings) -> Optional[bool]:
    if not _is_number(curr.get("rsi")):
        return None
    return bool(
        curr["rsi"] < settings.rsi_overbought
        if side == BUY
        else curr["rsi"] > settings.rsi_oversold
    )


def score_confidence(
    setup: str, side: str, curr: pd.Series, prev: pd.Series, settings
) -> tuple[int, List[str]]:
    """Score a signal from 1 to 3 based on aligned confirmations.

    The trigger itself is always worth 1 point. Each setup then gets three
    checks that are *independent of its own trigger* (so we never award a point
    for something the setup already guarantees), and every passing check adds a
    point, capped at :data:`MAX_CONFIDENCE`. A check whose inputs are still
    warming up counts as "not confirmed" rather than failing the signal.

    Returns:
        ``(confidence, ["human readable confirmation", ...])``
    """
    trend_len = curr.get("_trend_ema_length")
    trend_label = (
        f"Trend aligned (EMA{int(trend_len)})"
        if _is_number(trend_len)
        else "Trend aligned"
    )

    checks: Sequence[tuple[str, Optional[bool]]]
    if setup == "rsi_reversal":
        checks = (
            ("Volume spike", _volume_spike(curr, settings)),
            (trend_label, _trend_aligned(curr, side)),
            ("MACD momentum turning", _momentum_aligned(curr, prev, side)),
        )
    elif setup == "macd_crossover":
        checks = (
            ("Volume spike", _volume_spike(curr, settings)),
            (trend_label, _trend_aligned(curr, side)),
            ("RSI has room to run", _rsi_has_room(curr, side, settings)),
        )
    else:  # bb_breakout
        checks = (
            ("Volume spike", _volume_spike(curr, settings)),
            (trend_label, _trend_aligned(curr, side)),
            ("MACD momentum aligned", _momentum_aligned(curr, prev, side)),
        )

    passed = [label for label, ok in checks if ok]
    return min(MAX_CONFIDENCE, 1 + len(passed)), passed


# ---------------------------------------------------------------------------
# Risk levels
# ---------------------------------------------------------------------------
def build_levels(side: str, entry: float, atr_value: float, settings) -> tuple[float, float]:
    """Return ``(stop_loss, take_profit)`` from the ATR multipliers."""
    sl_distance = settings.atr_sl_multiplier * atr_value
    tp_distance = settings.atr_tp_multiplier * atr_value
    if side == BUY:
        return entry - sl_distance, entry + tp_distance
    return entry + sl_distance, entry - tp_distance


def position_size(entry: float, stop_loss: float, settings) -> tuple[float, float, float]:
    """Fixed-fractional position size.

    ``size = (capital * risk%) / |entry - stop_loss|``

    Returns:
        ``(size_in_base_asset, notional_in_quote, risk_amount_in_quote)``.
    """
    risk_amount = settings.capital * settings.risk_fraction
    stop_distance = abs(entry - stop_loss)
    if stop_distance <= 0:
        return 0.0, 0.0, risk_amount
    size = risk_amount / stop_distance
    return size, size * entry, risk_amount


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def detect_signals(symbol: str, df: pd.DataFrame, settings) -> List[Signal]:
    """Evaluate every enabled setup against the last closed candle.

    Args:
        symbol: Market symbol, e.g. ``"BTC/USDT"``.
        df: Frame returned by :func:`indicators.calculate_indicators`. The
            caller is responsible for having already dropped the still-forming
            candle, so ``df.iloc[-1]`` is a closed candle.
        settings: A :class:`config.Settings` instance.

    Returns:
        Zero or more :class:`Signal` objects, filtered by ``MIN_CONFIDENCE``.
    """
    if df is None or len(df) < 2:
        LOG.debug("%s: not enough candles to evaluate setups", symbol)
        return []

    curr = df.iloc[-1].copy()
    prev = df.iloc[-2]

    unusable = [c for c in REQUIRED_COLUMNS if not _is_number(curr.get(c))]
    if unusable:
        LOG.debug("%s: indicators still warming up (%s)", symbol, ", ".join(unusable))
        return []
    if not all(_is_number(prev.get(c)) for c in REQUIRED_COLUMNS):
        LOG.debug("%s: previous candle has warm-up NaNs, skipping", symbol)
        return []
    if curr["atr"] <= 0:
        LOG.warning("%s: ATR is %s, cannot size a stop", symbol, curr["atr"])
        return []

    # Surface the frame-level trend EMA length to the scoring helpers.
    curr["_trend_ema_length"] = df.attrs.get("trend_ema_length")

    candle_time = _candle_time(curr, df.index[-1])
    signals: List[Signal] = []

    for setup in settings.enabled_setups:
        detector = DETECTORS.get(setup)
        if detector is None:
            LOG.warning("Unknown setup %r in ENABLED_SETUPS, ignoring", setup)
            continue

        side = detector(curr, prev, settings)
        if side is None:
            continue

        entry = float(curr["close"])
        atr_value = float(curr["atr"])
        stop_loss, take_profit = build_levels(side, entry, atr_value, settings)
        if stop_loss <= 0 or take_profit <= 0:
            LOG.warning(
                "%s/%s: ATR %.6f produced a non-positive level, skipping",
                symbol,
                setup,
                atr_value,
            )
            continue

        confidence, confirmations = score_confidence(setup, side, curr, prev, settings)
        if confidence < settings.min_confidence:
            LOG.info(
                "%s %s %s: confidence %d < MIN_CONFIDENCE %d, skipped",
                symbol,
                setup,
                side,
                confidence,
                settings.min_confidence,
            )
            continue

        signal = Signal(
            symbol=symbol,
            setup=setup,
            side=side,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr_value,
            timeframe=settings.timeframe,
            candle_time=candle_time,
            confidence=confidence,
            confirmations=confirmations,
        )

        if settings.show_position_size and settings.capital > 0:
            size, notional, risk_amount = position_size(entry, stop_loss, settings)
            signal.position_size = size
            signal.position_notional = notional
            signal.risk_amount = risk_amount

        signals.append(signal)

    return signals
