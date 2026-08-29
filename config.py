"""Configuration loading for the crypto trading signal bot.

Every tunable knob is an environment variable so the exact same image can run
locally (via a ``.env`` file) and on Railway / a VPS (via real env vars).
``python-dotenv`` is used only to populate ``os.environ`` from ``.env`` when the
file exists; real environment variables always win.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

# Load .env if present. override=False => real env vars take precedence, which
# is what we want in the cloud where no .env file is shipped.
load_dotenv(override=False)

LOG = logging.getLogger(__name__)

DEFAULT_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "ADA/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "MATIC/USDT",
    "AVAX/USDT",
    "LINK/USDT",
]


class ConfigError(RuntimeError):
    """Raised when the environment is not usable for a live run."""


def _get_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _get_int(name: str, default: int) -> int:
    raw = _get_str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        LOG.warning("Env %s=%r is not an int, falling back to %s", name, raw, default)
        return default


def _get_float(name: str, default: float) -> float:
    raw = _get_str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        LOG.warning("Env %s=%r is not a float, falling back to %s", name, raw, default)
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = _get_str(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _get_list(name: str, default: List[str]) -> List[str]:
    raw = _get_str(name)
    if not raw:
        return list(default)
    items = [part.strip().upper() for part in raw.split(",") if part.strip()]
    return items or list(default)


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the bot configuration."""

    # --- Telegram ---------------------------------------------------------
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    send_startup_message: bool = True

    # --- Exchange / data --------------------------------------------------
    exchange_id: str = "binance"
    symbols: List[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    timeframe: str = "15m"
    candle_limit: int = 300
    fetch_retries: int = 4
    fetch_backoff_seconds: float = 2.0

    # --- Indicator periods ------------------------------------------------
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    trend_ema_period: int = 200
    volume_sma_period: int = 20
    volume_spike_multiplier: float = 1.5

    # --- Risk / signal shaping -------------------------------------------
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 2.0
    capital: float = 1000.0
    risk_percent: float = 1.0
    fee_percent: float = 0.1
    backtest_candles: int = 1000
    show_position_size: bool = True
    beginner_mode: bool = True
    min_confidence: int = 1
    enabled_setups: List[str] = field(
        default_factory=lambda: ["rsi_reversal", "macd_crossover", "bb_breakout"]
    )

    # --- Scheduling / state ----------------------------------------------
    poll_interval_minutes: int = 15
    candle_close_buffer_seconds: int = 15
    state_file: str = "signal_state.json"
    heartbeat_file: str = ""
    tracker_file: str = "signal_outcomes.json"
    track_outcomes: bool = True
    signal_cooldown_minutes: int = 45
    state_retention_days: int = 7
    log_level: str = "INFO"
    indicator_backend: str = "auto"

    @property
    def risk_fraction(self) -> float:
        """Risk per trade as a fraction of capital (1.0% -> 0.01)."""
        return self.risk_percent / 100.0

    def require_telegram(self) -> None:
        """Fail fast when Telegram credentials are missing."""
        missing = [
            name
            for name, value in (
                ("TELEGRAM_BOT_TOKEN", self.telegram_bot_token),
                ("TELEGRAM_CHAT_ID", self.telegram_chat_id),
            )
            if not value
        ]
        if missing:
            raise ConfigError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ". Copy .env.example to .env (local) or set them in your host's "
                "variables panel (Railway / systemd / docker-compose)."
            )


def load_settings() -> Settings:
    """Build a :class:`Settings` instance from the process environment."""
    settings = Settings(
        telegram_bot_token=_get_str("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_get_str("TELEGRAM_CHAT_ID"),
        send_startup_message=_get_bool("SEND_STARTUP_MESSAGE", True),
        exchange_id=_get_str("EXCHANGE_ID", "binance").lower() or "binance",
        symbols=_get_list("SYMBOLS", DEFAULT_SYMBOLS),
        timeframe=_get_str("TIMEFRAME", "15m") or "15m",
        candle_limit=_get_int("CANDLE_LIMIT", 300),
        fetch_retries=_get_int("FETCH_RETRIES", 4),
        fetch_backoff_seconds=_get_float("FETCH_BACKOFF_SECONDS", 2.0),
        rsi_period=_get_int("RSI_PERIOD", 14),
        rsi_oversold=_get_float("RSI_OVERSOLD", 30.0),
        rsi_overbought=_get_float("RSI_OVERBOUGHT", 70.0),
        macd_fast=_get_int("MACD_FAST", 12),
        macd_slow=_get_int("MACD_SLOW", 26),
        macd_signal=_get_int("MACD_SIGNAL", 9),
        bb_period=_get_int("BB_PERIOD", 20),
        bb_std=_get_float("BB_STD", 2.0),
        atr_period=_get_int("ATR_PERIOD", 14),
        trend_ema_period=_get_int("TREND_EMA_PERIOD", 200),
        volume_sma_period=_get_int("VOLUME_SMA_PERIOD", 20),
        volume_spike_multiplier=_get_float("VOLUME_SPIKE_MULTIPLIER", 1.5),
        atr_sl_multiplier=_get_float("ATR_SL_MULTIPLIER", 1.5),
        atr_tp_multiplier=_get_float("ATR_TP_MULTIPLIER", 2.0),
        capital=_get_float("CAPITAL", 1000.0),
        risk_percent=_get_float("RISK_PERCENT", 1.0),
        fee_percent=_get_float("FEE_PERCENT", 0.1),
        backtest_candles=_get_int("BACKTEST_CANDLES", 1000),
        show_position_size=_get_bool("SHOW_POSITION_SIZE", True),
        beginner_mode=_get_bool("BEGINNER_MODE", True),
        min_confidence=_get_int("MIN_CONFIDENCE", 1),
        enabled_setups=[
            s.lower()
            for s in _get_list(
                "ENABLED_SETUPS", ["rsi_reversal", "macd_crossover", "bb_breakout"]
            )
        ],
        poll_interval_minutes=_get_int("POLL_INTERVAL_MINUTES", 15),
        candle_close_buffer_seconds=_get_int("CANDLE_CLOSE_BUFFER_SECONDS", 15),
        state_file=_get_str("STATE_FILE", "signal_state.json") or "signal_state.json",
        heartbeat_file=_get_str("HEARTBEAT_FILE"),
        tracker_file=_get_str("TRACKER_FILE", "signal_outcomes.json")
        or "signal_outcomes.json",
        track_outcomes=_get_bool("TRACK_OUTCOMES", True),
        signal_cooldown_minutes=_get_int("SIGNAL_COOLDOWN_MINUTES", 45),
        state_retention_days=_get_int("STATE_RETENTION_DAYS", 7),
        log_level=_get_str("LOG_LEVEL", "INFO").upper() or "INFO",
        indicator_backend=_get_str("INDICATOR_BACKEND", "auto").lower() or "auto",
    )
    return settings


def configure_logging(level: str = "INFO") -> None:
    """Log to stdout so Railway / Docker / journald capture everything."""
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # ccxt and httpx are extremely chatty at DEBUG level.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
