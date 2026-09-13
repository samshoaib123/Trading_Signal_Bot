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
from typing import Dict, List

from dotenv import load_dotenv

from broker_mt5 import parse_symbol_map

# Load .env if present. override=False => real env vars take precedence, which
# is what we want in the cloud where no .env file is shipped.
LOG = logging.getLogger(__name__)

load_dotenv(override=False)


def _apply_local_config() -> None:
    """Apply values hard-coded in an optional, gitignored ``local_config.py``.

    Editing a Python file is the least fiddly way to get the bot running on your
    own machine, but a hard-coded token in a tracked file is one push away from
    being public. This reads a file git ignores instead, and only fills gaps -
    a real environment variable of the same name always wins, so nothing here
    can silently override Railway's Variables or a GitHub secret.
    """
    try:
        import local_config  # noqa: PLC0415 - optional, may not exist
    except ImportError:
        return
    except Exception as exc:  # noqa: BLE001 - a broken file must not kill the bot
        LOG.warning("local_config.py could not be imported (%s); ignoring it", exc)
        return

    applied = []
    for name in dir(local_config):
        if not name.isupper() or name.startswith("_"):
            continue
        if os.environ.get(name):
            continue                      # the real environment wins
        value = getattr(local_config, name)
        if value is None or callable(value):
            continue
        os.environ[name] = str(value)
        applied.append(name)

    if applied:
        LOG.info("Applied %d setting(s) from local_config.py: %s",
                 len(applied), ", ".join(sorted(applied)))


_apply_local_config()

# Signals score 1-3; mirrored here so require_mt5() can reject an
# unreachable threshold without importing strategies (which imports pandas).
MAX_SIGNAL_CONFIDENCE = 3

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


def _get_number_list(name: str, default: list, cast) -> list:
    """Parse a comma-separated list of numbers, falling back wholesale on error.

    Partial parsing would be worse than useless here: half a take-profit ladder
    is not a smaller ladder, it is a different strategy.
    """
    raw = _get_str(name)
    if not raw:
        return list(default)
    try:
        items = [cast(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError:
        LOG.warning("Env %s=%r is not a list of numbers, using %s",
                    name, raw, default)
        return list(default)
    return items or list(default)


def _get_int_list(name: str, default: list) -> list:
    return _get_number_list(name, default, int)


def _get_float_list(name: str, default: list) -> list:
    return _get_number_list(name, default, float)


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

    # --- Trend Sniper --------------------------------------------------
    # A multi-filter trend setup: an EMA ribbon for direction, a Bollinger
    # squeeze for timing, MACD and Stoch-RSI for momentum, ADX for whether the
    # market is trending enough to bother, and four scaled take-profits.
    stoch_rsi_period: int = 14
    stoch_rsi_k: int = 3
    stoch_rsi_d: int = 3
    stoch_rsi_oversold: float = 20.0
    stoch_rsi_overbought: float = 80.0
    adx_period: int = 14
    adx_trending: float = 25.0
    adx_building: float = 20.0
    kc_period: int = 20
    kc_multiplier: float = 1.5
    ribbon_lengths: List[int] = field(default_factory=lambda: [8, 13, 21, 34, 55])
    # How far the ribbon must be open, as a fraction of price, before a
    # stack counts as a trend rather than noise happening to line up.
    ribbon_min_width: float = 0.002
    # Take-profit ladder, in ATR multiples of the entry. Four targets, scaled so
    # the first is reachable often and the last pays for the ones that miss.
    tp_atr_multipliers: List[float] = field(
        default_factory=lambda: [1.0, 2.0, 3.0, 4.0]
    )
    # Fraction of the position closed at each target. The remainder rides to the
    # final one. Must be the same length as tp_atr_multipliers.
    tp_close_fractions: List[float] = field(
        default_factory=lambda: [0.4, 0.3, 0.2, 0.1]
    )
    # Move the stop to break-even once this target is reached (1-based; 0 = off).
    breakeven_after_target: int = 1
    # Higher timeframes shown in the multi-timeframe context panel.
    context_timeframes: List[str] = field(
        default_factory=lambda: ["15m", "1h", "4h", "1d"]
    )

    # --- MetaTrader 5 execution -------------------------------------------
    # Off by default. See broker_mt5.py for why this is the only module that
    # can move money, and what guards it.
    mt5_enabled: bool = False
    mt5_allow_live: bool = False
    mt5_dry_run: bool = False
    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = ""
    mt5_path: str = ""
    mt5_symbol_map: Dict[str, str] = field(default_factory=dict)
    mt5_symbol_suffix: str = ""
    mt5_magic: int = 907001
    mt5_deviation_points: int = 20
    mt5_filling_mode: str = ""
    mt5_max_open_positions: int = 3
    mt5_max_orders_per_day: int = 10
    mt5_max_lot: float = 0.0
    mt5_min_confidence: int = 2
    mt5_execution_file: str = "mt5_executions.json"

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

    def require_mt5(self) -> None:
        """Fail fast when MT5 execution is on but under-configured.

        A blank login is legitimate - it means "use whatever account the running
        terminal is already logged into" - but a partial set of credentials is
        always a mistake, and one that silently trades the wrong account.
        """
        if not self.mt5_enabled:
            return
        provided = [
            name
            for name, value in (
                ("MT5_LOGIN", self.mt5_login),
                ("MT5_PASSWORD", self.mt5_password),
                ("MT5_SERVER", self.mt5_server),
            )
            if value
        ]
        if provided and len(provided) != 3:
            missing = {"MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER"} - set(provided)
            raise ConfigError(
                "MT5_ENABLED is on with an incomplete login: "
                + ", ".join(sorted(missing))
                + " not set. Either set all three, or clear all three to trade "
                "the account the terminal is already logged into."
            )
        if self.mt5_min_confidence > MAX_SIGNAL_CONFIDENCE:
            raise ConfigError(
                f"MT5_MIN_CONFIDENCE={self.mt5_min_confidence} can never be met; "
                f"signals score at most {MAX_SIGNAL_CONFIDENCE}."
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
        stoch_rsi_period=_get_int("STOCH_RSI_PERIOD", 14),
        stoch_rsi_k=_get_int("STOCH_RSI_K", 3),
        stoch_rsi_d=_get_int("STOCH_RSI_D", 3),
        stoch_rsi_oversold=_get_float("STOCH_RSI_OVERSOLD", 20.0),
        stoch_rsi_overbought=_get_float("STOCH_RSI_OVERBOUGHT", 80.0),
        adx_period=_get_int("ADX_PERIOD", 14),
        adx_trending=_get_float("ADX_TRENDING", 25.0),
        adx_building=_get_float("ADX_BUILDING", 20.0),
        kc_period=_get_int("KC_PERIOD", 20),
        kc_multiplier=_get_float("KC_MULTIPLIER", 1.5),
        ribbon_lengths=_get_int_list("RIBBON_LENGTHS", [8, 13, 21, 34, 55]),
        ribbon_min_width=_get_float("RIBBON_MIN_WIDTH", 0.002),
        tp_atr_multipliers=_get_float_list("TP_ATR_MULTIPLIERS", [1.0, 2.0, 3.0, 4.0]),
        tp_close_fractions=_get_float_list("TP_CLOSE_FRACTIONS", [0.4, 0.3, 0.2, 0.1]),
        breakeven_after_target=_get_int("BREAKEVEN_AFTER_TARGET", 1),
        context_timeframes=[
            tf.lower() for tf in _get_list("CONTEXT_TIMEFRAMES",
                                           ["15m", "1h", "4h", "1d"])
        ],
        mt5_enabled=_get_bool("MT5_ENABLED", False),
        mt5_allow_live=_get_bool("MT5_ALLOW_LIVE", False),
        mt5_dry_run=_get_bool("MT5_DRY_RUN", False),
        mt5_login=_get_int("MT5_LOGIN", 0),
        mt5_password=_get_str("MT5_PASSWORD"),
        mt5_server=_get_str("MT5_SERVER"),
        mt5_path=_get_str("MT5_PATH"),
        mt5_symbol_map=parse_symbol_map(_get_str("MT5_SYMBOL_MAP")),
        mt5_symbol_suffix=_get_str("MT5_SYMBOL_SUFFIX"),
        mt5_magic=_get_int("MT5_MAGIC", 907001),
        mt5_deviation_points=_get_int("MT5_DEVIATION_POINTS", 20),
        mt5_filling_mode=_get_str("MT5_FILLING_MODE").upper(),
        mt5_max_open_positions=_get_int("MT5_MAX_OPEN_POSITIONS", 3),
        mt5_max_orders_per_day=_get_int("MT5_MAX_ORDERS_PER_DAY", 10),
        mt5_max_lot=_get_float("MT5_MAX_LOT", 0.0),
        mt5_min_confidence=_get_int("MT5_MIN_CONFIDENCE", 2),
        mt5_execution_file=_get_str("MT5_EXECUTION_FILE", "mt5_executions.json")
        or "mt5_executions.json",
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
