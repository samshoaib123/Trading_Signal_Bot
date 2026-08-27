"""One-shot deployment self-check.

``python main.py --preflight`` runs every step the bot depends on, in the order
it depends on them, and prints a PASS/FAIL line per step with a concrete fix for
whatever failed. Run it once locally before deploying, and once more from the
cloud host's shell if alerts are not arriving.

Each check is independent: a failure is reported and the run continues, so one
command tells you everything that is wrong rather than only the first problem.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

LOG = logging.getLogger(__name__)

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


@dataclass
class CheckResult:
    """Outcome of one preflight step."""

    name: str
    status: str
    detail: str = ""
    fix: str = ""

    @property
    def ok(self) -> bool:
        return self.status != FAIL


def _config_check(settings) -> CheckResult:
    """Are the two required variables present and plausibly shaped?"""
    missing = [
        name
        for name, value in (
            ("TELEGRAM_BOT_TOKEN", settings.telegram_bot_token),
            ("TELEGRAM_CHAT_ID", settings.telegram_chat_id),
        )
        if not value
    ]
    if missing:
        return CheckResult(
            "Configuration", FAIL,
            f"missing {', '.join(missing)}",
            "Locally: cp .env.example .env and fill them in. "
            "On Railway: Service -> Variables. On a VPS: edit /opt/trading-signal-bot/.env",
        )

    token = settings.telegram_bot_token
    if ":" not in token or not token.split(":")[0].isdigit():
        return CheckResult(
            "Configuration", FAIL,
            "TELEGRAM_BOT_TOKEN does not look like a BotFather token",
            "It must look like 123456789:AAE... - copy the whole line from @BotFather.",
        )

    chat_id = settings.telegram_chat_id.lstrip("-")
    if not chat_id.isdigit() and not settings.telegram_chat_id.startswith("@"):
        return CheckResult(
            "Configuration", WARN,
            f"TELEGRAM_CHAT_ID={settings.telegram_chat_id!r} is not numeric",
            "Usually a number from @userinfobot, or a -100... group id.",
        )

    return CheckResult(
        "Configuration", PASS,
        f"{len(settings.symbols)} symbols, {settings.timeframe}, "
        f"risk {settings.risk_percent}% of {settings.capital:,.0f}",
    )


def _exchange_check(settings) -> Tuple[CheckResult, Optional[object], List[str]]:
    """Can we reach the exchange and list the configured symbols?"""
    from exchange import ExchangeError, create_exchange, load_valid_symbols

    try:
        exchange = create_exchange(settings)
    except ExchangeError as exc:
        return CheckResult("Exchange client", FAIL, str(exc),
                           "Set EXCHANGE_ID to a valid ccxt id."), None, []

    # load_markets() is called directly rather than through load_valid_symbols():
    # that helper deliberately degrades to "use the configured list as-is" when
    # the venue is unreachable, which keeps the bot alive but would make this
    # check report a reachable exchange when nothing was actually reached.
    try:
        exchange.load_markets()
    except Exception as exc:  # noqa: BLE001 - network, DNS, geo-block, TLS...
        return CheckResult(
            "Exchange markets", FAIL, f"{type(exc).__name__}: {exc}",
            "If this mentions 451 or a restricted location, your host's region is "
            "geo-blocked by this exchange - set EXCHANGE_ID=binanceus, kucoin, okx "
            "or bybit, or deploy in another region. Otherwise check outbound "
            "network access from this machine.",
        ), exchange, []

    try:
        symbols = load_valid_symbols(exchange, settings.symbols)
    except ExchangeError as exc:
        return CheckResult(
            "Exchange markets", FAIL, str(exc),
            "Check SYMBOLS against what this exchange actually lists.",
        ), exchange, []

    dropped = len(settings.symbols) - len(symbols)
    detail = f"{exchange.id} reachable, {len(symbols)} tradable symbol(s)"
    if dropped:
        return CheckResult(
            "Exchange markets", WARN, f"{detail}, {dropped} skipped",
            "Listings change (MATIC/USDT is now POL/USDT on Binance). "
            "Update SYMBOLS to silence this.",
        ), exchange, symbols
    return CheckResult("Exchange markets", PASS, detail), exchange, symbols


def _data_check(settings, exchange, symbols) -> Tuple[CheckResult, Optional[object]]:
    """Can we actually pull candles, and are there enough of them?"""
    from exchange import fetch_ohlcv

    if not exchange or not symbols:
        return CheckResult("Candle download", FAIL, "skipped, no usable exchange",
                           "Fix the exchange check above first."), None

    symbol = symbols[0]
    df = fetch_ohlcv(exchange, symbol, settings)
    if df is None or df.empty:
        return CheckResult(
            "Candle download", FAIL, f"no candles returned for {symbol}",
            "Check the logs above for the underlying ccxt error.",
        ), None

    needed = max(settings.macd_slow + settings.macd_signal, settings.bb_period,
                 settings.rsi_period, settings.atr_period) + 5
    if len(df) < needed:
        return CheckResult(
            "Candle download", FAIL,
            f"only {len(df)} closed candles for {symbol}, need >= {needed}",
            f"Raise CANDLE_LIMIT (currently {settings.candle_limit}).",
        ), df

    return CheckResult(
        "Candle download", PASS,
        f"{len(df)} closed {settings.timeframe} candles for {symbol}, "
        f"last close {df['close'].iloc[-1]:g} @ {df.index[-1]:%Y-%m-%d %H:%M} UTC",
    ), df


def _indicator_check(settings, df) -> CheckResult:
    """Do the indicators produce usable numbers on that data?"""
    if df is None:
        return CheckResult("Indicators", FAIL, "skipped, no candle data",
                           "Fix the candle download above first.")
    from indicators import calculate_indicators, resolve_backend

    backend = resolve_backend(settings.indicator_backend)
    try:
        enriched = calculate_indicators(df, settings)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("Indicators", FAIL, f"{type(exc).__name__}: {exc}",
                           "This is a bug - please open an issue with this output.")

    last = enriched.iloc[-1]
    warming = [c for c in ("rsi", "macd", "bb_upper", "atr") if last[[c]].isna().any()]
    if warming:
        return CheckResult(
            "Indicators", FAIL, f"still NaN: {', '.join(warming)}",
            f"Not enough history - raise CANDLE_LIMIT (currently {settings.candle_limit}).",
        )

    trend = enriched.attrs.get("trend_ema_length")
    return CheckResult(
        "Indicators", PASS,
        f"{backend} backend | RSI {last['rsi']:.1f} | ATR {last['atr']:.6g} | "
        f"trend EMA{trend if trend else 'n/a'}",
    )


def _state_check(settings) -> CheckResult:
    """Is the state file path writable? (Silent data loss otherwise.)"""
    from state import load_state, save_state

    path = settings.state_file
    directory = os.path.dirname(os.path.abspath(path)) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, delete=True):
            pass
    except OSError as exc:
        return CheckResult(
            "State file", FAIL, f"{directory} is not writable ({exc})",
            "Point STATE_FILE at a writable path. On Railway attach a Volume and "
            "set STATE_FILE=/data/signal_state.json.",
        )

    existing = load_state(path)
    if existing:
        return CheckResult("State file", PASS,
                           f"{path} readable, {len(existing)} remembered signal(s)")

    save_state(path, {})
    return CheckResult("State file", PASS, f"{path} writable (new file)")


def _telegram_check(settings, notifier) -> CheckResult:
    """Will Telegram actually accept a message for this chat?"""
    ok = notifier.send(
        "\u2705 <b>Preflight check passed</b>\n"
        "Your crypto signal bot can reach this chat. "
        "Alerts will arrive here on the next qualifying candle close."
    )
    if ok:
        return CheckResult("Telegram delivery", PASS, "test message sent")
    return CheckResult(
        "Telegram delivery", FAIL, "see the error above",
        "Most common causes: (1) you never sent /start to your bot - open the chat "
        "and send it; (2) wrong TELEGRAM_CHAT_ID - get it from @userinfobot; "
        "(3) wrong token - re-copy it from @BotFather.",
    )


def run_preflight(settings, notifier) -> int:
    """Run every check and print a report. Returns a process exit code."""
    results: List[CheckResult] = []

    config = _config_check(settings)
    results.append(config)

    exchange_result, exchange, symbols = _exchange_check(settings)
    results.append(exchange_result)

    data_result, df = _data_check(settings, exchange, symbols)
    results.append(data_result)

    results.append(_indicator_check(settings, df))
    results.append(_state_check(settings))

    if config.status == FAIL:
        results.append(CheckResult(
            "Telegram delivery", FAIL, "skipped, credentials missing",
            "Fix the configuration check above first.",
        ))
    else:
        results.append(_telegram_check(settings, notifier))

    _print_report(results)
    return 0 if all(r.ok for r in results) else 1


def _print_report(results: List[CheckResult]) -> None:
    """Print the human-facing summary. Uses print, not logging, on purpose."""
    icons = {PASS: "\u2705", WARN: "\u26a0\ufe0f ", FAIL: "\u274c"}
    width = max(len(r.name) for r in results)

    print("")
    print("=" * 72)
    print("PREFLIGHT REPORT")
    print("=" * 72)
    for result in results:
        print(f"{icons[result.status]} {result.name.ljust(width)}  {result.detail}")
        if result.fix and result.status != PASS:
            for line in _wrap(result.fix, 66):
                print(f"   {' ' * width}  -> {line}")
    print("=" * 72)

    failures = [r for r in results if r.status == FAIL]
    warnings = [r for r in results if r.status == WARN]
    if failures:
        print(f"{len(failures)} check(s) FAILED - the bot will not work yet.")
        print("Fix the '->' items above, then run --preflight again.")
    elif warnings:
        print("All checks passed, with warnings. Safe to deploy.")
    else:
        print("All checks passed. You are ready to deploy: python main.py")
    print("=" * 72)
    print("")


def _wrap(text: str, width: int) -> List[str]:
    """Tiny word-wrapper so remediation hints stay readable in a narrow terminal."""
    words, lines, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines
