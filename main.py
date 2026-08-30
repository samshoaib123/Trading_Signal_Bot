#!/usr/bin/env python3
"""Crypto Trading Signal Bot — entry point.

Every 15 minutes (aligned to the candle close) the bot:

1. pulls 15m OHLCV candles for each configured pair from Binance via ccxt,
2. computes RSI / MACD / Bollinger Bands / ATR,
3. checks the last **closed** candle for three setups,
4. sizes the trade off ATR, scores confidence 1–3, and
5. pushes anything new to Telegram, de-duplicated through ``signal_state.json``.

It never places an order and never needs an API key — public endpoints only.

Usage::

    python main.py                 # run forever, aligned to candle closes
    python main.py --once          # single scan, useful for cron or a smoke test
    python main.py --dry-run       # scan and log messages without sending them
    python main.py --test-telegram # verify the token / chat id wiring
    python main.py --preflight     # check everything at once before deploying
    python main.py --backtest      # how these setups actually performed on history
    python main.py --report        # send the running win/loss scoreboard
    python main.py --find-chat-id  # print your chat id (needs only the token)
"""

from __future__ import annotations

import argparse
import logging
import os
import signal as signal_module
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import List

from config import ConfigError, Settings, configure_logging, load_settings
from exchange import ExchangeError, create_exchange, fetch_ohlcv, load_valid_symbols
from indicators import calculate_indicators, resolve_backend
from notifier import (TelegramNotifier, discover_chats, format_outcome,
                      format_scoreboard)
from backtest import run_backtest
from preflight import run_preflight
from state import load_state, prune_state, record_signal, save_state, should_send
from strategies import Signal, detect_signals
from tracker import load_outcomes, resolve_open, save_outcomes, track_signal

LOG = logging.getLogger("signal_bot")

_SHUTDOWN = False


def _handle_signal(signum, _frame) -> None:
    """Flip the shutdown flag so the loop can exit between cycles."""
    global _SHUTDOWN
    LOG.info("Received signal %s, shutting down after this cycle", signum)
    _SHUTDOWN = True


def next_candle_close(now: datetime, interval_minutes: int, buffer_seconds: int) -> datetime:
    """Return the next candle boundary (plus a small buffer).

    15m candles close at :00, :15, :30 and :45. We wake a few seconds *after*
    the boundary so the exchange has definitely published the closed candle.
    """
    minutes_since_hour = now.minute + now.second / 60 + now.microsecond / 60_000_000
    completed = int(minutes_since_hour // interval_minutes)
    boundary = now.replace(minute=0, second=0, microsecond=0) + timedelta(
        minutes=(completed + 1) * interval_minutes
    )
    target = boundary + timedelta(seconds=buffer_seconds)
    if target <= now:  # we were already past the buffer for this boundary
        target += timedelta(minutes=interval_minutes)
    return target


def sleep_until(target: datetime) -> None:
    """Sleep until ``target``, waking every few seconds to notice SIGTERM."""
    while not _SHUTDOWN:
        remaining = (target - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 5.0))


def scan_once(exchange, symbols: List[str], settings: Settings, notifier: TelegramNotifier) -> int:
    """Run one full scan across every symbol. Returns signals sent."""
    state = prune_state(load_state(settings.state_file), settings.state_retention_days)
    ledger = load_outcomes(settings.tracker_file)
    fresh: List[Signal] = []
    resolved = []
    scanned = 0

    for symbol in symbols:
        if _SHUTDOWN:
            break
        try:
            df = fetch_ohlcv(exchange, symbol, settings)
            if df is None or df.empty:
                continue
            # Close out any tracked position using the candles we just fetched,
            # before looking for new entries. Costs no extra API calls.
            resolved.extend(resolve_open(symbol, df, ledger, settings))
            df = calculate_indicators(df, settings)
            found = detect_signals(symbol, df, settings)
            scanned += 1
        except Exception as exc:  # noqa: BLE001 - one bad pair must not stop the scan
            LOG.exception("%s: scan failed (%s)", symbol, exc)
            continue

        for sig in found:
            LOG.info(
                "%s %s %s | entry %.8g SL %.8g TP %.8g | confidence %d/3 %s",
                sig.symbol, sig.setup, sig.side, sig.entry, sig.stop_loss,
                sig.take_profit, sig.confidence,
                f"({', '.join(sig.confirmations)})" if sig.confirmations else "",
            )
            if should_send(sig, state, settings):
                fresh.append(sig)
            else:
                LOG.debug("%s: duplicate suppressed", sig.dedupe_key)

    LOG.info(
        "Scan complete: %d/%d symbols OK, %d new signal(s), %d closed position(s)",
        scanned, len(symbols), len(fresh), len(resolved),
    )

    # Report closed positions first: knowing the last call was wrong is context
    # for the next one.
    for outcome in resolved:
        LOG.info(
            "%s %s %s closed: %s %+.2fR",
            outcome.symbol, outcome.setup, outcome.side,
            outcome.result, outcome.r_multiple,
        )
        notifier.send(format_outcome(outcome, ledger))

    if not fresh:
        save_outcomes(settings.tracker_file, ledger)
        save_state(settings.state_file, state)
        return 0

    # Highest-confidence first, so the most interesting alert leads the digest.
    fresh.sort(key=lambda s: (-s.confidence, s.symbol, s.setup))
    delivered = notifier.send_signals(fresh)

    # Record only what Telegram actually accepted. A batch can split into several
    # messages and any one of them can fail; recording the whole batch would let
    # de-duplication suppress a signal the user never received.
    for sig in delivered:
        record_signal(sig, state)

    if len(delivered) < len(fresh):
        LOG.error(
            "%d of %d signal(s) were not delivered; left unrecorded so the next "
            "cycle retries them",
            len(fresh) - len(delivered), len(fresh),
        )

    for sig in delivered:
        track_signal(sig, ledger, settings)

    save_outcomes(settings.tracker_file, ledger)
    save_state(settings.state_file, state)
    return len(delivered)


def touch_heartbeat(settings: Settings) -> None:
    """Record that a scan cycle completed, for external liveness checks.

    Written only when ``HEARTBEAT_FILE`` is set. A monitor (the container
    healthcheck, a cron job, an uptime probe) can treat a file older than a
    couple of cycles as "the bot is wedged". Failures here are logged and
    ignored - a monitoring convenience must never take the bot down.
    """
    if not settings.heartbeat_file:
        return
    try:
        directory = os.path.dirname(os.path.abspath(settings.heartbeat_file))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(settings.heartbeat_file, "w", encoding="utf-8") as fh:
            fh.write(datetime.now(timezone.utc).isoformat())
    except OSError as exc:
        LOG.warning("Could not write heartbeat to %s: %s", settings.heartbeat_file, exc)


def build_startup_message(settings: Settings, symbols: List[str], backend: str) -> str:
    """Human-readable summary of the running configuration."""
    setups = ", ".join(settings.enabled_setups)
    return (
        "✅ <b>Crypto Signal Bot started</b>\n\n"
        f"<b>Exchange:</b> {settings.exchange_id}\n"
        f"<b>Timeframe:</b> {settings.timeframe}\n"
        f"<b>Pairs ({len(symbols)}):</b> {', '.join(symbols)}\n"
        f"<b>Setups:</b> {setups}\n"
        f"<b>Risk:</b> {settings.risk_percent:.2f}% of "
        f"{settings.capital:,.0f} per trade\n"
        f"<b>SL/TP:</b> {settings.atr_sl_multiplier}x / {settings.atr_tp_multiplier}x "
        f"ATR({settings.atr_period})\n"
        f"<b>Min confidence:</b> {settings.min_confidence}/3\n"
        f"<b>Indicators:</b> {backend}\n\n"
        "<i>You will get an alert on the next qualifying candle close.</i>"
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crypto trading signal bot")
    parser.add_argument("--once", action="store_true",
                        help="run a single scan and exit (cron / smoke test)")
    parser.add_argument("--dry-run", action="store_true",
                        help="log Telegram messages instead of sending them")
    parser.add_argument("--test-telegram", action="store_true",
                        help="send a test message and exit")
    parser.add_argument("--find-chat-id", action="store_true",
                        help="list the chat ids that have messaged your bot "
                             "(only the token is needed)")
    parser.add_argument("--report", action="store_true",
                        help="send the cumulative win/loss scoreboard and exit")
    parser.add_argument("--backtest", action="store_true",
                        help="replay history for the configured pairs and print "
                             "win rate, expectancy and profit factor per setup")
    parser.add_argument("--preflight", action="store_true",
                        help="check config, exchange, data, state and Telegram, "
                             "then print a report and exit")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_level)

    LOG.info("Crypto Trading Signal Bot starting up")

    if args.preflight:
        return run_preflight(settings, TelegramNotifier(settings, dry_run=args.dry_run))

    if args.find_chat_id:
        return discover_chats(settings)

    if args.backtest:
        # No Telegram credentials needed: this only reads public candles.
        try:
            exchange = create_exchange(settings)
            symbols = load_valid_symbols(exchange, settings.symbols)
        except ExchangeError as exc:
            LOG.error("%s", exc)
            return 2
        print(run_backtest(exchange, symbols, settings,
                           settings.backtest_candles, settings.fee_percent))
        return 0

    if not args.dry_run:
        try:
            settings.require_telegram()
        except ConfigError as exc:
            LOG.error("%s", exc)
            return 2

    notifier = TelegramNotifier(settings, dry_run=args.dry_run)

    if args.test_telegram:
        ok = notifier.send(
            "🔔 <b>Test message</b>\nYour crypto signal bot is wired up correctly."
        )
        LOG.info("Telegram test %s", "succeeded" if ok else "FAILED")
        return 0 if ok else 1

    if args.report:
        ledger = load_outcomes(settings.tracker_file)
        ok = notifier.send(format_scoreboard(ledger, settings))
        LOG.info("Scoreboard %s", "sent" if ok else "FAILED to send")
        return 0 if ok else 1

    try:
        exchange = create_exchange(settings)
        symbols = load_valid_symbols(exchange, settings.symbols)
    except ExchangeError as exc:
        LOG.error("%s", exc)
        return 2

    backend = resolve_backend(settings.indicator_backend)
    LOG.info(
        "Watching %d pair(s) on %s %s using the %s indicator backend: %s",
        len(symbols), settings.exchange_id, settings.timeframe, backend,
        ", ".join(symbols),
    )

    signal_module.signal(signal_module.SIGTERM, _handle_signal)
    signal_module.signal(signal_module.SIGINT, _handle_signal)

    if settings.send_startup_message and not args.once:
        notifier.send(build_startup_message(settings, symbols, backend))

    if args.once:
        scan_once(exchange, symbols, settings, notifier)
        touch_heartbeat(settings)
        return 0

    # Scan immediately so a fresh deploy does not sit idle for up to 15 minutes,
    # then align to candle closes from there on.
    while not _SHUTDOWN:
        cycle_started = datetime.now(timezone.utc)
        try:
            scan_once(exchange, symbols, settings, notifier)
        except Exception as exc:  # noqa: BLE001 - the loop must never die
            LOG.exception("Unhandled error during scan: %s", exc)
        touch_heartbeat(settings)

        if _SHUTDOWN:
            break

        target = next_candle_close(
            datetime.now(timezone.utc),
            settings.poll_interval_minutes,
            settings.candle_close_buffer_seconds,
        )
        LOG.info(
            "Cycle took %.1fs; next scan at %s UTC",
            (datetime.now(timezone.utc) - cycle_started).total_seconds(),
            target.strftime("%Y-%m-%d %H:%M:%S"),
        )
        sleep_until(target)

    LOG.info("Bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
