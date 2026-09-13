#!/usr/bin/env python3
"""Crypto Trading Signal Bot — entry point.

Every 15 minutes (aligned to the candle close) the bot:

1. pulls 15m OHLCV candles for each configured pair from Binance via ccxt,
2. computes RSI / MACD / Bollinger Bands / ATR,
3. checks the last **closed** candle for three setups,
4. sizes the trade off ATR, scores confidence 1–3, and
5. pushes anything new to Telegram, de-duplicated through ``signal_state.json``.

Market data is read from public endpoints only, so no exchange API key is
needed. Orders are optional and go to a MetaTrader 5 terminal instead: set
``MT5_ENABLED=true`` to turn execution on. It is off by default, and even
when on it refuses a live account unless ``MT5_ALLOW_LIVE=true`` as well.

Usage::

    python main.py                 # run forever, aligned to candle closes
    python main.py --once          # single scan, useful for cron or a smoke test
    python main.py --dry-run       # scan and log messages without sending them
    python main.py --test-telegram # verify the token / chat id wiring
    python main.py --preflight     # check everything at once before deploying
    python main.py --backtest      # how these setups actually performed on history
    python main.py --report        # send the running win/loss scoreboard
    python main.py --find-chat-id  # print your chat id (needs only the token)
    python main.py --test-mt5      # verify the MetaTrader 5 connection
    python main.py --mt5-close-all # flatten every position this bot opened
"""

from __future__ import annotations

import argparse
import logging
import os
import signal as signal_module
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from config import ConfigError, Settings, configure_logging, load_settings
from exchange import ExchangeError, create_exchange, fetch_ohlcv, load_valid_symbols
from indicators import calculate_indicators, resolve_backend
from notifier import (TelegramNotifier, discover_chats, format_execution,
                      format_outcome, format_scoreboard)
from broker_mt5 import (MT5Broker, MT5Error, load_executions,
                        save_executions)
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


def scan_once(exchange, symbols: List[str], settings: Settings,
              notifier: TelegramNotifier, broker: Optional[MT5Broker] = None) -> int:
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

    execute_signals(delivered, settings, notifier, broker)

    save_outcomes(settings.tracker_file, ledger)
    save_state(settings.state_file, state)
    return len(delivered)


def execute_signals(signals: List[Signal], settings: Settings,
                    notifier: TelegramNotifier,
                    broker: Optional[MT5Broker]) -> None:
    """Send the strongest signals to MetaTrader 5, if execution is enabled.

    Only signals at or above ``MT5_MIN_CONFIDENCE`` are traded, which is a
    higher bar than the one for alerting: it costs nothing to read an alert you
    end up ignoring, and it costs a spread to open a position you end up closing.

    Broker trouble is logged and reported, never fatal. Alerting is the bot's
    primary job; losing the terminal must not stop the next scan from telling
    you what it found.
    """
    if broker is None:
        return

    tradable = [s for s in signals if s.confidence >= settings.mt5_min_confidence]
    skipped = len(signals) - len(tradable)
    if skipped:
        LOG.info(
            "MT5: %d signal(s) below the %d/3 execution threshold, alert only",
            skipped, settings.mt5_min_confidence,
        )
    if not tradable:
        return

    log = load_executions(settings.mt5_execution_file)
    placed = 0
    for sig in tradable:
        try:
            execution = broker.place(sig)
        except MT5Error as exc:
            LOG.error("MT5 execution failed for %s: %s", sig.symbol, exc)
            notifier.send(
                f"\u26a0\ufe0f <b>MT5 execution failed</b>\n{sig.symbol} {sig.side}\n"
                f"<code>{exc}</code>"
            )
            break          # the terminal is unwell; do not hammer it this cycle
        except Exception as exc:  # noqa: BLE001 - one bad order must not stop the scan
            LOG.exception("Unexpected MT5 error on %s: %s", sig.symbol, exc)
            continue

        if execution is None:
            continue       # deliberately skipped; broker.place() already said why
        log.append(execution.to_dict())
        if execution.ok:
            placed += 1
            notifier.send(format_execution(execution))
        else:
            notifier.send(
                f"\u26d4 <b>MT5 order rejected</b>\n{execution.pair} {execution.side}\n"
                f"retcode {execution.retcode}: <code>{execution.comment}</code>"
            )

    if log:
        save_executions(settings.mt5_execution_file, log)
    LOG.info("MT5: placed %d order(s) from %d tradable signal(s)",
             placed, len(tradable))


def open_broker(settings: Settings, dry_run: bool) -> Optional[MT5Broker]:
    """Connect to MetaTrader 5, or return ``None`` and keep alerting.

    A broken terminal downgrades the bot to alert-only rather than taking it
    down: an unexecuted signal you can still act on by hand beats no signal.
    """
    if not settings.mt5_enabled:
        return None
    try:
        settings.require_mt5()
        broker = MT5Broker(settings, dry_run=dry_run)
        broker.connect()
        return broker
    except (MT5Error, ConfigError) as exc:
        LOG.error("MT5 execution is enabled but unavailable: %s", exc)
        LOG.error("Continuing in alert-only mode.")
        return None


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


def build_startup_message(settings: Settings, symbols: List[str], backend: str,
                          broker: Optional[MT5Broker] = None) -> str:
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
        f"<b>Indicators:</b> {backend}\n"
        f"<b>Execution:</b> {_execution_line(settings, broker)}\n\n"
        "<i>You will get an alert on the next qualifying candle close.</i>"
    )


def _execution_line(settings: Settings, broker: Optional[MT5Broker]) -> str:
    """One line describing whether, and how, orders will actually be placed."""
    if broker is None:
        return "alert only (MT5 off)" if not settings.mt5_enabled else (
            "alert only (MT5 enabled but unavailable)"
        )
    try:
        account = broker.account_summary()
    except MT5Error:
        return "MT5 connected, account unreadable"
    suffix = " [DRY RUN]" if broker.dry_run else ""
    return (
        f"MT5 {account['mode']} {account['login']} — "
        f"{account['equity']:,.2f} {account['currency']}, "
        f"trades at {settings.mt5_min_confidence}/3+{suffix}"
    )


def run_mt5_check(settings: Settings, dry_run: bool = False) -> int:
    """Print the MT5 account and the symbol each pair maps to.

    Symbol naming is where MT5 setups fail most often - every broker spells
    ``BTCUSD`` differently - so this prints the resolution for each configured
    pair rather than only the connection status.
    """
    if not settings.mt5_enabled:
        print("MT5_ENABLED is not set. Set MT5_ENABLED=true to use execution.")
        return 1
    try:
        settings.require_mt5()
        broker = MT5Broker(settings, dry_run=dry_run)
        summary = broker.account_summary()
    except (MT5Error, ConfigError) as exc:
        print(f"MT5 check FAILED: {exc}")
        return 1

    print("=" * 72)
    print(f"Account   : {summary['login']} on {summary['server']} [{summary['mode']}]")
    print(f"Balance   : {summary['balance']:,.2f} {summary['currency']}")
    print(f"Equity    : {summary['equity']:,.2f} {summary['currency']}")
    print(f"Free margin: {summary['margin_free']:,.2f} {summary['currency']}")
    print(f"Open (this bot): {summary['open_positions']}")
    print("-" * 72)

    unresolved = []
    for pair in settings.symbols:
        resolved = broker.resolve_symbol(pair)
        print(f"  {pair:<14} -> {resolved or 'NO MATCH'}")
        if resolved is None:
            unresolved.append(pair)
    print("=" * 72)

    if unresolved:
        print(f"{len(unresolved)} pair(s) have no broker symbol and will never be")
        print("traded. Map them explicitly, for example:")
        print("  MT5_SYMBOL_MAP=" + ",".join(f"{p}=YOURSYMBOL" for p in unresolved[:2]))
    if summary["mode"] == "LIVE":
        print("This is a LIVE account. Real money will move.")
    broker.shutdown()
    return 0


def run_mt5_close_all(settings: Settings, dry_run: bool = False) -> int:
    """Flatten every position this bot opened. The panic button."""
    if not settings.mt5_enabled:
        print("MT5_ENABLED is not set; there is nothing for this bot to close.")
        return 1
    try:
        broker = MT5Broker(settings, dry_run=dry_run)
        broker.connect()
        positions = broker.open_positions()
    except (MT5Error, ConfigError) as exc:
        print(f"Could not reach MetaTrader 5: {exc}")
        return 1

    if not positions:
        print(f"No open positions with magic {settings.mt5_magic}.")
        broker.shutdown()
        return 0

    failed = 0
    for position in positions:
        if dry_run:
            print(f"[DRY RUN] would close {position.symbol} ticket {position.ticket}")
            continue
        execution = broker.close(position)
        if execution is None or not execution.ok:
            failed += 1
            print(f"FAILED to close {position.symbol} ticket {position.ticket}")
        else:
            print(f"Closed {position.symbol} ticket {position.ticket} "
                  f"at {execution.filled_price}")
    broker.shutdown()

    if failed:
        print(f"{failed} of {len(positions)} position(s) could not be closed. "
              "Close them by hand in the terminal.")
    return 1 if failed else 0


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
    parser.add_argument("--test-mt5", action="store_true",
                        help="connect to MetaTrader 5, print the account and the "
                             "resolved symbol for each pair, then exit")
    parser.add_argument("--mt5-close-all", action="store_true",
                        help="close every position this bot opened (matched by "
                             "magic number) and exit")
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

    if args.test_mt5:
        return run_mt5_check(settings, dry_run=args.dry_run)

    if args.mt5_close_all:
        return run_mt5_close_all(settings, dry_run=args.dry_run)

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

    broker = open_broker(settings, dry_run=args.dry_run)

    if settings.send_startup_message and not args.once:
        notifier.send(build_startup_message(settings, symbols, backend, broker))

    if args.once:
        try:
            scan_once(exchange, symbols, settings, notifier, broker)
        finally:
            if broker is not None:
                broker.shutdown()
        touch_heartbeat(settings)
        return 0

    # Scan immediately so a fresh deploy does not sit idle for up to 15 minutes,
    # then align to candle closes from there on.
    while not _SHUTDOWN:
        cycle_started = datetime.now(timezone.utc)
        try:
            scan_once(exchange, symbols, settings, notifier, broker)
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

    if broker is not None:
        broker.shutdown()
    LOG.info("Bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
