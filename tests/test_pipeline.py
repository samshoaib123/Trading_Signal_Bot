"""End-to-end pipeline test with a fake exchange.

Covers everything except the network call itself: fetch -> indicators ->
setup detection -> de-duplication -> Telegram formatting.
"""

import logging
import os
import sys
import tempfile
import unittest
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The modules under test log warnings for the error paths we exercise on purpose.
logging.disable(logging.CRITICAL)

import ccxt  # noqa: E402

from config import Settings  # noqa: E402
from exchange import fetch_ohlcv, load_valid_symbols  # noqa: E402
from main import scan_once  # noqa: E402
from state import load_state  # noqa: E402

CANDLE_MS = 15 * 60 * 1000
START_MS = 1_700_000_000_000


def bounce_series(rows: int = 262) -> list:
    """Long decline, then a sharp bounce that pops RSI back above 30.

    The bounce is the *second to last* candle so that it becomes the last
    CLOSED candle once :func:`fetch_ohlcv` discards the still-forming one.
    """
    close = [100.0]
    for _ in range(rows - 3):
        close.append(close[-1] * 0.995)
    close.append(close[-1] * 1.06)   # the bounce: last closed candle
    close.append(close[-1])          # still-forming candle, dropped on fetch
    candles = []
    for i, price in enumerate(close):
        volume = 500.0 if i >= rows - 2 else 100.0
        candles.append(
            [START_MS + i * CANDLE_MS, price, price * 1.002, price * 0.998, price, volume]
        )
    return candles


def flat_series(rows: int = 262) -> list:
    """A dead-flat market: no setup can possibly trigger."""
    return [
        [START_MS + i * CANDLE_MS, 100.0, 100.0, 100.0, 100.0, 100.0]
        for i in range(rows)
    ]


class FakeExchange:
    """Minimal stand-in for a ccxt exchange."""

    id = "fakeexchange"
    rateLimit = 50

    def __init__(self, candles_by_symbol, markets=None, fail_times=0, fail_with=None):
        self.candles_by_symbol = candles_by_symbol
        self._markets = markets
        self.fail_times = fail_times
        self.fail_with = fail_with or ccxt.NetworkError("boom")
        self.calls = 0

    def load_markets(self):
        if self._markets is None:
            return {s: {"symbol": s, "active": True} for s in self.candles_by_symbol}
        return self._markets

    def fetch_ohlcv(self, symbol, timeframe="15m", limit=300):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.fail_with
        return list(self.candles_by_symbol[symbol])[-limit:]


class RecordingNotifier:
    """Captures messages instead of talking to Telegram."""

    def __init__(self, fail=False):
        self.messages = []
        self.fail = fail

    def send(self, text):
        self.messages.append(text)
        return not self.fail

    def send_signals(self, signals):
        """Mirrors TelegramNotifier: returns the signals actually delivered."""
        signals = list(signals)
        if not signals:
            return []
        from notifier import format_digest_batches

        delivered = []
        for message, batch in format_digest_batches(signals, Settings()):
            if self.send(message):
                delivered.extend(batch)
        return delivered


class TestFetch(unittest.TestCase):
    def setUp(self):
        self.settings = replace(Settings(), fetch_backoff_seconds=0.0)

    def test_drops_the_still_forming_candle(self):
        candles = bounce_series(rows=50)
        ex = FakeExchange({"BTC/USDT": candles})
        df = fetch_ohlcv(ex, "BTC/USDT", self.settings)
        self.assertEqual(len(df), len(candles) - 1)
        self.assertEqual(df["close"].iloc[-1], candles[-2][4])

    def test_keeps_the_forming_candle_when_asked(self):
        candles = bounce_series(rows=50)
        ex = FakeExchange({"BTC/USDT": candles})
        df = fetch_ohlcv(ex, "BTC/USDT", self.settings, drop_unclosed=False)
        self.assertEqual(len(df), len(candles))

    def test_index_is_utc_and_sorted(self):
        ex = FakeExchange({"BTC/USDT": bounce_series(rows=50)})
        df = fetch_ohlcv(ex, "BTC/USDT", self.settings)
        self.assertEqual(str(df.index.tz), "UTC")
        self.assertTrue(df.index.is_monotonic_increasing)

    def test_retries_then_succeeds(self):
        ex = FakeExchange({"BTC/USDT": bounce_series(rows=50)}, fail_times=2)
        df = fetch_ohlcv(ex, "BTC/USDT", self.settings)
        self.assertIsNotNone(df)
        self.assertEqual(ex.calls, 3)

    def test_returns_none_after_exhausting_retries(self):
        settings = replace(self.settings, fetch_retries=3)
        ex = FakeExchange({"BTC/USDT": bounce_series(rows=50)}, fail_times=99)
        self.assertIsNone(fetch_ohlcv(ex, "BTC/USDT", settings))
        self.assertEqual(ex.calls, 3)

    def test_bad_symbol_is_not_retried(self):
        ex = FakeExchange(
            {"BTC/USDT": bounce_series(rows=50)},
            fail_times=99,
            fail_with=ccxt.BadSymbol("nope"),
        )
        self.assertIsNone(fetch_ohlcv(ex, "BTC/USDT", self.settings))
        self.assertEqual(ex.calls, 1)

    def test_geo_blocked_error_is_not_retried(self):
        ex = FakeExchange(
            {"BTC/USDT": bounce_series(rows=50)},
            fail_times=99,
            fail_with=ccxt.ExchangeError("451 restricted location"),
        )
        self.assertIsNone(fetch_ohlcv(ex, "BTC/USDT", self.settings))
        self.assertEqual(ex.calls, 1)


class TestSymbolValidation(unittest.TestCase):
    def test_unlisted_and_inactive_symbols_are_dropped(self):
        ex = FakeExchange(
            {},
            markets={
                "BTC/USDT": {"symbol": "BTC/USDT", "active": True},
                "MATIC/USDT": {"symbol": "MATIC/USDT", "active": False},
            },
        )
        valid = load_valid_symbols(ex, ["BTC/USDT", "MATIC/USDT", "GHOST/USDT"])
        self.assertEqual(valid, ["BTC/USDT"])

    def test_raises_when_nothing_is_tradable(self):
        from exchange import ExchangeError

        ex = FakeExchange({}, markets={})
        with self.assertRaises(ExchangeError):
            load_valid_symbols(ex, ["GHOST/USDT"])


class TestScanOnce(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.settings = replace(
            Settings(),
            state_file=os.path.join(self.tmpdir.name, "state.json"),
            tracker_file=os.path.join(self.tmpdir.name, "outcomes.json"),
            fetch_backoff_seconds=0.0,
            symbols=["BTC/USDT"],
        )
        self.exchange = FakeExchange({"BTC/USDT": bounce_series()})

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_a_signal_is_detected_formatted_and_recorded(self):
        notifier = RecordingNotifier()
        sent = scan_once(self.exchange, ["BTC/USDT"], self.settings, notifier)

        self.assertEqual(sent, 1)
        self.assertEqual(len(notifier.messages), 1)
        message = notifier.messages[0]
        self.assertIn("BUY Signal", message)
        self.assertIn("BTC/USDT", message)
        self.assertIn("RSI Reversal", message)
        self.assertIn("Suggested size:", message)

        state = load_state(self.settings.state_file)
        self.assertIn("BTC/USDT|rsi_reversal|BUY", state)

    def test_second_scan_of_the_same_candle_sends_nothing(self):
        notifier = RecordingNotifier()
        scan_once(self.exchange, ["BTC/USDT"], self.settings, notifier)
        scan_once(self.exchange, ["BTC/USDT"], self.settings, notifier)
        self.assertEqual(len(notifier.messages), 1)

    def test_state_is_not_recorded_when_telegram_fails(self):
        failing = RecordingNotifier(fail=True)
        scan_once(self.exchange, ["BTC/USDT"], self.settings, failing)
        self.assertEqual(load_state(self.settings.state_file), {})

        # Next cycle retries and succeeds, so no alert is lost.
        working = RecordingNotifier()
        scan_once(self.exchange, ["BTC/USDT"], self.settings, working)
        self.assertEqual(len(working.messages), 1)

    def test_a_flat_market_produces_no_alerts(self):
        exchange = FakeExchange({"BTC/USDT": flat_series()})
        notifier = RecordingNotifier()
        self.assertEqual(scan_once(exchange, ["BTC/USDT"], self.settings, notifier), 0)
        self.assertEqual(notifier.messages, [])

    def test_one_broken_pair_does_not_stop_the_others(self):
        class HalfBrokenExchange(FakeExchange):
            def fetch_ohlcv(self, symbol, timeframe="15m", limit=300):
                if symbol == "ETH/USDT":
                    raise ccxt.BadSymbol("delisted")
                return super().fetch_ohlcv(symbol, timeframe, limit)

        exchange = HalfBrokenExchange(
            {"BTC/USDT": bounce_series(), "ETH/USDT": bounce_series()}
        )
        notifier = RecordingNotifier()
        sent = scan_once(
            exchange, ["ETH/USDT", "BTC/USDT"], self.settings, notifier
        )
        self.assertEqual(sent, 1)
        self.assertIn("BTC/USDT", notifier.messages[0])

    def test_min_confidence_three_still_lets_the_strong_signal_through(self):
        settings = replace(self.settings, min_confidence=3)
        notifier = RecordingNotifier()
        self.assertEqual(
            scan_once(self.exchange, ["BTC/USDT"], settings, notifier), 1
        )

    def test_all_setups_disabled_yields_nothing(self):
        settings = replace(self.settings, enabled_setups=[])
        notifier = RecordingNotifier()
        self.assertEqual(
            scan_once(self.exchange, ["BTC/USDT"], settings, notifier), 0
        )


if __name__ == "__main__":
    unittest.main()


class TestHeartbeat(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_heartbeat_is_written_when_configured(self):
        from main import touch_heartbeat

        path = os.path.join(self.tmpdir.name, "nested", "heartbeat")
        touch_heartbeat(replace(Settings(), heartbeat_file=path))
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as fh:
            self.assertIn("T", fh.read())  # an ISO-8601 timestamp

    def test_heartbeat_is_skipped_when_unset(self):
        from main import touch_heartbeat

        touch_heartbeat(replace(Settings(), heartbeat_file=""))
        self.assertEqual(os.listdir(self.tmpdir.name), [])

    def test_unwritable_heartbeat_does_not_raise(self):
        from main import touch_heartbeat

        # A path whose parent is a regular file can never be created.
        blocker = os.path.join(self.tmpdir.name, "blocker")
        open(blocker, "w").close()
        touch_heartbeat(
            replace(Settings(), heartbeat_file=os.path.join(blocker, "heartbeat"))
        )


class TestPartialDelivery(unittest.TestCase):
    """A signal in a message Telegram rejected must not be marked as sent."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.settings = replace(
            Settings(),
            state_file=os.path.join(self.tmpdir.name, "state.json"),
            tracker_file=os.path.join(self.tmpdir.name, "outcomes.json"),
            fetch_backoff_seconds=0.0,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_only_delivered_signals_are_recorded(self):
        class FlakyNotifier(RecordingNotifier):
            """Accepts the first message, rejects every later one."""

            def send(self, text):
                self.messages.append(text)
                return len(self.messages) == 1

        # Force a split: tiny limit means one message per signal.
        import notifier as notifier_mod

        original = notifier_mod.MAX_MESSAGE_CHARS
        notifier_mod.MAX_MESSAGE_CHARS = 600
        try:
            exchange = FakeExchange(
                {"BTC/USDT": bounce_series(), "ETH/USDT": bounce_series()}
            )
            flaky = FlakyNotifier()
            sent = scan_once(
                exchange, ["BTC/USDT", "ETH/USDT"], self.settings, flaky
            )
        finally:
            notifier_mod.MAX_MESSAGE_CHARS = original

        self.assertGreater(len(flaky.messages), 1, "expected the batch to split")
        self.assertEqual(sent, 1, "only the accepted message's signal counts")

        state = load_state(self.settings.state_file)
        self.assertEqual(len(state), 1, "the rejected signal must stay unrecorded")

    def test_undelivered_signal_is_retried_next_cycle(self):
        class FailOnce(RecordingNotifier):
            calls = 0

            def send(self, text):
                FailOnce.calls += 1
                self.messages.append(text)
                return FailOnce.calls > 1   # first send fails, later ones succeed

        exchange = FakeExchange({"BTC/USDT": bounce_series()})
        first = FailOnce()
        self.assertEqual(scan_once(exchange, ["BTC/USDT"], self.settings, first), 0)
        self.assertEqual(load_state(self.settings.state_file), {})

        second = RecordingNotifier()
        self.assertEqual(scan_once(exchange, ["BTC/USDT"], self.settings, second), 1)
        self.assertEqual(len(second.messages), 1)


class TestOutcomeTrackingInTheLoop(unittest.TestCase):
    """A signal sent in one scan must be closed out by a later scan."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.settings = replace(
            Settings(),
            state_file=os.path.join(self.tmpdir.name, "state.json"),
            tracker_file=os.path.join(self.tmpdir.name, "outcomes.json"),
            fetch_backoff_seconds=0.0,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_signal_is_tracked_then_resolved_on_a_later_scan(self):
        from tracker import load_outcomes

        base = bounce_series()
        exchange = FakeExchange({"BTC/USDT": base})
        notifier = RecordingNotifier()

        # Scan 1: the bounce fires a BUY, which is recorded as an open position.
        self.assertEqual(scan_once(exchange, ["BTC/USDT"], self.settings, notifier), 1)
        ledger = load_outcomes(self.settings.tracker_file)
        self.assertEqual(len(ledger["open"]), 1)
        self.assertEqual(ledger["closed"], [])

        entry = ledger["open"][0]["entry"]
        target = ledger["open"][0]["take_profit"]

        # Scan 2: append candles that reach the target, then rescan.
        last_ts = base[-1][0]
        reach = [
            [last_ts + 900000, entry, target * 1.01, entry * 0.999, target, 500.0],
            [last_ts + 1800000, target, target, target, target, 500.0],
        ]
        exchange.candles_by_symbol["BTC/USDT"] = base + reach

        notifier2 = RecordingNotifier()
        scan_once(exchange, ["BTC/USDT"], self.settings, notifier2)

        ledger = load_outcomes(self.settings.tracker_file)
        self.assertEqual(ledger["open"], [], "the position should be closed")
        self.assertEqual(len(ledger["closed"]), 1)
        self.assertEqual(ledger["closed"][0]["result"], "win")

        outcome_messages = [m for m in notifier2.messages if "Target hit" in m]
        self.assertEqual(len(outcome_messages), 1)
        self.assertIn("R after fees", outcome_messages[0])
        self.assertIn("Record so far", outcome_messages[0])

    def test_tracking_can_be_switched_off(self):
        from tracker import load_outcomes

        settings = replace(self.settings, track_outcomes=False)
        exchange = FakeExchange({"BTC/USDT": bounce_series()})
        scan_once(exchange, ["BTC/USDT"], settings, RecordingNotifier())
        self.assertEqual(load_outcomes(settings.tracker_file)["open"], [])
