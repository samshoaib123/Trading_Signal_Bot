"""Outcome tracking: opening, resolving, scoring and reporting positions."""

import logging
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The modules under test log warnings for the error paths we exercise on purpose.
logging.disable(logging.CRITICAL)

from config import Settings  # noqa: E402
from notifier import format_outcome, format_scoreboard  # noqa: E402
from strategies import BUY, SELL, Signal  # noqa: E402
from tracker import (  # noqa: E402
    LOSS,
    WIN,
    load_outcomes,
    position_id,
    resolve_open,
    save_outcomes,
    scoreboard,
    track_signal,
)

SETTINGS = Settings()
OPENED = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


def make_signal(side=BUY, **kw):
    base = dict(
        symbol="BTC/USDT", setup="rsi_reversal", side=side,
        entry=100.0, stop_loss=97.0, take_profit=104.0, atr=2.0,
        timeframe="15m", candle_time=OPENED, confidence=2,
    )
    if side == SELL:
        base.update(stop_loss=103.0, take_profit=96.0)
    base.update(kw)
    return Signal(**base)


def candles(rows, start_offset_minutes=15):
    """Frame of future candles, indexed after the signal's own candle."""
    idx = [OPENED + timedelta(minutes=start_offset_minutes + 15 * i)
           for i in range(len(rows))]
    return pd.DataFrame(
        [{"open": h, "high": h, "low": l, "close": l, "volume": 1.0} for h, l in rows],
        index=pd.DatetimeIndex(idx, tz="UTC"),
    )


class TestTracking(unittest.TestCase):
    def setUp(self):
        self.ledger = {"open": [], "closed": []}

    def test_signal_is_recorded_as_open(self):
        track_signal(make_signal(), self.ledger, SETTINGS)
        self.assertEqual(len(self.ledger["open"]), 1)
        self.assertEqual(self.ledger["open"][0]["symbol"], "BTC/USDT")

    def test_tracking_the_same_signal_twice_is_idempotent(self):
        sig = make_signal()
        track_signal(sig, self.ledger, SETTINGS)
        track_signal(sig, self.ledger, SETTINGS)
        self.assertEqual(len(self.ledger["open"]), 1)

    def test_tracking_is_skipped_when_disabled(self):
        track_signal(make_signal(), self.ledger, replace(SETTINGS, track_outcomes=False))
        self.assertEqual(self.ledger["open"], [])

    def test_position_id_includes_the_candle(self):
        self.assertIn(OPENED.isoformat(), position_id(make_signal()))


class TestResolution(unittest.TestCase):
    def setUp(self):
        self.ledger = {"open": [], "closed": []}
        track_signal(make_signal(), self.ledger, SETTINGS)

    def test_target_hit_is_a_win(self):
        out = resolve_open("BTC/USDT", candles([(101, 99), (105, 100)]),
                           self.ledger, SETTINGS)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].result, WIN)
        self.assertEqual(out[0].exit_price, 104.0)
        self.assertEqual(self.ledger["open"], [])
        self.assertEqual(len(self.ledger["closed"]), 1)

    def test_stop_hit_is_a_loss(self):
        out = resolve_open("BTC/USDT", candles([(101, 99), (101, 96)]),
                           self.ledger, SETTINGS)
        self.assertEqual(out[0].result, LOSS)
        self.assertEqual(out[0].exit_price, 97.0)

    def test_stop_wins_a_tie_inside_one_candle(self):
        # A candle spanning both levels is scored a loss; we cannot know the order.
        out = resolve_open("BTC/USDT", candles([(105, 96)]), self.ledger, SETTINGS)
        self.assertEqual(out[0].result, LOSS)

    def test_unresolved_position_stays_open(self):
        out = resolve_open("BTC/USDT", candles([(101, 99)]), self.ledger, SETTINGS)
        self.assertEqual(out, [])
        self.assertEqual(len(self.ledger["open"]), 1)

    def test_candles_at_or_before_the_signal_are_ignored(self):
        # A candle stamped at the signal's own time must not resolve it.
        frame = candles([(105, 96)], start_offset_minutes=0)
        self.assertEqual(resolve_open("BTC/USDT", frame, self.ledger, SETTINGS), [])
        self.assertEqual(len(self.ledger["open"]), 1)

    def test_other_symbols_are_left_alone(self):
        resolve_open("ETH/USDT", candles([(105, 96)]), self.ledger, SETTINGS)
        self.assertEqual(len(self.ledger["open"]), 1)

    def test_sell_resolves_in_the_opposite_direction(self):
        ledger = {"open": [], "closed": []}
        track_signal(make_signal(side=SELL), ledger, SETTINGS)
        out = resolve_open("BTC/USDT", candles([(99, 95)]), ledger, SETTINGS)
        self.assertEqual(out[0].result, WIN)
        self.assertEqual(out[0].exit_price, 96.0)

    def test_fees_make_a_win_worth_less_than_its_raw_reward(self):
        out = resolve_open("BTC/USDT", candles([(105, 100)]), self.ledger, SETTINGS)
        raw_r = 4.0 / 3.0                                   # reward / risk
        fee_r = 2 * (SETTINGS.fee_percent / 100) * 100 / 3  # both sides, on notional
        self.assertAlmostEqual(out[0].r_multiple, raw_r - fee_r, places=4)

    def test_fees_make_a_loss_worse_than_one_r(self):
        out = resolve_open("BTC/USDT", candles([(101, 96)]), self.ledger, SETTINGS)
        self.assertLess(out[0].r_multiple, -1.0)

    def test_percentage_is_signed_by_direction(self):
        ledger = {"open": [], "closed": []}
        track_signal(make_signal(side=SELL), ledger, SETTINGS)
        out = resolve_open("BTC/USDT", candles([(99, 95)]), ledger, SETTINGS)
        self.assertGreater(out[0].pct, 0, "a profitable short is a positive result")


class TestScoreboard(unittest.TestCase):
    def test_empty_ledger(self):
        self.assertEqual(scoreboard({"open": [], "closed": []}), (0, 0, 0.0, 0.0))

    def test_counts_and_totals(self):
        ledger = {"open": [], "closed": [
            {"result": WIN, "r_multiple": 1.27},
            {"result": LOSS, "r_multiple": -1.07},
            {"result": WIN, "r_multiple": 1.27},
        ]}
        wins, losses, total_r, rate = scoreboard(ledger)
        self.assertEqual((wins, losses), (2, 1))
        self.assertAlmostEqual(total_r, 1.47, places=4)
        self.assertAlmostEqual(rate, 200 / 3, places=4)

    def test_message_for_an_empty_ledger_explains_itself(self):
        text = format_scoreboard({"open": [], "closed": []}, SETTINGS)
        self.assertIn("No positions have closed yet", text)

    def test_losing_record_says_so_plainly(self):
        ledger = {"open": [], "closed": [{"result": LOSS, "r_multiple": -1.07}]}
        text = format_scoreboard(ledger, SETTINGS)
        self.assertIn("lost money", text)
        self.assertIn("do not increase your size", text)

    def test_outcome_message_has_the_essentials(self):
        ledger = {"open": [], "closed": []}
        track_signal(make_signal(), ledger, SETTINGS)
        out = resolve_open("BTC/USDT", candles([(105, 100)]), ledger, SETTINGS)[0]
        text = format_outcome(out, ledger)
        self.assertIn("Target hit", text)
        self.assertIn("BTC/USDT", text)
        self.assertIn("R after fees", text)
        self.assertIn("Record so far", text)


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "outcomes.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_is_an_empty_ledger(self):
        self.assertEqual(load_outcomes(self.path), {"open": [], "closed": []})

    def test_round_trip(self):
        ledger = {"open": [], "closed": []}
        track_signal(make_signal(), ledger, SETTINGS)
        save_outcomes(self.path, ledger)
        self.assertEqual(len(load_outcomes(self.path)["open"]), 1)

    def test_corrupt_file_does_not_raise(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("{broken")
        self.assertEqual(load_outcomes(self.path), {"open": [], "closed": []})

    def test_closed_history_is_capped(self):
        from tracker import MAX_CLOSED_KEPT

        ledger = {"open": [], "closed": [
            {"result": WIN, "r_multiple": 1.0} for _ in range(MAX_CLOSED_KEPT + 50)
        ]}
        save_outcomes(self.path, ledger)
        self.assertEqual(len(load_outcomes(self.path)["closed"]), MAX_CLOSED_KEPT)


if __name__ == "__main__":
    unittest.main()
