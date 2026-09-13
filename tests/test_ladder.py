"""Scaled exits: the take-profit ladder, break-even, and how it is scored.

A single target makes one decision be right twice — near enough to hit often,
far enough to pay for the misses. The ladder splits that, and these tests pin
down the arithmetic, because a scaling bug does not crash, it just quietly
reports the wrong record.
"""

import logging
import os
import sys
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.disable(logging.CRITICAL)

from config import Settings  # noqa: E402
from strategies import BUY, SELL, Signal, build_targets  # noqa: E402
from tracker import (  # noqa: E402
    LOSS,
    WIN,
    load_outcomes,
    resolve_open,
    save_outcomes,
    track_signal,
)

OPENED = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
# Zero fees keep the R arithmetic exact; fees are covered on their own below.
SETTINGS = replace(Settings(), fee_percent=0.0, track_outcomes=True)


def make_signal(side=BUY, entry=100.0, stop=98.0, targets=None, fractions=None):
    targets = targets if targets is not None else [102.0, 104.0, 106.0, 108.0]
    fractions = fractions if fractions is not None else [0.4, 0.3, 0.2, 0.1]
    return Signal(
        symbol="BTC/USDT", setup="trend_sniper", side=side, entry=entry,
        stop_loss=stop, take_profit=targets[-1], atr=2.0, timeframe="15m",
        candle_time=OPENED, confidence=3, targets=list(targets),
        target_fractions=list(fractions),
    )


def candles(bars):
    """``bars`` is a list of (high, low) after the signal candle."""
    index = [OPENED + timedelta(minutes=15 * (i + 1)) for i in range(len(bars))]
    return pd.DataFrame(
        {"high": [b[0] for b in bars], "low": [b[1] for b in bars],
         "open": [b[1] for b in bars], "close": [b[0] for b in bars],
         "volume": [1.0] * len(bars)},
        index=pd.DatetimeIndex(index, tz="UTC"),
    )


def run(signal, bars, settings=SETTINGS):
    ledger = {"open": [], "closed": []}
    track_signal(signal, ledger, settings)
    resolved = resolve_open("BTC/USDT", candles(bars), ledger, settings)
    return ledger, (resolved[0] if resolved else None)


# --- building the ladder --------------------------------------------------


class BuildTests(unittest.TestCase):
    def test_targets_step_away_from_entry_in_atr_multiples(self):
        targets, _ = build_targets(BUY, 100.0, 2.0, SETTINGS)
        self.assertEqual([round(t, 6) for t in targets], [102.0, 104.0, 106.0, 108.0])

    def test_a_short_ladder_steps_downwards(self):
        targets, _ = build_targets(SELL, 100.0, 2.0, SETTINGS)
        self.assertEqual([round(t, 6) for t in targets], [98.0, 96.0, 94.0, 92.0])

    def test_fractions_are_normalised_to_one(self):
        settings = replace(SETTINGS, tp_close_fractions=[2.0, 2.0, 2.0, 2.0])
        _, fractions = build_targets(BUY, 100.0, 2.0, settings)
        self.assertAlmostEqual(sum(fractions), 1.0)

    def test_mismatched_lengths_fall_back_to_one_target(self):
        # Half a ladder is not a smaller ladder, it is a different strategy.
        settings = replace(SETTINGS, tp_close_fractions=[0.5, 0.5])
        targets, fractions = build_targets(BUY, 100.0, 2.0, settings)
        self.assertEqual(len(targets), 1)
        self.assertEqual(fractions, [1.0])

    def test_a_signal_without_targets_gets_a_one_rung_ladder(self):
        signal = Signal(
            symbol="BTC/USDT", setup="rsi_reversal", side=BUY, entry=100.0,
            stop_loss=98.0, take_profit=104.0, atr=2.0, timeframe="15m",
            candle_time=OPENED,
        )
        self.assertEqual(signal.targets, [104.0])
        self.assertEqual(signal.target_fractions, [1.0])

    def test_target_r_multiples_are_reported_in_units_of_risk(self):
        signal = make_signal()          # risk 2.0, targets at +2 +4 +6 +8
        self.assertEqual([round(r, 6) for r in signal.target_r_multiples],
                         [1.0, 2.0, 3.0, 4.0])


# --- walking the ladder ---------------------------------------------------


class WalkTests(unittest.TestCase):
    def test_an_immediate_stop_is_a_full_loss(self):
        _, outcome = run(make_signal(), [(100.5, 97.0)])
        self.assertEqual(outcome.result, LOSS)
        self.assertAlmostEqual(outcome.r_multiple, -1.0)
        self.assertEqual(outcome.targets_hit, 0)

    def test_running_the_whole_ladder_pays_the_weighted_sum(self):
        # 0.4*1R + 0.3*2R + 0.2*3R + 0.1*4R = 2.0R
        _, outcome = run(make_signal(), [(109.0, 99.5)])
        self.assertEqual(outcome.result, WIN)
        self.assertAlmostEqual(outcome.r_multiple, 2.0)
        self.assertEqual(outcome.targets_hit, 4)
        self.assertEqual(outcome.targets_total, 4)

    def test_two_targets_then_a_reversal_keeps_what_was_banked(self):
        # TP1 and TP2 bank 0.4*1 + 0.3*2 = 1.0R; the rest stops at break-even
        # for 0R rather than giving back a full R.
        _, outcome = run(make_signal(), [(104.5, 99.5), (101.0, 99.9)])
        self.assertAlmostEqual(outcome.r_multiple, 1.0)
        self.assertEqual(outcome.targets_hit, 2)
        self.assertTrue(outcome.stopped_at_breakeven)
        self.assertEqual(outcome.result, WIN)

    def test_a_breakeven_stop_before_any_target_is_still_a_full_loss(self):
        _, outcome = run(make_signal(), [(101.5, 99.5), (100.0, 97.5)])
        self.assertAlmostEqual(outcome.r_multiple, -1.0)
        self.assertFalse(outcome.stopped_at_breakeven)

    def test_the_stop_moves_to_entry_once_the_first_target_is_in(self):
        # Reaching TP1 then falling to 99 must not cost a full R.
        _, outcome = run(make_signal(), [(102.5, 99.5), (101.0, 99.0)])
        self.assertGreater(outcome.r_multiple, 0.0)
        self.assertTrue(outcome.stopped_at_breakeven)

    def test_breakeven_can_be_switched_off(self):
        settings = replace(SETTINGS, breakeven_after_target=0)
        _, outcome = run(make_signal(), [(102.5, 99.5), (101.0, 97.5)], settings)
        # TP1 banked 0.4R, the remaining 0.6 lost a full R each.
        self.assertAlmostEqual(outcome.r_multiple, 0.4 * 1.0 + 0.6 * -1.0)
        self.assertFalse(outcome.stopped_at_breakeven)

    def test_several_rungs_inside_one_candle_are_all_booked(self):
        _, outcome = run(make_signal(), [(106.5, 99.5), (101.0, 99.9)])
        self.assertEqual(outcome.targets_hit, 3)

    def test_the_stop_still_wins_ties_inside_a_candle(self):
        # A candle that touches both must be scored as the loss: with no tick
        # data, assuming the win would flatter the record.
        _, outcome = run(make_signal(), [(109.0, 97.0)])
        self.assertEqual(outcome.result, LOSS)
        self.assertAlmostEqual(outcome.r_multiple, -1.0)

    def test_a_short_ladder_walks_downwards(self):
        signal = make_signal(side=SELL, entry=100.0, stop=102.0,
                             targets=[98.0, 96.0, 94.0, 92.0])
        _, outcome = run(signal, [(100.5, 91.0)])
        self.assertEqual(outcome.result, WIN)
        self.assertAlmostEqual(outcome.r_multiple, 2.0)

    def test_a_short_stops_out_on_a_rally(self):
        signal = make_signal(side=SELL, entry=100.0, stop=102.0,
                             targets=[98.0, 96.0, 94.0, 92.0])
        _, outcome = run(signal, [(103.0, 99.5)])
        self.assertEqual(outcome.result, LOSS)

    def test_fees_are_charged_once_on_the_round_trip(self):
        paid = replace(SETTINGS, fee_percent=0.1)
        _, free_outcome = run(make_signal(), [(109.0, 99.5)])
        _, paid_outcome = run(make_signal(), [(109.0, 99.5)], paid)
        self.assertLess(paid_outcome.r_multiple, free_outcome.r_multiple)


# --- positions that have not resolved yet ---------------------------------


class OpenPositionTests(unittest.TestCase):
    def test_a_running_position_stays_open(self):
        ledger, outcome = run(make_signal(), [(101.0, 99.5)])
        self.assertIsNone(outcome)
        self.assertEqual(len(ledger["open"]), 1)

    def test_partial_progress_is_reported_on_the_open_position(self):
        ledger, outcome = run(make_signal(), [(104.5, 99.5)])
        self.assertIsNone(outcome)
        position = ledger["open"][0]
        self.assertEqual(position["targets_hit"], 2)
        self.assertAlmostEqual(position["open_r"], 1.0)
        self.assertTrue(position["at_breakeven"])

    def test_the_recorded_stop_is_never_overwritten_by_the_breakeven_move(self):
        # Writing the moved stop back would make risk read as zero on the next
        # scan, and the position could then never resolve at all.
        ledger, _ = run(make_signal(), [(104.5, 99.5)])
        self.assertEqual(ledger["open"][0]["stop_loss"], 98.0)
        self.assertEqual(ledger["open"][0]["stop_now"], 100.0)

    def test_a_position_that_moved_to_breakeven_still_resolves_on_a_later_scan(self):
        settings = SETTINGS
        ledger = {"open": [], "closed": []}
        track_signal(make_signal(), ledger, settings)

        first = candles([(104.5, 99.5)])
        self.assertEqual(resolve_open("BTC/USDT", first, ledger, settings), [])

        # The next scan sees the same history plus one more candle.
        both = candles([(104.5, 99.5), (101.0, 99.9)])
        resolved = resolve_open("BTC/USDT", both, ledger, settings)
        self.assertEqual(len(resolved), 1)
        self.assertAlmostEqual(resolved[0].r_multiple, 1.0)

    def test_replaying_the_same_candles_does_not_double_count_rungs(self):
        settings = SETTINGS
        ledger = {"open": [], "closed": []}
        track_signal(make_signal(), ledger, settings)
        bars = candles([(104.5, 99.5)])
        resolve_open("BTC/USDT", bars, ledger, settings)
        resolve_open("BTC/USDT", bars, ledger, settings)
        self.assertEqual(ledger["open"][0]["targets_hit"], 2)
        self.assertAlmostEqual(ledger["open"][0]["open_r"], 1.0)


# --- compatibility with positions recorded before ladders existed ---------


class LegacyTests(unittest.TestCase):
    def legacy_position(self):
        return {
            "id": "BTC/USDT|rsi_reversal|BUY|x", "symbol": "BTC/USDT",
            "setup": "rsi_reversal", "side": BUY, "entry": 100.0,
            "stop_loss": 98.0, "take_profit": 104.0, "confidence": 2,
            "opened_at": OPENED.isoformat(),
        }

    def test_a_single_target_position_resolves_as_it_always_did(self):
        ledger = {"open": [self.legacy_position()], "closed": []}
        resolved = resolve_open("BTC/USDT", candles([(104.5, 99.5)]),
                                ledger, SETTINGS)
        self.assertEqual(resolved[0].result, WIN)
        self.assertAlmostEqual(resolved[0].r_multiple, 2.0)

    def test_a_single_target_position_still_loses_a_full_r(self):
        ledger = {"open": [self.legacy_position()], "closed": []}
        resolved = resolve_open("BTC/USDT", candles([(101.0, 97.0)]),
                                ledger, SETTINGS)
        self.assertAlmostEqual(resolved[0].r_multiple, -1.0)

    def test_the_ladder_survives_a_save_and_reload(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "outcomes.json")
            ledger = {"open": [], "closed": []}
            track_signal(make_signal(), ledger, SETTINGS)
            save_outcomes(path, ledger)
            reloaded = load_outcomes(path)
            self.assertEqual(len(reloaded["open"][0]["targets"]), 4)
            resolved = resolve_open("BTC/USDT", candles([(109.0, 99.5)]),
                                    reloaded, SETTINGS)
            self.assertAlmostEqual(resolved[0].r_multiple, 2.0)


if __name__ == "__main__":
    unittest.main()
