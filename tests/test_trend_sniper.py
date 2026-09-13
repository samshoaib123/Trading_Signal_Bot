"""The Trend Sniper setup: four conditions that all have to agree.

Every other setup fires on one condition, so its tests are about the trigger.
This one is mostly about what it *refuses*, so each test holds a firing row
fixed and breaks exactly one leg — which is the only way to show a leg is load
bearing rather than decoration.
"""

import logging
import os
import sys
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.disable(logging.CRITICAL)

from config import Settings  # noqa: E402
from indicators import calculate_indicators  # noqa: E402
from strategies import (  # noqa: E402
    BUY,
    SELL,
    SETUP_REQUIREMENTS,
    _detect_trend_sniper,
    detect_signals,
)

SETTINGS = replace(Settings(), enabled_setups=["trend_sniper"], min_confidence=1)


def firing_row(**overrides):
    """A row where every leg of a long setup is satisfied."""
    row = {
        "close": 100.0, "atr": 2.0,
        "squeeze_off": True, "squeeze_on": False,
        "adx": 30.0, "plus_di": 28.0, "minus_di": 12.0,
        "macd_hist": 0.5, "stochrsi_k": 55.0, "stochrsi_d": 50.0,
        "ribbon_bull": True, "ribbon_bear": False, "ribbon_width": 0.01,
        "bb_bandwidth": 0.03,
    }
    row.update(overrides)
    return pd.Series(row)


def short_row(**overrides):
    row = firing_row(
        plus_di=12.0, minus_di=28.0, macd_hist=-0.5, stochrsi_k=45.0,
        ribbon_bull=False, ribbon_bear=True,
    )
    row.update(overrides)
    return pd.Series(row)


def fire(row, settings=SETTINGS):
    return _detect_trend_sniper(row, firing_row(), settings)


class TriggerTests(unittest.TestCase):
    def test_all_legs_aligned_gives_a_long(self):
        self.assertEqual(fire(firing_row()), BUY)

    def test_the_mirror_image_gives_a_short(self):
        self.assertEqual(fire(short_row()), SELL)


class RefusalTests(unittest.TestCase):
    """Each test breaks exactly one leg of an otherwise firing setup."""

    def test_no_squeeze_release_means_no_trade(self):
        self.assertIsNone(fire(firing_row(squeeze_off=False)))

    def test_a_squeeze_still_on_is_not_a_release(self):
        # Entering inside the range means waiting for the break while exposed.
        self.assertIsNone(fire(firing_row(squeeze_off=False, squeeze_on=True)))

    def test_a_directionless_market_is_refused_however_good_the_rest_looks(self):
        self.assertIsNone(fire(firing_row(adx=18.0)))

    def test_adx_exactly_at_the_threshold_is_accepted(self):
        self.assertEqual(fire(firing_row(adx=SETTINGS.adx_trending)), BUY)

    def test_a_tangled_ribbon_is_refused(self):
        self.assertIsNone(fire(firing_row(ribbon_bull=False)))

    def test_a_ribbon_claiming_both_directions_is_refused(self):
        # Should be impossible from the indicator; refusing beats guessing.
        self.assertIsNone(fire(firing_row(ribbon_bull=True, ribbon_bear=True)))

    def test_directional_movement_disagreeing_with_the_ribbon_is_refused(self):
        self.assertIsNone(fire(firing_row(plus_di=10.0, minus_di=30.0)))

    def test_macd_on_the_wrong_side_of_zero_is_refused(self):
        self.assertIsNone(fire(firing_row(macd_hist=-0.2)))

    def test_an_already_exhausted_move_is_refused(self):
        # Buying a breakout that has already run is the expensive way to be right.
        self.assertIsNone(fire(firing_row(stochrsi_k=95.0)))

    def test_a_short_into_an_exhausted_selloff_is_refused(self):
        self.assertIsNone(fire(short_row(stochrsi_k=5.0)))

    def test_warmup_nans_refuse_rather_than_guess(self):
        for column in ("adx", "plus_di", "stochrsi_k", "macd_hist", "ribbon_width"):
            with self.subTest(column=column):
                self.assertIsNone(fire(firing_row(**{column: np.nan})))

    def test_a_missing_column_refuses_rather_than_raising(self):
        row = firing_row()
        self.assertIsNone(fire(row.drop("adx")))


class ThresholdTests(unittest.TestCase):
    def test_the_adx_threshold_is_configurable(self):
        lenient = replace(SETTINGS, adx_trending=15.0)
        self.assertIsNone(fire(firing_row(adx=18.0)))
        self.assertEqual(fire(firing_row(adx=18.0), lenient), BUY)

    def test_the_exhaustion_threshold_is_configurable(self):
        strict = replace(SETTINGS, stoch_rsi_overbought=50.0)
        self.assertEqual(fire(firing_row(stochrsi_k=55.0)), BUY)
        self.assertIsNone(fire(firing_row(stochrsi_k=55.0), strict))


class IntegrationTests(unittest.TestCase):
    """Through calculate_indicators and detect_signals, on real frames."""

    def frame(self, rows=400, seed=7):
        rng = np.random.default_rng(seed)
        close = np.maximum(100 + np.cumsum(rng.normal(0, 1.0, rows)), 1.0)
        spread = np.abs(rng.normal(0, 0.5, rows)) + 0.1
        index = pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC")
        return pd.DataFrame(
            {"timestamp": index, "open": close, "high": close + spread,
             "low": close - spread, "close": close,
             "volume": rng.uniform(100, 1000, rows)},
            index=index,
        )

    def test_the_setup_is_declared_with_its_extra_requirements(self):
        self.assertIn("trend_sniper", SETUP_REQUIREMENTS)
        self.assertIn("adx", SETUP_REQUIREMENTS["trend_sniper"])

    def test_a_warming_up_frame_produces_nothing_rather_than_raising(self):
        frame = calculate_indicators(self.frame(rows=70), SETTINGS)
        self.assertEqual(detect_signals("BTC/USDT", frame, SETTINGS), [])

    def test_the_sniper_warming_up_does_not_mute_the_other_setups(self):
        # The ribbon needs 55 candles; RSI needs 14. Checking requirements
        # globally would silence RSI until the slowest indicator was ready, so
        # adding a setup would quietly cost you the ones you already had.
        settings = replace(SETTINGS,
                           enabled_setups=["rsi_reversal", "trend_sniper"])
        frame = calculate_indicators(self.frame(rows=45), settings)
        self.assertTrue(pd.isna(frame["ribbon_bull"].iloc[-1]),
                        "fixture must be short enough for the ribbon to be NaN")

        # Force a clean RSI exit from oversold on the last closed candle.
        forced = frame.copy()
        forced.loc[forced.index[-2], "rsi"] = 28.0
        forced.loc[forced.index[-1], "rsi"] = 35.0

        signals = detect_signals("BTC/USDT", forced, settings)
        self.assertEqual([s.setup for s in signals], ["rsi_reversal"])

    def test_a_fired_signal_carries_the_full_ladder(self):
        # Drive the detector directly, then confirm detect_signals' plumbing by
        # forcing the last row to a firing state.
        frame = calculate_indicators(self.frame(), SETTINGS)
        forced = frame.copy()
        for column, value in firing_row().items():
            if column in ("close", "atr"):
                continue
            forced.loc[forced.index[-1], column] = value
        forced.loc[forced.index[-1], "squeeze_off"] = True

        signals = detect_signals("BTC/USDT", forced, SETTINGS)
        self.assertEqual(len(signals), 1)
        signal = signals[0]
        self.assertEqual(signal.setup, "trend_sniper")
        self.assertEqual(len(signal.targets), 4)
        self.assertEqual(len(signal.target_fractions), 4)
        self.assertAlmostEqual(sum(signal.target_fractions), 1.0)
        # take_profit stays the single number every existing consumer reads.
        self.assertEqual(signal.take_profit, signal.targets[-1])

    def test_targets_march_away_from_entry_in_order(self):
        frame = calculate_indicators(self.frame(), SETTINGS)
        forced = frame.copy()
        for column, value in firing_row().items():
            if column in ("close", "atr"):
                continue
            forced.loc[forced.index[-1], column] = value
        signal = detect_signals("BTC/USDT", forced, SETTINGS)[0]
        self.assertEqual(signal.targets, sorted(signal.targets))
        self.assertGreater(signal.targets[0], signal.entry)
        self.assertLess(signal.stop_loss, signal.entry)


if __name__ == "__main__":
    unittest.main()
