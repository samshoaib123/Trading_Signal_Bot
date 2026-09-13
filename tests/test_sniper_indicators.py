"""ADX, Stoch-RSI, Keltner / squeeze and the EMA ribbon.

Unlike RSI or MACD these have no single canonical hand-computable value to
check against — implementations differ in smoothing and in what they do at the
edges. So they are tested by the properties that actually have to hold for the
strategy layer to be safe: bounds, direction on unambiguous input, behaviour on
degenerate input, and where the warm-up NaNs stop.
"""

import logging
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.disable(logging.CRITICAL)

from config import Settings  # noqa: E402
from indicators import (  # noqa: E402
    adx,
    calculate_indicators,
    ema_ribbon,
    keltner_channels,
    squeeze,
    stoch_rsi,
)

SETTINGS = Settings()


def series(values) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="15min", tz="UTC")
    return pd.Series([float(v) for v in values], index=index)


def ohlc(closes, spread=0.5):
    """Build high/low/close around a close path with a constant bar range."""
    close = series(closes)
    return close + spread, close - spread, close


def trend(rows=200, step=1.0, start=100.0):
    return [start + step * i for i in range(rows)]


def noisy_trend(rows=200, step=1.0, start=100.0, seed=2):
    """A rising path that still pulls back, so RSI actually varies.

    A perfectly linear ramp pins RSI at 100 for every bar, which leaves
    Stoch-RSI with no range to normalise against — a degenerate case worth
    testing on purpose, but not what "an uptrend" means.
    """
    rng = np.random.default_rng(seed)
    return list(start + step * np.arange(rows) + rng.normal(0, abs(step) * 2.0, rows))


def flat(rows=200, level=100.0):
    return [level] * rows


def choppy(rows=200, level=100.0, amplitude=1.0):
    return [level + amplitude * (1 if i % 2 else -1) for i in range(rows)]


# --- ADX ------------------------------------------------------------------


class ADXTests(unittest.TestCase):
    def test_strong_uptrend_scores_high_with_plus_di_leading(self):
        high, low, close = ohlc(trend())
        out = adx(high, low, close, 14).iloc[-1]
        self.assertGreater(out["adx"], 50.0)
        self.assertGreater(out["plus_di"], out["minus_di"])

    def test_strong_downtrend_puts_minus_di_ahead(self):
        high, low, close = ohlc(trend(step=-1.0, start=400.0))
        out = adx(high, low, close, 14).iloc[-1]
        self.assertGreater(out["adx"], 50.0)
        self.assertGreater(out["minus_di"], out["plus_di"])

    def test_chop_scores_far_lower_than_a_trend(self):
        # This is the whole reason ADX is in the setup: it must separate these.
        _, _, trending_close = ohlc(trend())
        trending = adx(*ohlc(trend()), 14)["adx"].iloc[-1]
        chopping = adx(*ohlc(choppy()), 14)["adx"].iloc[-1]
        self.assertGreater(trending, chopping)
        self.assertLess(chopping, 25.0)

    def test_adx_stays_inside_zero_to_one_hundred(self):
        rng = np.random.default_rng(3)
        closes = 100 + np.cumsum(rng.normal(0, 1.0, 500))
        out = adx(*ohlc(closes), 14).dropna()
        self.assertTrue((out["adx"] >= 0).all() and (out["adx"] <= 100).all())

    def test_a_perfectly_flat_market_does_not_divide_by_zero(self):
        out = adx(*ohlc(flat(), spread=0.0), 14)
        self.assertFalse(np.isinf(out["adx"].dropna()).any())

    def test_warmup_rows_are_nan_not_zero(self):
        # Zero would read as "no trend" and let a setup fire on no data at all.
        out = adx(*ohlc(trend(rows=60)), 14)
        self.assertTrue(out["adx"].iloc[:14].isna().all())


# --- Stochastic RSI -------------------------------------------------------


class StochRSITests(unittest.TestCase):
    def test_output_is_bounded(self):
        rng = np.random.default_rng(11)
        closes = 100 + np.cumsum(rng.normal(0, 1.0, 500))
        out = stoch_rsi(series(closes), 14, 14, 3, 3).dropna()
        # Exactly [0, 100], not "about" — a caller comparing against an 80
        # threshold should not have to think about float error.
        self.assertTrue((out["stochrsi_k"] >= 0.0).all())
        self.assertTrue((out["stochrsi_k"] <= 100.0).all())
        self.assertTrue((out["stochrsi_d"].dropna() <= 100.0).all())

    def test_it_reads_high_somewhere_in_an_uptrend(self):
        out = stoch_rsi(series(noisy_trend()), 14, 14, 3, 3).dropna()
        self.assertGreater(out["stochrsi_k"].max(), 80.0)

    def test_it_reads_low_somewhere_in_a_downtrend(self):
        out = stoch_rsi(series(noisy_trend(step=-1.0, start=600.0)),
                        14, 14, 3, 3).dropna()
        self.assertLess(out["stochrsi_k"].min(), 20.0)

    def test_a_perfectly_linear_ramp_is_the_degenerate_case(self):
        # RSI is a flat 100 throughout, so there is genuinely no range to
        # normalise against and neutral is the only honest answer.
        out = stoch_rsi(series(trend()), 14, 14, 3, 3)
        self.assertAlmostEqual(out["stochrsi_k"].iloc[-1], 50.0)

    def test_a_flat_rsi_window_reads_as_neutral_not_extreme(self):
        # No range to normalise against must mean "no information", not 0 or 100.
        out = stoch_rsi(series(flat()), 14, 14, 3, 3)
        self.assertAlmostEqual(out["stochrsi_k"].iloc[-1], 50.0)

    def test_d_lags_k(self):
        out = stoch_rsi(series(trend()), 14, 14, 3, 3)
        self.assertTrue(out["stochrsi_d"].isna().sum() > out["stochrsi_k"].isna().sum())


# --- Keltner and the squeeze ---------------------------------------------


class KeltnerTests(unittest.TestCase):
    def test_bands_are_ordered(self):
        out = keltner_channels(*ohlc(trend()), 20, 1.5).dropna()
        self.assertTrue((out["kc_upper"] > out["kc_mid"]).all())
        self.assertTrue((out["kc_mid"] > out["kc_lower"]).all())

    def test_a_wider_multiplier_widens_the_channel(self):
        narrow = keltner_channels(*ohlc(trend()), 20, 1.0).iloc[-1]
        wide = keltner_channels(*ohlc(trend()), 20, 3.0).iloc[-1]
        self.assertGreater(wide["kc_upper"] - wide["kc_lower"],
                           narrow["kc_upper"] - narrow["kc_lower"])


class SqueezeTests(unittest.TestCase):
    def quiet_then_loud(self):
        """A long calm stretch, then a violent expansion."""
        rng = np.random.default_rng(5)
        quiet = 100 + np.cumsum(rng.normal(0, 0.02, 160))
        loud = quiet[-1] + np.cumsum(rng.normal(0, 3.0, 80))
        return np.concatenate([quiet, loud])

    def test_a_calm_market_is_squeezed(self):
        closes = self.quiet_then_loud()
        out = squeeze(*ohlc(closes[:160], spread=0.02), 20, 2.0, 20, 1.5)
        self.assertTrue(bool(out["squeeze_on"].iloc[-1]))

    def test_an_expanding_market_is_not(self):
        closes = self.quiet_then_loud()
        out = squeeze(*ohlc(closes, spread=0.02), 20, 2.0, 20, 1.5)
        self.assertFalse(bool(out["squeeze_on"].iloc[-1]))

    def test_squeeze_off_marks_the_release_not_the_whole_calm_stretch(self):
        # The fire signal is the transition. If this fired every quiet bar the
        # strategy would enter at the start of the range instead of the break.
        closes = self.quiet_then_loud()
        out = squeeze(*ohlc(closes, spread=0.02), 20, 2.0, 20, 1.5)
        self.assertLess(out["squeeze_off"].sum(), out["squeeze_on"].sum())
        self.assertGreaterEqual(out["squeeze_off"].sum(), 1)

    def test_squeeze_off_never_fires_without_a_preceding_squeeze(self):
        out = squeeze(*ohlc(self.quiet_then_loud(), spread=0.02), 20, 2.0, 20, 1.5)
        fired = out.index[out["squeeze_off"].fillna(False)]
        for stamp in fired:
            previous = out["squeeze_on"].shift(1).loc[stamp]
            self.assertTrue(bool(previous))

    def test_bandwidth_is_smaller_when_quiet(self):
        closes = self.quiet_then_loud()
        quiet = squeeze(*ohlc(closes[:160], spread=0.02)).iloc[-1]["bb_bandwidth"]
        loud = squeeze(*ohlc(closes, spread=0.02)).iloc[-1]["bb_bandwidth"]
        self.assertLess(quiet, loud)

    def test_warmup_is_nan_not_false(self):
        # False would read as "no squeeze", which is a claim we cannot make yet.
        out = squeeze(*ohlc(trend(rows=60)), 20, 2.0, 20, 1.5)
        self.assertTrue(out["squeeze_on"].iloc[:19].isna().all())


# --- EMA ribbon -----------------------------------------------------------


class RibbonTests(unittest.TestCase):
    LENGTHS = [8, 13, 21, 34, 55]

    def test_a_clean_uptrend_stacks_bullish(self):
        out = ema_ribbon(series(trend()), self.LENGTHS, min_width=0.002).iloc[-1]
        self.assertTrue(bool(out["ribbon_bull"]))
        self.assertFalse(bool(out["ribbon_bear"]))

    def test_a_clean_downtrend_stacks_bearish(self):
        out = ema_ribbon(series(trend(step=-1.0, start=400.0)), self.LENGTHS,
                         min_width=0.002).iloc[-1]
        self.assertTrue(bool(out["ribbon_bear"]))
        self.assertFalse(bool(out["ribbon_bull"]))

    def test_chop_is_rejected_by_the_width_floor(self):
        # Ordering alone is not enough: in a flat market the EMAs sit on top of
        # each other and their order flips at random, so noise reads as a clean
        # stack about half the time. The floor is what makes this safe.
        out = ema_ribbon(series(choppy()), self.LENGTHS, min_width=0.002).iloc[-1]
        self.assertFalse(bool(out["ribbon_bull"]))
        self.assertFalse(bool(out["ribbon_bear"]))

    def test_chop_is_orders_of_magnitude_narrower_than_a_trend(self):
        chop = ema_ribbon(series(choppy()), self.LENGTHS)["ribbon_width"].iloc[-1]
        real = ema_ribbon(series(trend()), self.LENGTHS)["ribbon_width"].iloc[-1]
        self.assertGreater(real, chop * 20)

    def test_the_floor_is_off_by_default(self):
        # The raw indicator reports ordering; judgement belongs to the caller.
        out = ema_ribbon(series(choppy()), self.LENGTHS).iloc[-1]
        self.assertTrue(bool(out["ribbon_bull"]) or bool(out["ribbon_bear"]))

    def test_a_real_trend_clears_the_floor_comfortably(self):
        out = ema_ribbon(series(trend()), self.LENGTHS, min_width=0.002).iloc[-1]
        self.assertTrue(bool(out["ribbon_bull"]))

    def test_every_requested_ema_is_emitted(self):
        out = ema_ribbon(series(trend()), self.LENGTHS)
        for length in self.LENGTHS:
            self.assertIn(f"ribbon_{length}", out.columns)

    def test_lengths_are_sorted_so_stacking_is_well_defined(self):
        shuffled = ema_ribbon(series(trend()), [34, 8, 55, 13, 21]).iloc[-1]
        ordered = ema_ribbon(series(trend()), self.LENGTHS).iloc[-1]
        self.assertEqual(bool(shuffled["ribbon_bull"]), bool(ordered["ribbon_bull"]))

    def test_warmup_is_nan_until_the_slowest_ema_exists(self):
        out = ema_ribbon(series(trend(rows=80)), self.LENGTHS)
        self.assertTrue(out["ribbon_bull"].iloc[:54].isna().all())
        self.assertTrue(out["ribbon_bull"].iloc[54:].notna().all())

    def test_width_is_relative_so_it_compares_across_instruments(self):
        cheap = ema_ribbon(series(trend(step=0.01, start=1.0)), self.LENGTHS)
        dear = ema_ribbon(series(trend(step=10.0, start=1000.0)), self.LENGTHS)
        self.assertAlmostEqual(cheap["ribbon_width"].iloc[-1],
                               dear["ribbon_width"].iloc[-1], places=3)


# --- integration into calculate_indicators --------------------------------


class PipelineTests(unittest.TestCase):
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

    def test_every_sniper_column_is_attached(self):
        out = calculate_indicators(self.frame(), SETTINGS)
        for column in ("stochrsi_k", "stochrsi_d", "adx", "plus_di", "minus_di",
                       "squeeze_on", "squeeze_off", "bb_bandwidth",
                       "ribbon_bull", "ribbon_bear", "ribbon_width"):
            self.assertIn(column, out.columns, column)

    def test_ribbon_columns_follow_the_configured_lengths(self):
        from dataclasses import replace

        out = calculate_indicators(self.frame(),
                                   replace(SETTINGS, ribbon_lengths=[5, 10]))
        self.assertIn("ribbon_5", out.columns)
        self.assertIn("ribbon_10", out.columns)
        self.assertNotIn("ribbon_55", out.columns)

    def test_the_original_columns_still_survive(self):
        out = calculate_indicators(self.frame(), SETTINGS)
        for column in ("rsi", "macd", "bb_upper", "atr", "trend_ema"):
            self.assertIn(column, out.columns, column)

    def test_the_input_frame_is_not_mutated(self):
        frame = self.frame()
        calculate_indicators(frame, SETTINGS)
        self.assertNotIn("adx", frame.columns)


if __name__ == "__main__":
    unittest.main()
