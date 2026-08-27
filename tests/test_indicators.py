"""Indicator correctness checks against hand-computed reference values."""

import os
import sys
import logging
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The modules under test log warnings for the error paths we exercise on purpose.
logging.disable(logging.CRITICAL)

from config import Settings  # noqa: E402
from indicators import (  # noqa: E402
    atr,
    bollinger_bands,
    calculate_indicators,
    ema,
    macd,
    rsi,
    sma,
    true_range,
)


def synthetic_ohlcv(rows: int = 400, seed: int = 7) -> pd.DataFrame:
    """Deterministic random walk with a realistic high/low/volume structure."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1.0, rows))
    close = np.maximum(close, 1.0)
    spread = np.abs(rng.normal(0, 0.5, rows)) + 0.1
    index = pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": close + rng.normal(0, 0.2, rows),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.uniform(100, 1000, rows),
        },
        index=index,
    )


class TestBuildingBlocks(unittest.TestCase):
    def test_sma_matches_manual_mean(self):
        s = pd.Series([1.0, 2, 3, 4, 5])
        out = sma(s, 3)
        self.assertTrue(np.isnan(out.iloc[1]))
        self.assertAlmostEqual(out.iloc[2], 2.0)
        self.assertAlmostEqual(out.iloc[4], 4.0)

    def test_ema_matches_recursive_definition(self):
        # adjust=False means EMA_t = a*x_t + (1-a)*EMA_(t-1), seeded on x_0.
        s = pd.Series([1.0, 2, 3, 4, 5, 6])
        alpha = 2 / (3 + 1)
        manual = s.iloc[0]
        for value in s.iloc[1:]:
            manual = alpha * value + (1 - alpha) * manual
        out = ema(s, 3)
        self.assertAlmostEqual(out.iloc[-1], manual, places=10)
        self.assertTrue(np.isnan(out.iloc[0]))

    def test_rsi_all_up_candles_is_100(self):
        s = pd.Series(np.arange(1.0, 40.0))
        self.assertAlmostEqual(rsi(s, 14).iloc[-1], 100.0)

    def test_rsi_all_down_candles_is_zero(self):
        s = pd.Series(np.arange(40.0, 1.0, -1.0))
        self.assertAlmostEqual(rsi(s, 14).iloc[-1], 0.0, places=6)

    def test_rsi_stays_within_bounds(self):
        df = synthetic_ohlcv()
        values = rsi(df["close"], 14).dropna()
        self.assertTrue((values >= 0).all() and (values <= 100).all())
        self.assertGreater(len(values), 300)

    def test_true_range_uses_previous_close(self):
        high = pd.Series([10.0, 12.0])
        low = pd.Series([9.0, 11.5])
        close = pd.Series([9.5, 11.8])
        # Second bar: high-low = 0.5, |high-prev_close| = 2.5 -> TR = 2.5
        self.assertAlmostEqual(true_range(high, low, close).iloc[1], 2.5)

    def test_atr_is_positive_and_warms_up(self):
        df = synthetic_ohlcv()
        values = atr(df["high"], df["low"], df["close"], 14)
        self.assertTrue(np.isnan(values.iloc[10]))
        self.assertTrue((values.dropna() > 0).all())

    def test_macd_hist_is_line_minus_signal(self):
        df = synthetic_ohlcv()
        out = macd(df["close"])
        tail = out.dropna()
        self.assertTrue(
            np.allclose(tail["macd_hist"], tail["macd"] - tail["macd_signal"])
        )

    def test_bollinger_band_ordering_and_width(self):
        df = synthetic_ohlcv()
        bb = bollinger_bands(df["close"], 20, 2.0).dropna()
        self.assertTrue((bb["bb_upper"] > bb["bb_mid"]).all())
        self.assertTrue((bb["bb_mid"] > bb["bb_lower"]).all())
        # Upper/lower must be symmetric around the mid band.
        self.assertTrue(
            np.allclose(
                bb["bb_upper"] - bb["bb_mid"], bb["bb_mid"] - bb["bb_lower"]
            )
        )

    def test_bollinger_constant_series_has_zero_width(self):
        s = pd.Series([50.0] * 30)
        bb = bollinger_bands(s, 20, 2.0).dropna()
        self.assertTrue(np.allclose(bb["bb_upper"], bb["bb_lower"]))


class TestCalculateIndicators(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()

    def test_adds_every_required_column(self):
        out = calculate_indicators(synthetic_ohlcv(), self.settings)
        for column in (
            "rsi", "macd", "macd_signal", "macd_hist",
            "bb_lower", "bb_mid", "bb_upper", "atr",
            "volume_sma", "trend_ema",
        ):
            self.assertIn(column, out.columns, column)
        self.assertTrue(out[["rsi", "atr", "macd"]].iloc[-1].notna().all())

    def test_does_not_mutate_the_input(self):
        df = synthetic_ohlcv(rows=250)
        before = list(df.columns)
        calculate_indicators(df, self.settings)
        self.assertEqual(before, list(df.columns))

    def test_trend_ema_steps_down_when_history_is_short(self):
        # 120 rows cannot support a 200 EMA, so it should fall back to 100.
        out = calculate_indicators(synthetic_ohlcv(rows=120), self.settings)
        self.assertEqual(out.attrs["trend_ema_length"], 100)
        self.assertTrue(out["trend_ema"].notna().iloc[-1])

    def test_trend_ema_is_nan_when_history_is_tiny(self):
        out = calculate_indicators(synthetic_ohlcv(rows=40), self.settings)
        self.assertIsNone(out.attrs["trend_ema_length"])

    def test_rejects_frames_missing_columns(self):
        df = synthetic_ohlcv(rows=60).drop(columns=["volume"])
        with self.assertRaises(ValueError):
            calculate_indicators(df, self.settings)

    def test_rejects_empty_frames(self):
        with self.assertRaises(ValueError):
            calculate_indicators(pd.DataFrame(), self.settings)


if __name__ == "__main__":
    unittest.main()
