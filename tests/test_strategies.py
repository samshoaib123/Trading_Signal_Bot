"""Setup detection, risk levels, confidence scoring and de-duplication."""

import os
import sys
import logging
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The modules under test log warnings for the error paths we exercise on purpose.
logging.disable(logging.CRITICAL)

from config import Settings  # noqa: E402
from indicators import calculate_indicators  # noqa: E402
from state import (  # noqa: E402
    load_state,
    prune_state,
    record_signal,
    save_state,
    should_send,
)
from strategies import (  # noqa: E402
    BUY,
    SELL,
    Signal,
    build_levels,
    detect_signals,
    position_size,
    score_confidence,
)

SETTINGS = Settings()


def frame_from_rows(rows):
    """Build a two-row indicator frame straight from explicit values.

    Bypassing the indicator maths lets each test state exactly the crossing it
    wants to exercise instead of reverse-engineering a price series.
    """
    index = pd.date_range("2024-06-01 12:00", periods=len(rows), freq="15min", tz="UTC")
    df = pd.DataFrame(rows, index=index)
    df["timestamp"] = index
    df.attrs["trend_ema_length"] = 200
    return df


def base_row(**overrides):
    """A neutral candle: no setup triggers, no confirmations pass."""
    row = {
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": 100.0, "volume_sma": 100.0,
        "rsi": 50.0,
        "macd": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
        "bb_lower": 95.0, "bb_mid": 100.0, "bb_upper": 105.0,
        "atr": 2.0, "trend_ema": 100.0,
    }
    row.update(overrides)
    return row


class TestSetupDetection(unittest.TestCase):
    def test_rsi_reversal_buy_on_exit_from_oversold(self):
        df = frame_from_rows([base_row(rsi=28.0), base_row(rsi=32.0)])
        signals = detect_signals("BTC/USDT", df, SETTINGS)
        self.assertEqual([(s.setup, s.side) for s in signals],
                         [("rsi_reversal", BUY)])

    def test_rsi_reversal_sell_on_exit_from_overbought(self):
        df = frame_from_rows([base_row(rsi=72.0), base_row(rsi=68.0)])
        signals = detect_signals("BTC/USDT", df, SETTINGS)
        self.assertEqual([(s.setup, s.side) for s in signals],
                         [("rsi_reversal", SELL)])

    def test_rsi_staying_above_30_does_not_refire(self):
        df = frame_from_rows([base_row(rsi=35.0), base_row(rsi=42.0)])
        self.assertEqual(detect_signals("BTC/USDT", df, SETTINGS), [])

    def test_rsi_dipping_below_30_alone_is_not_a_signal(self):
        df = frame_from_rows([base_row(rsi=35.0), base_row(rsi=25.0)])
        self.assertEqual(detect_signals("BTC/USDT", df, SETTINGS), [])

    def test_macd_crossover_buy(self):
        df = frame_from_rows([
            base_row(macd=-1.0, macd_signal=-0.5, macd_hist=-0.5),
            base_row(macd=0.4, macd_signal=0.1, macd_hist=0.3),
        ])
        signals = detect_signals("ETH/USDT", df, SETTINGS)
        self.assertEqual([(s.setup, s.side) for s in signals],
                         [("macd_crossover", BUY)])

    def test_macd_crossover_sell(self):
        df = frame_from_rows([
            base_row(macd=1.0, macd_signal=0.5, macd_hist=0.5),
            base_row(macd=-0.2, macd_signal=0.1, macd_hist=-0.3),
        ])
        signals = detect_signals("ETH/USDT", df, SETTINGS)
        self.assertEqual([(s.setup, s.side) for s in signals],
                         [("macd_crossover", SELL)])

    def test_macd_staying_above_signal_does_not_refire(self):
        df = frame_from_rows([
            base_row(macd=1.0, macd_signal=0.5),
            base_row(macd=1.4, macd_signal=0.7),
        ])
        self.assertEqual(detect_signals("ETH/USDT", df, SETTINGS), [])

    def test_bollinger_breakout_buy(self):
        df = frame_from_rows([
            base_row(close=104.0, bb_upper=105.0),
            base_row(close=106.0, bb_upper=105.0),
        ])
        signals = detect_signals("SOL/USDT", df, SETTINGS)
        self.assertEqual([(s.setup, s.side) for s in signals],
                         [("bb_breakout", BUY)])

    def test_bollinger_breakdown_sell(self):
        df = frame_from_rows([
            base_row(close=96.0, bb_lower=95.0),
            base_row(close=94.0, bb_lower=95.0),
        ])
        signals = detect_signals("SOL/USDT", df, SETTINGS)
        self.assertEqual([(s.setup, s.side) for s in signals],
                         [("bb_breakout", SELL)])

    def test_price_riding_the_upper_band_does_not_refire(self):
        df = frame_from_rows([
            base_row(close=106.0, bb_upper=105.0),
            base_row(close=107.0, bb_upper=105.5),
        ])
        self.assertEqual(detect_signals("SOL/USDT", df, SETTINGS), [])

    def test_multiple_setups_can_fire_on_one_candle(self):
        df = frame_from_rows([
            base_row(rsi=28.0, macd=-1.0, macd_signal=-0.5, close=104.0),
            base_row(rsi=32.0, macd=0.4, macd_signal=0.1, close=106.0),
        ])
        setups = {s.setup for s in detect_signals("BTC/USDT", df, SETTINGS)}
        self.assertEqual(setups, {"rsi_reversal", "macd_crossover", "bb_breakout"})

    def test_disabled_setups_are_skipped(self):
        settings = replace(SETTINGS, enabled_setups=["macd_crossover"])
        df = frame_from_rows([base_row(rsi=28.0), base_row(rsi=32.0)])
        self.assertEqual(detect_signals("BTC/USDT", df, settings), [])

    def test_warming_up_candles_produce_nothing(self):
        df = frame_from_rows([base_row(rsi=np.nan), base_row(rsi=32.0)])
        self.assertEqual(detect_signals("BTC/USDT", df, SETTINGS), [])

    def test_zero_atr_is_rejected(self):
        df = frame_from_rows([base_row(rsi=28.0, atr=0.0), base_row(rsi=32.0, atr=0.0)])
        self.assertEqual(detect_signals("BTC/USDT", df, SETTINGS), [])

    def test_single_candle_frame_is_safe(self):
        self.assertEqual(detect_signals("BTC/USDT", frame_from_rows([base_row()]), SETTINGS), [])


class TestRiskLevels(unittest.TestCase):
    def test_buy_levels_use_the_atr_multipliers(self):
        sl, tp = build_levels(BUY, 100.0, 2.0, SETTINGS)
        self.assertAlmostEqual(sl, 97.0)   # 100 - 1.5 * 2
        self.assertAlmostEqual(tp, 104.0)  # 100 + 2.0 * 2

    def test_sell_levels_are_mirrored(self):
        sl, tp = build_levels(SELL, 100.0, 2.0, SETTINGS)
        self.assertAlmostEqual(sl, 103.0)
        self.assertAlmostEqual(tp, 96.0)

    def test_risk_reward_is_one_to_one_third_three(self):
        df = frame_from_rows([base_row(rsi=28.0), base_row(rsi=32.0)])
        signal = detect_signals("BTC/USDT", df, SETTINGS)[0]
        self.assertAlmostEqual(signal.risk_reward, 2.0 / 1.5, places=6)

    def test_position_size_matches_the_formula(self):
        settings = replace(SETTINGS, capital=1000.0, risk_percent=1.0)
        size, notional, risk = position_size(100.0, 97.0, settings)
        self.assertAlmostEqual(risk, 10.0)          # 1% of 1000
        self.assertAlmostEqual(size, 10.0 / 3.0)    # risk / stop distance
        self.assertAlmostEqual(notional, size * 100.0)

    def test_position_size_handles_a_zero_stop_distance(self):
        size, notional, risk = position_size(100.0, 100.0, SETTINGS)
        self.assertEqual((size, notional), (0.0, 0.0))
        self.assertAlmostEqual(risk, 10.0)

    def test_signal_carries_position_size_when_enabled(self):
        df = frame_from_rows([base_row(rsi=28.0), base_row(rsi=32.0)])
        signal = detect_signals("BTC/USDT", df, SETTINGS)[0]
        self.assertIsNotNone(signal.position_size)
        self.assertAlmostEqual(signal.risk_amount, 10.0)

    def test_position_size_omitted_when_disabled(self):
        settings = replace(SETTINGS, show_position_size=False)
        df = frame_from_rows([base_row(rsi=28.0), base_row(rsi=32.0)])
        self.assertIsNone(detect_signals("BTC/USDT", df, settings)[0].position_size)


class TestConfidence(unittest.TestCase):
    def test_no_confirmations_scores_one(self):
        curr = pd.Series(base_row(rsi=32.0, _trend_ema_length=200))
        prev = pd.Series(base_row(rsi=28.0))
        score, passed = score_confidence("rsi_reversal", BUY, curr, prev, SETTINGS)
        self.assertEqual((score, passed), (1, []))

    def test_every_confirmation_scores_three(self):
        curr = pd.Series(base_row(
            rsi=32.0, volume=500.0, volume_sma=100.0,
            close=110.0, trend_ema=100.0, macd_hist=0.5, _trend_ema_length=200,
        ))
        prev = pd.Series(base_row(rsi=28.0, macd_hist=0.1))
        score, passed = score_confidence("rsi_reversal", BUY, curr, prev, SETTINGS)
        self.assertEqual(score, 3)
        self.assertEqual(len(passed), 3)

    def test_one_confirmation_scores_two(self):
        curr = pd.Series(base_row(
            rsi=32.0, volume=500.0, volume_sma=100.0, _trend_ema_length=200,
        ))
        prev = pd.Series(base_row(rsi=28.0))
        score, passed = score_confidence("rsi_reversal", BUY, curr, prev, SETTINGS)
        self.assertEqual((score, passed), (2, ["Volume spike"]))

    def test_confidence_is_capped_at_three(self):
        curr = pd.Series(base_row(
            volume=9999.0, volume_sma=1.0, close=200.0, trend_ema=100.0,
            macd_hist=5.0, rsi=55.0, _trend_ema_length=200,
        ))
        prev = pd.Series(base_row(macd_hist=0.0))
        for setup in ("rsi_reversal", "macd_crossover", "bb_breakout"):
            score, _ = score_confidence(setup, BUY, curr, prev, SETTINGS)
            self.assertLessEqual(score, 3, setup)

    def test_missing_trend_ema_does_not_crash_scoring(self):
        curr = pd.Series(base_row(trend_ema=np.nan, _trend_ema_length=None))
        prev = pd.Series(base_row())
        score, passed = score_confidence("bb_breakout", BUY, curr, prev, SETTINGS)
        self.assertEqual(score, 1)
        self.assertNotIn("Trend aligned", passed)

    def test_min_confidence_filters_weak_signals(self):
        settings = replace(SETTINGS, min_confidence=3)
        df = frame_from_rows([base_row(rsi=28.0), base_row(rsi=32.0)])
        self.assertEqual(detect_signals("BTC/USDT", df, settings), [])


class TestEndToEndOnRealShapedData(unittest.TestCase):
    """detect_signals must survive a full indicator pipeline without raising."""

    def test_pipeline_runs_over_a_random_walk(self):
        rng = np.random.default_rng(3)
        rows = 400
        close = 100 + np.cumsum(rng.normal(0, 1.2, rows))
        close = np.maximum(close, 1.0)
        index = pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": index,
                "open": close,
                "high": close + 0.8,
                "low": close - 0.8,
                "close": close,
                "volume": rng.uniform(100, 900, rows),
            },
            index=index,
        )
        # Walk the whole series so many different candles get evaluated.
        enriched = calculate_indicators(df, SETTINGS)
        total = 0
        for end in range(250, rows):
            window = enriched.iloc[: end + 1]
            window.attrs["trend_ema_length"] = enriched.attrs["trend_ema_length"]
            for sig in detect_signals("BTC/USDT", window, SETTINGS):
                total += 1
                self.assertIn(sig.side, (BUY, SELL))
                self.assertGreater(sig.entry, 0)
                self.assertGreater(sig.take_profit, 0)
                self.assertGreater(sig.stop_loss, 0)
                self.assertIn(sig.confidence, (1, 2, 3))
                if sig.side == BUY:
                    self.assertLess(sig.stop_loss, sig.entry)
                    self.assertGreater(sig.take_profit, sig.entry)
                else:
                    self.assertGreater(sig.stop_loss, sig.entry)
                    self.assertLess(sig.take_profit, sig.entry)
        self.assertGreater(total, 0, "expected at least one signal over 150 candles")


class TestDeduplication(unittest.TestCase):
    def setUp(self):
        self.candle = datetime(2024, 6, 1, 12, 15, tzinfo=timezone.utc)
        self.signal = Signal(
            symbol="BTC/USDT", setup="rsi_reversal", side=BUY,
            entry=100.0, stop_loss=97.0, take_profit=104.0, atr=2.0,
            timeframe="15m", candle_time=self.candle, confidence=2,
        )

    def test_first_signal_is_always_sent(self):
        self.assertTrue(should_send(self.signal, {}, SETTINGS))

    def test_same_candle_is_suppressed_after_a_restart(self):
        state = {}
        record_signal(self.signal, state)
        self.assertFalse(should_send(self.signal, state, SETTINGS))

    def test_new_candle_within_cooldown_is_suppressed(self):
        state = {}
        record_signal(self.signal, state)
        later = replace(self.signal, candle_time=self.candle + timedelta(minutes=15))
        self.assertFalse(should_send(later, state, SETTINGS))

    def test_new_candle_after_cooldown_is_sent(self):
        state = {}
        record_signal(self.signal, state)
        state[self.signal.dedupe_key]["sent_at"] = (
            datetime.now(timezone.utc) - timedelta(minutes=120)
        ).isoformat()
        later = replace(self.signal, candle_time=self.candle + timedelta(hours=2))
        self.assertTrue(should_send(later, state, SETTINGS))

    def test_opposite_side_is_a_different_key(self):
        state = {}
        record_signal(self.signal, state)
        opposite = replace(self.signal, side=SELL)
        self.assertTrue(should_send(opposite, state, SETTINGS))

    def test_other_setup_on_same_pair_is_a_different_key(self):
        state = {}
        record_signal(self.signal, state)
        other = replace(self.signal, setup="macd_crossover")
        self.assertTrue(should_send(other, state, SETTINGS))

    def test_prune_drops_stale_entries_only(self):
        state = {
            "old": {"sent_at": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()},
            "new": {"sent_at": datetime.now(timezone.utc).isoformat()},
        }
        kept = prune_state(state, retention_days=7)
        self.assertEqual(set(kept), {"new"})


class TestStatePersistence(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "state.json")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_missing_file_loads_as_empty(self):
        self.assertEqual(load_state(self.path), {})

    def test_round_trip(self):
        payload = {"BTC/USDT|rsi_reversal|BUY": {"sent_at": "2024-06-01T12:00:00+00:00"}}
        save_state(self.path, payload)
        self.assertEqual(load_state(self.path), payload)

    def test_corrupt_file_loads_as_empty_instead_of_raising(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("{not json at all")
        self.assertEqual(load_state(self.path), {})

    def test_save_leaves_no_temp_files_behind(self):
        save_state(self.path, {"a": {"sent_at": "2024-06-01T12:00:00+00:00"}})
        leftovers = [f for f in os.listdir(self.tmpdir.name) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()


class TestTimeframeParsing(unittest.TestCase):
    def test_known_units(self):
        from strategies import timeframe_minutes

        self.assertEqual(timeframe_minutes("15m"), 15)
        self.assertEqual(timeframe_minutes("4h"), 240)
        self.assertEqual(timeframe_minutes("1d"), 1440)
        self.assertEqual(timeframe_minutes("1w"), 10080)

    def test_unknown_timeframe_is_zero(self):
        from strategies import timeframe_minutes

        self.assertEqual(timeframe_minutes("banana"), 0)
        self.assertEqual(timeframe_minutes(""), 0)

    def test_candle_close_time_is_one_timeframe_after_open(self):
        signal = Signal(
            symbol="BTC/USDT", setup="rsi_reversal", side=BUY,
            entry=100.0, stop_loss=97.0, take_profit=104.0, atr=2.0,
            timeframe="15m",
            candle_time=datetime(2025, 4, 10, 14, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(
            signal.candle_close_time,
            datetime(2025, 4, 10, 14, 15, tzinfo=timezone.utc),
        )
