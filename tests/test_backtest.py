"""Backtest simulation maths, aggregation and CLI wiring."""

import logging
import os
import sys
import unittest
from datetime import datetime, timezone
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The modules under test log warnings for the error paths we exercise on purpose.
logging.disable(logging.CRITICAL)

from backtest import (  # noqa: E402
    LOSS,
    OPEN,
    WIN,
    Stats,
    Trade,
    _fee_r_cost,
    aggregate,
    backtest_symbol,
    overall,
    render_report,
    simulate_trade,
)
from config import Settings  # noqa: E402
from strategies import BUY, SELL, Signal  # noqa: E402

SETTINGS = Settings()
T0 = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


def signal(side=BUY, **kw):
    base = dict(symbol="BTC/USDT", setup="rsi_reversal", side=side, entry=100.0,
                stop_loss=97.0, take_profit=104.0, atr=2.0, timeframe="15m",
                candle_time=T0)
    if side == SELL:
        base.update(stop_loss=103.0, take_profit=96.0)
    base.update(kw)
    return Signal(**base)


def future(rows):
    idx = pd.date_range(T0, periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame([{"high": h, "low": l} for h, l in rows], index=idx)


class TestSimulate(unittest.TestCase):
    def test_target_first_is_a_win(self):
        t = simulate_trade(signal(), future([(101, 99), (105, 100)]), 0.0)
        self.assertEqual(t.outcome, WIN)
        self.assertAlmostEqual(t.r_multiple, 4 / 3, places=6)
        self.assertEqual(t.candles_held, 2)

    def test_stop_first_is_a_loss(self):
        t = simulate_trade(signal(), future([(101, 99), (101, 96)]), 0.0)
        self.assertEqual(t.outcome, LOSS)
        self.assertAlmostEqual(t.r_multiple, -1.0, places=6)

    def test_stop_wins_ties_within_one_candle(self):
        t = simulate_trade(signal(), future([(105, 96)]), 0.0)
        self.assertEqual(t.outcome, LOSS)

    def test_never_touched_stays_open(self):
        t = simulate_trade(signal(), future([(101, 99)]), 0.0)
        self.assertEqual(t.outcome, OPEN)
        self.assertEqual(t.r_multiple, 0.0)

    def test_sell_mirrors_the_directions(self):
        t = simulate_trade(signal(side=SELL), future([(99, 95)]), 0.0)
        self.assertEqual(t.outcome, WIN)

    def test_sell_stop_is_above_entry(self):
        t = simulate_trade(signal(side=SELL), future([(104, 99)]), 0.0)
        self.assertEqual(t.outcome, LOSS)

    def test_zero_risk_signal_is_not_simulated(self):
        t = simulate_trade(signal(stop_loss=100.0), future([(105, 95)]), 0.0)
        self.assertEqual(t.outcome, OPEN)


class TestFees(unittest.TestCase):
    def test_fee_cost_formula(self):
        # Two sides, on notional, expressed in units of the risk taken.
        self.assertAlmostEqual(_fee_r_cost(signal(), 0.1), 2 * 0.001 * 100 / 3, places=9)

    def test_fees_shrink_a_win(self):
        clean = simulate_trade(signal(), future([(105, 100)]), 0.0)
        charged = simulate_trade(signal(), future([(105, 100)]), 0.1)
        self.assertLess(charged.r_multiple, clean.r_multiple)

    def test_fees_deepen_a_loss(self):
        charged = simulate_trade(signal(), future([(101, 96)]), 0.1)
        self.assertLess(charged.r_multiple, -1.0)

    def test_a_tighter_stop_pays_proportionally_more_in_fees(self):
        wide = _fee_r_cost(signal(stop_loss=90.0), 0.1)
        tight = _fee_r_cost(signal(stop_loss=99.5), 0.1)
        self.assertGreater(tight, wide)


class TestStats(unittest.TestCase):
    def make(self, rs):
        s = Stats("t")
        for r in rs:
            s.add(Trade(signal(), WIN if r > 0 else LOSS, r_multiple=r))
        return s

    def test_win_rate_and_expectancy(self):
        s = self.make([1.0, 1.0, -1.0, -1.0])
        self.assertEqual(s.win_rate, 50.0)
        self.assertAlmostEqual(s.expectancy, 0.0)
        self.assertAlmostEqual(s.profit_factor, 1.0)

    def test_profit_factor_above_one_when_winning(self):
        self.assertGreater(self.make([2.0, -1.0, 2.0, -1.0]).profit_factor, 1.0)

    def test_profit_factor_with_no_losses_is_infinite(self):
        self.assertEqual(self.make([1.0, 2.0]).profit_factor, float("inf"))

    def test_max_consecutive_losses(self):
        self.assertEqual(self.make([1.0, -1, -1, -1, 1.0, -1]).max_consecutive_losses, 3)

    def test_open_trades_are_excluded_from_rates(self):
        s = Stats("t")
        s.add(Trade(signal(), WIN, r_multiple=1.0))
        s.add(Trade(signal(), OPEN))
        self.assertEqual(s.closed, 1)
        self.assertEqual(s.total, 2)
        self.assertEqual(s.win_rate, 100.0)

    def test_aggregate_buckets_by_key(self):
        trades = [
            Trade(signal(setup="rsi_reversal"), WIN, r_multiple=1.0),
            Trade(signal(setup="macd_crossover"), LOSS, r_multiple=-1.0),
        ]
        buckets = aggregate(trades, lambda t: t.setup)
        self.assertEqual(set(buckets), {"rsi_reversal", "macd_crossover"})
        self.assertEqual(overall(trades).closed, 2)


class TestReplay(unittest.TestCase):
    def frame(self, rows=600, seed=5):
        rng = np.random.default_rng(seed)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, rows)))
        span = np.abs(rng.normal(0, 0.003, rows)) + 0.001
        idx = pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC")
        return pd.DataFrame(
            {"timestamp": idx, "open": close, "high": close * (1 + span),
             "low": close * (1 - span), "close": close,
             "volume": rng.uniform(100, 900, rows)},
            index=idx,
        )

    def test_replay_produces_resolvable_trades(self):
        trades = backtest_symbol("BTC/USDT", self.frame(), SETTINGS, 0.1)
        self.assertGreater(len(trades), 0)
        self.assertTrue(all(t.outcome in (WIN, LOSS, OPEN) for t in trades))
        self.assertTrue(any(t.outcome in (WIN, LOSS) for t in trades))

    def test_replay_is_deterministic(self):
        a = backtest_symbol("BTC/USDT", self.frame(), SETTINGS, 0.1)
        b = backtest_symbol("BTC/USDT", self.frame(), SETTINGS, 0.1)
        self.assertEqual([t.outcome for t in a], [t.outcome for t in b])

    def test_cooldown_limits_repeat_signals(self):
        from dataclasses import replace

        frame = self.frame()
        few = backtest_symbol("BTC/USDT", frame,
                              replace(SETTINGS, signal_cooldown_minutes=600), 0.1)
        many = backtest_symbol("BTC/USDT", frame,
                               replace(SETTINGS, signal_cooldown_minutes=0), 0.1)
        self.assertLess(len(few), len(many))

    def test_too_little_history_returns_nothing(self):
        self.assertEqual(backtest_symbol("BTC/USDT", self.frame(rows=40), SETTINGS, 0.1), [])

    def test_report_renders_with_trades(self):
        trades = backtest_symbol("BTC/USDT", self.frame(), SETTINGS, 0.1)
        text = render_report(trades, SETTINGS, 0.1, 600)
        for fragment in ("BACKTEST REPORT", "WIN%", "HOW TO READ THIS", "CAVEATS"):
            self.assertIn(fragment, text)

    def test_report_handles_no_signals_without_pretending(self):
        text = render_report([], SETTINGS, 0.1, 600)
        self.assertIn("No signals fired", text)
        self.assertNotIn("WIN%", text)


class TestCliWiring(unittest.TestCase):
    """Every flag must reach its own code — this catches use-before-assignment."""

    def test_report_flag_does_not_crash_before_the_notifier_exists(self):
        import main

        sent = []

        class Fake:
            def __init__(self, *a, **k): pass
            def send(self, text):
                sent.append(text)
                return True

        with mock.patch.object(main, "TelegramNotifier", Fake), \
             mock.patch.object(main.Settings, "require_telegram", lambda self: None):
            code = main.main(["--report"])

        self.assertEqual(code, 0)
        self.assertEqual(len(sent), 1)
        self.assertIn("Scoreboard", sent[0])


if __name__ == "__main__":
    unittest.main()
