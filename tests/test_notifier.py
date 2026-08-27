"""Message rendering and candle-close scheduling."""

import logging
import os
import sys
import unittest
from dataclasses import replace
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The modules under test log warnings for the error paths we exercise on purpose.
logging.disable(logging.CRITICAL)

from config import Settings  # noqa: E402
from main import next_candle_close  # noqa: E402
from notifier import (  # noqa: E402
    DISCLAIMER,
    MAX_MESSAGE_CHARS,
    confidence_stars,
    format_digest,
    format_price,
    format_quantity,
    format_signal,
)
from strategies import BUY, SELL, Signal  # noqa: E402

SETTINGS = Settings()


def make_signal(**overrides) -> Signal:
    defaults = dict(
        symbol="BTC/USDT", setup="rsi_reversal", side=BUY,
        entry=65000.0, stop_loss=64200.0, take_profit=66600.0, atr=533.21,
        timeframe="15m",
        candle_time=datetime(2025, 4, 10, 14, 0, tzinfo=timezone.utc),
        confidence=2, confirmations=["Volume spike", "Trend aligned (EMA200)"],
        position_size=0.0125, position_notional=812.5, risk_amount=10.0,
    )
    defaults.update(overrides)
    return Signal(**defaults)


class TestFormatting(unittest.TestCase):
    def test_price_decimals_scale_with_magnitude(self):
        self.assertEqual(format_price(65000.0), "65,000.00")
        self.assertEqual(format_price(0.00001234), "0.00001234")
        self.assertEqual(format_price(2.5), "2.5000")

    def test_quantity_keeps_small_sizes_readable(self):
        self.assertEqual(format_quantity(0.0125), "0.0125")
        self.assertEqual(format_quantity(1234.5), "1,234.50")

    def test_confidence_stars(self):
        self.assertEqual(confidence_stars(2), "★★☆ (2/3)")
        self.assertEqual(confidence_stars(3), "★★★ (3/3)")
        self.assertEqual(confidence_stars(0), "☆☆☆ (0/3)")

    def test_buy_message_contains_every_required_field(self):
        text = format_signal(make_signal(), SETTINGS)
        for fragment in (
            "BUY Signal", "BTC/USDT", "RSI Reversal", "65,000.00",
            "64,200.00", "66,600.00", "15m", "2025-04-10 14:15 UTC",
            "★★☆ (2/3)", "Volume spike",
        ):
            self.assertIn(fragment, text, fragment)

    def test_sell_message_says_sell(self):
        text = format_signal(make_signal(side=SELL, stop_loss=65800.0,
                                         take_profit=63900.0), SETTINGS)
        self.assertIn("SELL Signal", text)
        self.assertNotIn("BUY Signal", text)

    def test_stop_and_target_percentages_have_the_right_sign(self):
        text = format_signal(make_signal(), SETTINGS)
        self.assertIn("(-1.23%)", text)  # stop below entry
        self.assertIn("(+2.46%)", text)  # target above entry

    def test_position_size_line_is_present(self):
        self.assertIn("Suggested size:", format_signal(make_signal(), SETTINGS))

    def test_position_size_line_omitted_when_not_computed(self):
        signal = make_signal(position_size=None, position_notional=None,
                             risk_amount=None)
        self.assertNotIn("Suggested size:", format_signal(signal, SETTINGS))

    def test_leverage_warning_when_notional_exceeds_capital(self):
        signal = make_signal(position_notional=4000.0)
        text = format_signal(signal, replace(SETTINGS, capital=1000.0))
        self.assertIn("needs leverage", text)

    def test_no_leverage_warning_within_capital(self):
        text = format_signal(make_signal(), replace(SETTINGS, capital=1000.0))
        self.assertNotIn("needs leverage", text)

    def test_html_in_a_symbol_is_escaped(self):
        text = format_signal(make_signal(symbol="<b>/USDT"), SETTINGS)
        self.assertIn("&lt;b&gt;/USDT", text)
        self.assertNotIn("<b>/USDT", text)


class TestDigest(unittest.TestCase):
    def test_empty_input_produces_no_messages(self):
        self.assertEqual(format_digest([], SETTINGS), [])

    def test_single_signal_has_no_count_header(self):
        messages = format_digest([make_signal()], SETTINGS)
        self.assertEqual(len(messages), 1)
        self.assertNotIn("new signals", messages[0])
        self.assertTrue(messages[0].endswith(DISCLAIMER))

    def test_several_signals_share_one_message_with_a_header(self):
        signals = [make_signal(symbol=s) for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT")]
        messages = format_digest(signals, SETTINGS)
        self.assertEqual(len(messages), 1)
        self.assertIn("3 new signals", messages[0])
        for symbol in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
            self.assertIn(symbol, messages[0])

    def test_large_batches_split_below_the_telegram_limit(self):
        signals = [make_signal(symbol=f"COIN{i}/USDT") for i in range(40)]
        messages = format_digest(signals, SETTINGS)
        self.assertGreater(len(messages), 1)
        for message in messages:
            self.assertLessEqual(len(message), MAX_MESSAGE_CHARS + len(DISCLAIMER) + 200)
        # Nothing may be dropped in the split.
        joined = "".join(messages)
        for i in range(40):
            self.assertIn(f"COIN{i}/USDT", joined)

    def test_every_message_carries_the_disclaimer(self):
        signals = [make_signal(symbol=f"COIN{i}/USDT") for i in range(40)]
        for message in format_digest(signals, SETTINGS):
            self.assertTrue(message.endswith(DISCLAIMER))


class TestScheduling(unittest.TestCase):
    def test_wakes_just_after_the_next_quarter_hour(self):
        now = datetime(2025, 4, 10, 14, 3, 30, tzinfo=timezone.utc)
        self.assertEqual(
            next_candle_close(now, 15, 15),
            datetime(2025, 4, 10, 14, 15, 15, tzinfo=timezone.utc),
        )

    def test_exactly_on_a_boundary_targets_that_boundary_plus_buffer(self):
        now = datetime(2025, 4, 10, 14, 15, 0, tzinfo=timezone.utc)
        self.assertEqual(
            next_candle_close(now, 15, 15),
            datetime(2025, 4, 10, 14, 30, 15, tzinfo=timezone.utc),
        )

    def test_inside_the_buffer_window_does_not_return_the_past(self):
        now = datetime(2025, 4, 10, 14, 15, 10, tzinfo=timezone.utc)
        target = next_candle_close(now, 15, 15)
        self.assertGreater(target, now)

    def test_last_candle_of_the_hour_rolls_over(self):
        now = datetime(2025, 4, 10, 14, 47, 0, tzinfo=timezone.utc)
        self.assertEqual(
            next_candle_close(now, 15, 15),
            datetime(2025, 4, 10, 15, 0, 15, tzinfo=timezone.utc),
        )

    def test_last_candle_of_the_day_rolls_over(self):
        now = datetime(2025, 4, 10, 23, 52, 0, tzinfo=timezone.utc)
        self.assertEqual(
            next_candle_close(now, 15, 15),
            datetime(2025, 4, 11, 0, 0, 15, tzinfo=timezone.utc),
        )

    def test_other_intervals_are_supported(self):
        now = datetime(2025, 4, 10, 14, 3, 0, tzinfo=timezone.utc)
        self.assertEqual(
            next_candle_close(now, 5, 0),
            datetime(2025, 4, 10, 14, 5, 0, tzinfo=timezone.utc),
        )


if __name__ == "__main__":
    unittest.main()
