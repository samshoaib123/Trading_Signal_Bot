"""Preflight self-check behaviour, including its failure reporting."""

import logging
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The modules under test log warnings for the error paths we exercise on purpose.
logging.disable(logging.CRITICAL)

import ccxt  # noqa: E402

from config import Settings  # noqa: E402
from preflight import FAIL, PASS, WARN, _config_check, _state_check, run_preflight  # noqa: E402
from test_pipeline import FakeExchange, RecordingNotifier, bounce_series  # noqa: E402

GOOD = dict(telegram_bot_token="123456789:AAEabcdef", telegram_chat_id="987654321")


class TestConfigCheck(unittest.TestCase):
    def test_missing_credentials_fail(self):
        result = _config_check(Settings())
        self.assertEqual(result.status, FAIL)
        self.assertIn("TELEGRAM_BOT_TOKEN", result.detail)
        self.assertTrue(result.fix)

    def test_malformed_token_fails(self):
        result = _config_check(replace(Settings(), **{**GOOD, "telegram_bot_token": "nope"}))
        self.assertEqual(result.status, FAIL)
        self.assertIn("BotFather", result.fix)

    def test_non_numeric_chat_id_warns_but_does_not_block(self):
        result = _config_check(replace(Settings(), **{**GOOD, "telegram_chat_id": "me"}))
        self.assertEqual(result.status, WARN)
        self.assertTrue(result.ok)

    def test_good_config_passes(self):
        result = _config_check(replace(Settings(), **GOOD))
        self.assertEqual(result.status, PASS)


class TestStateCheck(unittest.TestCase):
    def test_writable_directory_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = replace(Settings(), state_file=os.path.join(tmp, "state.json"))
            self.assertEqual(_state_check(settings).status, PASS)

    def test_unwritable_path_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = os.path.join(tmp, "blocker")
            open(blocker, "w").close()
            settings = replace(Settings(), state_file=os.path.join(blocker, "state.json"))
            result = _state_check(settings)
            self.assertEqual(result.status, FAIL)
            self.assertIn("STATE_FILE", result.fix)


class TestFullRun(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.settings = replace(
            Settings(), **GOOD,
            symbols=["BTC/USDT"],
            state_file=os.path.join(self.tmpdir.name, "state.json"),
            fetch_backoff_seconds=0.0,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_everything_healthy_exits_zero(self):
        exchange = FakeExchange({"BTC/USDT": bounce_series()})
        with mock.patch("exchange.create_exchange", return_value=exchange):
            code = run_preflight(self.settings, RecordingNotifier())
        self.assertEqual(code, 0)

    def test_unreachable_exchange_is_reported_not_masked(self):
        """load_valid_symbols degrades gracefully; preflight must not inherit that."""

        class Unreachable(FakeExchange):
            def load_markets(self):
                raise ccxt.NetworkError("dns failure")

        with mock.patch("exchange.create_exchange",
                        return_value=Unreachable({"BTC/USDT": bounce_series()})):
            code = run_preflight(self.settings, RecordingNotifier())
        self.assertEqual(code, 1)

    def test_telegram_failure_exits_nonzero(self):
        exchange = FakeExchange({"BTC/USDT": bounce_series()})
        with mock.patch("exchange.create_exchange", return_value=exchange):
            code = run_preflight(self.settings, RecordingNotifier(fail=True))
        self.assertEqual(code, 1)

    def test_missing_credentials_skip_the_telegram_call(self):
        exchange = FakeExchange({"BTC/USDT": bounce_series()})
        notifier = RecordingNotifier()
        with mock.patch("exchange.create_exchange", return_value=exchange):
            code = run_preflight(replace(self.settings, telegram_bot_token=""), notifier)
        self.assertEqual(code, 1)
        self.assertEqual(notifier.messages, [])


if __name__ == "__main__":
    unittest.main()
