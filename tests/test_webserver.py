"""Dashboard API: every endpoint answers, with the shape the page expects."""

import logging
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The modules under test log warnings for the error paths we exercise on purpose.
logging.disable(logging.CRITICAL)

from config import Settings  # noqa: E402
from test_pipeline import FakeExchange, bounce_series  # noqa: E402


def client_and_module(settings=None, symbols=("BTC/USDT", "ETH/USDT")):
    """A TestClient wired to a fake exchange, with temp state files."""
    from fastapi.testclient import TestClient

    import webserver

    tmp = tempfile.TemporaryDirectory()
    webserver.SETTINGS = settings or replace(
        Settings(),
        symbols=list(symbols),
        state_file=os.path.join(tmp.name, "state.json"),
        tracker_file=os.path.join(tmp.name, "outcomes.json"),
        fetch_backoff_seconds=0.0,
    )
    webserver.EXCHANGE = FakeExchange({s: bounce_series() for s in symbols})
    webserver.SYMBOLS = list(symbols)
    webserver._CACHE.update(at=0.0, rows=[], failed=[], fetched_at=None)
    # Module-level caches outlive a TestClient, so a later test would otherwise
    # be served an answer computed against a previous test's fake exchange.
    webserver._CONTEXT_CACHE.clear()

    # Skip the startup hook: it would build a real ccxt client.
    with mock.patch.object(webserver.app.router, "on_startup", []):
        client = TestClient(webserver.app)
    return client, webserver, tmp


class TestEndpoints(unittest.TestCase):
    def setUp(self):
        self.client, self.web, self.tmp = client_and_module()

    def tearDown(self):
        self.tmp.cleanup()

    def test_status_reports_the_running_configuration(self):
        body = self.client.get("/api/status").json()
        self.assertEqual(body["timeframe"], "15m")
        self.assertEqual(body["symbols"], ["BTC/USDT", "ETH/USDT"])
        self.assertIn("indicator_backend", body)
        self.assertFalse(body["bot_loop_running"])

    def test_market_returns_live_indicator_values(self):
        body = self.client.get("/api/market").json()
        self.assertEqual(len(body["rows"]), 2)
        row = body["rows"][0]
        for field in ("symbol", "price", "rsi", "macd_hist", "bb_position",
                      "atr", "atr_pct", "trend", "volume_ratio", "signals"):
            self.assertIn(field, row, field)
        self.assertIsInstance(row["signals"], list)
        self.assertGreater(row["price"], 0)

    def test_market_detects_the_same_signal_the_bot_would_send(self):
        # bounce_series ends on an RSI reversal, so the dashboard must show it.
        body = self.client.get("/api/market").json()
        fired = [s for r in body["rows"] for s in r["signals"]]
        self.assertTrue(fired, "expected the bounce to fire a setup")
        self.assertEqual(fired[0]["side"], "BUY")
        self.assertEqual(fired[0]["setup"], "rsi_reversal")
        self.assertIn(fired[0]["confidence"], (1, 2, 3))

    def test_market_is_cached_then_refreshed_on_demand(self):
        self.client.get("/api/market")
        calls_after_first = self.web.EXCHANGE.calls
        self.client.get("/api/market")
        self.assertEqual(self.web.EXCHANGE.calls, calls_after_first, "should serve cache")
        self.assertTrue(self.client.get("/api/market").json()["cached"])
        self.client.get("/api/market?refresh=true")
        self.assertGreater(self.web.EXCHANGE.calls, calls_after_first, "refresh must refetch")

    def test_bb_position_is_bounded(self):
        for row in self.client.get("/api/market").json()["rows"]:
            if row["bb_position"] is not None:
                self.assertGreaterEqual(row["bb_position"], -25)
                self.assertLessEqual(row["bb_position"], 125)

    def test_positions_and_history_start_empty(self):
        self.assertEqual(self.client.get("/api/positions").json()["count"], 0)
        board = self.client.get("/api/history").json()["scoreboard"]
        self.assertEqual((board["wins"], board["losses"], board["closed"]), (0, 0, 0))

    def test_positions_are_marked_to_the_live_price(self):
        from strategies import BUY, Signal
        from tracker import save_outcomes, track_signal
        from datetime import datetime, timezone

        ledger = {"open": [], "closed": []}
        track_signal(
            Signal(symbol="BTC/USDT", setup="rsi_reversal", side=BUY, entry=10.0,
                   stop_loss=9.0, take_profit=12.0, atr=1.0, timeframe="15m",
                   candle_time=datetime(2024, 1, 1, tzinfo=timezone.utc)),
            ledger, self.web.SETTINGS,
        )
        save_outcomes(self.web.SETTINGS.tracker_file, ledger)

        body = self.client.get("/api/positions").json()
        self.assertEqual(body["count"], 1)
        row = body["open"][0]
        self.assertIsNotNone(row["price"], "should be marked to the cached price")
        self.assertIsNotNone(row["excursion_r"])
        self.assertIsNotNone(row["live_r"])

    def test_history_scoreboard_totals(self):
        from tracker import save_outcomes

        save_outcomes(self.web.SETTINGS.tracker_file, {"open": [], "closed": [
            {"result": "win", "r_multiple": 1.27, "symbol": "BTC/USDT",
             "setup": "rsi_reversal", "side": "BUY", "entry": 100.0,
             "exit_price": 104.0, "closed_at": "2024-06-01T13:00:00+00:00"},
            {"result": "loss", "r_multiple": -1.07, "symbol": "ETH/USDT",
             "setup": "macd_crossover", "side": "BUY", "entry": 50.0,
             "exit_price": 48.0, "closed_at": "2024-06-01T14:00:00+00:00"},
        ]})
        board = self.client.get("/api/history").json()["scoreboard"]
        self.assertEqual((board["wins"], board["losses"]), (1, 1))
        self.assertAlmostEqual(board["total_r"], 0.2, places=2)
        self.assertEqual(board["win_rate"], 50.0)

    def test_history_returns_newest_first(self):
        from tracker import save_outcomes

        save_outcomes(self.web.SETTINGS.tracker_file, {"open": [], "closed": [
            {"result": "win", "r_multiple": 1.0, "closed_at": "2024-06-01T10:00:00+00:00"},
            {"result": "win", "r_multiple": 1.0, "closed_at": "2024-06-02T10:00:00+00:00"},
        ]})
        closed = self.client.get("/api/history").json()["closed"]
        self.assertEqual(closed[0]["closed_at"], "2024-06-02T10:00:00+00:00")

    def test_backtest_endpoint_returns_per_setup_stats(self):
        body = self.client.get("/api/backtest?candles=400").json()
        for key in ("by_setup", "by_symbol", "overall", "candles", "fee_percent"):
            self.assertIn(key, body)
        self.assertEqual(body["candles"], 400)
        for bucket in body["by_setup"]:
            for field in ("name", "signals", "closed", "win_rate", "expectancy",
                          "total_r", "max_consecutive_losses"):
                self.assertIn(field, bucket, field)

    def test_dashboard_page_is_served(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Sniper Board", res.text)
        # The page is only useful if it actually calls the API it renders.
        for endpoint in ("/api/status", "/api/market", "/api/positions",
                         "/api/history", "/api/account", "/api/context"):
            self.assertIn(endpoint, res.text, endpoint)

    def test_healthz(self):
        self.assertTrue(self.client.get("/healthz").json()["ok"])


class TestResilience(unittest.TestCase):
    def tearDown(self):
        self.tmp.cleanup()

    def test_one_failing_pair_does_not_blank_the_dashboard(self):
        import ccxt

        class HalfBroken(FakeExchange):
            def fetch_ohlcv(self, symbol, timeframe="15m", limit=300):
                if symbol == "ETH/USDT":
                    raise ccxt.BadSymbol("delisted")
                return super().fetch_ohlcv(symbol, timeframe, limit)

        self.client, web, self.tmp = client_and_module()
        web.EXCHANGE = HalfBroken({s: bounce_series() for s in web.SYMBOLS})
        web._CACHE.update(at=0.0, rows=[], failed=[], fetched_at=None)

        body = self.client.get("/api/market").json()
        self.assertEqual([r["symbol"] for r in body["rows"]], ["BTC/USDT"])
        self.assertIn("ETH/USDT", body["failed"])


if __name__ == "__main__":
    unittest.main()


class TestDashboardToken(unittest.TestCase):
    """DASHBOARD_TOKEN gates the page without breaking the health check."""

    def setUp(self):
        import webserver

        self._saved_token = webserver.DASHBOARD_TOKEN

    def tearDown(self):
        import webserver

        # Restore it: leaking the token would 401 every later test.
        webserver.DASHBOARD_TOKEN = self._saved_token
        self.tmp.cleanup()

    def _client(self, token):
        from fastapi.testclient import TestClient

        import webserver

        self.client, self.web, self.tmp = client_and_module()
        webserver.DASHBOARD_TOKEN = token
        return TestClient(webserver.app)

    def test_no_token_configured_means_open_access(self):
        client = self._client("")
        self.assertEqual(client.get("/api/status").status_code, 200)

    def test_wrong_key_is_rejected(self):
        client = self._client("s3cret")
        self.assertEqual(client.get("/api/status").status_code, 401)
        self.assertEqual(client.get("/api/status?key=nope").status_code, 401)

    def test_correct_key_in_query_is_accepted_and_sets_a_cookie(self):
        client = self._client("s3cret")
        res = client.get("/?key=s3cret")
        self.assertEqual(res.status_code, 200)
        self.assertIn("signalbot_key", res.cookies)
        # The cookie now carries subsequent API calls made by the page itself.
        self.assertEqual(client.get("/api/status").status_code, 200)

    def test_header_is_accepted(self):
        client = self._client("s3cret")
        res = client.get("/api/status", headers={"X-Dashboard-Key": "s3cret"})
        self.assertEqual(res.status_code, 200)

    def test_healthz_stays_open_for_platform_probes(self):
        client = self._client("s3cret")
        self.assertEqual(client.get("/healthz").status_code, 200)


class TestBoardEndpoints(unittest.TestCase):
    """The panels the board renders: account, context, and ladder progress."""

    def setUp(self):
        self.client, self.web, self.tmp = client_and_module()

    def tearDown(self):
        self.tmp.cleanup()

    # --- account ----------------------------------------------------------

    def test_account_reports_alert_only_rather_than_an_error(self):
        # Running without a broker is a supported mode, not a broken one, and
        # the board has to say so without inventing a balance.
        body = self.client.get("/api/account").json()
        self.assertFalse(body["enabled"])
        self.assertFalse(body["connected"])
        self.assertIn("reason", body)

    def test_account_reports_a_broker_it_cannot_reach(self):
        self.web.SETTINGS = replace(self.web.SETTINGS, mt5_enabled=True)
        body = self.client.get("/api/account").json()
        self.assertTrue(body["enabled"])
        self.assertFalse(body["connected"])
        self.assertIn("reason", body)

    # --- market watch -----------------------------------------------------

    def test_market_rows_carry_the_sniper_state_the_board_shows(self):
        row = self.client.get("/api/market").json()["rows"][0]
        for field in ("adx", "adx_state", "stochrsi_k", "squeeze",
                      "squeeze_released", "ribbon", "ribbon_width"):
            self.assertIn(field, row, field)

    def test_adx_state_is_a_word_the_page_can_style(self):
        row = self.client.get("/api/market").json()["rows"][0]
        self.assertIn(row["adx_state"], {"trending", "building", "flat", "unknown"})

    def test_ribbon_state_is_never_a_raw_nan(self):
        # NaN is not valid JSON and silently becomes null or breaks the parse.
        for row in self.client.get("/api/market").json()["rows"]:
            self.assertIn(row["ribbon"], {"bull", "bear", "mixed", "unknown"})

    # --- context ----------------------------------------------------------

    def test_context_answers_for_the_lead_pair_by_default(self):
        body = self.client.get("/api/context").json()
        self.assertEqual(body["symbol"], "BTC/USDT")
        self.assertEqual(len(body["timeframes"]), len(self.web.SETTINGS.context_timeframes))

    def test_context_reports_alignment_out_of_the_timeframes_it_read(self):
        body = self.client.get("/api/context").json()
        self.assertLessEqual(body["aligned"], body["total"])
        self.assertIn(body["verdict"].split()[0], {"leaning", "mixed", "unknown"})

    def test_every_context_row_has_a_bias_the_page_can_style(self):
        for row in self.client.get("/api/context").json()["timeframes"]:
            self.assertIn(row["bias"], {"bull", "bear", "mixed", "unknown"})

    def test_an_unwatched_pair_is_a_404_not_a_blank_panel(self):
        self.assertEqual(self.client.get("/api/context?symbol=DOGE/USDT").status_code, 404)

    def test_context_is_cached_between_calls(self):
        # Four fetches per pair is too many to repeat on every dashboard poll.
        first = self.client.get("/api/context").json()
        second = self.client.get("/api/context").json()
        self.assertFalse(first.get("cached", False))
        self.assertTrue(second.get("cached"))

    # --- positions --------------------------------------------------------

    def open_a_laddered_position(self):
        from datetime import datetime, timezone

        from strategies import BUY, Signal
        from tracker import load_outcomes, save_outcomes, track_signal

        signal = Signal(
            symbol="BTC/USDT", setup="trend_sniper", side=BUY, entry=100.0,
            stop_loss=98.0, take_profit=108.0, atr=2.0, timeframe="15m",
            candle_time=datetime(2024, 6, 1, tzinfo=timezone.utc), confidence=3,
            targets=[102.0, 104.0, 106.0, 108.0],
            target_fractions=[0.4, 0.3, 0.2, 0.1],
        )
        ledger = load_outcomes(self.web.SETTINGS.tracker_file)
        track_signal(signal, ledger, self.web.SETTINGS)
        save_outcomes(self.web.SETTINGS.tracker_file, ledger)

    def test_open_positions_carry_the_whole_ladder(self):
        self.open_a_laddered_position()
        position = self.client.get("/api/positions").json()["open"][0]
        self.assertEqual(position["targets_total"], 4)
        self.assertEqual(position["targets_hit"], 0)
        self.assertEqual(position["next_target"], 102.0)
        for field in ("banked_r", "peak_r", "at_breakeven", "bars_held",
                      "stop_now", "live_r", "remaining_fraction"):
            self.assertIn(field, position, field)

    def test_next_target_is_null_once_every_rung_is_reached(self):
        self.open_a_laddered_position()
        from tracker import load_outcomes, save_outcomes

        ledger = load_outcomes(self.web.SETTINGS.tracker_file)
        ledger["open"][0]["targets_hit"] = 4
        save_outcomes(self.web.SETTINGS.tracker_file, ledger)
        position = self.client.get("/api/positions").json()["open"][0]
        self.assertIsNone(position["next_target"])

    def test_a_legacy_single_target_position_still_renders(self):
        from tracker import save_outcomes

        save_outcomes(self.web.SETTINGS.tracker_file, {
            "open": [{"id": "x", "symbol": "BTC/USDT", "setup": "rsi_reversal",
                      "side": "BUY", "entry": 100.0, "stop_loss": 98.0,
                      "take_profit": 104.0, "confidence": 2,
                      "opened_at": "2024-06-01T00:00:00+00:00"}],
            "closed": [],
        })
        position = self.client.get("/api/positions").json()["open"][0]
        self.assertEqual(position["targets_total"], 1)
        self.assertEqual(position["targets"], [104.0])


class TestLivePLMaths(unittest.TestCase):
    """Live P/L has to account for what the ladder already closed."""

    def setUp(self):
        self.client, self.web, self.tmp = client_and_module()

    def tearDown(self):
        self.tmp.cleanup()

    def seed(self, targets_hit, open_r):
        from tracker import save_outcomes

        save_outcomes(self.web.SETTINGS.tracker_file, {
            "open": [{
                "id": "x", "symbol": "BTC/USDT", "setup": "trend_sniper",
                "side": "BUY", "entry": 10.0, "stop_loss": 9.0,
                "take_profit": 14.0, "targets": [11.0, 12.0, 13.0, 14.0],
                "target_fractions": [0.4, 0.3, 0.2, 0.1],
                "targets_hit": targets_hit, "open_r": open_r,
                "confidence": 3, "opened_at": "2024-01-01T00:00:00+00:00",
            }],
            "closed": [],
        })
        return self.client.get("/api/positions").json()["open"][0]

    def test_a_untouched_position_rides_its_whole_size(self):
        position = self.seed(0, 0.0)
        self.assertAlmostEqual(position["remaining_fraction"], 1.0)
        self.assertAlmostEqual(position["live_r"], position["excursion_r"])

    def test_banked_rungs_shrink_what_is_still_exposed(self):
        position = self.seed(2, 1.0)
        self.assertAlmostEqual(position["remaining_fraction"], 0.3)

    def test_live_pl_is_banked_plus_what_the_remainder_is_doing(self):
        # Marking the full size after two rungs closed would report a P/L the
        # trade can no longer make or lose.
        position = self.seed(2, 1.0)
        self.assertAlmostEqual(
            position["live_r"], 1.0 + 0.3 * position["excursion_r"], places=6
        )
        self.assertNotAlmostEqual(position["live_r"], position["excursion_r"])

    def test_a_fully_closed_ladder_has_nothing_left_riding(self):
        position = self.seed(4, 2.0)
        self.assertAlmostEqual(position["remaining_fraction"], 0.0)
        self.assertAlmostEqual(position["live_r"], 2.0)
