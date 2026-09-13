"""MetaTrader 5 execution: sizing, safety guards, symbol resolution, ordering.

The real ``MetaTrader5`` package is Windows-only and drives a GUI terminal, so
none of it can run in CI. Every test here drives ``MT5Broker`` against
``FakeMT5``, a stand-in that mimics the parts of the API the broker touches -
including the retcodes and the "returns None on failure" habit that make the
real thing awkward.
"""

import logging
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The broker logs errors on the refusal paths we exercise on purpose.
logging.disable(logging.CRITICAL)

from broker_mt5 import (  # noqa: E402
    MT5Broker,
    MT5Error,
    load_executions,
    parse_symbol_map,
    round_to_step,
    save_executions,
    symbol_candidates,
)
from config import ConfigError, Settings  # noqa: E402
from strategies import BUY, SELL, Signal  # noqa: E402

CANDLE = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


def make_signal(side=BUY, **kw):
    base = dict(
        symbol="BTC/USDT", setup="rsi_reversal", side=side,
        entry=100.0, stop_loss=98.0, take_profit=104.0, atr=2.0,
        timeframe="15m", candle_time=CANDLE, confidence=3,
    )
    base.update(kw)
    return Signal(**base)


# --- the fake terminal ----------------------------------------------------


@dataclass
class FakeSymbolInfo:
    name: str = "BTCUSD"
    visible: bool = True
    digits: int = 2
    point: float = 0.01
    trade_tick_size: float = 0.01
    trade_tick_value: float = 0.1       # 1 lot loses 10.00 per 1.00 of price
    volume_min: float = 0.01
    volume_max: float = 50.0
    volume_step: float = 0.01
    filling_mode: int = 1               # FOK


@dataclass
class FakeTick:
    bid: float = 99.9
    ask: float = 100.1


@dataclass
class FakeAccount:
    login: int = 123456
    server: str = "Demo-Server"
    currency: str = "USD"
    balance: float = 10_000.0
    equity: float = 10_000.0
    margin_free: float = 9_000.0
    trade_mode: int = 0                 # ACCOUNT_TRADE_MODE_DEMO
    trade_allowed: bool = True


@dataclass
class FakePosition:
    ticket: int = 555
    symbol: str = "BTCUSD"
    volume: float = 0.1
    type: int = 0                       # POSITION_TYPE_BUY
    magic: int = 907001


@dataclass
class FakeResult:
    retcode: int = 10009
    order: int = 777
    price: float = 100.1
    volume: float = 0.0
    comment: str = ""


class FakeMT5:
    """Just enough MetaTrader5 surface for MT5Broker."""

    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    POSITION_TYPE_BUY = 0
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    SYMBOL_FILLING_FOK = 1
    SYMBOL_FILLING_IOC = 2
    TRADE_RETCODE_DONE = 10009
    ACCOUNT_TRADE_MODE_DEMO = 0

    def __init__(self, symbols=("BTCUSD",), account=None, result=None):
        self.symbols: Dict[str, FakeSymbolInfo] = {
            name: FakeSymbolInfo(name=name) for name in symbols
        }
        self.account = account if account is not None else FakeAccount()
        self.result = result if result is not None else FakeResult()
        self.positions: List[FakePosition] = []
        self.requests: List[dict] = []
        self.initialized = False
        self.init_ok = True
        self.shutdowns = 0

    # connection
    def initialize(self, **kwargs):
        self.init_kwargs = kwargs
        self.initialized = self.init_ok
        return self.init_ok

    def shutdown(self):
        self.shutdowns += 1
        self.initialized = False

    def last_error(self):
        return (-1, "fake error")

    def account_info(self):
        return self.account

    # symbols
    def symbol_info(self, name):
        return self.symbols.get(name)

    def symbol_info_tick(self, name):
        return FakeTick() if name in self.symbols else None

    def symbol_select(self, name, enable=True):
        info = self.symbols.get(name)
        if info is None:
            return False
        info.visible = enable
        return True

    # trading
    def positions_get(self, **_kw):
        return list(self.positions)

    def order_send(self, request):
        self.requests.append(request)
        return self.result


def broker_for(mt5=None, **overrides):
    settings = replace(Settings(), mt5_enabled=True, **overrides)
    return MT5Broker(settings, mt5_module=mt5 or FakeMT5()), settings


# --- pure helpers ---------------------------------------------------------


class SymbolMappingTests(unittest.TestCase):
    def test_parse_symbol_map_reads_pairs(self):
        self.assertEqual(
            parse_symbol_map("BTC/USDT=BTCUSD, ETH/USDT=ETHUSD.r"),
            {"BTC/USDT": "BTCUSD", "ETH/USDT": "ETHUSD.r"},
        )

    def test_parse_symbol_map_skips_malformed_entries(self):
        # One typo must cost you one pair, not the whole configuration.
        self.assertEqual(parse_symbol_map("BTC/USDT=BTCUSD,garbage,=X,Y="),
                         {"BTC/USDT": "BTCUSD"})

    def test_parse_symbol_map_handles_blank(self):
        self.assertEqual(parse_symbol_map(""), {})

    def test_candidates_prefer_exact_then_usd(self):
        self.assertEqual(symbol_candidates("BTC/USDT"), ["BTCUSDT", "BTCUSD", "BTC"])

    def test_candidates_apply_broker_suffix_first(self):
        self.assertEqual(
            symbol_candidates("ETH/USDT", suffix=".r")[:3],
            ["ETHUSDT.r", "ETHUSDT", "ETHUSD.r"],
        )

    def test_candidates_strip_ccxt_settlement_suffix(self):
        self.assertIn("BTCUSDT", symbol_candidates("BTC/USDT:USDT"))


class RoundingTests(unittest.TestCase):
    def test_rounds_down_never_up(self):
        # Rounding up would risk more than the caller budgeted for.
        self.assertEqual(round_to_step(0.079, 0.01), 0.07)
        self.assertEqual(round_to_step(0.6999, 0.1), 0.6)

    def test_exact_multiples_survive_float_noise(self):
        self.assertEqual(round_to_step(0.3, 0.1), 0.3)
        self.assertEqual(round_to_step(0.07, 0.01), 0.07)

    def test_zero_step_is_a_passthrough(self):
        self.assertEqual(round_to_step(1.234, 0.0), 1.234)


# --- connection guards ----------------------------------------------------


class ConnectionTests(unittest.TestCase):
    def test_live_account_is_refused_without_allow_live(self):
        account = FakeAccount(trade_mode=1)          # not DEMO
        broker, _ = broker_for(FakeMT5(account=account))
        with self.assertRaises(MT5Error) as ctx:
            broker.connect()
        self.assertIn("MT5_ALLOW_LIVE", str(ctx.exception))

    def test_live_account_is_allowed_when_opted_in(self):
        account = FakeAccount(trade_mode=1)
        broker, _ = broker_for(FakeMT5(account=account), mt5_allow_live=True)
        broker.connect()
        self.assertTrue(broker._connected)

    def test_refused_live_account_does_not_leak_the_connection(self):
        mt5 = FakeMT5(account=FakeAccount(trade_mode=1))
        broker, _ = broker_for(mt5)
        with self.assertRaises(MT5Error):
            broker.connect()
        self.assertEqual(mt5.shutdowns, 1)

    def test_failed_initialize_raises_with_guidance(self):
        mt5 = FakeMT5()
        mt5.init_ok = False
        broker, _ = broker_for(mt5)
        with self.assertRaises(MT5Error) as ctx:
            broker.connect()
        self.assertIn("initialize", str(ctx.exception))

    def test_trading_disabled_account_is_refused(self):
        mt5 = FakeMT5(account=FakeAccount(trade_allowed=False))
        broker, _ = broker_for(mt5)
        with self.assertRaises(MT5Error) as ctx:
            broker.connect()
        self.assertIn("Algo Trading", str(ctx.exception))

    def test_credentials_are_passed_through_when_set(self):
        mt5 = FakeMT5()
        broker, _ = broker_for(mt5, mt5_login=42, mt5_password="pw",
                               mt5_server="Broker-Demo")
        broker.connect()
        self.assertEqual(mt5.init_kwargs["login"], 42)
        self.assertEqual(mt5.init_kwargs["server"], "Broker-Demo")

    def test_connect_is_idempotent(self):
        mt5 = FakeMT5()
        broker, _ = broker_for(mt5)
        broker.connect()
        broker.connect()
        self.assertTrue(broker._connected)

    def test_context_manager_shuts_down(self):
        mt5 = FakeMT5()
        broker, _ = broker_for(mt5)
        with broker:
            pass
        self.assertEqual(mt5.shutdowns, 1)


# --- symbol resolution ----------------------------------------------------


class ResolutionTests(unittest.TestCase):
    def test_falls_back_from_usdt_to_usd(self):
        broker, _ = broker_for(FakeMT5(symbols=("BTCUSD",)))
        self.assertEqual(broker.resolve_symbol("BTC/USDT"), "BTCUSD")

    def test_explicit_map_wins_over_guessing(self):
        mt5 = FakeMT5(symbols=("BTCUSDT", "BTC.CFD"))
        broker, _ = broker_for(mt5, mt5_symbol_map={"BTC/USDT": "BTC.CFD"})
        self.assertEqual(broker.resolve_symbol("BTC/USDT"), "BTC.CFD")

    def test_unmapped_symbol_returns_none_rather_than_raising(self):
        broker, _ = broker_for(FakeMT5(symbols=("EURUSD",)))
        self.assertIsNone(broker.resolve_symbol("BTC/USDT"))

    def test_hidden_symbol_is_selected_into_market_watch(self):
        mt5 = FakeMT5(symbols=("BTCUSD",))
        mt5.symbols["BTCUSD"].visible = False
        broker, _ = broker_for(mt5)
        self.assertEqual(broker.resolve_symbol("BTC/USDT"), "BTCUSD")
        self.assertTrue(mt5.symbols["BTCUSD"].visible)

    def test_misses_are_cached_too(self):
        mt5 = FakeMT5(symbols=("EURUSD",))
        broker, _ = broker_for(mt5)
        broker.resolve_symbol("BTC/USDT")
        mt5.symbols["BTCUSD"] = FakeSymbolInfo()     # appears after the first miss
        self.assertIsNone(broker.resolve_symbol("BTC/USDT"))


# --- position sizing ------------------------------------------------------


class SizingTests(unittest.TestCase):
    def setUp(self):
        self.broker, _ = broker_for(FakeMT5())
        self.info = FakeSymbolInfo()

    def test_risk_budget_drives_the_lot_size(self):
        # 1% of 10,000 = 100 to risk. The fixture loses 10.00 per lot per
        # 1.00 of price (0.1 tick value / 0.01 tick size), so a 2.00 stop costs
        # 20.00 per lot => 100 / 20 = 5 lots.
        self.assertAlmostEqual(
            self.broker.lots_for(self.info, 100.0, 98.0, 10_000.0), 5.0
        )

    def test_wider_stop_means_smaller_size(self):
        tight = self.broker.lots_for(self.info, 100.0, 99.0, 10_000.0)
        wide = self.broker.lots_for(self.info, 100.0, 96.0, 10_000.0)
        self.assertGreater(tight, wide)

    def test_risk_is_independent_of_stop_distance(self):
        for stop in (99.0, 98.0, 95.0, 90.0):
            lots = self.broker.lots_for(self.info, 100.0, stop, 10_000.0)
            risked = lots * abs(100.0 - stop) * (
                self.info.trade_tick_value / self.info.trade_tick_size
            )
            self.assertAlmostEqual(risked, 100.0, places=6)

    def test_tick_value_is_respected_not_assumed(self):
        # A contract worth 10x as much per point must be sized 10x smaller.
        fat = replace(self.info, trade_tick_value=1.0)
        self.assertAlmostEqual(
            self.broker.lots_for(fat, 100.0, 98.0, 10_000.0), 0.5
        )

    def test_zero_stop_distance_is_refused(self):
        self.assertEqual(self.broker.lots_for(self.info, 100.0, 100.0, 10_000.0), 0.0)

    def test_unusable_broker_metadata_is_refused(self):
        broken = replace(self.info, trade_tick_size=0.0, point=0.0, trade_tick_value=0.0)
        self.assertEqual(self.broker.lots_for(broken, 100.0, 98.0, 10_000.0), 0.0)

    def test_trade_is_skipped_when_the_minimum_lot_overshoots_the_budget(self):
        # A tiny account cannot afford even 0.01 lots here, so it trades nothing
        # rather than silently risking more than RISK_PERCENT.
        self.assertEqual(self.broker.lots_for(self.info, 100.0, 98.0, 1.0), 0.0)

    def test_max_lot_cap_is_applied(self):
        broker, _ = broker_for(FakeMT5(), mt5_max_lot=0.1)
        self.assertAlmostEqual(broker.lots_for(self.info, 100.0, 98.0, 10_000.0), 0.1)

    def test_result_is_a_whole_number_of_volume_steps(self):
        # 100 budget / 30.00 per lot = 3.333 lots, which a 0.1 step broker
        # can only fill as 3.3. Compared via the step count rather than a float
        # modulo, which is itself inexact.
        chunky = replace(self.info, volume_step=0.1, volume_min=0.1)
        lots = self.broker.lots_for(chunky, 100.0, 97.0, 10_000.0)
        self.assertAlmostEqual(lots, 3.3)
        self.assertAlmostEqual(lots, round(lots / 0.1) * 0.1)


# --- order placement ------------------------------------------------------


class PlacementTests(unittest.TestCase):
    def test_buy_crosses_the_spread_at_the_ask(self):
        mt5 = FakeMT5()
        broker, _ = broker_for(mt5)
        broker.place(make_signal(BUY))
        self.assertEqual(mt5.requests[0]["price"], 100.1)
        self.assertEqual(mt5.requests[0]["type"], FakeMT5.ORDER_TYPE_BUY)

    def test_sell_crosses_the_spread_at_the_bid(self):
        mt5 = FakeMT5()
        broker, _ = broker_for(mt5)
        broker.place(make_signal(SELL, stop_loss=102.0, take_profit=96.0))
        self.assertEqual(mt5.requests[0]["price"], 99.9)
        self.assertEqual(mt5.requests[0]["type"], FakeMT5.ORDER_TYPE_SELL)

    def test_stops_keep_their_distance_from_the_real_fill(self):
        # The signal was computed on Binance at 100.00; the broker fills at
        # 100.10. The stop must sit 2.00 below the *fill*, not below 100.00,
        # or the risk budget is wrong by the size of the spread.
        mt5 = FakeMT5()
        broker, _ = broker_for(mt5)
        broker.place(make_signal(BUY))
        request = mt5.requests[0]
        self.assertAlmostEqual(request["price"] - request["sl"], 2.0, places=6)
        self.assertAlmostEqual(request["tp"] - request["price"], 4.0, places=6)

    def test_sell_stops_sit_above_the_fill(self):
        mt5 = FakeMT5()
        broker, _ = broker_for(mt5)
        broker.place(make_signal(SELL, stop_loss=102.0, take_profit=96.0))
        request = mt5.requests[0]
        self.assertGreater(request["sl"], request["price"])
        self.assertLess(request["tp"], request["price"])

    def test_order_carries_the_magic_number(self):
        mt5 = FakeMT5()
        broker, settings = broker_for(mt5)
        broker.place(make_signal())
        self.assertEqual(mt5.requests[0]["magic"], settings.mt5_magic)

    def test_comment_is_truncated_to_the_mt5_limit(self):
        mt5 = FakeMT5()
        broker, _ = broker_for(mt5)
        broker.place(make_signal(setup="x" * 60))
        self.assertLessEqual(len(mt5.requests[0]["comment"]), 31)

    def test_successful_order_reports_the_ticket(self):
        broker, _ = broker_for(FakeMT5())
        execution = broker.place(make_signal())
        self.assertTrue(execution.ok)
        self.assertEqual(execution.ticket, 777)

    def test_rejected_order_comes_back_not_ok_instead_of_raising(self):
        mt5 = FakeMT5(result=FakeResult(retcode=10030, comment="Unsupported filling"))
        broker, _ = broker_for(mt5)
        execution = broker.place(make_signal())
        self.assertFalse(execution.ok)
        self.assertEqual(execution.retcode, 10030)

    def test_order_send_returning_none_raises(self):
        mt5 = FakeMT5()
        mt5.order_send = lambda request: None
        broker, _ = broker_for(mt5)
        with self.assertRaises(MT5Error):
            broker.place(make_signal())

    def test_dry_run_never_sends_an_order(self):
        mt5 = FakeMT5()
        settings = replace(Settings(), mt5_enabled=True)
        broker = MT5Broker(settings, dry_run=True, mt5_module=mt5)
        execution = broker.place(make_signal())
        self.assertEqual(mt5.requests, [])
        self.assertEqual(execution.comment, "dry-run")

    def test_mt5_dry_run_setting_also_suppresses_orders(self):
        mt5 = FakeMT5()
        broker, _ = broker_for(mt5, mt5_dry_run=True)
        broker.place(make_signal())
        self.assertEqual(mt5.requests, [])

    def test_unmapped_pair_is_skipped_quietly(self):
        mt5 = FakeMT5(symbols=("EURUSD",))
        broker, _ = broker_for(mt5)
        self.assertIsNone(broker.place(make_signal()))
        self.assertEqual(mt5.requests, [])

    def test_open_position_cap_blocks_further_orders(self):
        mt5 = FakeMT5(symbols=("BTCUSD", "ETHUSD"))
        mt5.positions = [FakePosition(symbol="ETHUSD"), FakePosition(symbol="XAUUSD")]
        broker, _ = broker_for(mt5, mt5_max_open_positions=2)
        self.assertIsNone(broker.place(make_signal()))
        self.assertEqual(mt5.requests, [])

    def test_positions_from_other_bots_do_not_count_towards_the_cap(self):
        mt5 = FakeMT5()
        mt5.positions = [FakePosition(symbol="ETHUSD", magic=1),
                         FakePosition(symbol="XAUUSD", magic=2)]
        broker, _ = broker_for(mt5, mt5_max_open_positions=2)
        self.assertIsNotNone(broker.place(make_signal()))

    def test_a_second_position_on_the_same_symbol_is_refused(self):
        mt5 = FakeMT5()
        mt5.positions = [FakePosition(symbol="BTCUSD")]
        broker, _ = broker_for(mt5, mt5_max_open_positions=0)
        self.assertIsNone(broker.place(make_signal()))

    def test_daily_order_cap_stops_the_run(self):
        mt5 = FakeMT5()
        broker, _ = broker_for(mt5, mt5_max_orders_per_day=2)
        for _ in range(4):
            broker.place(make_signal())
        self.assertEqual(len(mt5.requests), 2)

    def test_filling_mode_follows_the_symbol_mask(self):
        mt5 = FakeMT5()
        mt5.symbols["BTCUSD"].filling_mode = FakeMT5.SYMBOL_FILLING_IOC
        broker, _ = broker_for(mt5)
        broker.place(make_signal())
        self.assertEqual(mt5.requests[0]["type_filling"], FakeMT5.ORDER_FILLING_IOC)

    def test_configured_filling_mode_overrides_the_mask(self):
        mt5 = FakeMT5()
        broker, _ = broker_for(mt5, mt5_filling_mode="RETURN")
        broker.place(make_signal())
        self.assertEqual(mt5.requests[0]["type_filling"], FakeMT5.ORDER_FILLING_RETURN)

    def test_missing_quote_skips_rather_than_sending_a_blind_order(self):
        mt5 = FakeMT5()
        mt5.symbol_info_tick = lambda name: None
        broker, _ = broker_for(mt5)
        self.assertIsNone(broker.place(make_signal()))


class ClosingTests(unittest.TestCase):
    def test_closing_a_long_sells_at_the_bid(self):
        mt5 = FakeMT5()
        broker, _ = broker_for(mt5)
        broker.close(FakePosition(type=FakeMT5.POSITION_TYPE_BUY))
        self.assertEqual(mt5.requests[0]["type"], FakeMT5.ORDER_TYPE_SELL)
        self.assertEqual(mt5.requests[0]["price"], 99.9)

    def test_closing_a_short_buys_at_the_ask(self):
        mt5 = FakeMT5()
        broker, _ = broker_for(mt5)
        broker.close(FakePosition(type=1))
        self.assertEqual(mt5.requests[0]["type"], FakeMT5.ORDER_TYPE_BUY)
        self.assertEqual(mt5.requests[0]["price"], 100.1)

    def test_close_references_the_position_ticket(self):
        mt5 = FakeMT5()
        broker, _ = broker_for(mt5)
        broker.close(FakePosition(ticket=4242))
        self.assertEqual(mt5.requests[0]["position"], 4242)

    def test_open_positions_are_filtered_by_magic(self):
        mt5 = FakeMT5()
        mt5.positions = [FakePosition(magic=907001), FakePosition(magic=999)]
        broker, _ = broker_for(mt5)
        self.assertEqual(len(broker.open_positions()), 1)


class AccountSummaryTests(unittest.TestCase):
    def test_summary_labels_a_demo_account(self):
        broker, _ = broker_for(FakeMT5())
        self.assertEqual(broker.account_summary()["mode"], "DEMO")

    def test_summary_labels_a_live_account(self):
        mt5 = FakeMT5(account=FakeAccount(trade_mode=1))
        broker, _ = broker_for(mt5, mt5_allow_live=True)
        self.assertEqual(broker.account_summary()["mode"], "LIVE")


# --- execution log --------------------------------------------------------


class ExecutionLogTests(unittest.TestCase):
    def test_round_trips_through_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "executions.json")
            save_executions(path, [{"symbol": "BTCUSD", "ticket": 1}])
            self.assertEqual(load_executions(path)[0]["ticket"], 1)

    def test_missing_file_reads_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_executions(os.path.join(tmp, "nope.json")), [])

    def test_corrupt_file_reads_as_empty_rather_than_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "executions.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not json")
            self.assertEqual(load_executions(path), [])

    def test_log_is_trimmed_so_it_cannot_grow_without_bound(self):
        from broker_mt5 import MAX_EXECUTIONS_KEPT

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "executions.json")
            save_executions(path, [{"n": i} for i in range(MAX_EXECUTIONS_KEPT + 50)])
            kept = load_executions(path)
            self.assertEqual(len(kept), MAX_EXECUTIONS_KEPT)
            # The newest entries are the ones worth keeping.
            self.assertEqual(kept[-1]["n"], MAX_EXECUTIONS_KEPT + 49)


# --- configuration --------------------------------------------------------


class SettingsTests(unittest.TestCase):
    def test_execution_is_off_by_default(self):
        self.assertFalse(Settings().mt5_enabled)
        self.assertFalse(Settings().mt5_allow_live)

    def test_partial_credentials_are_rejected(self):
        settings = replace(Settings(), mt5_enabled=True, mt5_login=1)
        with self.assertRaises(ConfigError) as ctx:
            settings.require_mt5()
        self.assertIn("MT5_PASSWORD", str(ctx.exception))

    def test_no_credentials_means_use_the_logged_in_terminal(self):
        replace(Settings(), mt5_enabled=True).require_mt5()   # must not raise

    def test_full_credentials_pass(self):
        replace(Settings(), mt5_enabled=True, mt5_login=1,
                mt5_password="pw", mt5_server="s").require_mt5()

    def test_unreachable_confidence_threshold_is_rejected(self):
        settings = replace(Settings(), mt5_enabled=True, mt5_min_confidence=4)
        with self.assertRaises(ConfigError):
            settings.require_mt5()

    def test_disabled_execution_skips_validation_entirely(self):
        replace(Settings(), mt5_login=1).require_mt5()         # must not raise



class DryRunAccountingTests(unittest.TestCase):
    """A dry run should predict the live run, including how much it stops short."""

    def test_dry_run_orders_count_against_the_daily_cap(self):
        mt5 = FakeMT5()
        settings = replace(Settings(), mt5_enabled=True, mt5_max_orders_per_day=2)
        broker = MT5Broker(settings, dry_run=True, mt5_module=mt5)
        results = [broker.place(make_signal()) for _ in range(4)]
        self.assertEqual(sum(r is not None for r in results), 2)


if __name__ == "__main__":
    unittest.main()
