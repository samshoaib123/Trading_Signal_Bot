"""Historical replay of the same setups the live bot fires on.

The point of this module is honesty. The three setups are popular, not proven,
and the only way to know whether they are worth following is to measure them on
real candles. ``python main.py --backtest`` replays history exactly as the bot
would have seen it and reports win rate, expectancy and profit factor per setup.

Method
------
* Indicators are computed once over the whole frame. Every indicator here is
  causal - RSI, MACD, Bollinger, ATR, EMA and SMA all look backwards only - so
  evaluating candle *i* against a frame that also contains later candles cannot
  leak the future. Detection then walks forward one candle at a time.
* De-duplication mirrors the live bot (same candle suppression, same cooldown),
  so the trade count is what you would actually have been sent, not every raw
  cross.
* A trade opens at the signal candle's close and is resolved by walking later
  candles until the stop or the target is touched.
* **When one candle's range contains both the stop and the target, the stop is
  counted.** Without tick data there is no way to know which came first, and
  assuming the win would flatter every result.
* Round-trip fees are charged at ``FEE_PERCENT`` per side (default 0.1%, the
  Binance taker rate). On a 15m ATR stop this is not a rounding error - it is
  often a fifth of the risk.

What it cannot model
--------------------
Slippage beyond fees, partial fills, funding on perpetuals, spread, or the fact
that a stop *gap* through your level fills worse than your level. Treat the
numbers as an optimistic ceiling, not a forecast.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence

import pandas as pd

from indicators import calculate_indicators
from strategies import BUY, SETUP_LABELS, Signal, detect_signals

LOG = logging.getLogger(__name__)

WIN, LOSS, OPEN = "win", "loss", "open"


@dataclass
class Trade:
    """One simulated trade, from signal to resolution."""

    signal: Signal
    outcome: str
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    candles_held: int = 0
    r_multiple: float = 0.0        # net profit / initial risk, after fees

    @property
    def setup(self) -> str:
        return self.signal.setup

    @property
    def symbol(self) -> str:
        return self.signal.symbol


@dataclass
class Stats:
    """Aggregated performance for one bucket (a setup, a symbol, or all)."""

    name: str
    wins: int = 0
    losses: int = 0
    unresolved: int = 0
    r_multiples: List[float] = field(default_factory=list)

    @property
    def closed(self) -> int:
        return self.wins + self.losses

    @property
    def total(self) -> int:
        return self.closed + self.unresolved

    @property
    def win_rate(self) -> float:
        return (self.wins / self.closed * 100.0) if self.closed else 0.0

    @property
    def expectancy(self) -> float:
        """Average R per closed trade. Positive means an edge, on this data."""
        return (sum(self.r_multiples) / len(self.r_multiples)) if self.r_multiples else 0.0

    @property
    def total_r(self) -> float:
        return sum(self.r_multiples)

    @property
    def profit_factor(self) -> float:
        """Gross win R divided by gross loss R. Above 1.0 is profitable."""
        gains = sum(r for r in self.r_multiples if r > 0)
        pains = -sum(r for r in self.r_multiples if r < 0)
        if pains <= 0:
            return float("inf") if gains > 0 else 0.0
        return gains / pains

    @property
    def max_consecutive_losses(self) -> int:
        worst = run = 0
        for r in self.r_multiples:
            run = run + 1 if r < 0 else 0
            worst = max(worst, run)
        return worst

    def add(self, trade: Trade) -> None:
        if trade.outcome == WIN:
            self.wins += 1
            self.r_multiples.append(trade.r_multiple)
        elif trade.outcome == LOSS:
            self.losses += 1
            self.r_multiples.append(trade.r_multiple)
        else:
            self.unresolved += 1


def _fee_r_cost(signal: Signal, fee_percent: float) -> float:
    """Round-trip fees expressed in R (fractions of the initial risk).

    Fees are charged on notional, both entering and exiting, so the cost in R is
    ``2 * fee * entry / stop_distance``. A tight stop makes this large.
    """
    risk_per_unit = abs(signal.entry - signal.stop_loss)
    if risk_per_unit <= 0:
        return 0.0
    return 2.0 * (fee_percent / 100.0) * signal.entry / risk_per_unit


def simulate_trade(
    signal: Signal, future: pd.DataFrame, fee_percent: float
) -> Trade:
    """Walk candles after the signal until the stop or the target is touched."""
    risk = abs(signal.entry - signal.stop_loss)
    reward = abs(signal.take_profit - signal.entry)
    if risk <= 0:
        return Trade(signal=signal, outcome=OPEN)

    fee_r = _fee_r_cost(signal, fee_percent)
    target_r = reward / risk

    for held, (ts, row) in enumerate(future.iterrows(), start=1):
        high, low = row["high"], row["low"]

        if signal.side == BUY:
            hit_stop = low <= signal.stop_loss
            hit_target = high >= signal.take_profit
        else:
            hit_stop = high >= signal.stop_loss
            hit_target = low <= signal.take_profit

        # Stop wins ties: without tick data we cannot know the order, and
        # assuming the target would inflate every number in this report.
        if hit_stop:
            return Trade(signal, LOSS, signal.stop_loss, ts, held, -1.0 - fee_r)
        if hit_target:
            return Trade(signal, WIN, signal.take_profit, ts, held, target_r - fee_r)

    return Trade(signal=signal, outcome=OPEN, candles_held=len(future))


def backtest_symbol(
    symbol: str, df: pd.DataFrame, settings, fee_percent: float
) -> List[Trade]:
    """Replay one symbol's history and return every trade the bot would have taken."""
    enriched = calculate_indicators(df, settings)
    trend_len = enriched.attrs.get("trend_ema_length")

    warmup = max(
        settings.macd_slow + settings.macd_signal,
        settings.bb_period,
        settings.rsi_period,
        settings.atr_period,
        trend_len or 0,
    ) + 2

    if len(enriched) <= warmup + 5:
        LOG.warning("%s: only %d candles, not enough to backtest", symbol, len(enriched))
        return []

    cooldown = timedelta(minutes=settings.signal_cooldown_minutes)
    last_fired: Dict[str, datetime] = {}
    trades: List[Trade] = []

    for i in range(warmup, len(enriched) - 1):
        window = enriched.iloc[: i + 1]
        window.attrs["trend_ema_length"] = trend_len

        for signal in detect_signals(symbol, window, settings):
            # Same suppression the live bot applies, so the trade count matches
            # what would actually have reached Telegram.
            previous = last_fired.get(signal.dedupe_key)
            if previous is not None and signal.candle_time - previous < cooldown:
                continue
            last_fired[signal.dedupe_key] = signal.candle_time
            trades.append(
                simulate_trade(signal, enriched.iloc[i + 1:], fee_percent)
            )

    return trades


def aggregate(trades: Sequence[Trade], key) -> Dict[str, Stats]:
    """Bucket trades by any key function and compute stats per bucket."""
    buckets: Dict[str, Stats] = {}
    for trade in trades:
        name = key(trade)
        buckets.setdefault(name, Stats(name)).add(trade)
    return buckets


def overall(trades: Sequence[Trade]) -> Stats:
    stats = Stats("ALL")
    for trade in trades:
        stats.add(trade)
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _row(stats: Stats, width: int) -> str:
    pf = stats.profit_factor
    pf_text = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    return (
        f"{stats.name.ljust(width)}  "
        f"{stats.total:>5}  "
        f"{stats.closed:>6}  "
        f"{stats.win_rate:>6.1f}%  "
        f"{stats.expectancy:>+7.3f}  "
        f"{stats.total_r:>+8.1f}  "
        f"{pf_text}  "
        f"{stats.max_consecutive_losses:>4}"
    )


def render_report(
    trades: Sequence[Trade], settings, fee_percent: float, candles: int
) -> str:
    """Build the plain-text performance report."""
    lines: List[str] = []
    add = lines.append

    add("=" * 78)
    add("BACKTEST REPORT")
    add("=" * 78)

    if not trades:
        add("")
        add("No signals fired over this history.")
        add("That is a result too: these setups are rarer than most people expect.")
        add("Try more symbols, a longer history, or MIN_CONFIDENCE=1.")
        add("=" * 78)
        return "\n".join(lines)

    total = overall(trades)
    add(f"Timeframe      {settings.timeframe}   |   ~{candles} candles per symbol")
    add(f"Stop / target  {settings.atr_sl_multiplier}x / {settings.atr_tp_multiplier}x "
        f"ATR({settings.atr_period})   |   fees {fee_percent}% per side")
    add(f"Filters        MIN_CONFIDENCE={settings.min_confidence}, "
        f"cooldown {settings.signal_cooldown_minutes}m")
    add("")

    header_width = 26
    head = (
        f"{'BUCKET'.ljust(header_width)}  "
        f"{'SIGS':>5}  {'CLOSED':>6}  {'WIN%':>7}  "
        f"{'EXP(R)':>7}  {'TOTAL R':>8}  {'  PF':>5}  {'MAXL':>4}"
    )
    add(head)
    add("-" * len(head))

    add("By setup")
    by_setup = aggregate(trades, lambda t: SETUP_LABELS.get(t.setup, t.setup))
    for name in sorted(by_setup, key=lambda n: -by_setup[n].total_r):
        add("  " + _row(by_setup[name], header_width - 2))

    add("")
    add("By pair")
    by_symbol = aggregate(trades, lambda t: t.symbol)
    for name in sorted(by_symbol, key=lambda n: -by_symbol[n].total_r):
        add("  " + _row(by_symbol[name], header_width - 2))

    add("-" * len(head))
    add(_row(total, header_width))
    add("")

    unresolved = sum(1 for t in trades if t.outcome == OPEN)
    if unresolved:
        add(f"{unresolved} signal(s) never reached stop or target inside the sample "
            "and are excluded from every rate above.")

    add("")
    add("HOW TO READ THIS")
    add("  WIN%     share of closed trades that reached the target first.")
    add("  EXP(R)   average result per trade in units of the risk you took.")
    add("           +0.10 means each signal returned 10% of what it risked, on")
    add("           average. Below 0.00 the setup lost money on this history.")
    add("  TOTAL R  sum of all results. At 1% risk per trade, +10 R is roughly")
    add("           +10% on the account, before compounding.")
    add("  PF       gross wins / gross losses. Under 1.00 the setup lost money.")
    add("  MAXL     longest run of consecutive losses. This is the number that")
    add("           decides whether you can actually follow the bot in practice.")
    add("")
    add("CAVEATS - read before trusting any of this")
    add("  * Ties go to the stop: when a single candle contained both the stop")
    add("    and the target, it is scored a loss. Real fills may differ.")
    add("  * Fees are modelled, slippage and spread are not. Real results will")
    add("    be worse than this, not better.")
    add("  * One sample of recent history is not an edge. A positive number here")
    add("    is a reason to paper-trade, never a reason to size up.")
    add("=" * 78)
    return "\n".join(lines)


def run_backtest(exchange, symbols: Sequence[str], settings, candles: int,
                 fee_percent: float) -> str:
    """Fetch history for every symbol, replay it, and return the report text."""
    from exchange import fetch_ohlcv

    deep = _with_limit(settings, candles)
    all_trades: List[Trade] = []

    for symbol in symbols:
        LOG.info("Backtesting %s over ~%d candles...", symbol, candles)
        try:
            df = fetch_ohlcv(exchange, symbol, deep)
            if df is None or df.empty:
                continue
            trades = backtest_symbol(symbol, df, settings, fee_percent)
            LOG.info("  %s: %d signal(s)", symbol, len(trades))
            all_trades.extend(trades)
        except Exception as exc:  # noqa: BLE001 - one bad pair must not stop the run
            LOG.exception("%s: backtest failed (%s)", symbol, exc)

    return render_report(all_trades, settings, fee_percent, candles)


def _with_limit(settings, candles: int):
    """Copy of ``settings`` asking the exchange for a deeper history."""
    from dataclasses import replace

    return replace(settings, candle_limit=candles)
