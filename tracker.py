"""Follow every sent signal to its stop or target, and keep score.

A signal bot that only ever announces entries is unaccountable: you never learn
whether following it worked. This module closes that loop. Each alert is
recorded as an open position; on every later scan the same candles the bot
already downloaded are checked for a touch of the stop or the target, an outcome
message is sent, and a running scoreboard is updated.

The scoring rules match ``backtest.py`` exactly, so live results and historical
results are directly comparable:

* the stop wins ties inside one candle (no tick data, and assuming the win would
  flatter the record),
* round-trip fees are charged at ``FEE_PERCENT`` per side, and
* results are reported in R - multiples of the risk taken on that trade.

State lives in ``TRACKER_FILE`` (default ``signal_outcomes.json``) next to the
de-duplication state, and is written atomically the same way.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from strategies import BUY, SETUP_LABELS, Signal

LOG = logging.getLogger(__name__)

TRACKER_VERSION = 1
MAX_CLOSED_KEPT = 500

WIN, LOSS = "win", "loss"


@dataclass
class Outcome:
    """A resolved position, ready to be reported."""

    symbol: str
    setup: str
    side: str
    entry: float
    exit_price: float
    result: str
    r_multiple: float
    opened_at: str
    closed_at: str
    candles_held: int

    @property
    def setup_label(self) -> str:
        return SETUP_LABELS.get(self.setup, self.setup)

    @property
    def pct(self) -> float:
        """Raw price move in percent, before fees, signed by direction."""
        if not self.entry:
            return 0.0
        move = (self.exit_price - self.entry) / self.entry * 100.0
        return move if self.side == BUY else -move


def _empty() -> Dict[str, list]:
    return {"open": [], "closed": []}


def load_outcomes(path: str) -> Dict[str, list]:
    """Read the tracker file, returning an empty ledger when unusable."""
    if not os.path.exists(path):
        return _empty()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("Could not read tracker file %s (%s); starting fresh", path, exc)
        return _empty()

    if not isinstance(payload, dict):
        return _empty()
    ledger = {
        "open": payload.get("open") or [],
        "closed": payload.get("closed") or [],
    }
    if not isinstance(ledger["open"], list) or not isinstance(ledger["closed"], list):
        return _empty()
    return ledger


def save_outcomes(path: str, ledger: Dict[str, list]) -> None:
    """Atomically persist the ledger, trimming closed history."""
    payload = {
        "version": TRACKER_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "open": ledger.get("open", []),
        "closed": (ledger.get("closed") or [])[-MAX_CLOSED_KEPT:],
    }
    directory = os.path.dirname(os.path.abspath(path)) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, delete=False, suffix=".tmp"
        ) as tmp:
            json.dump(payload, tmp, indent=2, sort_keys=True)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name
        os.replace(tmp_path, path)
    except OSError as exc:
        LOG.error("Failed to persist tracker state to %s: %s", path, exc)


def position_id(signal: Signal) -> str:
    return f"{signal.dedupe_key}|{signal.candle_time.isoformat()}"


def track_signal(signal: Signal, ledger: Dict[str, list], settings) -> None:
    """Record a freshly sent signal as an open position."""
    if not settings.track_outcomes:
        return
    pid = position_id(signal)
    if any(p.get("id") == pid for p in ledger["open"]):
        return
    ledger["open"].append(
        {
            "id": pid,
            "symbol": signal.symbol,
            "setup": signal.setup,
            "side": signal.side,
            "entry": signal.entry,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "confidence": signal.confidence,
            "opened_at": signal.candle_time.isoformat(),
        }
    )
    LOG.debug("Tracking %s (%d open)", pid, len(ledger["open"]))


def _fee_r_cost(entry: float, stop_loss: float, fee_percent: float) -> float:
    """Round-trip fees expressed in R. Matches backtest._fee_r_cost."""
    risk = abs(entry - stop_loss)
    if risk <= 0:
        return 0.0
    return 2.0 * (fee_percent / 100.0) * entry / risk


def resolve_open(
    symbol: str, df: pd.DataFrame, ledger: Dict[str, list], settings
) -> List[Outcome]:
    """Close any tracked position for ``symbol`` that hit its stop or target.

    Uses the candles the scan already downloaded, so this costs no extra API
    calls. Only candles strictly after the signal's own candle are considered.
    """
    if not settings.track_outcomes or df is None or df.empty:
        return []

    still_open: List[dict] = []
    resolved: List[Outcome] = []

    for pos in ledger["open"]:
        if pos.get("symbol") != symbol:
            still_open.append(pos)
            continue

        opened_at = _parse_dt(pos.get("opened_at"))
        if opened_at is None:
            LOG.warning("Dropping tracked position with an unreadable date: %s", pos)
            continue

        future = df[df.index > opened_at]
        if future.empty:
            still_open.append(pos)
            continue

        outcome = _walk(pos, future, settings)
        if outcome is None:
            still_open.append(pos)
        else:
            resolved.append(outcome)
            ledger["closed"].append(outcome.__dict__)

    ledger["open"] = still_open
    return resolved


def _walk(pos: dict, future: pd.DataFrame, settings) -> Optional[Outcome]:
    """Return the outcome if this position resolved inside ``future``."""
    entry = float(pos["entry"])
    stop = float(pos["stop_loss"])
    target = float(pos["take_profit"])
    side = pos["side"]

    risk = abs(entry - stop)
    if risk <= 0:
        return None
    target_r = abs(target - entry) / risk
    fee_r = _fee_r_cost(entry, stop, settings.fee_percent)

    for held, (ts, row) in enumerate(future.iterrows(), start=1):
        if side == BUY:
            hit_stop = row["low"] <= stop
            hit_target = row["high"] >= target
        else:
            hit_stop = row["high"] >= stop
            hit_target = row["low"] <= target

        # The stop wins ties, exactly as in the backtest.
        if hit_stop:
            return _outcome(pos, stop, LOSS, -1.0 - fee_r, ts, held)
        if hit_target:
            return _outcome(pos, target, WIN, target_r - fee_r, ts, held)

    return None


def _outcome(pos, exit_price, result, r, ts, held) -> Outcome:
    return Outcome(
        symbol=pos["symbol"], setup=pos["setup"], side=pos["side"],
        entry=float(pos["entry"]), exit_price=float(exit_price),
        result=result, r_multiple=round(float(r), 4),
        opened_at=pos["opened_at"],
        closed_at=ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
        candles_held=held,
    )


def _parse_dt(value) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return pd.Timestamp(parsed).tz_convert("UTC")


def scoreboard(ledger: Dict[str, list]) -> Tuple[int, int, float, float]:
    """``(wins, losses, total_R, win_rate_pct)`` over all closed positions."""
    closed = ledger.get("closed") or []
    wins = sum(1 for c in closed if c.get("result") == WIN)
    losses = sum(1 for c in closed if c.get("result") == LOSS)
    total_r = sum(float(c.get("r_multiple", 0.0)) for c in closed)
    rate = (wins / (wins + losses) * 100.0) if (wins + losses) else 0.0
    return wins, losses, total_r, rate
