"""MetaTrader 5 order execution.

Everything else in this bot is read-only: ``exchange.py`` touches public Binance
endpoints and cannot place an order even if it wanted to. This module is the one
place that can move real money, so it is deliberately paranoid:

* it is **off** unless ``MT5_ENABLED=true``,
* it refuses a **live** account unless ``MT5_ALLOW_LIVE=true`` as well, so the
  default blast radius of a misconfiguration is a demo balance,
* every order is sized from the signal's own stop distance, never a fixed lot,
* per-cycle and per-day limits cap how much damage a bad run can do.

``MetaTrader5`` is a Windows-only package (the terminal it drives is a Windows
application), so the import is lazy and the failure message says so. The rest of
the bot - and this module's own tests - run fine on Linux without it.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

LOG = logging.getLogger(__name__)

# Order lifetime hints. MT5 exposes these as terminal constants; we keep local
# names so the module is importable (and testable) without the package.
FILLING_MODES = ("FOK", "IOC", "RETURN")


class MT5Error(RuntimeError):
    """Raised when MetaTrader 5 is unusable or rejects an order."""


@dataclass
class Execution:
    """The result of trying to place one order."""

    symbol: str            # the MT5 symbol actually traded, e.g. "BTCUSD"
    pair: str              # the bot's pair notation, e.g. "BTC/USDT"
    side: str              # BUY / SELL
    volume: float          # lots
    requested_price: float
    filled_price: float
    stop_loss: float
    take_profit: float
    ticket: int
    retcode: int
    comment: str
    placed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def ok(self) -> bool:
        return self.ticket > 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "pair": self.pair,
            "side": self.side,
            "volume": self.volume,
            "requested_price": self.requested_price,
            "filled_price": self.filled_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "ticket": self.ticket,
            "retcode": self.retcode,
            "comment": self.comment,
            "placed_at": self.placed_at.isoformat(),
        }


def _import_mt5():
    """Import the MetaTrader5 package with an actionable error message."""
    try:
        import MetaTrader5 as mt5  # noqa: PLC0415 - optional, platform-specific
    except ImportError as exc:
        raise MT5Error(
            "The 'MetaTrader5' package is not installed, or this is not Windows. "
            "MetaTrader 5's Python API only runs on Windows against a locally "
            "installed terminal. Install it there with "
            "'pip install -r requirements-mt5.txt', or leave MT5_ENABLED=false "
            "and keep using Telegram alerts."
        ) from exc
    return mt5


def parse_symbol_map(raw: str) -> Dict[str, str]:
    """Parse ``"BTC/USDT=BTCUSD,ETH/USDT=ETHUSD"`` into a dict.

    Malformed entries are logged and skipped rather than raising: a typo in one
    mapping should cost you that one pair, not the whole run.
    """
    mapping: Dict[str, str] = {}
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            LOG.warning("MT5_SYMBOL_MAP entry %r has no '='; ignoring it", chunk)
            continue
        pair, _, broker_symbol = chunk.partition("=")
        pair, broker_symbol = pair.strip().upper(), broker_symbol.strip()
        if not pair or not broker_symbol:
            LOG.warning("MT5_SYMBOL_MAP entry %r is incomplete; ignoring it", chunk)
            continue
        mapping[pair] = broker_symbol
    return mapping


def symbol_candidates(pair: str, suffix: str = "") -> List[str]:
    """Broker symbol guesses for a ccxt pair, most likely first.

    Brokers spell the same instrument half a dozen ways - ``BTCUSD``,
    ``BTCUSDT``, ``BTCUSD.r``, ``BTCUSDm`` - so rather than demand a full map up
    front we try the obvious forms and let ``MT5Broker`` keep the one the
    terminal recognises.
    """
    base, _, quote = pair.upper().partition("/")
    quote = quote.split(":")[0]           # "BTC/USDT:USDT" -> "USDT"
    if not base:
        return []
    stems = [f"{base}{quote}"] if quote else []
    # Most CFD brokers quote crypto against USD, not USDT.
    if quote in {"USDT", "USDC", "BUSD"}:
        stems.append(f"{base}USD")
    stems.append(base)

    seen, out = set(), []
    for stem in stems:
        for candidate in ((stem + suffix, stem) if suffix else (stem,)):
            if candidate and candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
    return out


def round_to_step(volume: float, step: float) -> float:
    """Round ``volume`` *down* to a whole multiple of ``step``.

    Down, never nearest: rounding up would quietly risk more than the caller
    asked for. The result is re-rounded to the step's own precision so floating
    point cannot produce 0.30000000000000004 lots.
    """
    if step <= 0:
        return volume
    steps = int(volume / step + 1e-9)
    decimals = max(0, len(f"{step:.8f}".rstrip("0").split(".")[1]))
    return round(steps * step, decimals)


class MT5Broker:
    """Places risk-sized market orders on a MetaTrader 5 terminal."""

    def __init__(self, settings, dry_run: bool = False, mt5_module=None):
        self.settings = settings
        # --dry-run must never reach a broker, exactly as it never reaches Telegram.
        self.dry_run = dry_run or settings.mt5_dry_run
        self._mt5 = mt5_module          # injected in tests; imported lazily otherwise
        self._connected = False
        self._symbol_cache: Dict[str, Optional[str]] = {}
        self._orders_today = 0
        self._orders_day: Optional[date] = None

    # --- connection -------------------------------------------------------

    @property
    def mt5(self):
        if self._mt5 is None:
            self._mt5 = _import_mt5()
        return self._mt5

    def __enter__(self) -> "MT5Broker":
        self.connect()
        return self

    def __exit__(self, *_exc) -> None:
        self.shutdown()

    def connect(self) -> None:
        """Initialise the terminal, log in, and enforce the demo-only guard."""
        if self._connected:
            return
        mt5 = self.mt5
        kwargs: Dict[str, object] = {}
        if self.settings.mt5_path:
            kwargs["path"] = self.settings.mt5_path
        if self.settings.mt5_login:
            kwargs.update(
                login=self.settings.mt5_login,
                password=self.settings.mt5_password,
                server=self.settings.mt5_server,
            )
        if not mt5.initialize(**kwargs):
            raise MT5Error(
                f"MetaTrader 5 initialize() failed: {self._last_error()}. "
                "Check that the terminal is installed and running, that "
                "MT5_PATH points at terminal64.exe if it is in a custom "
                "location, and that MT5_LOGIN / MT5_PASSWORD / MT5_SERVER match "
                "the account."
            )
        self._connected = True

        account = mt5.account_info()
        if account is None:
            self.shutdown()
            raise MT5Error(
                f"Connected to the terminal but account_info() is empty: "
                f"{self._last_error()}. The terminal is probably not logged in."
            )

        demo = getattr(account, "trade_mode", None) == getattr(
            mt5, "ACCOUNT_TRADE_MODE_DEMO", 0
        )
        if not demo and not self.settings.mt5_allow_live:
            self.shutdown()
            raise MT5Error(
                f"Account {account.login} on {account.server} is a LIVE account "
                "and MT5_ALLOW_LIVE is not set. Refusing to trade real money by "
                "accident. Set MT5_ALLOW_LIVE=true only once you have watched "
                "this bot run on a demo account and you accept the risk."
            )
        if not getattr(account, "trade_allowed", True):
            self.shutdown()
            raise MT5Error(
                "The terminal reports trading is not allowed for this account. "
                "Enable 'Algo Trading' in the MT5 toolbar and check the account "
                "has trading rights."
            )

        LOG.info(
            "MT5 connected: account %s on %s (%s), balance %.2f %s, equity %.2f%s",
            account.login, account.server, "DEMO" if demo else "LIVE",
            account.balance, account.currency, account.equity,
            " [DRY RUN - no orders will be sent]" if self.dry_run else "",
        )

    def shutdown(self) -> None:
        if self._connected:
            try:
                self.mt5.shutdown()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                LOG.warning("MT5 shutdown() raised %s; ignoring", exc)
            self._connected = False

    def _last_error(self) -> str:
        try:
            return str(self.mt5.last_error())
        except Exception:  # noqa: BLE001
            return "unknown error"

    # --- symbols ----------------------------------------------------------

    def resolve_symbol(self, pair: str) -> Optional[str]:
        """Map a bot pair to a broker symbol, or ``None`` if it has none.

        The answer is cached (misses included) because a scan asks for the same
        ten pairs every 15 minutes and each miss costs a round trip per guess.
        """
        key = pair.upper()
        if key in self._symbol_cache:
            return self._symbol_cache[key]

        explicit = self.settings.mt5_symbol_map.get(key)
        candidates = [explicit] if explicit else symbol_candidates(
            key, self.settings.mt5_symbol_suffix
        )

        resolved = None
        for candidate in candidates:
            info = self.mt5.symbol_info(candidate)
            if info is None:
                continue
            # A symbol hidden from Market Watch returns info but cannot be traded
            # until it is selected.
            if not getattr(info, "visible", True) and not self.mt5.symbol_select(
                candidate, True
            ):
                LOG.warning("MT5 symbol %s exists but could not be selected", candidate)
                continue
            resolved = candidate
            break

        if resolved is None:
            LOG.warning(
                "No MT5 symbol for %s (tried %s). Add an explicit mapping via "
                "MT5_SYMBOL_MAP, e.g. MT5_SYMBOL_MAP=%s=YOURSYMBOL",
                pair, ", ".join(candidates) or "nothing", key,
            )
        else:
            LOG.info("MT5 symbol for %s resolved to %s", pair, resolved)
        self._symbol_cache[key] = resolved
        return resolved

    # --- sizing -----------------------------------------------------------

    def lots_for(self, info, entry: float, stop_loss: float, equity: float) -> float:
        """Lots that put ``risk_percent`` of equity between entry and stop.

        Money lost per lot is ``(stop distance / tick size) * tick value``, which
        is the only formulation that stays correct across instruments whose tick
        size is not one point (indices, metals, JPY pairs, crypto CFDs).

        Returns ``0.0`` when the broker's minimum lot would risk more than the
        budget - refusing the trade is the whole point of position sizing.
        """
        distance = abs(entry - stop_loss)
        if distance <= 0:
            LOG.error("Refusing to size a trade with a zero stop distance")
            return 0.0

        tick_size = getattr(info, "trade_tick_size", 0.0) or getattr(info, "point", 0.0)
        tick_value = getattr(info, "trade_tick_value", 0.0)
        if tick_size <= 0 or tick_value <= 0:
            LOG.error(
                "%s: broker reports tick size %s / tick value %s; cannot size a "
                "trade safely", getattr(info, "name", "?"), tick_size, tick_value,
            )
            return 0.0

        risk_budget = equity * self.settings.risk_fraction
        loss_per_lot = (distance / tick_size) * tick_value
        if loss_per_lot <= 0:
            return 0.0

        step = getattr(info, "volume_step", 0.01) or 0.01
        vol_min = getattr(info, "volume_min", step)
        vol_max = getattr(info, "volume_max", 100.0)
        if self.settings.mt5_max_lot > 0:
            vol_max = min(vol_max, self.settings.mt5_max_lot)

        volume = round_to_step(risk_budget / loss_per_lot, step)

        if volume < vol_min:
            min_risk = vol_min * loss_per_lot
            LOG.warning(
                "%s: skipping - the minimum %.2f lot would risk %.2f, over the "
                "%.2f budget (%.2f%% of %.2f equity). Lower RISK_PERCENT only if "
                "you mean to; the safer fix is a smaller account exposure.",
                getattr(info, "name", "?"), vol_min, min_risk, risk_budget,
                self.settings.risk_percent, equity,
            )
            return 0.0

        if volume > vol_max:
            LOG.info(
                "%s: sizing capped at %.2f lots (wanted %.2f)",
                getattr(info, "name", "?"), vol_max, volume,
            )
            volume = round_to_step(vol_max, step)
        return volume

    # --- trading ----------------------------------------------------------

    def open_positions(self) -> List:
        """Positions opened by *this* bot, identified by its magic number."""
        positions = self.mt5.positions_get()
        if not positions:
            return []
        magic = self.settings.mt5_magic
        return [p for p in positions if getattr(p, "magic", None) == magic]

    def _budget_allows_another_order(self) -> bool:
        """Check the per-day order cap, resetting it when the date rolls over."""
        today = datetime.now(timezone.utc).date()
        if self._orders_day != today:
            self._orders_day, self._orders_today = today, 0
        cap = self.settings.mt5_max_orders_per_day
        if cap > 0 and self._orders_today >= cap:
            LOG.warning(
                "MT5 daily order cap reached (%d); no more orders until UTC "
                "midnight", cap,
            )
            return False
        return True

    def place(self, signal) -> Optional[Execution]:
        """Place one risk-sized market order with SL and TP attached.

        Returns ``None`` when the trade was deliberately skipped (no symbol, no
        room, size rounds to zero); raises :class:`MT5Error` only when the
        terminal itself is unusable. A rejected order comes back as an
        ``Execution`` with ``ok == False`` so the caller can report it.
        """
        self.connect()
        mt5 = self.mt5

        if not self._budget_allows_another_order():
            return None

        symbol = self.resolve_symbol(signal.symbol)
        if symbol is None:
            return None

        open_now = self.open_positions()
        cap = self.settings.mt5_max_open_positions
        if cap > 0 and len(open_now) >= cap:
            LOG.warning(
                "%s: skipping - already holding %d position(s), cap is %d",
                signal.symbol, len(open_now), cap,
            )
            return None

        # Never stack a second position on one symbol: two entries on the same
        # instrument double the risk the sizing calculation just budgeted for.
        if any(getattr(p, "symbol", "") == symbol for p in open_now):
            LOG.info(
                "%s: skipping - this bot already has a position on it",
                signal.symbol,
            )
            return None

        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            LOG.error("%s: no live quote from the broker; skipping", symbol)
            return None

        account = mt5.account_info()
        if account is None:
            raise MT5Error(f"account_info() went empty mid-run: {self._last_error()}")

        is_buy = signal.side.upper() == "BUY"
        # Cross the spread: buy at ask, sell at bid.
        price = float(tick.ask if is_buy else tick.bid)
        if price <= 0:
            LOG.error("%s: broker quoted a non-positive price; skipping", symbol)
            return None

        # Size off the *signal's* geometry but place at the *broker's* price. The
        # stop keeps its ATR distance from the real fill, so the risk budget holds
        # even when the CFD price differs from Binance's.
        stop_distance = abs(signal.entry - signal.stop_loss)
        target_distance = abs(signal.take_profit - signal.entry)
        stop_loss = price - stop_distance if is_buy else price + stop_distance
        take_profit = price + target_distance if is_buy else price - target_distance

        digits = int(getattr(info, "digits", 5))
        stop_loss, take_profit = round(stop_loss, digits), round(take_profit, digits)

        volume = self.lots_for(info, price, stop_loss, float(account.equity))
        if volume <= 0:
            return None

        if self.dry_run:
            LOG.info(
                "[DRY RUN] would %s %.2f lots of %s at %.*f, SL %.*f TP %.*f",
                signal.side, volume, symbol, digits, price,
                digits, stop_loss, digits, take_profit,
            )
            self._orders_today += 1
            return Execution(
                symbol=symbol, pair=signal.symbol, side=signal.side, volume=volume,
                requested_price=price, filled_price=price, stop_loss=stop_loss,
                take_profit=take_profit, ticket=0, retcode=0, comment="dry-run",
            )

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": self.settings.mt5_deviation_points,
            "magic": self.settings.mt5_magic,
            "comment": f"{signal.setup}"[:31],   # MT5 truncates past 31 chars
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(info),
        }

        result = mt5.order_send(request)
        if result is None:
            raise MT5Error(
                f"order_send() returned nothing for {symbol}: {self._last_error()}"
            )

        retcode = int(getattr(result, "retcode", -1))
        ok = retcode == getattr(mt5, "TRADE_RETCODE_DONE", 10009)
        execution = Execution(
            symbol=symbol,
            pair=signal.symbol,
            side=signal.side,
            volume=float(getattr(result, "volume", volume) or volume),
            requested_price=price,
            filled_price=float(getattr(result, "price", price) or price),
            stop_loss=stop_loss,
            take_profit=take_profit,
            ticket=int(getattr(result, "order", 0) or 0) if ok else 0,
            retcode=retcode,
            comment=str(getattr(result, "comment", "") or ""),
        )

        if ok:
            self._orders_today += 1
            LOG.info(
                "MT5 %s %.2f %s filled at %.*f (ticket %d), SL %.*f TP %.*f",
                signal.side, execution.volume, symbol, digits,
                execution.filled_price, execution.ticket,
                digits, stop_loss, digits, take_profit,
            )
        else:
            LOG.error(
                "MT5 rejected %s %s: retcode %d (%s)",
                signal.side, symbol, retcode, execution.comment,
            )
        return execution

    def _filling_mode(self, info):
        """Pick a filling mode the symbol actually supports.

        Brokers differ, and sending an unsupported mode earns retcode 10030
        ("Unsupported filling mode") on every single order - a failure that looks
        like a credentials problem but is not.
        """
        mt5 = self.mt5
        configured = (self.settings.mt5_filling_mode or "").upper()
        if configured in FILLING_MODES:
            return getattr(mt5, f"ORDER_FILLING_{configured}")

        allowed = int(getattr(info, "filling_mode", 0))
        # filling_mode is a bit mask of SYMBOL_FILLING_* flags.
        if allowed & int(getattr(mt5, "SYMBOL_FILLING_FOK", 1)):
            return mt5.ORDER_FILLING_FOK
        if allowed & int(getattr(mt5, "SYMBOL_FILLING_IOC", 2)):
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def close(self, position) -> Optional[Execution]:
        """Close an open position at market. Used by ``--mt5-close-all``."""
        self.connect()
        mt5 = self.mt5
        symbol = getattr(position, "symbol", "")
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            LOG.error("%s: no quote; cannot close position %s", symbol,
                      getattr(position, "ticket", "?"))
            return None

        # Closing is the opposite deal: a long is closed by selling at the bid.
        was_buy = getattr(position, "type", 0) == getattr(mt5, "POSITION_TYPE_BUY", 0)
        price = float(tick.bid if was_buy else tick.ask)
        info = mt5.symbol_info(symbol)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(position.volume),
            "type": mt5.ORDER_TYPE_SELL if was_buy else mt5.ORDER_TYPE_BUY,
            "position": int(position.ticket),
            "price": price,
            "deviation": self.settings.mt5_deviation_points,
            "magic": self.settings.mt5_magic,
            "comment": "closed by bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(info),
        }
        result = mt5.order_send(request)
        retcode = int(getattr(result, "retcode", -1)) if result is not None else -1
        ok = retcode == getattr(mt5, "TRADE_RETCODE_DONE", 10009)
        if not ok:
            LOG.error("Failed to close position %s: retcode %d",
                      getattr(position, "ticket", "?"), retcode)
        return Execution(
            symbol=symbol, pair=symbol, side="SELL" if was_buy else "BUY",
            volume=float(position.volume), requested_price=price,
            filled_price=float(getattr(result, "price", price) or price),
            stop_loss=0.0, take_profit=0.0,
            ticket=int(position.ticket) if ok else 0, retcode=retcode,
            comment=str(getattr(result, "comment", "") or ""),
        )

    def account_summary(self) -> Dict[str, object]:
        """Balance / equity snapshot for the dashboard and ``--test-mt5``."""
        self.connect()
        account = self.mt5.account_info()
        if account is None:
            raise MT5Error(f"account_info() is empty: {self._last_error()}")
        demo = getattr(account, "trade_mode", None) == getattr(
            self.mt5, "ACCOUNT_TRADE_MODE_DEMO", 0
        )
        return {
            "login": account.login,
            "server": account.server,
            "mode": "DEMO" if demo else "LIVE",
            "currency": account.currency,
            "balance": float(account.balance),
            "equity": float(account.equity),
            "margin_free": float(getattr(account, "margin_free", 0.0)),
            "open_positions": len(self.open_positions()),
        }


# --- execution log --------------------------------------------------------
#
# Kept separate from ``tracker.py``: that ledger scores the *signals* against
# Binance candles, which is what tells you whether the strategy works. This one
# records what the *broker* actually did - fills, slippage, rejections - which is
# what tells you whether execution is working. They disagree often enough that
# collapsing them would hide the interesting cases.

EXECUTION_LOG_VERSION = 1
MAX_EXECUTIONS_KEPT = 500


def load_executions(path: str) -> List[dict]:
    """Read the execution log, returning ``[]`` when it is missing or unusable."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("Could not read execution log %s (%s); starting fresh", path, exc)
        return []
    executions = payload.get("executions") if isinstance(payload, dict) else None
    return executions if isinstance(executions, list) else []


def save_executions(path: str, executions: List[dict]) -> None:
    """Atomically persist the execution log, newest last, oldest trimmed."""
    payload = {
        "version": EXECUTION_LOG_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "executions": executions[-MAX_EXECUTIONS_KEPT:],
    }
    directory = os.path.dirname(os.path.abspath(path)) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, delete=False, suffix=".tmp"
        ) as tmp:
            json.dump(payload, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name
        os.replace(tmp_path, path)
    except OSError as exc:
        LOG.error("Failed to persist the execution log to %s: %s", path, exc)
