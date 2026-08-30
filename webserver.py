"""Live web dashboard for the signal bot.

Serves the same information the Telegram alerts carry, plus the live state of
every watched pair, from the *same* Python modules the bot itself runs. There is
no second implementation of the indicators or the setups, so what the dashboard
shows and what Telegram sends can never disagree.

Market data is fetched **server side** through ccxt. A browser cannot call
Binance directly - exchange APIs do not send CORS headers, so the request is
blocked before it leaves the page - which is why the live-data path has to run
here and be handed to the browser as JSON.

Run it on its own::

    python webserver.py                 # dashboard only, on :8000

or let it also run the scanning loop in a background thread, so a single
deployment gives you both Telegram alerts and the dashboard::

    RUN_BOT_IN_WEBAPP=true python webserver.py
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from config import Settings, configure_logging, load_settings
from exchange import ExchangeError, create_exchange, fetch_ohlcv, load_valid_symbols
from indicators import calculate_indicators, resolve_backend
from state import load_state
from strategies import BUY, SETUP_LABELS, detect_signals
from tracker import load_outcomes, scoreboard

LOG = logging.getLogger("webserver")

app = FastAPI(title="Crypto Signal Bot", docs_url=None, redoc_url=None)

# Optional shared secret. The dashboard carries no credentials, but it does show
# your trading record, and a Railway URL is public the moment a domain is
# attached. Set DASHBOARD_TOKEN and the page is reachable only with the key.
DASHBOARD_TOKEN = os.getenv("DASHBOARD_TOKEN", "").strip()
COOKIE_NAME = "signalbot_key"


@app.middleware("http")
async def require_token(request: Request, call_next):
    """Gate every route behind DASHBOARD_TOKEN when one is configured.

    The key may arrive as ?key=, an X-Dashboard-Key header, or the cookie set
    after a successful query - so opening the URL once keeps the page's own
    fetches working without putting the key in every request you type.
    """
    if not DASHBOARD_TOKEN or request.url.path == "/healthz":
        return await call_next(request)

    supplied = (
        request.query_params.get("key")
        or request.headers.get("x-dashboard-key")
        or request.cookies.get(COOKIE_NAME)
        or ""
    )
    if not secrets.compare_digest(supplied, DASHBOARD_TOKEN):
        return PlainTextResponse(
            "Unauthorised. Append ?key=YOUR_DASHBOARD_TOKEN to the URL.",
            status_code=401,
        )

    response = await call_next(request)
    if request.query_params.get("key"):
        # Remember it so the page's own API calls do not need the parameter.
        response.set_cookie(
            COOKIE_NAME, DASHBOARD_TOKEN, httponly=True, samesite="strict",
            max_age=60 * 60 * 24 * 30,
        )
    return response

# Module state, populated once at startup.
SETTINGS: Settings = load_settings()
EXCHANGE = None
SYMBOLS: List[str] = []
STARTED_AT = datetime.now(timezone.utc)

# Live market snapshots are cached: ten pairs on a 15m timeframe do not change
# meaningfully within a minute, and every browser refresh would otherwise cost
# ten exchange calls.
_CACHE: Dict[str, Any] = {"at": 0.0, "rows": [], "error": None}
_CACHE_LOCK = threading.Lock()
CACHE_SECONDS = int(os.getenv("DASHBOARD_CACHE_SECONDS", "45"))


# ---------------------------------------------------------------------------
# Market snapshot
# ---------------------------------------------------------------------------
def _bb_position(row) -> Optional[float]:
    """Where price sits inside the Bollinger envelope: 0 = lower, 100 = upper."""
    lower, upper = row.get("bb_lower"), row.get("bb_upper")
    try:
        width = float(upper) - float(lower)
        if width <= 0:
            return None
        return max(-25.0, min(125.0, (float(row["close"]) - float(lower)) / width * 100.0))
    except (TypeError, ValueError):
        return None


def _num(value) -> Optional[float]:
    try:
        out = float(value)
        return out if out == out else None       # NaN check without importing math
    except (TypeError, ValueError):
        return None


def build_snapshot() -> Dict[str, Any]:
    """Fetch every watched pair and describe its current state."""
    rows: List[Dict[str, Any]] = []
    failures: List[str] = []

    for symbol in SYMBOLS:
        try:
            df = fetch_ohlcv(EXCHANGE, symbol, SETTINGS)
            if df is None or df.empty:
                failures.append(symbol)
                continue
            enriched = calculate_indicators(df, SETTINGS)
            last = enriched.iloc[-1]
            prev = enriched.iloc[-2] if len(enriched) > 1 else last

            live = [
                {
                    "setup": s.setup,
                    "setup_label": s.setup_label,
                    "side": s.side,
                    "entry": s.entry,
                    "stop_loss": s.stop_loss,
                    "take_profit": s.take_profit,
                    "confidence": s.confidence,
                    "confirmations": s.confirmations,
                }
                for s in detect_signals(symbol, enriched, SETTINGS)
            ]

            close = _num(last["close"])
            prev_close = _num(prev["close"])
            change = ((close - prev_close) / prev_close * 100.0) if close and prev_close else None
            trend_ema = _num(last.get("trend_ema"))
            volume, volume_sma = _num(last.get("volume")), _num(last.get("volume_sma"))

            rows.append({
                "symbol": symbol,
                "price": close,
                "change_pct": change,
                "candle_time": enriched.index[-1].isoformat(),
                "rsi": _num(last.get("rsi")),
                "macd_hist": _num(last.get("macd_hist")),
                "macd_above_signal": (
                    None if _num(last.get("macd")) is None or _num(last.get("macd_signal")) is None
                    else bool(last["macd"] > last["macd_signal"])
                ),
                "bb_position": _bb_position(last),
                "atr": _num(last.get("atr")),
                "atr_pct": (
                    (_num(last.get("atr")) / close * 100.0)
                    if close and _num(last.get("atr")) else None
                ),
                "trend": (
                    None if trend_ema is None or close is None
                    else ("up" if close > trend_ema else "down")
                ),
                "trend_ema": trend_ema,
                "volume_ratio": (
                    (volume / volume_sma) if volume and volume_sma else None
                ),
                "signals": live,
            })
        except Exception as exc:  # noqa: BLE001 - one bad pair must not blank the page
            LOG.exception("%s: dashboard fetch failed (%s)", symbol, exc)
            failures.append(symbol)

    return {
        "rows": rows,
        "failed": failures,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def cached_snapshot(force: bool = False) -> Dict[str, Any]:
    """Return the market snapshot, refreshing it at most every CACHE_SECONDS."""
    with _CACHE_LOCK:
        fresh_enough = (time.time() - _CACHE["at"]) < CACHE_SECONDS
        if _CACHE["rows"] and fresh_enough and not force:
            return {
                "rows": _CACHE["rows"],
                "failed": _CACHE.get("failed", []),
                "fetched_at": _CACHE["fetched_at"],
                "cached": True,
            }

    snapshot = build_snapshot()
    with _CACHE_LOCK:
        _CACHE.update(at=time.time(), **snapshot)
    snapshot["cached"] = False
    return snapshot


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/status")
def api_status() -> Dict[str, Any]:
    """Configuration and health, enough to tell a live bot from a stalled one."""
    return {
        "exchange": SETTINGS.exchange_id,
        "timeframe": SETTINGS.timeframe,
        "symbols": SYMBOLS,
        "setups": [SETUP_LABELS.get(s, s) for s in SETTINGS.enabled_setups],
        "min_confidence": SETTINGS.min_confidence,
        "capital": SETTINGS.capital,
        "risk_percent": SETTINGS.risk_percent,
        "atr_sl": SETTINGS.atr_sl_multiplier,
        "atr_tp": SETTINGS.atr_tp_multiplier,
        "fee_percent": SETTINGS.fee_percent,
        "indicator_backend": resolve_backend(SETTINGS.indicator_backend),
        "bot_loop_running": _BOT_THREAD is not None and _BOT_THREAD.is_alive(),
        "started_at": STARTED_AT.isoformat(),
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/market")
def api_market(refresh: bool = Query(False, description="bypass the cache")) -> Dict[str, Any]:
    """Live indicator values and any setup currently firing, per pair."""
    return cached_snapshot(force=refresh)


@app.get("/api/positions")
def api_positions() -> Dict[str, Any]:
    """Open tracked positions, marked to the latest cached price."""
    ledger = load_outcomes(SETTINGS.tracker_file)
    prices = {r["symbol"]: r["price"] for r in cached_snapshot()["rows"] if r["price"]}

    out = []
    for pos in ledger.get("open", []):
        price = prices.get(pos["symbol"])
        entry = float(pos["entry"])
        risk = abs(entry - float(pos["stop_loss"]))
        unrealised = None
        if price and risk > 0:
            move = (price - entry) if pos["side"] == BUY else (entry - price)
            unrealised = move / risk           # in R, before fees
        out.append({**pos, "price": price, "unrealised_r": unrealised})

    return {"open": out, "count": len(out)}


@app.get("/api/history")
def api_history(limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    """Closed positions, newest first, plus the cumulative scoreboard."""
    ledger = load_outcomes(SETTINGS.tracker_file)
    closed = list(reversed(ledger.get("closed", [])))[:limit]
    wins, losses, total_r, rate = scoreboard(ledger)
    return {
        "closed": closed,
        "scoreboard": {
            "wins": wins, "losses": losses, "total_r": round(total_r, 3),
            "win_rate": round(rate, 1), "closed": wins + losses,
            "open": len(ledger.get("open", [])),
            "account_pct": round(total_r * SETTINGS.risk_percent, 2),
        },
    }


@app.get("/api/alerts")
def api_alerts() -> Dict[str, Any]:
    """What the bot has already sent, from the de-duplication state."""
    state = load_state(SETTINGS.state_file)
    alerts = sorted(state.values(), key=lambda e: e.get("sent_at", ""), reverse=True)
    return {"alerts": alerts, "count": len(alerts)}


@app.get("/api/backtest")
def api_backtest(candles: int = Query(0, ge=0, le=1500)) -> Dict[str, Any]:
    """Replay history and return per-setup performance as JSON.

    Slow by nature: it downloads deep history for every pair, so the dashboard
    calls it only when asked.
    """
    from backtest import aggregate, backtest_symbol, overall
    from dataclasses import replace

    depth = candles or SETTINGS.backtest_candles
    deep = replace(SETTINGS, candle_limit=depth)
    trades = []
    for symbol in SYMBOLS:
        try:
            df = fetch_ohlcv(EXCHANGE, symbol, deep)
            if df is not None and not df.empty:
                trades.extend(backtest_symbol(symbol, df, SETTINGS, SETTINGS.fee_percent))
        except Exception as exc:  # noqa: BLE001
            LOG.exception("%s: backtest failed (%s)", symbol, exc)

    def pack(stats):
        pf = stats.profit_factor
        return {
            "name": stats.name, "signals": stats.total, "closed": stats.closed,
            "win_rate": round(stats.win_rate, 1),
            "expectancy": round(stats.expectancy, 3),
            "total_r": round(stats.total_r, 2),
            "profit_factor": (None if pf == float("inf") else round(pf, 2)),
            "max_consecutive_losses": stats.max_consecutive_losses,
        }

    by_setup = aggregate(trades, lambda t: SETUP_LABELS.get(t.setup, t.setup))
    by_symbol = aggregate(trades, lambda t: t.symbol)
    return {
        "candles": depth,
        "fee_percent": SETTINGS.fee_percent,
        "by_setup": [pack(s) for s in sorted(by_setup.values(), key=lambda s: -s.total_r)],
        "by_symbol": [pack(s) for s in sorted(by_symbol.values(), key=lambda s: -s.total_r)],
        "overall": pack(overall(trades)),
    }


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse({"ok": True, "time": datetime.now(timezone.utc).isoformat()})


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    try:
        with open(os.path.join(os.path.dirname(__file__), "dashboard.html"),
                  encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"dashboard.html missing: {exc}")


# ---------------------------------------------------------------------------
# Optional background scanning loop
# ---------------------------------------------------------------------------
_BOT_THREAD: Optional[threading.Thread] = None


def _run_bot_loop() -> None:
    """Run the normal scan loop in a daemon thread.

    Lets one deployment serve the dashboard AND send Telegram alerts, instead of
    paying for two services. The loop owns its own exchange client so a slow
    dashboard request can never delay a scan.
    """
    from main import main as bot_main

    LOG.info("Starting the scanning loop in a background thread")
    try:
        bot_main([])
    except Exception as exc:  # noqa: BLE001 - the web server must survive
        LOG.exception("Background bot loop stopped: %s", exc)


@app.on_event("startup")
def startup() -> None:
    global EXCHANGE, SYMBOLS, _BOT_THREAD

    configure_logging(SETTINGS.log_level)
    try:
        EXCHANGE = create_exchange(SETTINGS)
        SYMBOLS = load_valid_symbols(EXCHANGE, SETTINGS.symbols)
    except ExchangeError as exc:
        LOG.error("%s", exc)
        SYMBOLS = list(SETTINGS.symbols)

    LOG.info("Dashboard ready for %d pair(s) on %s", len(SYMBOLS), SETTINGS.exchange_id)

    if os.getenv("RUN_BOT_IN_WEBAPP", "").strip().lower() in {"1", "true", "yes", "on"}:
        _BOT_THREAD = threading.Thread(target=_run_bot_loop, daemon=True,
                                       name="signal-bot-loop")
        _BOT_THREAD.start()


def run() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=SETTINGS.log_level.lower())


if __name__ == "__main__":
    run()
