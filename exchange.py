"""Binance (or any other ccxt) market data access with retries.

Only public endpoints are used, so no API key is required and no order can ever
be placed by this bot.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable, List, Optional

import ccxt
import pandas as pd

from indicators import OHLCV_COLUMNS

LOG = logging.getLogger(__name__)

# ccxt exceptions that are worth retrying: transient network/venue problems.
RETRYABLE = (
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RequestTimeout,
    ccxt.DDoSProtection,
    ccxt.RateLimitExceeded,
    ccxt.OnMaintenance,
)


class ExchangeError(RuntimeError):
    """Raised when the exchange cannot be used at all."""


def create_exchange(settings) -> ccxt.Exchange:
    """Instantiate a rate-limited, read-only ccxt exchange client.

    ``enableRateLimit=True`` makes ccxt space out requests according to the
    venue's published limits, which is the simplest correct way to stay under
    Binance's weight budget when polling ten symbols every 15 minutes.
    """
    exchange_id = settings.exchange_id
    if not hasattr(ccxt, exchange_id):
        raise ExchangeError(
            f"Unknown EXCHANGE_ID={exchange_id!r}. Pick any ccxt id, e.g. "
            "'binance', 'binanceus', 'kucoin', 'okx', 'bybit'."
        )
    klass = getattr(ccxt, exchange_id)
    exchange = klass(
        {
            "enableRateLimit": True,
            "timeout": 30_000,
            "options": {"defaultType": "spot", "adjustForTimeDifference": True},
        }
    )
    LOG.info("Using exchange %s (rate limit %s ms)", exchange.id, exchange.rateLimit)
    return exchange


def load_valid_symbols(exchange: ccxt.Exchange, symbols: Iterable[str]) -> List[str]:
    """Drop symbols the exchange does not list, with a warning for each.

    Listings change (MATIC/USDT became POL/USDT on Binance, for example). A
    missing pair should cost one log line, not crash the whole bot.
    """
    requested = list(symbols)
    try:
        markets = exchange.load_markets()
    except Exception as exc:  # noqa: BLE001 - we degrade rather than crash
        LOG.warning(
            "Could not load markets from %s (%s); using the configured symbol "
            "list as-is.",
            exchange.id,
            exc,
        )
        return requested

    valid, invalid = [], []
    for symbol in requested:
        market = markets.get(symbol)
        if market is None:
            invalid.append(symbol)
        elif market.get("active") is False:
            invalid.append(symbol)
        else:
            valid.append(symbol)

    if invalid:
        LOG.warning(
            "Skipping %d symbol(s) not tradable on %s: %s",
            len(invalid),
            exchange.id,
            ", ".join(invalid),
        )
    if not valid:
        raise ExchangeError(
            f"None of the configured symbols are tradable on {exchange.id}: "
            f"{', '.join(requested)}"
        )
    return valid


def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    settings,
    drop_unclosed: bool = True,
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV candles for one symbol and return them as a DataFrame.

    Args:
        exchange: A client from :func:`create_exchange`.
        symbol: Market symbol, e.g. ``"BTC/USDT"``.
        settings: A :class:`config.Settings` instance.
        drop_unclosed: Binance returns the *currently forming* candle as the
            last row. Its close price is not final, so evaluating setups on it
            would produce alerts that vanish moments later. When ``True`` (the
            default) that row is discarded and every downstream consumer can
            assume ``df.iloc[-1]`` is a closed candle.

    Returns:
        A UTC-indexed OHLCV frame, or ``None`` when every retry failed.
    """
    delay = settings.fetch_backoff_seconds
    last_error: Optional[Exception] = None

    for attempt in range(1, settings.fetch_retries + 1):
        try:
            raw = exchange.fetch_ohlcv(
                symbol, timeframe=settings.timeframe, limit=settings.candle_limit
            )
            if not raw:
                LOG.warning("%s: exchange returned no candles", symbol)
                return None

            df = pd.DataFrame(raw, columns=OHLCV_COLUMNS)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp", drop=False).sort_index()
            df = df[~df.index.duplicated(keep="last")]

            if drop_unclosed and len(df) > 1:
                df = df.iloc[:-1]

            LOG.debug(
                "%s: %d closed candles, last close %s @ %s",
                symbol,
                len(df),
                df["close"].iloc[-1],
                df.index[-1],
            )
            return df

        except ccxt.BadSymbol as exc:
            LOG.error("%s: not listed on %s (%s)", symbol, exchange.id, exc)
            return None
        except RETRYABLE as exc:
            last_error = exc
            if attempt == settings.fetch_retries:
                break
            LOG.warning(
                "%s: fetch attempt %d/%d failed (%s: %s); retrying in %.1fs",
                symbol,
                attempt,
                settings.fetch_retries,
                type(exc).__name__,
                exc,
                delay,
            )
            time.sleep(delay)
            delay *= 2  # exponential backoff
        except ccxt.ExchangeError as exc:
            # Includes HTTP 451 (geo-restricted) which retrying will not fix.
            LOG.error(
                "%s: exchange rejected the request (%s: %s). If this is a 451, "
                "your host's region is geo-blocked by %s - try EXCHANGE_ID="
                "binanceus, kucoin or okx, or deploy in another region.",
                symbol,
                type(exc).__name__,
                exc,
                exchange.id,
            )
            return None

    LOG.error(
        "%s: giving up after %d attempts (%s)",
        symbol,
        settings.fetch_retries,
        last_error,
    )
    return None
