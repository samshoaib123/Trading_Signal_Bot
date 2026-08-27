"""Telegram delivery and message formatting.

Uses ``python-telegram-bot`` v20+ (async API). The rest of the bot is plain
synchronous code, so :class:`TelegramNotifier` exposes a blocking ``send()``
that drives the async client through :func:`asyncio.run`.
"""

from __future__ import annotations

import asyncio
import html
import logging
import math
from datetime import datetime, timezone
from typing import Iterable, List, Sequence

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import Forbidden, InvalidToken, RetryAfter, TelegramError

from strategies import BUY, MAX_CONFIDENCE, Signal

LOG = logging.getLogger(__name__)

# Telegram hard-limits a text message to 4096 characters.
MAX_MESSAGE_CHARS = 4000
SEPARATOR = "\n➖➖➖➖➖➖➖➖➖➖\n\n"
DISCLAIMER = "<i>Signals only — not financial advice. Always manage your own risk.</i>"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def format_price(value: float) -> str:
    """Format a price with a sensible number of decimals for its magnitude.

    BTC at 65,000 wants 2 decimals; SHIB at 0.000012 wants 8. Picking by
    magnitude avoids both "65000.00000000" and "0.00".
    """
    if value is None or not math.isfinite(value):
        return "n/a"
    magnitude = abs(value)
    if magnitude >= 1000:
        decimals = 2
    elif magnitude >= 10:
        decimals = 3
    elif magnitude >= 1:
        decimals = 4
    elif magnitude >= 0.01:
        decimals = 6
    else:
        decimals = 8
    return f"{value:,.{decimals}f}"


def format_quantity(value: float) -> str:
    """Format a base-asset quantity, keeping small sizes readable."""
    if value is None or not math.isfinite(value):
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    if abs(value) >= 1:
        return f"{value:,.4f}"
    return f"{value:.8f}".rstrip("0").rstrip(".")


def confidence_stars(confidence: int) -> str:
    """``2`` -> ``★★☆ (2/3)``."""
    filled = max(0, min(MAX_CONFIDENCE, confidence))
    return f"{'★' * filled}{'☆' * (MAX_CONFIDENCE - filled)} ({filled}/{MAX_CONFIDENCE})"


def _pct(from_price: float, to_price: float) -> str:
    if not from_price:
        return ""
    change = (to_price - from_price) / from_price * 100.0
    return f"{change:+.2f}%"


def format_signal(signal: Signal, settings=None) -> str:
    """Render one signal as a Telegram HTML message body.

    Args:
        signal: The signal to render.
        settings: Optional :class:`config.Settings`; when given it is used to
            warn if the risk-based position size implies more notional than the
            configured capital.
    """
    emoji = "🟢 🚀" if signal.side == BUY else "🔴 🔻"
    esc = html.escape

    lines = [
        f"{emoji} <b>{esc(signal.side)} Signal</b>",
        "",
        f"<b>Pair:</b> {esc(signal.symbol)}",
        f"<b>Setup:</b> {esc(signal.setup_label)}",
        f"<b>Entry:</b> <code>{format_price(signal.entry)}</code>",
        f"<b>Stop Loss:</b> <code>{format_price(signal.stop_loss)}</code> "
        f"({_pct(signal.entry, signal.stop_loss)})",
        f"<b>Take Profit:</b> <code>{format_price(signal.take_profit)}</code> "
        f"({_pct(signal.entry, signal.take_profit)})",
        f"<b>Risk : Reward:</b> 1 : {signal.risk_reward:.2f}",
        f"<b>Timeframe:</b> {esc(signal.timeframe)}",
        f"<b>Candle close:</b> "
        f"{signal.candle_close_time.strftime('%Y-%m-%d %H:%M')} UTC",
        f"<b>Confidence:</b> {confidence_stars(signal.confidence)}",
    ]

    if signal.confirmations:
        lines.append(
            "<b>Confirmations:</b> "
            + esc(", ".join(signal.confirmations))
        )

    if signal.position_size:
        size = format_quantity(signal.position_size)
        notional = signal.position_notional or 0.0
        risk = signal.risk_amount or 0.0
        quote = esc(signal.quote_asset)
        lines.append(
            f"<b>Suggested size:</b> {size} {esc(signal.base_asset)} "
            f"(≈ {notional:,.2f} {quote} notional, risking {risk:,.2f} {quote})"
        )
        hint = _leverage_hint(notional, settings)
        if hint:
            lines.append(hint)

    return "\n".join(lines)


def _leverage_hint(notional: float, settings) -> str:
    """Warn when the risk-based size implies more notional than the capital.

    A 1.5x ATR stop on a 15m candle is often well under 1% wide, so risking 1%
    of the account can call for a position several times the account size. That
    is only reachable with leverage, and saying so is more useful than silently
    printing an impossible quantity.
    """
    capital = getattr(settings, "capital", 0.0) if settings else 0.0
    if capital <= 0 or notional <= capital:
        return ""
    return (
        f"⚠️ <i>Notional is {notional / capital:.1f}x your "
        f"{capital:,.0f} capital — needs leverage, or lower RISK_PERCENT.</i>"
    )


def format_digest(signals: Sequence[Signal], settings=None) -> List[str]:
    """Group signals into one or more Telegram-sized messages.

    Signals are concatenated with a visual separator until the next one would
    push the message past Telegram's 4096-character limit, then a new message
    is started.
    """
    if not signals:
        return []

    blocks = [format_signal(s, settings) for s in signals]
    reserve = len(DISCLAIMER) + 4

    messages: List[str] = []
    current: List[str] = []
    current_len = 0

    for block in blocks:
        addition = len(block) + len(SEPARATOR)
        if current and current_len + addition + reserve > MAX_MESSAGE_CHARS:
            messages.append(SEPARATOR.join(current) + "\n\n" + DISCLAIMER)
            current, current_len = [], 0
        current.append(block)
        current_len += addition

    if current:
        messages.append(SEPARATOR.join(current) + "\n\n" + DISCLAIMER)

    if len(signals) > 1 and messages:
        header = (
            f"📊 <b>{len(signals)} new signals</b> · "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"
        )
        messages[0] = header + "\n\n" + messages[0]
    return messages


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------
class TelegramNotifier:
    """Blocking wrapper around the async ``python-telegram-bot`` client."""

    def __init__(self, settings, dry_run: bool = False) -> None:
        self.settings = settings
        self.dry_run = dry_run
        self.max_attempts = 3

    def send(self, text: str) -> bool:
        """Send one HTML message. Returns ``True`` when Telegram accepted it."""
        if self.dry_run:
            LOG.info("[dry-run] would send Telegram message:\n%s", text)
            return True
        try:
            return asyncio.run(self._send_async(text))
        except (InvalidToken, Forbidden) as exc:
            # Unrecoverable: bad token, or the user never pressed /start.
            LOG.error(
                "Telegram rejected the credentials (%s). Check TELEGRAM_BOT_TOKEN "
                "and make sure you sent /start to the bot from TELEGRAM_CHAT_ID.",
                exc,
            )
            return False
        except Exception as exc:  # noqa: BLE001 - never kill the loop over an alert
            LOG.exception("Unexpected Telegram failure: %s", exc)
            return False

    def send_signals(self, signals: Iterable[Signal]) -> int:
        """Format and send a batch of signals. Returns messages delivered."""
        signals = list(signals)
        if not signals:
            return 0
        delivered = 0
        for message in format_digest(signals, self.settings):
            if self.send(message):
                delivered += 1
        return delivered

    async def _send_async(self, text: str) -> bool:
        """One message with retry/backoff, honouring Telegram's flood control.

        The client is built inside the retry loop on purpose: entering the
        ``Bot`` context manager performs a ``getMe`` call, so a network blip at
        that moment has to be retried just like a failed ``sendMessage``.
        """
        delay = 2.0
        for attempt in range(1, self.max_attempts + 1):
            try:
                async with Bot(self.settings.telegram_bot_token) as bot:
                    await bot.send_message(
                        chat_id=self.settings.telegram_chat_id,
                        text=text,
                        parse_mode=ParseMode.HTML,
                        disable_web_page_preview=True,
                    )
                return True
            except RetryAfter as exc:
                wait = float(getattr(exc, "retry_after", delay)) + 1.0
                LOG.warning("Telegram flood control: sleeping %.1fs", wait)
                await asyncio.sleep(wait)
            except (InvalidToken, Forbidden):
                raise
            except TelegramError as exc:
                if attempt == self.max_attempts:
                    LOG.error(
                        "Telegram send failed after %d attempts (%s: %s)",
                        self.max_attempts, type(exc).__name__, exc,
                    )
                    return False
                LOG.warning(
                    "Telegram send attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt, self.max_attempts, exc, delay,
                )
                await asyncio.sleep(delay)
                delay *= 2
        return False


def send_telegram(settings, text: str, dry_run: bool = False) -> bool:
    """Module-level convenience wrapper used by scripts and tests."""
    return TelegramNotifier(settings, dry_run=dry_run).send(text)
