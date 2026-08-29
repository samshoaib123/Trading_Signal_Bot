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
from typing import Iterable, List, Sequence, Tuple

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

    if getattr(settings, "beginner_mode", False):
        lines.append("")
        lines.extend(_beginner_block(signal))

    return "\n".join(lines)


def _beginner_block(signal: Signal) -> List[str]:
    """Plain-language instructions for someone who has never placed a trade.

    The numbers above are useless to a beginner who does not know that the stop
    is an *order* you place now, not a price you watch for. This spells out the
    three orders and, for a SELL, says outright that spot accounts cannot take
    it - which is the single most likely way a new user loses money following a
    signal they did not understand.
    """
    lines = ["<b>What to do</b>"]

    if signal.side == BUY:
        lines += [
            f"1. Buy near <code>{format_price(signal.entry)}</code> "
            "(skip it if price has already run far past this).",
            f"2. Immediately place a stop-loss order at "
            f"<code>{format_price(signal.stop_loss)}</code>.",
            f"3. Place a take-profit order at "
            f"<code>{format_price(signal.take_profit)}</code>.",
        ]
    else:
        lines += [
            "1. <b>Spot account? Skip this one.</b> A SELL means betting the "
            "price falls, which needs futures or margin. On spot it only "
            f"applies if you already hold {html.escape(signal.base_asset)}.",
            f"2. If you can short: enter near "
            f"<code>{format_price(signal.entry)}</code>.",
            f"3. Stop-loss <code>{format_price(signal.stop_loss)}</code>, "
            f"take-profit <code>{format_price(signal.take_profit)}</code>.",
        ]

    lines.append(
        "4. Then leave it alone. Both orders are set — moving a stop because "
        "the trade went against you is how small losses become large ones."
    )
    lines.append(
        "⚠️ <i>Never enter without the stop-loss order actually placed. "
        "This setup will lose often; only the size of each loss is in your "
        "control.</i>"
    )
    return lines


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


def format_digest_batches(
    signals: Sequence[Signal], settings=None
) -> List[Tuple[str, List[Signal]]]:
    """Group signals into Telegram-sized messages, keeping the mapping.

    Signals are concatenated with a visual separator until the next one would
    push the message past Telegram's 4096-character limit, then a new message is
    started. Each entry pairs the rendered message with the signals inside it, so
    the caller knows exactly which signals a failed send lost.
    """
    if not signals:
        return []

    reserve = len(DISCLAIMER) + 4
    batches: List[Tuple[str, List[Signal]]] = []
    blocks: List[str] = []
    batch: List[Signal] = []
    length = 0

    def flush():
        if blocks:
            batches.append(
                (SEPARATOR.join(blocks) + "\n\n" + DISCLAIMER, list(batch))
            )

    for signal in signals:
        block = format_signal(signal, settings)
        addition = len(block) + len(SEPARATOR)
        if blocks and length + addition + reserve > MAX_MESSAGE_CHARS:
            flush()
            blocks, batch, length = [], [], 0
        blocks.append(block)
        batch.append(signal)
        length += addition

    flush()

    if len(signals) > 1 and batches:
        header = (
            f"📊 <b>{len(signals)} new signals</b> · "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"
        )
        text, first = batches[0]
        batches[0] = (header + "\n\n" + text, first)
    return batches


def format_digest(signals: Sequence[Signal], settings=None) -> List[str]:
    """Rendered messages only — the text half of :func:`format_digest_batches`."""
    return [text for text, _ in format_digest_batches(signals, settings)]


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

    def send_signals(self, signals: Iterable[Signal]) -> List[Signal]:
        """Format and send a batch of signals.

        Returns the signals that actually reached Telegram. A batch can split
        across several messages and any one of them can fail on its own, so
        returning the delivered subset lets the caller record only those as
        sent - a signal in a failed message is then retried next cycle instead
        of being silently suppressed by de-duplication.
        """
        signals = list(signals)
        if not signals:
            return []

        delivered: List[Signal] = []
        for message, batch in format_digest_batches(signals, self.settings):
            if self.send(message):
                delivered.extend(batch)
            else:
                LOG.error(
                    "Telegram rejected the message carrying %d signal(s): %s",
                    len(batch), ", ".join(s.dedupe_key for s in batch),
                )
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


# ---------------------------------------------------------------------------
# Outcome and scoreboard messages
# ---------------------------------------------------------------------------
def format_outcome(outcome, ledger=None) -> str:
    """Report a position that reached its stop or target.

    Closing the loop matters more than it looks: a bot that only announces
    entries can never be judged, and a beginner has no way to tell a losing
    streak from a broken bot.
    """
    won = outcome.result == "win"
    icon = "✅" if won else "❌"
    verdict = "Target hit" if won else "Stop hit"
    esc = html.escape

    lines = [
        f"{icon} <b>{esc(verdict)}</b> — {esc(outcome.symbol)}",
        "",
        f"<b>Setup:</b> {esc(outcome.setup_label)} ({esc(outcome.side)})",
        f"<b>Entry:</b> <code>{format_price(outcome.entry)}</code>",
        f"<b>Exit:</b> <code>{format_price(outcome.exit_price)}</code> "
        f"({outcome.pct:+.2f}%)",
        f"<b>Result:</b> {outcome.r_multiple:+.2f}R after fees",
        f"<b>Held:</b> {outcome.candles_held} candle"
        f"{'s' if outcome.candles_held != 1 else ''}",
    ]

    if ledger is not None:
        from tracker import scoreboard

        wins, losses, total_r, rate = scoreboard(ledger)
        if wins + losses:
            lines += [
                "",
                f"<b>Record so far:</b> {wins}W / {losses}L "
                f"({rate:.0f}% win rate), {total_r:+.1f}R total",
            ]
    return "\n".join(lines)


def format_scoreboard(ledger, settings=None) -> str:
    """Cumulative results across every signal the bot has sent."""
    from tracker import scoreboard

    wins, losses, total_r, rate = scoreboard(ledger)
    closed = wins + losses
    open_count = len(ledger.get("open") or [])

    if not closed:
        return (
            "📋 <b>Scoreboard</b>\n\n"
            f"No positions have closed yet. {open_count} still open.\n"
            "<i>Results appear here as signals reach their stop or target.</i>"
        )

    risk_pct = getattr(settings, "risk_percent", 1.0) if settings else 1.0
    lines = [
        "📋 <b>Scoreboard</b>",
        "",
        f"<b>Closed:</b> {closed}  ({wins}W / {losses}L)",
        f"<b>Win rate:</b> {rate:.1f}%",
        f"<b>Total:</b> {total_r:+.2f}R after fees",
        f"<b>Average:</b> {total_r / closed:+.3f}R per signal",
        f"<b>Still open:</b> {open_count}",
    ]
    if risk_pct:
        lines.append(
            f"\n<i>At {risk_pct:g}% risk per trade that is roughly "
            f"{total_r * risk_pct:+.1f}% on the account, before compounding.</i>"
        )
    if total_r < 0:
        lines.append(
            "\n⚠️ <i>These signals have lost money so far. That is real "
            "information — do not increase your size to win it back.</i>"
        )
    return "\n".join(lines)
