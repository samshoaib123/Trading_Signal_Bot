"""Persistent de-duplication state.

The bot re-evaluates every pair every 15 minutes, and a restart re-evaluates
the same closed candle it may already have alerted on. Both would produce
duplicate Telegram messages, so the last alert for each ``pair|setup|side`` key
is written to a small JSON file (``STATE_FILE``, default ``signal_state.json``)
that survives restarts.

A signal is suppressed when either:

* it belongs to the same candle we already alerted on (restart-safety), or
* the previous alert for that key is newer than ``SIGNAL_COOLDOWN_MINUTES``.

The file is written atomically (temp file + ``os.replace``) so a crash mid-write
cannot leave truncated JSON behind.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Dict

LOG = logging.getLogger(__name__)

STATE_VERSION = 1


def load_state(path: str) -> Dict[str, dict]:
    """Read the de-duplication state, returning ``{}` when unusable.

    A missing file is normal on first run. A corrupt file is logged and treated
    as empty rather than crashing the bot - the worst case is one duplicate
    alert, which beats a crash loop.
    """
    if not os.path.exists(path):
        LOG.info("No state file at %s yet; starting fresh", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("Could not read state file %s (%s); starting fresh", path, exc)
        return {}

    if not isinstance(payload, dict):
        LOG.warning("State file %s has an unexpected shape; starting fresh", path)
        return {}

    signals = payload.get("signals", {})
    if not isinstance(signals, dict):
        return {}
    LOG.info("Loaded %d de-duplication entries from %s", len(signals), path)
    return signals


def save_state(path: str, signals: Dict[str, dict]) -> None:
    """Atomically persist the de-duplication state."""
    payload = {
        "version": STATE_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "signals": signals,
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
        LOG.error("Failed to persist state to %s: %s", path, exc)


def should_send(signal, state: Dict[str, dict], settings) -> bool:
    """Decide whether ``signal`` is new enough to be worth an alert."""
    entry = state.get(signal.dedupe_key)
    if not entry:
        return True

    candle_iso = signal.candle_time.isoformat()
    if entry.get("candle_time") == candle_iso:
        LOG.debug("%s: already alerted for candle %s", signal.dedupe_key, candle_iso)
        return False

    sent_at = _parse_dt(entry.get("sent_at"))
    if sent_at is None:
        return True

    cooldown = timedelta(minutes=settings.signal_cooldown_minutes)
    age = datetime.now(timezone.utc) - sent_at
    if age < cooldown:
        LOG.info(
            "%s: suppressed, last alert was %.0f min ago (cooldown %d min)",
            signal.dedupe_key,
            age.total_seconds() / 60,
            settings.signal_cooldown_minutes,
        )
        return False
    return True


def record_signal(signal, state: Dict[str, dict]) -> None:
    """Mark ``signal`` as alerted so later runs suppress repeats."""
    state[signal.dedupe_key] = {
        "symbol": signal.symbol,
        "setup": signal.setup,
        "side": signal.side,
        "entry": signal.entry,
        "confidence": signal.confidence,
        "candle_time": signal.candle_time.isoformat(),
        "sent_at": datetime.now(timezone.utc).isoformat(),
    }


def prune_state(state: Dict[str, dict], retention_days: int) -> Dict[str, dict]:
    """Drop entries older than ``retention_days`` to keep the file small."""
    if retention_days <= 0:
        return state
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    kept = {}
    for key, entry in state.items():
        sent_at = _parse_dt(entry.get("sent_at"))
        if sent_at is None or sent_at >= cutoff:
            kept[key] = entry
    dropped = len(state) - len(kept)
    if dropped:
        LOG.debug("Pruned %d stale de-duplication entries", dropped)
    return kept


def _parse_dt(value) -> "datetime | None":
    """Parse an ISO-8601 string into an aware UTC datetime, or ``None``."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
