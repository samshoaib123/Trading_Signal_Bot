# Running the bot free on GitHub Actions

`signals.yml` runs `python main.py --once` every 15 minutes on GitHub's runners.
No server, no Railway, no VPS — and on a public repository, no cost.

Read the caveats before you rely on it. This trades reliability for price, and
the trade is real.

## Setup

1. **Make the repository public.** Actions minutes are unlimited on public repos.
   On a private repo the free allowance is 2,000 minutes/month and this workflow
   needs roughly 2,900 (about 96 runs/day at ~1 minute each), so it would stop
   partway through the month. Nothing here is secret — `.env` is gitignored and
   has never been committed, and credentials come from repository secrets.

2. **Merge this branch into `main`.** GitHub only runs scheduled workflows from
   the repository's **default branch**. On any other branch the cron is ignored
   entirely — the workflow will look installed and simply never fire.

3. **Add your credentials** under *Settings → Secrets and variables → Actions →
   Secrets*:

   | Secret | Value |
   |---|---|
   | `TELEGRAM_BOT_TOKEN` | Token from @BotFather |
   | `TELEGRAM_CHAT_ID` | Your id from @userinfobot |

4. **Check it works** before waiting on the schedule: *Actions → signals → Run
   workflow → mode: `preflight`*. That reports config, exchange reachability,
   candle download, indicators, state file and Telegram delivery, each with a
   fix for whatever failed. Then try mode `dry-run` for a real scan that sends
   nothing.

## Settings you can change without editing the workflow

Add these under *Settings → Secrets and variables → Actions → **Variables***
(not Secrets — these are not sensitive):

| Variable | Default | Purpose |
|---|---|---|
| `EXCHANGE_ID` | `kucoin` | GitHub runners are in US datacentres and `binance.com` blocks US IPs, so the default is not Binance. Try `okx`, `bybit` or `binanceus` if kucoin fails preflight. |
| `SYMBOLS` | bot default (10 majors) | Comma-separated pairs |
| `CAPITAL` / `RISK_PERCENT` | `1000` / `1` | Position sizing |
| `MIN_CONFIDENCE` | `1` | Raise to `2` or `3` for fewer, stronger alerts |
| `SIGNAL_COOLDOWN_MINUTES` | `45` | Minimum gap between repeats of one setup |

## The caveats, honestly

**The schedule is best-effort.** GitHub's cron is not a guarantee. Runs are
commonly 5–20 minutes late under load and are sometimes skipped entirely. The
bot only ever evaluates the last *closed* candle and de-duplicates by candle
timestamp, so a late run costs you a late alert, not a wrong one — but a delay
long enough to skip a candle means that candle's signal never arrives. If you
need alerts to land promptly and reliably, pay for the always-on option.

**Scheduled workflows get disabled after 60 days of no repository activity.**
GitHub emails you first. Any push re-enables it.

**State lives in the Actions cache, not on disk.** Each run restores
`signal_state.json` from the most recent cache entry and saves a new one, since
cache keys are immutable. A cache miss (eviction, a cache cleared by hand) costs
you at most one duplicate alert — verified: with state present a repeat scan of
the same candle sends nothing; with state missing it sends once.

**Every run is a cold start.** Roughly 40 seconds of that minute is checkout and
`pip install`, not scanning. That is why this needs ~2,900 minutes a month.

## Turning it off

*Actions → signals → `···` → Disable workflow*. Deleting the file works too.
Neither affects a Railway or VPS deployment of the same repo — but do not run
both at once against the same Telegram chat, or you will get every alert twice.
