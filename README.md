# Crypto Trading Signal Bot

A production-ready signal bot that watches Binance 15-minute candles for three
classic technical setups and pushes alerts to Telegram — with ATR-based stop
loss / take profit, a 1–3 confidence score, and a suggested position size.

**It never places an order and never needs an exchange API key.** Public ccxt
endpoints only.

```
🟢 🚀 BUY Signal

Pair: BTC/USDT
Setup: RSI Reversal
Entry: 65,000.00
Stop Loss: 64,200.00 (-1.23%)
Take Profit: 66,600.00 (+2.46%)
Risk : Reward: 1 : 1.33
Timeframe: 15m
Candle close: 2025-04-10 14:15 UTC
Confidence: ★★☆ (2/3)
Confirmations: Volume spike, Trend aligned (EMA200)
Suggested size: 0.0125 BTC (≈ 812.50 USDT notional, risking 10.00 USDT)
```

---

## Contents

- [How it works](#how-it-works)
- [The three setups](#the-three-setups)
- [Risk levels and position sizing](#risk-levels-and-position-sizing)
- [Confidence score](#confidence-score)
- [Deduplication](#deduplication)
- [Quick start (local)](#quick-start-local)
- [Getting your Telegram token and chat id](#getting-your-telegram-token-and-chat-id)
- [Deploy on Railway](#deploy-on-railway)
- [Deploy on an Ubuntu VPS (systemd)](#deploy-on-an-ubuntu-vps-systemd)
- [Deploy with Docker / docker-compose](#deploy-with-docker--docker-compose)
- [Environment variables](#environment-variables)
- [Project layout](#project-layout)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)

---

## How it works

Every 15 minutes, aligned to the candle close (`:00`, `:15`, `:30`, `:45` UTC
plus a small buffer), the bot:

1. **Fetches** 300 × 15m OHLCV candles per pair from Binance via `ccxt`, with
   `enableRateLimit` on and exponential-backoff retries.
2. **Discards the still-forming candle** — only *closed* candles are evaluated,
   so alerts never flip-flop mid-candle.
3. **Computes** RSI(14), MACD(12,26,9), Bollinger Bands(20,2), ATR(14), a trend
   EMA and a volume SMA.
4. **Detects** the three setups on the last closed candle.
5. **Sizes** the trade off ATR and **scores** confidence from 1 to 3.
6. **De-duplicates** against `signal_state.json`, then **sends** whatever is new
   to Telegram (batched into one message when several fire at once).

Anything that goes wrong with one pair is logged and skipped; the loop itself
never dies.

### Indicator backend

Indicators are implemented directly on top of pandas in `indicators.py` — no
TA-Lib, no C toolchain, no extra wheels. This matters more than it used to:
`pandas-ta` now only publishes wheels for **Python ≥ 3.12**, and the old
`0.3.14b0` release that most tutorials pin has been removed from PyPI, so
`pip install pandas-ta==0.3.14b0` fails outright on Python 3.11 and below.

If you want `pandas-ta` anyway (Python 3.12+ only):

```bash
pip install -r requirements-optional.txt
export INDICATOR_BACKEND=pandas_ta
```

Both backends use Wilder's smoothing for RSI/ATR and agree to floating-point
noise. `INDICATOR_BACKEND=auto` (the default) uses `pandas-ta` when it imports
cleanly and silently falls back to the built-in implementations otherwise.

---

## The three setups

All three require a **fresh cross**: the condition must be false on the previous
closed candle and true on the current one. A price riding the upper band, or an
RSI parked above 30, therefore fires **once**, not every 15 minutes.

| Setup | BUY trigger | SELL trigger |
|---|---|---|
| `rsi_reversal` | RSI(14) was ≤ 30, closes back above 30 | RSI(14) was ≥ 70, closes back below 70 |
| `macd_crossover` | MACD line crosses **above** its signal line | MACD line crosses **below** its signal line |
| `bb_breakout` | Close breaks **above** the upper band(20, 2) | Close breaks **below** the lower band(20, 2) |

Turn individual setups off with `ENABLED_SETUPS=rsi_reversal,macd_crossover`.

---

## Risk levels and position sizing

Stops and targets are volatility-adjusted with ATR(14), so a quiet pair gets a
tight stop and a volatile one gets room to breathe:

| Side | Stop loss | Take profit |
|---|---|---|
| BUY | `entry − 1.5 × ATR` | `entry + 2.0 × ATR` |
| SELL | `entry + 1.5 × ATR` | `entry − 2.0 × ATR` |

That is a fixed **1 : 1.33** risk-to-reward. Both multipliers are configurable
(`ATR_SL_MULTIPLIER`, `ATR_TP_MULTIPLIER`).

Position size uses the standard fixed-fractional formula:

```
size = (CAPITAL × RISK_PERCENT / 100) / |entry − stop_loss|
```

With the defaults (`CAPITAL=1000`, `RISK_PERCENT=1`) you risk $10 per trade.

> **Note on notional.** A 1.5×ATR stop on a 15m candle is often well under 1%
> wide, so risking 1% of the account can call for a position several times the
> account size. When that happens the message says so explicitly
> (`⚠️ Notional is 3.2x your 1,000 capital — needs leverage, or lower
> RISK_PERCENT`) instead of quietly printing an impossible quantity.

Set `SHOW_POSITION_SIZE=false` to drop the sizing line entirely.

---

## Confidence score

Every signal starts at **1/3** for the trigger itself. Each setup then runs
three confirmation checks that are independent of its own trigger (so no point
is ever awarded for something the setup already guarantees), and each passing
check adds a point, capped at 3:

| Setup | Check 1 | Check 2 | Check 3 |
|---|---|---|---|
| `rsi_reversal` | Volume ≥ 1.5 × SMA(20) | Price on the right side of EMA(200) | MACD histogram turning in your direction |
| `macd_crossover` | Volume ≥ 1.5 × SMA(20) | Price on the right side of EMA(200) | RSI has room (< 70 for BUY, > 30 for SELL) |
| `bb_breakout` | Volume ≥ 1.5 × SMA(20) | Price on the right side of EMA(200) | MACD histogram aligned |

The passing checks are named in the alert. A check whose inputs are still
warming up counts as "not confirmed" rather than failing the signal. Use
`MIN_CONFIDENCE=2` (or `3`) to only get the stronger setups.

The trend EMA steps down to 100 or 50 automatically if a pair has less history
than `TREND_EMA_PERIOD` candles, so the trend check never silently disappears.

---

## Deduplication

State lives in `signal_state.json` (path configurable via `STATE_FILE`) and is
keyed by `pair | setup | side`. A signal is suppressed when either:

- it is the **same candle** already alerted on — this is what makes restarts
  safe, since a restart re-evaluates the same closed candle, or
- the previous alert for that key is newer than `SIGNAL_COOLDOWN_MINUTES`
  (default 45, i.e. three 15m candles).

The file is written atomically (temp file + `os.replace`), and **state is only
recorded after Telegram accepts the message** — if delivery fails, the next
cycle retries instead of dropping the alert. Entries older than
`STATE_RETENTION_DAYS` are pruned automatically.

On Railway and Docker, put the state file on a persistent volume (see below) or
you may get one duplicate alert per redeploy.

---

## Quick start (local)

Requires **Python 3.10+** (3.12 recommended).

```bash
git clone https://github.com/samshoaib123/Trading_Signal_Bot.git
cd Trading_Signal_Bot

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID

python main.py --preflight         # 1. check EVERYTHING at once (see below)
python main.py --once --dry-run    # 2. one real scan, messages logged not sent
python main.py                     # 3. run for real
```

### CLI flags

| Flag | What it does |
|---|---|
| *(none)* | Run forever, aligned to 15m candle closes |
| `--once` | Single scan then exit — handy for cron or a smoke test |
| `--dry-run` | Scan normally but log Telegram messages instead of sending |
| `--test-telegram` | Send one test message and exit |
| `--preflight` | Check config, exchange, candles, indicators, state file and Telegram, print a report, exit |

Exit codes: `0` success, `1` a check failed, `2` configuration error.

### `--preflight`

The single command to run before (and after) deploying. It walks every
dependency in order and tells you exactly what to fix:

```
========================================================================
PREFLIGHT REPORT
========================================================================
✅ Configuration      10 symbols, 15m, risk 1.0% of 1,000
✅ Exchange markets   binance reachable, 9 tradable symbol(s)
✅ Candle download    299 closed 15m candles for BTC/USDT, last close 64980.5
✅ Indicators         pandas backend | RSI 46.2 | ATR 121.4 | trend EMA200
✅ State file         signal_state.json writable (new file)
✅ Telegram delivery  test message sent
========================================================================
All checks passed. You are ready to deploy: python main.py
========================================================================
```

Failures print a `->` line with the fix. Checks are independent, so one run
reports every problem rather than only the first. Exit code is `0` only when
nothing failed, which makes it usable in a deploy script.

It deliberately calls `load_markets()` itself instead of reusing the bot's
degrade-gracefully symbol loader — otherwise an unreachable exchange would be
reported as healthy.

---

## Getting your Telegram token and chat id

1. Open Telegram, message **@BotFather**, send `/newbot`, follow the prompts.
   Copy the token — it looks like `123456789:AAE...`. That is
   `TELEGRAM_BOT_TOKEN`.
2. **Send `/start` to your new bot.** Telegram will not let a bot message you
   until you do; skipping this gives a `Forbidden: bot can't initiate
   conversation` error.
3. Message **@userinfobot** to get your numeric user id. That is
   `TELEGRAM_CHAT_ID`.
4. For a **group**: add the bot to the group, send any message there, then open
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` and read
   `message.chat.id` — group ids start with `-100`.

---

## Deploy on Railway

Railway's free tier is enough for this bot (it is idle 99% of the time).

1. **Push this repo to GitHub** (public or private, both work).

2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from
   GitHub repo** → pick your fork. Railway auto-detects Python via Nixpacks and
   uses `requirements.txt`; `railway.json` in this repo pins the start command
   and the restart policy.

3. Open the service → **Variables** → add at minimum:

   ```
   TELEGRAM_BOT_TOKEN = 123456789:AAE...
   TELEGRAM_CHAT_ID   = 123456789
   ```

   Add any of the [environment variables](#environment-variables) you want to
   override. Do **not** upload `.env` — Railway injects real env vars.

4. **Important — this is a worker, not a web service.** Railway only assigns a
   port to services that listen on one; this bot does not. Under
   **Settings → Networking**, leave the public domain **unassigned**. If Railway
   marks the deploy unhealthy waiting for a port, remove any healthcheck path
   under **Settings → Deploy**.

5. **Persist the state file** so redeploys do not replay old signals:
   **Settings → Volumes → Add Volume**, mount path `/data`, then set
   `STATE_FILE=/data/signal_state.json` in Variables. (Skip this and the worst
   case is one duplicate alert after each deploy.)

6. Watch **Deployments → View Logs**. You should see
   `Crypto Trading Signal Bot starting up`, the pair list, and a Telegram
   startup message in your chat.

   If nothing arrives, temporarily set the start command to
   `python main.py --preflight` (**Settings → Deploy → Custom Start Command**),
   redeploy, and read the report in the logs — it names the exact problem.
   Change it back to `python main.py` afterwards.

> **If you see HTTP 451 in the logs**, Railway's region is geo-blocked by
> `binance.com`. Set `EXCHANGE_ID=binanceus`, `kucoin`, `okx` or `bybit` — the
> bot is exchange-agnostic through ccxt. See
> [Troubleshooting](#troubleshooting).

**CLI alternative:**

```bash
npm i -g @railway/cli
railway login
railway init
railway variables --set TELEGRAM_BOT_TOKEN=... --set TELEGRAM_CHAT_ID=...
railway up
```

---

## Deploy on an Ubuntu VPS (systemd)

Tested on Ubuntu 22.04 / 24.04. Any $5 VPS is plenty.

```bash
# 1. System packages
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git

# 2. A dedicated unprivileged user
sudo useradd --system --create-home --shell /usr/sbin/nologin botuser

# 3. Code
sudo mkdir -p /opt/trading-signal-bot
sudo chown botuser:botuser /opt/trading-signal-bot
sudo -u botuser git clone https://github.com/samshoaib123/Trading_Signal_Bot.git \
    /opt/trading-signal-bot

# 4. Virtualenv
cd /opt/trading-signal-bot
sudo -u botuser python3 -m venv .venv
sudo -u botuser .venv/bin/pip install --upgrade pip
sudo -u botuser .venv/bin/pip install -r requirements.txt

# 5. Configuration (chmod 600 — it holds your bot token)
sudo -u botuser cp .env.example .env
sudo -u botuser nano .env
sudo chmod 600 .env

# 6. Smoke test before installing the service
sudo -u botuser .venv/bin/python main.py --preflight
sudo -u botuser .venv/bin/python main.py --once --dry-run

# 7. Install and start the service
sudo cp deploy/trading-signal-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trading-signal-bot

# 8. Verify
sudo systemctl status trading-signal-bot
sudo journalctl -u trading-signal-bot -f
```

The unit restarts the bot automatically (`Restart=always`, 15s backoff), starts
it on boot, and applies basic hardening (`NoNewPrivileges`, `ProtectSystem=strict`,
`ProtectHome`, write access limited to its own directory).

**Updating:**

```bash
cd /opt/trading-signal-bot
sudo -u botuser git pull
sudo -u botuser .venv/bin/pip install -r requirements.txt
sudo systemctl restart trading-signal-bot
```

**Keep the VPS clock accurate** — candle alignment depends on it:

```bash
timedatectl set-ntp true && timedatectl
```

---

## Deploy with Docker / docker-compose

```bash
cp .env.example .env      # fill in your token and chat id
docker compose up -d --build
docker compose logs -f
```

The compose file mounts a named volume at `/app/data` and sets
`STATE_FILE=/app/data/signal_state.json`, so deduplication state survives
`docker compose down`. Logs are capped at 3 × 10 MB. The container runs as a
non-root user and restarts unless explicitly stopped.

The image also sets `HEARTBEAT_FILE=/app/data/heartbeat`, which the bot touches
after every scan cycle. The container `HEALTHCHECK` reports unhealthy if that
file goes stale for two cycles — it checks the bot itself, so it stays correct
whatever `EXCHANGE_ID` you pick and costs no API calls. Check it with
`docker inspect --format '{{.State.Health.Status}}' crypto-signal-bot`.

Plain Docker:

```bash
docker build -t crypto-signal-bot .
docker run -d --name signal-bot --restart unless-stopped \
  --env-file .env \
  -e STATE_FILE=/app/data/signal_state.json \
  -v signal-state:/app/data \
  crypto-signal-bot
```

Useful commands:

```bash
docker compose restart              # restart after an .env change
docker compose up -d --build        # rebuild after a code change
docker compose down                 # stop (state volume is kept)
```

---

## Environment variables

Only the first two are required. Everything else has a working default — see
`.env.example` for the same list in copy-paste form.

### Telegram

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | — | **Required.** Token from @BotFather |
| `TELEGRAM_CHAT_ID` | — | **Required.** Your user id, or a `-100…` group id |
| `SEND_STARTUP_MESSAGE` | `true` | Send a config summary on boot |

### Market data

| Variable | Default | Description |
|---|---|---|
| `EXCHANGE_ID` | `binance` | Any ccxt id: `binanceus`, `kucoin`, `okx`, `bybit`… |
| `SYMBOLS` | 10 majors | Comma-separated, e.g. `BTC/USDT,ETH/USDT` |
| `TIMEFRAME` | `15m` | Any timeframe the exchange supports |
| `CANDLE_LIMIT` | `300` | Candles per fetch (must exceed your longest period) |
| `FETCH_RETRIES` | `4` | Attempts per pair before giving up on this cycle |
| `FETCH_BACKOFF_SECONDS` | `2` | Initial backoff; doubles each retry |

### Indicators

| Variable | Default | | Variable | Default |
|---|---|---|---|---|
| `RSI_PERIOD` | `14` | | `BB_PERIOD` | `20` |
| `RSI_OVERSOLD` | `30` | | `BB_STD` | `2.0` |
| `RSI_OVERBOUGHT` | `70` | | `ATR_PERIOD` | `14` |
| `MACD_FAST` | `12` | | `TREND_EMA_PERIOD` | `200` |
| `MACD_SLOW` | `26` | | `VOLUME_SMA_PERIOD` | `20` |
| `MACD_SIGNAL` | `9` | | `VOLUME_SPIKE_MULTIPLIER` | `1.5` |
| `INDICATOR_BACKEND` | `auto` | | | |

### Risk and filtering

| Variable | Default | Description |
|---|---|---|
| `ATR_SL_MULTIPLIER` | `1.5` | Stop distance in ATRs |
| `ATR_TP_MULTIPLIER` | `2.0` | Target distance in ATRs |
| `CAPITAL` | `1000` | Account size used for position sizing |
| `RISK_PERCENT` | `1` | Percent of capital risked per trade |
| `SHOW_POSITION_SIZE` | `true` | Include the sizing line in alerts |
| `ENABLED_SETUPS` | all three | `rsi_reversal,macd_crossover,bb_breakout` |
| `MIN_CONFIDENCE` | `1` | Drop signals scoring below this (1–3) |

### Scheduling, state and logging

| Variable | Default | Description |
|---|---|---|
| `POLL_INTERVAL_MINUTES` | `15` | Should match `TIMEFRAME` |
| `CANDLE_CLOSE_BUFFER_SECONDS` | `15` | Wait after the boundary before fetching |
| `STATE_FILE` | `signal_state.json` | Deduplication state path |
| `HEARTBEAT_FILE` | *(unset)* | Touched after each cycle for liveness monitoring |
| `SIGNAL_COOLDOWN_MINUTES` | `45` | Minimum gap between repeats of one key |
| `STATE_RETENTION_DAYS` | `7` | Prune entries older than this |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Project layout

```
main.py            Entry point: CLI, candle-aligned loop, scan orchestration
config.py          Environment loading, defaults, logging setup
exchange.py        ccxt client, retrying OHLCV fetch, symbol validation
indicators.py      RSI / MACD / Bollinger / ATR / EMA / SMA (+ pandas-ta backend)
strategies.py      Setup detection, ATR levels, confidence scoring, sizing
notifier.py        Telegram HTML formatting and delivery with retries
state.py           Atomic JSON state, deduplication rules, pruning
preflight.py       --preflight self-check with actionable failure hints
tests/             110 unit and pipeline tests (no network required)
deploy/            systemd unit file
.github/workflows/ CI: tests on Python 3.11 and 3.12
Dockerfile         Python 3.12 slim image, non-root
docker-compose.yml Compose service with a persistent state volume
railway.json       Railway build/deploy configuration
Procfile           Worker declaration for Procfile-based hosts
```

Required function names, as specified: `fetch_ohlcv` (`exchange.py`),
`calculate_indicators` (`indicators.py`), `detect_signals` (`strategies.py`),
`send_telegram` (`notifier.py`), `save_state` / `load_state` (`state.py`).

---

## Tests

```bash
python -m unittest discover -s tests -v
```

110 tests, no network needed. They cover indicator maths against hand-computed
values, every setup's trigger *and* its non-trigger (a price riding the band
must not re-fire), ATR levels and position sizing, confidence scoring, the
deduplication and cooldown rules, atomic state persistence including corrupt
files, Telegram message rendering and 4096-char splitting, candle-close
scheduling across hour and day boundaries, the heartbeat file, and a full
fake-exchange pipeline run from raw candles to formatted message.

---

## Troubleshooting

**`HTTP 451` / `Service unavailable from a restricted location`**
Your host's region is geo-blocked by `binance.com`. This is common on US cloud
providers. Set `EXCHANGE_ID=binanceus`, `kucoin`, `okx` or `bybit`, or deploy in
a different region. The bot does not retry 451 — it is not transient.

**`Forbidden: bot can't initiate conversation with a user`**
You never sent `/start` to your bot. Open the chat and send it.

**`Missing required environment variable(s)`**
`TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` are not set. Locally, check `.env`
exists in the working directory; in the cloud, check the Variables panel.

**`Skipping N symbol(s) not tradable`**
Listings change — `MATIC/USDT` became `POL/USDT` on Binance, for example. The
bot logs and skips unlisted pairs instead of crashing. Update `SYMBOLS` to match
what your exchange actually lists.

**No signals at all**
Normal. Fresh crosses on 15m are not that frequent, and `MIN_CONFIDENCE` filters
further. Run `python main.py --once --dry-run` with `LOG_LEVEL=DEBUG` to see
what is being evaluated. Widening `SYMBOLS` produces more alerts.

**Duplicate alerts after every deploy**
`signal_state.json` is not on a persistent volume. See
[Deduplication](#deduplication).

**`pip install pandas-ta` fails**
Expected on Python 3.11 and below — see
[Indicator backend](#indicator-backend). The bot does not need it.

---

## Disclaimer

This software generates **technical analysis signals for educational purposes
only**. It is not investment advice, it does not place trades, and no setup has
a guaranteed win rate regardless of what any tutorial claims. Backtest before
risking capital, never risk money you cannot afford to lose, and treat the
suggested position size as arithmetic, not a recommendation.
