# Crypto Trading Signal Bot

A production-ready signal bot that watches Binance 15-minute candles for three
classic technical setups and pushes alerts to Telegram — with ATR-based stop
loss / take profit, a 1–3 confidence score, and a suggested position size.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samshoaib123/Trading_Signal_Bot/blob/main/notebooks/colab_quickstart.ipynb)

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
- [Does it actually work? Backtest it](#does-it-actually-work-backtest-it)
- [Outcome tracking](#outcome-tracking)
- [Risk levels and position sizing](#risk-levels-and-position-sizing)
- [Confidence score](#confidence-score)
- [Deduplication](#deduplication)
- [Quick start (local)](#quick-start-local)
- [Try it in Google Colab](#try-it-in-google-colab)
- [Getting your Telegram token and chat id](#getting-your-telegram-token-and-chat-id)
- [Deploy free on GitHub Actions](#deploy-free-on-github-actions)
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

## Does it actually work? Backtest it

```bash
python main.py --backtest
```

Replays real history for your configured pairs and reports, per setup and per
pair: signals fired, win rate, expectancy in R, total R, profit factor, and the
longest losing streak.

Read that last column carefully. A setup can be profitable on paper and still be
unusable if it puts thirteen losses in a row between you and the profit.

The replay is deliberately conservative:

* **Indicators are causal**, so evaluating candle *i* against a frame that also
  holds later candles cannot leak the future.
* **De-duplication mirrors the live bot** (same candle suppression, same
  cooldown), so the trade count is what you would actually have been sent.
* **The stop wins ties.** When one candle's range contains both the stop and the
  target, it is scored a loss — without tick data there is no way to know which
  came first, and assuming the win would flatter every number.
* **Round-trip fees are charged** at `FEE_PERCENT` per side. On a 15m ATR stop
  fees are often a fifth of the risk, which is enough to turn a marginal setup
  negative.

It does not model slippage, spread, partial fills or funding, so real results
will be worse than the report, not better. A positive number here is a reason to
paper-trade, never a reason to size up.

## Outcome tracking

With `TRACK_OUTCOMES=true` (the default) every alert is recorded as an open
position, and later scans check whether it reached its stop or target using
candles the bot already downloaded — no extra API calls. When one resolves you
get a short message:

```
✅ Target hit — BTC/USDT
Setup: RSI Reversal (BUY)
Entry: 65,000.00
Exit: 66,600.00 (+2.46%)
Result: +1.27R after fees
Held: 6 candles

Record so far: 12W / 19L (39% win rate), -4.2R total
```

`python main.py --report` sends the cumulative scoreboard on demand.

The scoring rules are identical to `--backtest`, so live results and historical
results are directly comparable. A bot that only announces entries can never be
judged; this is what lets you tell a normal losing streak from a broken setup.

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
| `--backtest` | Replay history and print win rate / expectancy / profit factor per setup |
| `--report` | Send the cumulative win/loss scoreboard to Telegram and exit |

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

## Try it in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samshoaib123/Trading_Signal_Bot/blob/main/notebooks/colab_quickstart.ipynb)

`notebooks/colab_quickstart.ipynb` clones the repo, installs the dependencies,
takes your Telegram credentials via `getpass` (so the token never lands in saved
notebook output), runs `--preflight`, and lets you inspect raw indicator values
for any pair.

### The badge needs two things first

The badge above resolves to `blob/main/...` on a **public** repo. Two conditions
have to hold or Colab answers *"Notebook not found"*:

1. **The repo must be public.** Colab cannot read a private repo from a plain
   link. Nothing in this repository is secret — `.env` is gitignored and has
   never been committed, and credentials are read from environment variables at
   runtime — so making it public is safe. As a bonus, GitHub Actions minutes
   become free and unlimited on public repos.
2. **The notebook must be on `main`.** A branch name containing a slash (like
   `claude/crypto-trading-signal-bot-xmcatj`) breaks Colab's
   `blob/<branch>/<path>` URL: it cannot tell where the branch name ends and the
   file path begins. Merging the branch into `main` fixes it.

**Keeping the repo private?** Then skip the badge and load the notebook through
Colab's own GitHub integration: *File -> Open notebook -> GitHub tab -> tick
"Include private repositories" -> authorize*. The first cell will also ask for a
GitHub token so it can clone; a fine-grained token with **Contents: Read-only**
is enough. The cell reads it with `getpass`, scrubs it from any error message,
and strips it from the git remote after cloning.

**Colab is a testing tool here, not a host.** Use it to confirm the setup works,
see what alerts look like and tune thresholds. It cannot run the bot 24/7:

| Limit | Effect |
|---|---|
| Disconnects after ~90 min idle, ~12 h max | The bot stops; alerts stop |
| Browser tab must stay open | Close the laptop, lose the bot |
| Runtime resets wipe the filesystem | `signal_state.json` is lost, so you get duplicate alerts next run |

### Expect a Binance 451 in Colab

Colab runs on Google infrastructure in the US, and `binance.com` blocks US IP
addresses. Preflight will report it clearly. The fix is one line in the notebook:

```python
os.environ['EXCHANGE_ID'] = 'kucoin'   # or okx, bybit, binanceus
```

Everything else behaves identically — the bot is exchange-agnostic through ccxt.

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

## Deploy free on GitHub Actions

`.github/workflows/signals.yml` runs `python main.py --once` every 15 minutes on
GitHub's runners. No server, and on a public repository no cost. Full setup and
caveats: [`.github/workflows/README.md`](.github/workflows/README.md).

Three conditions have to hold:

1. **Public repository** — Actions minutes are unlimited on public repos. The
   private free allowance is 2,000 minutes/month and this needs about 2,900
   (~96 runs/day, ~1 minute each, most of it cold-start `pip install`).
2. **Workflow on the default branch** — GitHub runs scheduled workflows *only*
   from the default branch. Elsewhere the cron is silently ignored.
3. **`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` as repository secrets** —
   *Settings → Secrets and variables → Actions*.

Then run it manually once to check the wiring: *Actions → signals → Run workflow
→ mode `preflight`*.

State survives between runs through the Actions cache: each run restores
`signal_state.json` from the most recent entry and saves a new one under a fresh
key (cache keys are immutable). A cache miss costs at most one duplicate alert.

### What you are trading away

GitHub's cron is best-effort: runs are commonly 5–20 minutes late under load and
are sometimes skipped. The bot always evaluates the last *closed* candle and
de-duplicates by candle timestamp, so a late run gives you a late alert rather
than a wrong one — but a delay long enough to skip a candle means that candle's
signal never arrives at all. Scheduled workflows are also disabled after 60 days
without repository activity.

For alerts that land on time, every time, use Railway or a VPS.

> **Binance will not work here.** GitHub runners are in US datacentres and
> `binance.com` blocks US IPs, so the workflow defaults to `EXCHANGE_ID=kucoin`.
> Change it with an Actions *variable*, not by editing the workflow.

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
| `BEGINNER_MODE` | `true` | Add plain-language "what to do" steps to each alert |
| `FEE_PERCENT` | `0.1` | Taker fee per side, used by tracking and backtesting |
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
| `TRACK_OUTCOMES` | `true` | Follow each signal to its stop or target |
| `TRACKER_FILE` | `signal_outcomes.json` | Where outcomes and the scoreboard live |
| `BACKTEST_CANDLES` | `1000` | History depth for `--backtest` |
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
tracker.py         Follows sent signals to their stop or target, keeps the score
backtest.py        Historical replay with fees, per-setup performance report
preflight.py       --preflight self-check with actionable failure hints
tests/             161 unit and pipeline tests (no network required)
notebooks/         Colab quickstart notebook
deploy/            systemd unit file
.github/workflows/ CI (tests.yml) + free 15-minute scheduler (signals.yml)
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

161 tests, no network needed. They cover indicator maths against hand-computed
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
