"""Optional hard-coded settings for running the bot on your own machine.

Copy this file to ``local_config.py`` and fill in your values:

    cp local_config.example.py local_config.py

``local_config.py`` is gitignored, so it can never be pushed by accident. Any
UPPERCASE name here is applied as a default; a real environment variable of the
same name always wins, so the file changes nothing on Railway or GitHub Actions
(where it is not deployed at all - use their Variables / Secrets panels there).

Only the first two matter. Delete any line you do not want to override.
"""

TELEGRAM_BOT_TOKEN = "8123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw"
TELEGRAM_CHAT_ID = "1234567890"

# Anything else from .env.example can go here too, as a string:
# SYMBOLS = "BTC/USDT,ETH/USDT,SOL/USDT"
# CAPITAL = "1000"
# RISK_PERCENT = "1"
# MIN_CONFIDENCE = "2"
# EXCHANGE_ID = "binance"
