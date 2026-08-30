# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Unbuffered stdout so `docker logs` / Railway logs stream in real time.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

WORKDIR /app

# Install dependencies first so the layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user; /app/data holds the de-duplication state.
RUN useradd --create-home --uid 10001 botuser \
    && mkdir -p /app/data \
    && chown -R botuser:botuser /app
USER botuser

ENV STATE_FILE=/app/data/signal_state.json \
    HEARTBEAT_FILE=/app/data/heartbeat

# The bot touches HEARTBEAT_FILE after every scan cycle. Two missed 15m cycles
# means it is wedged. This checks the bot itself rather than the exchange, so it
# stays correct whatever EXCHANGE_ID is set to and costs no API calls.
HEALTHCHECK --interval=5m --timeout=10s --start-period=3m --retries=2 \
    CMD python -c "import os,sys,time; f=os.environ['HEARTBEAT_FILE']; \
sys.exit(0 if os.path.exists(f) and time.time()-os.path.getmtime(f) < 1860 else 1)"

EXPOSE 8000

# Default: Telegram alerts only.
# For the web dashboard instead, override the command and set
# RUN_BOT_IN_WEBAPP=true so one container serves the page AND scans:
#   docker run -e RUN_BOT_IN_WEBAPP=true -p 8000:8000 <image> python webserver.py
CMD ["python", "main.py"]
