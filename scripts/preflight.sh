#!/usr/bin/env bash
# Start every process this system has, at once, and say which came up.
#
# Run it on the box before enabling the systemd units, and again after any
# change to configuration. It exists because starting the processes
# *together* is a different test from starting them one at a time, and the
# first time that was actually tried it found a bug: all four call
# create_all() on the same SQLite file, two raced, and the gateway died
# with "table market_snapshots already exists" while the console came up
# fine — so webhooks were dead and the console looked healthy.
#
#   scripts/preflight.sh              # temp config, temp database
#   scripts/preflight.sh --use-env    # your real .env and var/
#
# Nothing here trades: the relay route it writes is disabled, and the
# temporary registry's credentials are throwaway.
set -uo pipefail

GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; DIM=$'\033[2m'; BOLD=$'\033[1m'; OFF=$'\033[0m'
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
USE_ENV=0
[[ "${1:-}" == "--use-env" ]] && USE_ENV=1

WORK="$(mktemp -d)"
PIDS=()
FAILED=0

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null
  done
  sleep 1
  [[ $USE_ENV -eq 0 ]] && rm -rf "$WORK"
}
trap cleanup EXIT

if [[ ! -x "$PY" ]]; then
  echo "${RED}No interpreter at $PY${OFF}  (set PY=/path/to/python)"
  exit 2
fi

if [[ $USE_ENV -eq 0 ]]; then
  mkdir -p "$WORK/var"
  cat > "$WORK/endpoints.json" <<'JSON'
{
  "tradingview": {"kind":"tradingview","ingest_secret":"preflight-ingest-0000001","read_token":"preflight-read-000000001"},
  "strategy":    {"kind":"strategy","ingest_secret":"preflight-strategy-000001","read_token":"preflight-read-000000001"}
}
JSON
  cat > "$WORK/routes.json" <<'JSON'
{"strategy":{"enabled":false,"destination":{"kind":"pull"},"symbol_map":{"NIFTY":"NIFTY.I"},"max_quantity":1}}
JSON
  chmod 600 "$WORK/endpoints.json" "$WORK/routes.json"
  export WEBHOOK_ENDPOINTS_FILE="$WORK/endpoints.json"
  export SIGNAL_ROUTES_FILE="$WORK/routes.json"
  export SQLITE_PATH="$WORK/var/brain.sqlite"
  export BAR_STORE_DIR="$WORK/var/bars"
  export SIGNAL_RELAY_KILL_FILE="$WORK/var/RELAY_KILLED"
  export TRADINGVIEW_WEBHOOK_SECRET="preflight-tv-secret-00001"
  export TRADINGVIEW_ALLOWED_IPS="any"
else
  # shellcheck disable=SC1091
  [[ -f "$REPO/.env" ]] && set -a && . "$REPO/.env" && set +a
fi
export WEBHOOK_TRUST_FORWARDED_FOR="${WEBHOOK_TRUST_FORWARDED_FOR:-1}"
export TRADINGVIEW_WEBHOOK_PORT="${TRADINGVIEW_WEBHOOK_PORT:-8787}"
export WEBHOOK_GATEWAY_PORT="${WEBHOOK_GATEWAY_PORT:-8788}"
export no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost"

echo "${BOLD}Preflight — starting every process at once${OFF}"
echo "${DIM}logs in $WORK${OFF}"
echo

start() {  # name, logfile, command...
  local name="$1" log="$2"; shift 2
  ( cd "$REPO" && exec "$@" ) > "$WORK/$log" 2>&1 < /dev/null &
  PIDS+=("$!")
  printf '  starting %-14s pid %s\n' "$name" "$!"
}

start console   console.log "$PY" -m uvicorn index_option_brain.app.main:app \
                  --host 127.0.0.1 --port 8000 --log-level warning
start tradingview tv.log    "$PY" -m index_option_brain.integrations.tradingview
start gateway   gateway.log "$PY" -m index_option_brain.integrations.webhooks
echo

# Long enough for the slowest of them: the console loads contract
# specifications and reaches the feed before it serves.
sleep 15

probe() {  # label, url, expected-substring
  local label="$1" url="$2" want="$3" body
  body="$(curl -sS --max-time 6 --noproxy 127.0.0.1 "$url" 2>&1)"
  if [[ "$body" == *"$want"* ]]; then
    printf '%s ok  %s %-12s %s%s%s\n' "$GREEN" "$OFF" "$label" "$DIM" "${body:0:70}" "$OFF"
  else
    printf '%s FAIL%s %-12s %s\n' "$RED" "$OFF" "$label" "${body:0:120}"
    FAILED=$((FAILED + 1))
  fi
}

probe console     "http://127.0.0.1:8000/health"                        '"status"'
probe receiver    "http://127.0.0.1:${TRADINGVIEW_WEBHOOK_PORT}/tv/health" '"status":"ok"'
probe gateway     "http://127.0.0.1:${WEBHOOK_GATEWAY_PORT}/health"     '"status":"ok"'
probe gateway-ui  "http://127.0.0.1:${WEBHOOK_GATEWAY_PORT}/"           '<!doctype html>'

# The chat bot needs a real Telegram token, so what is checked is that it
# refuses to start without an allowlist — the guard that matters.
echo
printf '  chat bot guards: '
if TELEGRAM_BOT_TOKEN="" "$PY" -m index_option_brain.chat >/dev/null 2>&1; then
  printf '%sFAIL%s started with no token\n' "$RED" "$OFF"
  FAILED=$((FAILED + 1))
elif TELEGRAM_BOT_TOKEN="123456:AAfake" TELEGRAM_ALLOWED_CHAT_IDS="" \
     "$PY" -m index_option_brain.chat >/dev/null 2>&1; then
  printf '%sFAIL%s started with no allowlist\n' "$RED" "$OFF"
  FAILED=$((FAILED + 1))
else
  printf '%sok%s   refuses without a token and without an allowlist\n' "$GREEN" "$OFF"
fi

echo
for log in console.log tv.log gateway.log; do
  if grep -qE "Traceback|CRITICAL" "$WORK/$log" 2>/dev/null; then
    printf '%s%s has a traceback:%s\n' "$YELLOW" "$log" "$OFF"
    grep -m1 -A2 "Traceback" "$WORK/$log" | sed 's/^/    /'
    FAILED=$((FAILED + 1))
  fi
done

echo
if [[ $FAILED -eq 0 ]]; then
  echo "${BOLD}${GREEN}All processes came up together.${OFF}"
else
  echo "${BOLD}${RED}$FAILED problem(s). Logs kept in $WORK${OFF}"
  USE_ENV=1   # keep the directory for inspection
fi
exit $((FAILED > 0))
