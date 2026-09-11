#!/usr/bin/env bash
# One command to a working TradingView webhook URL.
#
#   bash scripts/start_here.sh
#
# Installs into a venv if needed, generates credentials, starts the
# gateway, opens a Cloudflare tunnel, prints the URL to paste into
# TradingView, and proves the chain end to end before handing it to you.
#
# Needs: Python 3.12+, and cloudflared for a public URL (it says how to
# install it and still works locally without one).
#
# Needs NOT: a VPS, a domain, a DNS record, an open port, root, Docker, or
# a TradingView paid plan to test with (curl stands in).
set -uo pipefail

G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; D=$'\033[2m'; B=$'\033[1m'; O=$'\033[0m'
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PORT="${PORT:-8788}"
SLUG=tradingview
export no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost"

say()  { printf '%s\n' "$*"; }
step() { printf '\n%s==>%s %s\n' "$B" "$O" "$*"; }
die()  { printf '%s%s%s\n' "$R" "$*" "$O"; exit 1; }

# ---------------------------------------------------------------- python
step "Python"
PY=""
for candidate in "$REPO/.venv/bin/python" python3.13 python3.12 python3; do
  if command -v "$candidate" >/dev/null 2>&1 || [[ -x "$candidate" ]]; then
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)' 2>/dev/null; then
      PY="$candidate"; break
    fi
  fi
done
[[ -n "$PY" ]] || die "Python 3.12+ not found. Install it, then re-run."
say "  $($PY --version) at $PY"

if [[ ! -x "$REPO/.venv/bin/python" ]]; then
  step "Creating .venv (one minute, once)"
  "$PY" -m venv .venv || die "venv creation failed"
  PY="$REPO/.venv/bin/python"
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -e . || die "install failed"
  say "  installed"
else
  PY="$REPO/.venv/bin/python"
  "$PY" -c 'import index_option_brain' 2>/dev/null || {
    step "Installing the package into the existing venv"
    "$PY" -m pip install --quiet -e . || die "install failed"
  }
fi

# ----------------------------------------------------------- credentials
step "Credentials"
mkdir -p var
if [[ -f var/webhook-endpoints.json ]]; then
  say "  var/webhook-endpoints.json exists — leaving it alone"
else
  # Generated here rather than asked for, because the failure mode of
  # "choose a secret" is a short one, and the two must differ.
  ING="$("$PY" -c 'import secrets;print(secrets.token_hex(24))')"
  SIG="$("$PY" -c 'import secrets;print(secrets.token_hex(24))')"
  READ="$("$PY" -c 'import secrets;print(secrets.token_hex(24))')"
  cat > var/webhook-endpoints.json <<JSON
{
  "tradingview": {
    "kind": "tradingview",
    "description": "indicator alerts — an observation",
    "ingest_secret": "$ING",
    "read_token": "$READ",
    "retain": 500
  },
  "strategy": {
    "kind": "strategy",
    "description": "strategy alerts — an order intent",
    "ingest_secret": "$SIG",
    "read_token": "$READ",
    "retain": 500
  }
}
JSON
  chmod 600 var/webhook-endpoints.json
  say "  generated var/webhook-endpoints.json (mode 600)"
fi
if [[ ! -f var/signal-routes.json ]]; then
  # Disabled, and a pull destination: nothing can leave the machine. The
  # route has to exist because the gateway refuses to start with a
  # strategy endpoint that has none — an endpoint that accepts orders and
  # silently discards them is worse than a refusal.
  cat > var/signal-routes.json <<'JSON'
{
  "strategy": {
    "description": "held for an EA to poll. enabled:false = nothing is sent.",
    "enabled": false,
    "destination": { "kind": "pull" },
    "symbol_map": { "NIFTY": "NIFTY.I", "BANKNIFTY": "BANKNIFTY.I" },
    "allowed_actions": ["buy", "sell", "exit"],
    "max_quantity": 1,
    "max_orders_per_day": 20,
    "max_age_seconds": 90
  }
}
JSON
  chmod 600 var/signal-routes.json
  say "  generated var/signal-routes.json (disabled — nothing can trade)"
fi
read_field() { "$PY" -c "import json,sys;d=json.load(open('var/webhook-endpoints.json'));print(d[sys.argv[1]][sys.argv[2]])" "$1" "$2"; }
ING="$(read_field tradingview ingest_secret)"
READ="$(read_field tradingview read_token)"
SIG="$(read_field strategy ingest_secret 2>/dev/null || echo "$ING")"

# ------------------------------------------------------------- gateway
step "Gateway"
if curl -sS --max-time 3 --noproxy 127.0.0.1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  say "  already running on :$PORT"
  GW_PID=""
else
  LOG="$(mktemp)"
  WEBHOOK_ENDPOINTS_FILE="$REPO/var/webhook-endpoints.json" \
  SIGNAL_ROUTES_FILE="$REPO/var/signal-routes.json" \
  SQLITE_PATH="$REPO/var/index_brain.sqlite" \
  WEBHOOK_GATEWAY_PORT="$PORT" \
  WEBHOOK_TRUST_FORWARDED_FOR=1 \
    "$PY" -m index_option_brain.integrations.webhooks > "$LOG" 2>&1 &
  GW_PID=$!
  for _ in $(seq 1 25); do
    sleep 1
    curl -sS --max-time 2 --noproxy 127.0.0.1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$GW_PID" 2>/dev/null || { say "${R}  gateway exited:${O}"; tail -12 "$LOG" | sed 's/^/    /'; exit 1; }
  done
  curl -sS --max-time 3 --noproxy 127.0.0.1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
    || { say "${R}  gateway never answered${O}"; tail -12 "$LOG" | sed 's/^/    /'; exit 1; }
  say "  started on :$PORT  ${D}(log: $LOG)${O}"
fi

# -------------------------------------------------------------- tunnel
step "Public URL"
BASE="http://127.0.0.1:$PORT"
TUNNEL_PID=""

# cloudflared, or fetch it. The point of this script is that one command is
# enough, and "install cloudflared first" is not one command.
CFD=""
# Windows (Git Bash) will not exec a file without the extension.
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) CFD_NAME=cloudflared.exe ;; *) CFD_NAME=cloudflared ;; esac
if command -v cloudflared >/dev/null 2>&1; then
  CFD="$(command -v cloudflared)"
elif [[ -x "var/bin/$CFD_NAME" ]]; then
  CFD="$REPO/var/bin/$CFD_NAME"
else
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)              ASSET=cloudflared-linux-amd64 ;;
    Linux-aarch64|Linux-arm64) ASSET=cloudflared-linux-arm64 ;;
    Darwin-arm64)              ASSET=cloudflared-darwin-arm64.tgz ;;
    Darwin-x86_64)             ASSET=cloudflared-darwin-amd64.tgz ;;
    MINGW*|MSYS*|CYGWIN*)      ASSET=cloudflared-windows-amd64.exe ;;
    *)                         ASSET="" ;;
  esac
  if [[ -n "$ASSET" ]]; then
    say "  fetching cloudflared (${ASSET})"
    mkdir -p var/bin
    URL_CFD="https://github.com/cloudflare/cloudflared/releases/latest/download/$ASSET"
    if [[ "$ASSET" == *.tgz ]]; then
      curl -sSL --max-time 120 -o var/bin/cfd.tgz "$URL_CFD" 2>/dev/null \
        && tar -xzf var/bin/cfd.tgz -C var/bin cloudflared 2>/dev/null \
        && rm -f var/bin/cfd.tgz
    else
      curl -sSL --max-time 120 -o "var/bin/$CFD_NAME" "$URL_CFD" 2>/dev/null
    fi
    chmod +x "var/bin/$CFD_NAME" 2>/dev/null
    if "var/bin/$CFD_NAME" --version >/dev/null 2>&1; then
      CFD="$REPO/var/bin/$CFD_NAME"
      say "  $("$CFD" --version 2>&1 | head -1)"
    else
      rm -f "var/bin/$CFD_NAME"
      say "${Y}  could not fetch cloudflared — the URL below is local only${O}"
    fi
  else
    say "${Y}  no cloudflared build for $(uname -s)-$(uname -m) — the URL below is local only${O}"
  fi
fi

if [[ -n "$CFD" ]]; then
  TLOG="$(mktemp)"
  say "  opening a tunnel ${D}(up to ~90s; it is verified before you get it)${O}"
  "$CFD" tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > "$TLOG" 2>&1 &
  TUNNEL_PID=$!
  URL=""
  for _ in $(seq 1 30); do
    sleep 1
    URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TLOG" | head -1)"
    [[ -n "$URL" ]] && break
  done
  # The hostname is printed when it is *allocated*, which happens before
  # any tunnel connects — and on a network that blocks outbound 7844 no
  # connection ever does. Printing that URL as the answer hands over a
  # webhook target that answers 530 to TradingView and nothing to you. So
  # the URL is only an answer once a request has come back through it.
  if [[ -n "$URL" ]]; then
    LIVE=""
    for _ in $(seq 1 20); do
      if [[ "$(curl -sS --max-time 8 -o /dev/null -w '%{http_code}' "$URL/health" 2>/dev/null)" == "200" ]]; then
        LIVE=1; break
      fi
      sleep 2
    done
    if [[ -n "$LIVE" ]]; then
      BASE="$URL"
      say "  $BASE  ${D}(verified: a request came back through it)${O}"
    else
      say "${Y}  cloudflared allocated $URL but no request survives the round trip.${O}"
      say "${Y}  That is this machine's outbound port 7844 being blocked, not your"
      say "  config. TradingView would get 530. Using the local URL instead.${O}"
      if grep -q '7844' "$TLOG"; then
        say "${D}  $(grep -m1 '7844' "$TLOG" | sed -e 's/.*ERROR: //' -e 's/[[:space:]]*|[[:space:]]*$//')${O}"
      fi
      kill "$TUNNEL_PID" 2>/dev/null; TUNNEL_PID=""
    fi
  else
    say "${Y}  cloudflared printed no URL in 30s; using the local one${O}"
    tail -6 "$TLOG" | sed 's/^/    /'
  fi
fi

cleanup() {
  [[ -n "${TUNNEL_PID:-}" ]] && kill "$TUNNEL_PID" 2>/dev/null
  [[ -n "${GW_PID:-}" ]] && kill "$GW_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------- proof
step "Proving the chain"
# Both legs, each against the endpoint kind that matches it. Pointing the
# strategy cases at a `tradingview` endpoint made seven of them fail for
# the wrong reason — the harness was wrong, not the system, and a script
# that reports that to you is worse than no script.
PROOF="$("$PY" scripts/hook_test.py --base "$BASE" --ingest-secret "$ING" \
  --read-token "$READ" --slug "$SLUG" --strategy-slug strategy \
  --strategy-ingest-secret "$SIG" 2>&1)"
printf '%s\n' "$PROOF" | tail -30
if printf '%s' "$PROOF" | grep -q '0 failed'; then
  say "  ${G}every link in the chain works${O}"
else
  say "  ${Y}some checks failed — the URL below may still work, but read them${O}"
fi

# --------------------------------------------------------------- output
if [[ "$BASE" == http://127.0.0.1:* ]]; then
  HEAD="${B}${Y}No public URL — TradingView cannot reach this. Local only:${O}"
else
  HEAD="${B}${G}Paste this into TradingView -> Alert -> Webhook URL${O}"
fi
cat <<EOF

$HEAD
${B}  $BASE/hook/$SLUG${O}

${B}Paste this into the alert message${O} ${D}(or set it as the Pine indicator's secret)${O}
{
  "secret": "$ING",
  "kind": "BREAKOUT",
  "ticker": "{{ticker}}",
  "interval": "{{interval}}",
  "price": "{{close}}",
  "bar_time": "{{time}}",
  "fired_at": "{{timenow}}"
}

${B}Watch it arrive${O}
  open   $BASE/                       ${D}# the page; read token below${O}
  read token: $READ

${B}Or from a shell${O}
  curl -s -H "Authorization: Bearer $READ" "$BASE/v1/$SLUG/payload" | jq .

EOF
if [[ "$BASE" == https://*trycloudflare.com ]]; then
  say "${Y}This hostname dies when you press Ctrl-C and a different one appears"
  say "next time. Fine for testing an alert; for anything permanent use"
  say "deploy/cloudflared-config.yml or deploy/Caddyfile.${O}"
fi
say ""
say "${D}Ctrl-C to stop everything.${O}"
while true; do sleep 3600; done
