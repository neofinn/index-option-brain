#!/usr/bin/env bash
# Get a working webhook URL in about a minute, with no DNS and no domain.
#
# TradingView calls webhooks on ports 80 and 443 only, which normally means
# a DNS record and a certificate. A Cloudflare quick tunnel needs neither:
# it dials out from this machine, Cloudflare terminates TLS on 443, and it
# prints a hostname you can paste into an alert immediately. No account, no
# config file, no open port, and it works behind NAT.
#
#   scripts/quick_tunnel.sh                 # tunnels the gateway on :8788
#   scripts/quick_tunnel.sh --port 8000     # or the console
#   scripts/quick_tunnel.sh --slug strategy # name the endpoint in the URL
#
# THE URL CHANGES EVERY TIME THIS RESTARTS. That is the whole trade: a
# quick tunnel is for proving the chain works and for testing an alert
# template, not for leaving running. An alert pointing at a dead
# trycloudflare hostname fails silently from the chart's side, and
# TradingView will not tell you which of the five links broke. For anything
# you intend to leave up, use deploy/cloudflared-config.yml (a named tunnel
# with a stable hostname) or deploy/Caddyfile (a domain pointing at this
# box).
set -uo pipefail

GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; DIM=$'\033[2m'; BOLD=$'\033[1m'; OFF=$'\033[0m'
PORT=8788
SLUG=tradingview
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Extracted as a function so the parsing can be tested without running
# cloudflared — the format is Cloudflare's to change, and a silent failure
# to find the URL would leave this script hanging with no explanation.
extract_url() { grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1; }

# Checked before the argument loop, which would otherwise reject it — the
# first version had this after the loop, where it could never run.
if [[ "${1:-}" == "--self-test" ]]; then extract_url; exit 0; fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --slug) SLUG="$2"; shift 2 ;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "${RED}Unknown argument: $1${OFF}"; exit 2 ;;
  esac
done

if ! command -v cloudflared >/dev/null 2>&1; then
  cat <<EOF
${RED}cloudflared is not installed.${OFF}

  Debian/Ubuntu:
    curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \\
      | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
    echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] \\
      https://pkg.cloudflare.com/cloudflared any main" \\
      | sudo tee /etc/apt/sources.list.d/cloudflared.list
    sudo apt-get update && sudo apt-get install -y cloudflared
EOF
  exit 2
fi

# Fail here rather than after the tunnel is up: a tunnel to a dead port
# answers 502, which looks like a Cloudflare problem and is not one.
if ! curl -sS --max-time 4 --noproxy 127.0.0.1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "${RED}Nothing is answering on 127.0.0.1:$PORT${OFF}"
  echo "  Start it first, then re-run:"
  echo "    ${DIM}python -m index_option_brain.integrations.webhooks${OFF}   # :8788"
  echo "    ${DIM}python -m uvicorn index_option_brain.app.main:app --port 8000${OFF}"
  exit 2
fi

LOG="$(mktemp)"
cleanup() { [[ -n "${TUNNEL_PID:-}" ]] && kill "$TUNNEL_PID" 2>/dev/null; rm -f "$LOG"; }
trap cleanup EXIT INT TERM

echo "${BOLD}Opening a quick tunnel to 127.0.0.1:$PORT${OFF}"
cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > "$LOG" 2>&1 &
TUNNEL_PID=$!

URL=""
for _ in $(seq 1 30); do
  sleep 1
  URL="$(extract_url < "$LOG")"
  [[ -n "$URL" ]] && break
  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo "${RED}cloudflared exited before printing a URL:${OFF}"
    tail -15 "$LOG" | sed 's/^/    /'
    exit 1
  fi
done

if [[ -z "$URL" ]]; then
  echo "${RED}No trycloudflare URL after 30s.${OFF} Last output:"
  tail -15 "$LOG" | sed 's/^/    /'
  exit 1
fi

echo
echo "${GREEN}${BOLD}Webhook URL — paste this into TradingView${OFF}"
echo "${BOLD}  $URL/hook/$SLUG${OFF}"
echo
echo "${DIM}Read API:   $URL/v1/$SLUG?since=0${OFF}"
echo "${DIM}Page:       $URL/${OFF}"
echo
echo "${YELLOW}This hostname dies when you press Ctrl-C, and a different one${OFF}"
echo "${YELLOW}appears next time. An alert left pointing at a dead one fails${OFF}"
echo "${YELLOW}silently — TradingView only says the delivery failed.${OFF}"
echo
echo "${DIM}Verify the whole chain through the tunnel:${OFF}"
echo "${DIM}  scripts/hook_test.py --base $URL \\
      --ingest-secret \$INGEST --read-token \$READ --slug $SLUG${OFF}"
echo
echo "${DIM}Ctrl-C to stop. Tunnel log: $LOG${OFF}"
wait "$TUNNEL_PID"
