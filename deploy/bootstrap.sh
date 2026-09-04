#!/usr/bin/env bash
# Set up a fresh Ubuntu server to run the Index Brain, on its own tailnet,
# optionally alongside Clawdbot.
#
# Paste into the VPS's terminal (Hostinger's browser terminal works — no SSH
# client needed) and run as root:
#
#     curl -fsSL https://raw.githubusercontent.com/neofinn/index-option-brain/scaffold/architecture-contracts/deploy/bootstrap.sh | bash
#
# or download, read it, then run it — which is the better habit for anything
# that installs software as root.
#
# Idempotent: safe to re-run. It creates what is missing and leaves what is
# there alone.
#
# --------------------------------------------------------------------------
# Why two unix users
# --------------------------------------------------------------------------
# The brain and Clawdbot both run here, and Clawdbot executes shell commands
# chosen by a language model. That is fine for reading and reporting, and it
# is not fine anywhere near broker credentials: a prompt injection arriving
# in a WhatsApp message or a fetched page would otherwise be able to read
# the keys.
#
# So `indexbrain` owns the application and its .env (0600), and `clawdbot`
# runs as a separate user with no group in common and no read access to that
# file. Clawdbot reaches the brain over http://127.0.0.1:8000, whose every
# route is a GET. It can learn what the system thinks; it cannot change what
# the system does, and it cannot read what the system trades with.
#
# This is a boundary, not a guarantee. Anything running as root on this box
# can cross it, so do not add Clawdbot to sudoers.

set -euo pipefail

REPO="${REPO:-https://github.com/neofinn/index-option-brain.git}"
BRANCH="${BRANCH:-scaffold/architecture-contracts}"
APP_DIR="${APP_DIR:-/opt/index-brain}"
APP_USER="${APP_USER:-indexbrain}"
BOT_USER="${BOT_USER:-clawdbot}"
BOT_DIR="${BOT_DIR:-/opt/clawdbot}"
STATUS_BRANCH="${STATUS_BRANCH:-deploy-status}"
PORT="${PORT:-8000}"
INSTALL_CLAWDBOT="${INSTALL_CLAWDBOT:-ask}"

log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m !\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m x\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "Run as root."
. /etc/os-release 2>/dev/null || true
[ "${ID:-}" = "ubuntu" ] || warn "Tested on Ubuntu; ${ID:-unknown} may differ."

log "Base packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# sudo is used to verify the clawdbot boundary and is not guaranteed on a
# minimal template; python3 is needed by the deploy agent on the host.
apt-get install -y -qq ca-certificates curl gnupg git jq ufw sudo python3 >/dev/null

log "Docker"
if ! command -v docker >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin >/dev/null
fi
systemctl enable --now docker >/dev/null 2>&1 || true
docker --version

log "Tailscale"
if ! command -v tailscale >/dev/null 2>&1; then
  curl -fsSL https://tailscale.com/install.sh | sh >/dev/null
fi
systemctl enable --now tailscaled >/dev/null 2>&1 || true

log "Application user and checkout"
id -u "$APP_USER" >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"
usermod -aG docker "$APP_USER"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" remote set-url origin "$REPO"
  git -C "$APP_DIR" fetch --quiet origin "$BRANCH"
  git -C "$APP_DIR" checkout --quiet -B "$BRANCH" "origin/$BRANCH"
else
  git clone --quiet --branch "$BRANCH" "$REPO" "$APP_DIR"
fi
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# The .env is the only file on this box that will ever hold broker keys.
# 0600 and owned by the app user: root can read it, clawdbot cannot.
if [ ! -f "$APP_DIR/.env" ]; then
  cat > "$APP_DIR/.env" <<'ENVEOF'
# Broker credentials go here. This file is 0600 and owned by the app user
# specifically so the assistant running on this box cannot read it.
LLM_ENABLED=false
RUN_MODE=paper
CAPTURE_ENABLED=true
# DHAN_CLIENT_ID=
# DHAN_ACCESS_TOKEN=
# DELTA_API_KEY=
# DELTA_API_SECRET=
ENVEOF
fi
chown "$APP_USER:$APP_USER" "$APP_DIR/.env"
chmod 600 "$APP_DIR/.env"

log "Service"
install -m 0644 /dev/stdin /etc/systemd/system/index-brain.service <<UNITEOF
[Unit]
Description=Index Option Brain
After=network-online.target docker.service
Requires=docker.service

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
# Managed by the deploy agent. '-' so a missing file is not a boot failure.
EnvironmentFile=-/etc/index-brain/managed.env
ExecStart=/usr/bin/docker compose up --build
ExecStop=/usr/bin/docker compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNITEOF

for unit in index-brain-update.service index-brain-update.timer \
            index-brain-agent.service index-brain-agent.timer; do
  [ -f "$APP_DIR/deploy/$unit" ] && install -m 0644 "$APP_DIR/deploy/$unit" "/etc/systemd/system/$unit"
done
[ -f "$APP_DIR/deploy/self-update.sh" ] && install -m 0755 "$APP_DIR/deploy/self-update.sh" /usr/local/bin/index-brain-update
[ -f "$APP_DIR/deploy/agent.sh" ] && install -m 0755 "$APP_DIR/deploy/agent.sh" /usr/local/bin/index-brain-agent

# The agent reads config from a file it manages and the unit loads. It is
# separate from .env on purpose: .env holds credentials and nothing pushed
# from a repository may write to it.
mkdir -p /etc/index-brain
touch /etc/index-brain/managed.env
chmod 644 /etc/index-brain/managed.env

systemctl daemon-reload
systemctl enable --now index-brain.service
systemctl enable --now index-brain-update.timer 2>/dev/null || \
  warn "Update timer not installed; deploys will need a manual restart."
systemctl enable --now index-brain-agent.timer 2>/dev/null || \
  warn "Agent timer not installed; the box will not report its status back."

log "Firewall"
# Loopback and tailnet only. The console is never on the public internet —
# tailscale serve proxies to 127.0.0.1, so nothing needs an inbound port.
ufw --force reset >/dev/null 2>&1 || true
ufw default deny incoming >/dev/null
ufw default allow outgoing >/dev/null
ufw allow 22/tcp >/dev/null
ufw allow in on tailscale0 >/dev/null 2>&1 || true
ufw --force enable >/dev/null

log "Waiting for the app to answer"
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 5
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
  && echo "  health OK" \
  || warn "Not answering yet — 'journalctl -u index-brain -f' will say why."

# ---------------------------------------------------------------- clawdbot
if [ "$INSTALL_CLAWDBOT" = "ask" ]; then
  read -r -p "Install Clawdbot alongside? [y/N] " reply || reply=n
  case "$reply" in [yY]*) INSTALL_CLAWDBOT=yes ;; *) INSTALL_CLAWDBOT=no ;; esac
fi

if [ "$INSTALL_CLAWDBOT" = "yes" ]; then
  log "Clawdbot, as a separate user"
  id -u "$BOT_USER" >/dev/null 2>&1 || useradd --system --create-home --shell /bin/bash "$BOT_USER"
  mkdir -p "$BOT_DIR"
  chown "$BOT_USER:$BOT_USER" "$BOT_DIR"

  # Deliberately NOT in the docker group and NOT in sudoers: an assistant
  # that can run containers as root can read any file on the box, which
  # would make the .env permissions decorative.
  if id -nG "$BOT_USER" | grep -qw docker; then
    gpasswd -d "$BOT_USER" docker >/dev/null 2>&1 || true
    warn "Removed $BOT_USER from the docker group; that group is root-equivalent."
  fi

  cat > "$BOT_DIR/brain.md" <<'BRIEFEOF'
# Reading the trading engine

The Index Brain runs on this machine. Ask it things over HTTP:

    curl -s http://127.0.0.1:8000/api/brief/NIFTY      # plain English, best for chat
    curl -s http://127.0.0.1:8000/api/analysis/NIFTY   # full brain output as JSON
    curl -s http://127.0.0.1:8000/api/market/NIFTY     # spot, chain, capture status
    curl -s http://127.0.0.1:8000/api/history/NIFTY    # recorded decisions
    curl -s http://127.0.0.1:8000/health

Every one of those is read-only. There is no endpoint that places a trade,
and you cannot read the broker credentials — that is intentional, not a
misconfiguration to route around.

When relaying what the engine says, carry two things it reports and a
summary would drop: what it could NOT measure, and that nothing is
authorized. "NIFTY looks bullish" without those reads as a position the
system holds. It holds nothing.
BRIEFEOF
  chown "$BOT_USER:$BOT_USER" "$BOT_DIR/brain.md"

  cat > "$BOT_DIR/.env" <<'BOTENVEOF'
# Clawdbot's own key. Separate from the trading engine's credentials by
# design — nothing here should be able to reach those.
# ANTHROPIC_API_KEY=
BOTENVEOF
  chown "$BOT_USER:$BOT_USER" "$BOT_DIR/.env"
  chmod 600 "$BOT_DIR/.env"

  # Verify the boundary rather than assuming it.
  if sudo -u "$BOT_USER" test -r "$APP_DIR/.env" 2>/dev/null; then
    die "$BOT_USER can read $APP_DIR/.env — refusing to finish with that open."
  fi
  echo "  verified: $BOT_USER cannot read $APP_DIR/.env"
  echo "  install Clawdbot itself into $BOT_DIR as $BOT_USER, per its own docs"
fi

log "Tailscale"
if tailscale status >/dev/null 2>&1; then
  tailscale serve --bg --https=443 "http://127.0.0.1:$PORT" >/dev/null 2>&1 || true
  echo "  console: https://$(tailscale status --json | jq -r '.Self.DNSName' | sed 's/\.$//')"
else
  cat <<'TSEOF'
  Not connected yet. Run:

      tailscale up

  open the URL it prints, then:

      tailscale serve --bg --https=443 http://127.0.0.1:8000
TSEOF
fi

log "Done"
cat <<SUMMARYEOF
  app        $APP_DIR (user $APP_USER)
  service    systemctl status index-brain
  logs       journalctl -u index-brain -f
  updates    pull-based, health-checked, auto-rollback
  keys       $APP_DIR/.env  (0600, $APP_USER only)
  config     /etc/index-brain/managed.env  (agent-managed, no secrets)
  agent      systemctl status index-brain-agent.timer
  status     pushed to the '$STATUS_BRANCH' branch every 5 minutes

  The engine is analysis-only until a broker is configured AND its response
  mapping is verified. It will not place an order in this state.
SUMMARYEOF
