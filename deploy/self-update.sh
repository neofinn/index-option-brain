#!/usr/bin/env bash
# Pull the tracked branch, rebuild, and roll back if the result is unhealthy.
#
# This is what makes the box maintainable without anyone touching it: changes
# are pushed to the repository and the machine picks them up. Nothing needs
# inbound access to the server, which is the point — a trading box should not
# be accepting connections so that someone can administer it.
#
# The rollback is not optional decoration. Auto-deploying whatever is on a
# branch means a bad commit takes the system down at the worst possible
# moment, so every update is verified against /ready and reverted if it does
# not come up. A machine running yesterday's working code is fine; a machine
# running today's broken code and nobody watching is not.

set -Eeuo pipefail

APP_DIR="${APP_DIR:-/opt/index-option-brain}"
BRANCH="${DEPLOY_BRANCH:-main}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/ready}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"
COMPOSE=(docker compose)

log() { printf '%s self-update: %s\n' "$(date -Is)" "$*"; }

cd "$APP_DIR"

git fetch --quiet origin "$BRANCH"
PREVIOUS="$(git rev-parse HEAD)"
TARGET="$(git rev-parse "origin/${BRANCH}")"

if [[ "$PREVIOUS" == "$TARGET" ]]; then
    exit 0
fi

log "updating ${PREVIOUS:0:8} -> ${TARGET:0:8} on ${BRANCH}"

deploy() {
    git checkout --quiet --force "$1"
    "${COMPOSE[@]}" up -d --build
}

healthy() {
    # /ready, not /health: a process that is alive but blind reports healthy
    # from /health, and that is exactly the failure this must catch.
    local deadline=$((SECONDS + HEALTH_TIMEOUT))
    while (( SECONDS < deadline )); do
        if curl -fsS --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    return 1
}

if deploy "$TARGET" && healthy; then
    log "deployed ${TARGET:0:8}, ready"
    # Keep the branch pointer following the remote so `git log` reads
    # normally rather than showing a detached head forever.
    git checkout --quiet -B "$BRANCH" "$TARGET"
    exit 0
fi

log "ERROR ${TARGET:0:8} did not become ready — rolling back to ${PREVIOUS:0:8}"
if deploy "$PREVIOUS" && healthy; then
    log "rolled back to ${PREVIOUS:0:8}, ready"
else
    # Both the new and the old commit failed, so this is not a bad deploy —
    # it is the feed, the host, or something outside the repository. Say so
    # rather than looping.
    log "CRITICAL rollback also unhealthy; the fault is not in the deploy"
fi
exit 1
