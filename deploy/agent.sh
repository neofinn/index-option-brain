#!/usr/bin/env bash
# The control loop: read intent from git, reconcile the box, report back to git.
#
# self-update.sh deploys code. This does the other two thirds of remote
# control: it applies declarative *configuration* from deploy/desired_state.
# json, and — the part that actually matters — it pushes a status report to a
# branch so whoever pushed the intent can see what happened.
#
# Without the report, a pull-based deploy is a write-only channel: you push a
# commit and find out whether it worked by asking someone to look. With it,
# the loop is closed over HTTPS in both directions and the machine still
# accepts no inbound connections at all.
#
# What this cannot do
# -------------------
# Enable trading. index_option_brain/deploy/reconcile.py rejects any desired
# state naming a run mode, a dry-run flag, a kill switch or a credential —
# the whole document, not the offending key. Turning analysis into orders
# requires a human editing .env on this machine, which means it cannot be
# done by pushing a commit and therefore cannot be done by anyone who
# compromises the repository.
#
# That is the deployment-layer version of the rule the brains already
# follow: no automated actor may override risk or execution.

set -Eeuo pipefail

APP_DIR="${APP_DIR:-/opt/index-brain}"
STATUS_BRANCH="${STATUS_BRANCH:-deploy-status}"
STATE_FILE="${STATE_FILE:-deploy/desired_state.json}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/ready}"
APPLIED_FILE="${APPLIED_FILE:-/var/lib/index-brain/applied-revision}"
SERVICE="${SERVICE:-index-brain.service}"
PY="${PY:-docker compose exec -T app python}"

log() { printf '%s agent: %s\n' "$(date -Is)" "$*"; }

cd "$APP_DIR"
mkdir -p "$(dirname "$APPLIED_FILE")"
APPLIED="$(cat "$APPLIED_FILE" 2>/dev/null || true)"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
ERRORS=()

log "fetching intent"
git fetch --quiet origin || ERRORS+=("git fetch failed")

# Read the desired state from the *remote*, so a broken local checkout does
# not strand the box on stale intent.
git show "origin/${CURRENT_BRANCH}:${STATE_FILE}" > /tmp/desired_state.json 2>/dev/null \
  || cp "$STATE_FILE" /tmp/desired_state.json 2>/dev/null \
  || ERRORS+=("no desired state found")

REVISION=""; TARGET_BRANCH="$CURRENT_BRANCH"; RESTART=0; ENV_UPDATES="{}"; REASONS=""; REJECTED=""
if [[ -s /tmp/desired_state.json ]]; then
  # The plan is computed by tested Python. A rejected state exits non-zero
  # and sets REJECTED, and we deliberately keep running so the rejection
  # itself gets reported — a box that silently ignores bad intent looks
  # identical to one that never received it.
  if PLAN="$(python -m index_option_brain.deploy.cli plan \
        --file /tmp/desired_state.json \
        --current-branch "$CURRENT_BRANCH" \
        --applied-revision "$APPLIED" 2>/dev/null)"; then
    eval "$PLAN"
  else
    eval "$(python -m index_option_brain.deploy.cli plan \
        --file /tmp/desired_state.json \
        --current-branch "$CURRENT_BRANCH" \
        --applied-revision "$APPLIED" 2>/dev/null || true)"
    ERRORS+=("desired state rejected: ${REJECTED:-unknown}")
    log "REJECTED: ${REJECTED:-unknown}"
  fi
fi

if [[ -z "${REJECTED:-}" && -n "$REVISION" ]]; then
  log "revision ${REVISION}${REASONS:+ — $REASONS}"

  if [[ "$TARGET_BRANCH" != "$CURRENT_BRANCH" ]]; then
    log "switching to $TARGET_BRANCH"
    git checkout --quiet -B "$TARGET_BRANCH" "origin/$TARGET_BRANCH" || ERRORS+=("branch switch failed")
    CURRENT_BRANCH="$TARGET_BRANCH"
  fi

  # Config lands in an env file the unit reads, never in .env — that file
  # holds credentials and nothing pushed from a repository may write to it.
  if [[ "$ENV_UPDATES" != "{}" ]]; then
    python - "$ENV_UPDATES" <<'PYEOF'
import json, pathlib, sys
updates = json.loads(sys.argv[1])
path = pathlib.Path("/etc/index-brain/managed.env")
path.parent.mkdir(parents=True, exist_ok=True)
existing = {}
if path.exists():
    for line in path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            existing[k.strip()] = v.strip()
existing.update(updates)
path.write_text(
    "# Managed by the deploy agent from deploy/desired_state.json.\n"
    "# Credentials are NOT here: they live in .env, which this never writes.\n"
    + "".join(f"{k}={v}\n" for k, v in sorted(existing.items()))
)
PYEOF
    RESTART=1
  fi

  if [[ "$RESTART" == "1" ]]; then
    log "restarting $SERVICE"
    systemctl restart "$SERVICE" || ERRORS+=("restart failed")
  fi
  printf '%s' "$REVISION" > "$APPLIED_FILE"
fi

log "health"
HEALTHY=0
for _ in $(seq 1 24); do
  if curl -fsS --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then HEALTHY=1; break; fi
  sleep 5
done
(( HEALTHY )) || ERRORS+=("not ready at $HEALTH_URL")

SERVICE_ACTIVE=0
systemctl is-active --quiet "$SERVICE" && SERVICE_ACTIVE=1 || ERRORS+=("$SERVICE not active")

DETAIL="$(curl -fsS --max-time 10 "http://127.0.0.1:8000/api/market/NIFTY" 2>/dev/null \
  | python -c 'import json,sys
try:
    d=json.load(sys.stdin)
    print(json.dumps({"capture":d.get("capture",{}).get("coverage"),
                      "spot":d.get("index",{}).get("ltp"),
                      "as_of":d.get("as_of")}))
except Exception:
    print("{}")' 2>/dev/null || echo '{}')"

STATUS_ARGS=(--out /tmp/deploy-status.json --revision "${REVISION:-none}"
             --commit "$(git rev-parse --short HEAD)" --branch "$CURRENT_BRANCH"
             --detail "$DETAIL")
(( HEALTHY )) && STATUS_ARGS+=(--healthy)
(( SERVICE_ACTIVE )) && STATUS_ARGS+=(--service-active)
for e in ${ERRORS[@]+"${ERRORS[@]}"}; do STATUS_ARGS+=(--error "$e"); done
python -m index_option_brain.deploy.cli status "${STATUS_ARGS[@]}" >/dev/null

# Publish. An orphan branch holding one file per host: it shares no history
# with the code, so a status push can never touch a source branch and a
# force-push here cannot lose a commit anyone cares about.
log "publishing status to $STATUS_BRANCH"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
if git worktree add --quiet --detach "$WORK" 2>/dev/null; then
  (
    cd "$WORK"
    git checkout --quiet --orphan "$STATUS_BRANCH" 2>/dev/null || git checkout --quiet "$STATUS_BRANCH"
    git rm -rqf . 2>/dev/null || true
    mkdir -p status
    cp /tmp/deploy-status.json "status/$(hostname).json"
    git add -A
    git -c user.name="index-brain-agent" -c user.email="agent@localhost" \
        commit --quiet -m "status: $(hostname) $(date -Is)" || true
    git push --quiet --force origin "HEAD:$STATUS_BRANCH" || echo "status push failed (no write credential?)"
  ) || log "status publish failed"
  git worktree remove --force "$WORK" 2>/dev/null || true
fi

log "done — healthy=$HEALTHY active=$SERVICE_ACTIVE errors=${#ERRORS[@]}"
cat /tmp/deploy-status.json
