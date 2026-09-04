# Running it always-on, inside your tailnet

## What server, and what you actually have to do

**A 2 GB / 1–2 vCPU / 20 GB VPS is enough.** Measured: the process peaks at
**210 MB** after a full live cycle, and most of that is numpy, pandas and
scipy. Add Docker, Tailscale and the OS and the whole box sits near 550 MB.
Nothing here is CPU-bound — it polls a public endpoint every 20 seconds and
runs a few hundred Black-Scholes evaluations. The cheapest tier at any
provider will do; paying for more buys nothing until Postgres and a backtest
corpus arrive.

Latency does not matter either, and it is worth saying why: the poll interval
is 20 seconds and positions are held for hours or days, so the difference
between a host in Mumbai and one in Frankfurt is noise. Pick a region near
India if it costs the same — for a broker connection later it is marginally
tidier — but do not pay extra for it.

Providers that take a cloud-init file, which is what makes the setup
hands-off: Hetzner (cheapest), DigitalOcean (has a Bangalore region), AWS
Lightsail, Vultr, Linode. Prices move, so check them rather than trusting a
figure here; the relevant point is that this fits in the entry tier
everywhere.

### The three things nobody but you can do

Be clear-eyed about this: setup can be automated, **accounts cannot**.

1. **Create and pay for the server.** Card details, and often a phone
   verification.
2. **Create a Tailscale account** and generate an auth key. Free for personal
   use.
3. **Generate the Dhan credentials**, when you get to that — a browser login
   with 2FA behind it.

Everything after that is automated, and I never need access to the machine.

### The hands-off path

1. Sign up with a provider that accepts cloud-init.
2. Create a server: Ubuntu 24.04 LTS, smallest tier with 2 GB RAM.
3. Open `deploy/cloud-init.yaml`, paste your Tailscale auth key into the one
   `FILL IN` line, and paste the whole file into the provider's **user data**
   box.
4. Click create.

A few minutes later the console is at
`https://index-brain.<your-tailnet>.ts.net`. No SSH key is installed and no
password login is enabled — the only way in is your tailnet, over Tailscale
SSH. That is deliberate: a box holding broker credentials should not be
reachable from the internet by anyone, including you.

### How it stays current without anyone logging in

`deploy/self-update.sh`, on a ten-minute timer, pulls the tracked branch,
rebuilds, and **rolls back if the result does not become ready**. So changes
reach the machine by being pushed to the repository — nothing needs inbound
access to the server, which is the whole point.

The rollback is not decoration. Auto-deploying a branch means a bad commit
takes the system down while nobody is watching, so every update is verified
against `/ready` and reverted if it fails. If the previous commit is also
unhealthy the script says so and stops rather than looping, because that
means the fault is the feed or the host rather than the deploy.

Track a branch you control the merges into. Pointing it at a development
branch means every push is a deploy.


The engine has to run continuously, not on page load. NSE serves no
historical bars, so the only source of history is the aggregator observing
snapshots as the session proceeds — a process that starts when you open the
console and stops when you close it will never have a daily bar. The
background poller (`app/runner.py`) is what makes it accumulate; keeping that
process up is the whole point of this directory.

## Why tailnet-only, not public

The console will eventually hold broker credentials and be able to place
orders. Two consequences:

- It is served over **`tailscale serve`**, which is HTTPS reachable from your
  tailnet and nowhere else. There is no port published on the host, no
  reverse proxy to misconfigure, and no login page to get wrong — the
  identity check is the tailnet's.
- **`tailscale funnel`** would put the same page on the public internet. The
  config file leaves `AllowFunnel` empty on purpose. If you ever enable it,
  put real authentication in front first; right now there is none, because
  none was needed for a tailnet-only surface.

## Docker (recommended)

```bash
cp .env.example .env
# Put an ephemeral, pre-authorized, TAGGED auth key in TS_AUTHKEY:
#   https://login.tailscale.com/admin/settings/keys
docker compose up -d
```

The console is then at `https://index-brain.<your-tailnet>.ts.net`, from any
device signed into your tailnet.

Two details in the compose file worth knowing about:

- The brain container uses `network_mode: service:tailscale`, so it shares the
  sidecar's network namespace. That is what lets Tailscale reach
  `127.0.0.1:8000` while the app is not exposed on the host at all.
- **One worker, deliberately.** The bar aggregator holds observed history in
  memory. A second worker would build a second, different history, and the
  console would show whichever one happened to answer the request.

Tag the auth key (`--advertise-tags`) so the node is owned by the tag rather
than by your user account. An untagged node expires with your key and the
console silently drops off the tailnet months later.

## Bare metal / VPS

```bash
sudo useradd --system --create-home --home-dir /opt/index-option-brain brain
sudo -u brain git clone https://github.com/neofinn/index-option-brain /opt/index-option-brain
cd /opt/index-option-brain
sudo -u brain python3.12 -m venv .venv
sudo -u brain .venv/bin/pip install -e .
sudo -u brain mkdir -p var
sudo -u brain cp .env.example .env   # then edit it

sudo cp deploy/index-option-brain.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now index-option-brain

# Expose it on the tailnet. Not on 0.0.0.0 — the unit binds loopback only.
sudo tailscale serve --bg --https=443 http://127.0.0.1:8000
sudo tailscale serve status
```

The unit is hardened (`ProtectSystem=strict`, `NoNewPrivileges`,
`MemoryDenyWriteExecute`, outbound-only address families) because this process
holds credentials once a broker is connected. `RestartSec=15s` is long on
purpose: restarting fast against a rate-limited feed turns a temporary block
into a longer one.

## What to monitor

| Endpoint | Meaning |
|---|---|
| `/health` | Liveness. 200 while the process is up, **regardless of the feed**. |
| `/ready` | Readiness. 503 when the poll loop has failed its last three attempts. |
| `/api/runner` | Counted loop stats: cycles, failures, events, analyses run. |

Watch **`/ready`**, not `/health`. The distinction is deliberate: a process
that is alive but blind looks identical to a healthy one from `/health`, and
that is exactly the failure an unattended box needs to surface. Equally,
`/health` must *not* fail on a data outage — a process manager restarting the
app because NSE is rate-limiting would turn a data problem into an
availability problem and throw away every bar observed that session.

`/api/runner` is how you tell "running and quiet" from "running and broken".
They look the same in a single snapshot; `successful_cycles` climbing with
`events_significant` flat is the first, `consecutive_failures` climbing is the
second.

A useful expectation once it is up: `analyses_run` should be far *lower* than
`cycles`. That is the event engine working — it re-runs the brain when
something changed, not on a timer. Measured on a closed market: six cycles,
one significant event, one analysis.

## Bars survive restarts

Observed bars are snapshotted to `BAR_STORE_DIR` (default `var/bars`), one
file per symbol and interval, written atomically. They are reloaded on start,
so a deploy no longer costs a session of history — which matters because NSE
serves no history and a week of 5-minute bars is a week of uptime.

Snapshots happen every 20 successful cycles as well as on a clean shutdown,
because a crash is not a shutdown: periodic writes mean a `kill -9` costs the
bars since the last snapshot rather than the whole session.

A snapshot that cannot be read cleanly is discarded rather than partially
loaded, and the aggregator starts cold. That is deliberate: starting cold
costs a session, while a series seeded from a truncated file is a wrong
series that no downstream indicator can detect.

Mount it. In Docker the container root is read-only, so add a volume for it:

```yaml
    volumes:
      - ./var:/app/var
```

The full §27 Postgres schema is still to be built; this covers the specific
thing that hurt.

## Updating

```bash
git pull
docker compose up -d --build     # Docker
# or
sudo systemctl restart index-option-brain   # bare metal
```
