# Chatting to the system from Telegram

```bash
export TELEGRAM_BOT_TOKEN="123456:AA..."          # from @BotFather
export TELEGRAM_ALLOWED_CHAT_IDS="4242"           # required
python -m index_option_brain.chat
```

## Long polling, not a webhook

Telegram offers both. Long polling is the right one here for a concrete
reason: **no public URL, no certificate, no inbound port.** It works from
behind NAT and on a box whose DNS is not set up. The webhook gateway in
this repo is still waiting on an A record; this works today.

It also keeps the bot off the public internet entirely. A webhook receiver
for it would be a second public endpoint, and the one this repo already
has took three real bugs to get right.

## Getting your chat id

Message the bot once. It will ignore you — and write a log line saying
which chat id it refused. Put that number in `TELEGRAM_ALLOWED_CHAT_IDS`
and restart.

That is deliberate: an unknown chat gets **no reply at all**, not an
error, because an error confirms the bot is real and attached to something
worth probing. A bot token is a bearer credential — anyone holding it can
act as the bot, and anyone who finds the bot can message it — so the
allowlist is required and the process refuses to start without one.

## Commands

```
Reads
  /status          engine, poller and calendar state
  /brief NIFTY     what the engine sees and decided
  /alerts          chart alerts accepted, and how stale
  /cal             calendar: pending proposals and confirmed events

Decisions
  /confirm <id>    accept a calendar proposal
  /reject <id>     discard one

Safety
  /kill            stop every relay route now
  /killstatus      whether it is engaged
```

**There is no command that trades.** No buy, no sell, no size, no
enabling a route. Reads go through the console's read-only API over
loopback rather than by importing the engine — that API is provably
read-only (every route GET, HEAD or OPTIONS, with a test asserting it), so
a bot that only GETs it cannot make the system do anything whatever a
message says. A test asserts the command surface offers nothing that
trades, as a property rather than one handler at a time.

## `/kill` is one-way, and that is the point

It creates `var/RELAY_KILLED`. `kill_switch_engaged()` checks the
environment variable **and** that file on every dispatch, so the switch
takes effect immediately in a process the bot cannot otherwise reach.

There is no `/unkill`. Re-arming means deleting that file on the machine.
The asymmetry is the safety property: **chat can only ever move this
system toward doing less.** A switch that could be flipped back from chat
is one an argument in a group chat can turn off.

Two details worth knowing:

- A failure to create the file is reported loudly, never swallowed. A kill
  switch that silently failed to engage is worse than not having one,
  because you would stop watching.
- The file check uses `os.stat`, not `Path.exists()`. `exists()` returns
  False for *every* error, which makes "there is no kill file" and "I
  cannot see whether there is a kill file" the same answer. Only
  `FileNotFoundError` counts as absent; anything else engages the switch.

## The calendar: an agent proposes, you confirm

Spec §4 names four trigger types that are calendar facts rather than
measurements — RBI policy, the Budget, an index rebalance, a scheduled
release. `ScheduledEventCalendar` shipped with **no implementation** because
no free Indian source serves them, and inventing dates would have put
invented event risk into the Risk Engine's blackout logic.

An agent reading circulars is the right way to fill that gap, and
`calendar_entries` is where its output lands.

### Why a proposal changes nothing

An unverified date is dangerous in **both directions**. Believed, it makes
the system refuse to trade on a day nothing is happening. Missed, it lets
the system trade through a day something is. There is no conservative
default, because neither direction of a *wrong* date is the safe one —
which is exactly why the interface shipped empty rather than guessing.

So `StoredEventCalendar` returns `CONFIRMED` rows **and nothing else**. A
proposal is a message to you; it affects trading only once you answer. Same
rule `LearningEngine` already follows: agent output is an artifact a human
promotes, never a live write.

```
/cal
Pending — nothing below affects trading yet (1):
#3  Tue 07 Oct 2026 05:15 UTC  RBI  RBI MPC decision
     source: https://rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=…

/confirm 3
#3 confirmed: RBI MPC decision at 2026-10-07T05:15:00+00:00
It will take effect on the calendar's next refresh.
```

The **source is shown**, because confirming a date you cannot check is the
failure this table exists to avoid, and the first question about a bad
blackout is where the date came from.

A decision is not reversible from chat. An entry already decided comes
back unchanged — a blackout appearing and disappearing from a group chat
is a decision nobody can audit afterwards.

### Two thresholds, not one

The snapshot reloads every 10 minutes, so a confirmation takes effect in
minutes. It is reported *stale* only after 6 hours. Those are deliberately
different numbers: one answers "should I reload", the other answers
"should anyone trust this". Collapsing them makes a calendar merely due a
reload look untrustworthy, and one genuinely unreachable for hours look
fine.

Never refreshed counts as stale, because an empty calendar and an unloaded
one are indistinguishable from their contents and only one of them means
"nothing is scheduled".

### The proposer itself

Not built yet, and that is the honest state: the store, the state machine,
the deterministic calendar and the confirmation flow are done and tested,
and the LLM-backed proposer plugs in through the existing
`IntelligenceProvider` seam. Shipping an untested model call into a path
that can open a trading blackout would have been worse than shipping the
seam.

Everything it needs is in place: `CalendarStore.propose()` is idempotent
on `(name, starts_at)`, so a weekly re-run accumulates no duplicates and
cannot quietly reset a decided entry to PROPOSED — which is the dangerous
version of that bug, since it would silently drop a confirmed blackout.
