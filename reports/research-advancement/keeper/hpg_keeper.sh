#!/bin/bash
# hpg_keeper.sh -- keeps an existing, already authenticated HiPerGator SSH master connection healthy
# and tells you when you need to sign in again.
#
# What it does: checks the master connection, sends one trivial command through it, relays study
# events as notifications, and notifies you if the session has dropped.
# What it never does: log in, open a new connection, read, store or replay a password or a Duo
# code, or change any security setting. If the master is gone it only tells you.
set -u
PATH=/usr/bin:/bin:/usr/sbin:/sbin

CONF="$HOME/.tii/keeper/keeper.conf"
[ -f "$CONF" ] && . "$CONF"
HOST="${TII_HPG_HOST:-hpg}"
REMOTE_EVENTS="${TII_REMOTE_EVENTS:-}"
REMIND_HOURS="${TII_REMIND_HOURS:-0}"
RECOVERY_HINT="${TII_RECOVERY_HINT:-}"
SSH_BIN="${TII_SSH_BIN:-/usr/bin/ssh}"                 # override only for testing
OSA_BIN="${TII_OSASCRIPT_BIN:-/usr/bin/osascript}"     # override only for testing
STATE="${TII_KEEPER_STATE:-$HOME/Library/Application Support/tii-keeper}"
LOG="${TII_KEEPER_LOG:-$HOME/Library/Logs/tii-hpg-keeper.log}"
mkdir -p "$STATE" "$(dirname "$LOG")"

now() { date '+%Y-%m-%dT%H:%M:%S%z'; }
log() { printf '%s %s\n' "$(now)" "$*" >> "$LOG"; }

# Title and text are passed as arguments, never pasted into the AppleScript source.
notify() {
    "$OSA_BIN" -e 'on run argv' \
               -e 'display notification (item 2 of argv) with title (item 1 of argv)' \
               -e 'end run' -- "$1" "$2" >/dev/null 2>&1 || true
}

num() { case "${1:-}" in ''|*[!0-9]*) echo 0 ;; *) echo "$1" ;; esac; }

# Run a command, kill it after N seconds (macOS has no timeout command).
with_timeout() {
    local secs=$1; shift
    "$@" &
    local pid=$!
    ( sleep "$secs"; kill -TERM "$pid" 2>/dev/null ) >/dev/null 2>&1 &
    local watchdog=$!
    wait "$pid"; local rc=$?
    kill -TERM "$watchdog" 2>/dev/null; wait "$watchdog" 2>/dev/null
    return $rc
}

# True if a master connection for HOST is running. A control command; it contacts no server.
master_up() { "$SSH_BIN" -O check "$HOST" >/dev/null 2>&1; }

# A command through the master that cannot turn into a login attempt:
#   BatchMode=yes        no password or Duo prompt can be shown or answered
#   ControlMaster=no     never create a master
#   ProxyCommand=false   if the master is not usable, a direct connection fails locally at once
via_master() {
    "$SSH_BIN" -o BatchMode=yes -o ControlMaster=no -o ProxyCommand=false "$HOST" "$@"
}

relay_events() {
    [ -n "$REMOTE_EVENTS" ] || return 0
    local seen tmp n used line rest kind text
    seen=$(num "$(cat "$STATE/events.seen" 2>/dev/null)")
    tmp=$(mktemp "$STATE/ev.XXXXXX") || return 0
    if with_timeout 40 via_master "tail -n +$((seen + 1)) $(printf '%q' "$REMOTE_EVENTS") 2>/dev/null | head -n 20" > "$tmp" 2>/dev/null; then
        n=0
        used=0
        while IFS= read -r line || [ -n "$line" ]; do
            used=$((used + 1))
            case "$line" in *$'\t'*$'\t'*) ;; *) continue ;; esac
            rest=${line#*$'\t'}
            kind=${rest%%$'\t'*}
            text=${rest#*$'\t'}
            [ -n "$kind" ] || continue
            notify "TII: ${kind:0:60}" "${text:0:200}"
            n=$((n + 1))
        done < "$tmp"
        if [ "$used" -gt 0 ]; then echo $((seen + used)) > "$STATE/events.seen"; fi
        if [ "$n" -gt 0 ]; then log "relayed $n event(s)"; fi
    fi
    rm -f "$tmp"
}

# Keep the log small.
if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt 1000000 ]; then
    tail -n 2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi

if master_up; then
    if with_timeout 40 via_master true >/dev/null 2>&1; then
        rm -f "$STATE/fail.count"
        if [ -f "$STATE/down.since" ]; then
            rm -f "$STATE/down.since" "$STATE/down.notified"
            notify "HiPerGator" "Connection is back. The study loop will pick up from here."
            log "reconnected"
        fi
        log "ok"
        relay_events
        exit 0
    fi
    fails=$(( $(num "$(cat "$STATE/fail.count" 2>/dev/null)") + 1 ))
    echo "$fails" > "$STATE/fail.count"
    log "master present but did not answer ($fails in a row)"
    # A master that fails three checks in a row is dead weight; close it so your next sign-in is clean.
    if [ "$fails" -lt 3 ]; then exit 0; fi
    "$SSH_BIN" -O exit "$HOST" >/dev/null 2>&1 || true
    rm -f "$STATE/fail.count"
    log "closed unresponsive master"
fi

# No usable master: tell the person. Never try to connect.
now_s=$(date +%s)
[ -f "$STATE/down.since" ] || echo "$now_s" > "$STATE/down.since"
last=$(num "$(cat "$STATE/down.notified" 2>/dev/null)")
REMIND_HOURS=$(num "$REMIND_HOURS")
if [ "$last" -eq 0 ] || { [ "$REMIND_HOURS" -gt 0 ] && [ $((now_s - last)) -ge $((REMIND_HOURS * 3600)) ]; }; then
    notify "HiPerGator sign-in needed" "In Terminal run: ssh -fN $HOST   Your jobs on HiPerGator keep running meanwhile.${RECOVERY_HINT:+ $RECOVERY_HINT}"
    echo "$now_s" > "$STATE/down.notified"
    log "notified: sign-in needed"
else
    log "down, already notified"
fi
exit 0
