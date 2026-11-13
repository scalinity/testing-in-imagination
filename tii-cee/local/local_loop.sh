#!/bin/bash
# local_loop.sh -- the Mac side of study TII-CEE-1 (WATCHER_DESIGN.md section 5). launchd runs it
# every 600 seconds.
#
# What it does, only while an authenticated HiPerGator SSH master connection already exists:
# reads the new lines of the watcher's event queue and its STATUS.md into a local mirror, copies
# export/approved/ (aggregates that passed the export guard; nothing else is ever copied), shows a
# notification per new event, and starts ONE headless agent session in the repository when a stage
# completed, a gate was decided, something failed or something is blocked.
# What it never does: open a connection, log in, or touch a password or a Duo prompt. With no
# master it exits quietly; the work on HiPerGator does not depend on it.
set -u

CONF="${TII_LOCAL_LOOP_CONF:-$HOME/.tii/local-loop/local_loop.conf}"
TII_AGENT_CMD=()
[ -f "$CONF" ] || { echo "local_loop: no settings file at $CONF" >&2; exit 1; }
. "$CONF"
HOST="${TII_HPG_HOST:-hpg}"
REMOTE_ROOT="${TII_REMOTE_ROOT:?set TII_REMOTE_ROOT in local_loop.conf}"
REPO="${TII_REPO:?set TII_REPO in local_loop.conf}"
STATE="${TII_LOOP_STATE:-$HOME/Library/Application Support/tii-local-loop}"
APPROVED="${TII_APPROVED_DIR:-$STATE/approved}"
NOTIFY="${TII_NOTIFY:-1}"
LOG="${TII_LOOP_LOG:-$HOME/Library/Logs/tii-local-loop.log}"
SESSION_LOG="${TII_SESSION_LOG:-$HOME/Library/Logs/tii-agent-session.log}"
MIRROR="$STATE/mirror"
PIDFILE="$STATE/agent.pid"
LOCK="$STATE/lock"
BATCH=50   # events handled per run; the rest wait for the next one

# The fixed continuation prompt of WATCHER_DESIGN.md section 5, step 5.
PROMPT="Read STATUS.md, RESEARCH_LEDGER.md, BLOCKED.md, and the newest gate records. Continue study TII-CEE-1 under the version 3.1 kickoff from the first unfinished item. Do not repeat completed work."

mkdir -p "$STATE" "$MIRROR" "$APPROVED" "$(dirname "$LOG")" "$(dirname "$SESSION_LOG")"

now() { date '+%Y-%m-%dT%H:%M:%S%z'; }
log() { printf '%s %s\n' "$(now)" "$*" >> "$LOG"; }
num() { case "${1:-}" in ''|*[!0-9]*) echo 0 ;; *) echo "$1" ;; esac; }

# Title and text are passed as arguments, never pasted into the AppleScript source (hpg_keeper.sh).
notify() {
    osascript -e 'on run argv' \
              -e 'display notification (item 2 of argv) with title (item 1 of argv)' \
              -e 'end run' -- "$1" "$2" </dev/null >/dev/null 2>&1 || true
}

# Run a command, kill it after N seconds (macOS has no timeout command). Unlike the keeper's
# version, the watchdog also ends its own sleep, so nothing lingers after a five-minute limit.
with_timeout() {
    local secs=$1; shift
    "$@" &
    local pid=$!
    ( trap 'kill "$nap" 2>/dev/null' TERM
      sleep "$secs" & nap=$!
      wait "$nap" && kill -TERM "$pid" 2>/dev/null ) >/dev/null 2>&1 &
    local watchdog=$!
    wait "$pid"; local rc=$?
    kill -TERM "$watchdog" 2>/dev/null; wait "$watchdog" 2>/dev/null
    return $rc
}

# The options that make a command unable to turn into a login attempt (hpg_keeper.sh):
#   BatchMode=yes        no password or Duo prompt can be shown or answered
#   ControlMaster=no     never create a master
#   ProxyCommand=false   if the master is not usable, a direct connection fails locally at once
VIA_MASTER="-o BatchMode=yes -o ControlMaster=no -o ProxyCommand=false"
via_master() { ssh $VIA_MASTER "$HOST" "$@"; }

# Keep the log small.
if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt 1000000 ]; then
    tail -n 2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi

# 1. One run of this loop at a time, and nothing to do while an agent session for the study runs.
if ! mkdir "$LOCK" 2>/dev/null; then
    age=$(( $(date +%s) - $(num "$(stat -f %m "$LOCK" 2>/dev/null)") ))
    [ "$age" -gt 1800 ] || exit 0
    rmdir "$LOCK" 2>/dev/null
    mkdir "$LOCK" 2>/dev/null || exit 0
    log "took over a lock left by a run that never finished"
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

session=$(num "$(cat "$PIDFILE" 2>/dev/null)")
if [ "$session" -gt 0 ] && kill -0 "$session" 2>/dev/null; then
    log "agent session $session is still running"
    exit 0
fi

# 2. A control command that contacts no server. No master: exit quietly; never try to connect.
ssh -O check "$HOST" >/dev/null 2>&1 || exit 0

# 3. New lines of the event queue, and STATUS.md, into the local mirror. The first line of the
#    answer is the queue's length, so a queue that was started over is noticed.
seen=$(num "$(cat "$STATE/events.seen" 2>/dev/null)")
events=$(printf '%q' "$REMOTE_ROOT/state/events.tsv")
status=$(printf '%q' "$REMOTE_ROOT/state/STATUS.md")
tmp=$(mktemp "$STATE/ev.XXXXXX") || exit 0
trigger=0
if with_timeout 40 via_master "test -f $events || exit 0; wc -l < $events; tail -n +$((seen + 1)) $events | head -n $BATCH" > "$tmp" 2>/dev/null; then
    TIME_RE='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}[+-][0-9]{4}$'
    TEXT_RE='^[a-z][a-z0-9_]*=[-A-Za-z0-9_.:+/]+( [a-z][a-z0-9_]*=[-A-Za-z0-9_.:+/]+)*$'
    {
        IFS= read -r total || total=""
        total=$(num "${total//[[:space:]]/}")
        if [ "$total" -lt "$seen" ]; then
            echo 0 > "$STATE/events.seen"
            log "the remote event queue is shorter than what was read; starting again from its first line"
        else
            n=0; shown=0; dropped=0
            while IFS= read -r line; do
                n=$((n + 1))
                IFS=$'\t' read -r when kind text extra <<< "$line"
                # Only lines of the watcher's own shape are shown or kept: a time, a known kind,
                # and name=value tokens. Anything else is counted and dropped unseen.
                case "${kind:-}" in
                    "stage completed"|"gate decided"|"blocked"|"failure") wake=1 ;;
                    "study completed"|"all remaining items blocked") wake=0 ;;
                    *) dropped=$((dropped + 1)); continue ;;
                esac
                if ! [[ "${when:-}" =~ $TIME_RE && "${text:-}" =~ $TEXT_RE && -z "${extra:-}" ]]; then
                    dropped=$((dropped + 1)); continue
                fi
                printf '%s\n' "$line" >> "$MIRROR/events.tsv"
                [ "$NOTIFY" = 1 ] && notify "TII: $kind" "$text"
                shown=$((shown + 1))
                [ "$wake" -eq 1 ] && trigger=1
            done
            if [ "$n" -gt 0 ]; then
                echo $((seen + n)) > "$STATE/events.seen"
                log "read $n event line(s): $shown relayed, $dropped dropped"
            fi
        fi
    } < "$tmp"
fi
rm -f "$tmp"

if with_timeout 40 via_master "cat $status" > "$MIRROR/STATUS.md.tmp" 2>/dev/null && [ -s "$MIRROR/STATUS.md.tmp" ]; then
    mv "$MIRROR/STATUS.md.tmp" "$MIRROR/STATUS.md"
else
    rm -f "$MIRROR/STATUS.md.tmp"
fi

# 4. Approved aggregates, over the same master and the same options. The guard only ever approves
#    JSON files, so only JSON files are copied. No other path is ever copied.
with_timeout 300 rsync -a --delete --include='*/' --include='*.json' --exclude='*' \
    -e "ssh $VIA_MASTER" "$HOST:$REMOTE_ROOT/export/approved/" "$APPROVED/" >/dev/null 2>&1 \
    || log "approved aggregates were not copied this time"

# 5. One headless agent session, in the repository, with the fixed prompt. The command line comes
#    from the settings file; the prompt is its last argument. The pid file goes when the session ends.
if [ "$trigger" -eq 1 ]; then
    if [ "${#TII_AGENT_CMD[@]}" -eq 0 ]; then
        log "an event calls for an agent session but TII_AGENT_CMD is not set"
    elif [ ! -d "$REPO" ]; then
        log "an event calls for an agent session but the repository is not at TII_REPO"
    else
        ( trap - EXIT
          cd "$REPO" || exit 1
          nohup "${TII_AGENT_CMD[@]}" "$PROMPT" >> "$SESSION_LOG" 2>&1 < /dev/null
          rm -f "$PIDFILE" ) >/dev/null 2>&1 &
        echo $! > "$PIDFILE"
        log "started agent session $!"
    fi
fi

# 6. The lock is released by the EXIT trap.
exit 0
