#!/bin/bash
# deploy_code.sh -- sends the study's committed code to HiPerGator. Nothing is pushed to GitHub.
#
# It bundles the study branch (git bundle), copies the bundle and the deploy step to <TII>/deploy/
# on HiPerGator through the existing SSH master, and runs the deploy step there. That step
# (tiicee/pipeline/deploy.py) fetches the bundle into the cluster checkout as the same branch,
# writes a read-only code export, <TII>/code/<commit>/, and points <TII>/code/current at it.
# Stages already under way keep running from their own export; new stages use the new one.
#
# Only committed work is sent. Settings come from the local loop's settings file.
# What it never does: open a connection, log in, or touch a password or a Duo prompt. With no
# master it names the one command to run and stops.
#
# Exit codes: 0 deployed; 1 settings or repository; 3 no connection; 4 the copy failed;
# 5 HiPerGator refused the deployment (the reason is printed as "refused <code>").
set -u

CONF="${TII_LOCAL_LOOP_CONF:-$HOME/.tii/local-loop/local_loop.conf}"
TII_AGENT_CMD=()
[ -f "$CONF" ] || { echo "deploy_code: no settings file at $CONF" >&2; exit 1; }
. "$CONF"
HOST="${TII_HPG_HOST:-hpg}"
REMOTE_ROOT="${TII_REMOTE_ROOT:?set TII_REMOTE_ROOT in local_loop.conf}"
REPO="${TII_REPO:?set TII_REPO in local_loop.conf}"
BRANCH="${TII_BRANCH:-tii-cee-1/2026-11-13}"
CHECKOUT="${TII_REMOTE_CHECKOUT:-/blue/cai6734/daniel.escalante/testing-in-imagination}"
REMOTE_PYTHON="${TII_REMOTE_PYTHON:-/usr/bin/python3}"
DEPLOY_STEP="tii-cee/tiicee/pipeline/deploy.py"

say() { echo "deploy_code: $*"; }
fail() { echo "deploy_code: $2" >&2; exit "$1"; }
q() { printf '%q' "$1"; }

# Run a command, kill it after N seconds (macOS has no timeout command). The command keeps this
# shell's standard input, which a background command would otherwise lose.
with_timeout() {
    local secs=$1; shift
    "$@" <&0 &
    local pid=$!
    ( trap 'kill "$nap" 2>/dev/null' TERM
      sleep "$secs" & nap=$!
      wait "$nap" && kill -TERM "$pid" 2>/dev/null ) </dev/null >/dev/null 2>&1 &
    local watchdog=$!
    wait "$pid"; local rc=$?
    kill -TERM "$watchdog" 2>/dev/null; wait "$watchdog" 2>/dev/null
    return $rc
}

# The options that make a command unable to turn into a login attempt (hpg_keeper.sh).
VIA_MASTER="-o BatchMode=yes -o ControlMaster=no -o ProxyCommand=false"
via_master() { ssh $VIA_MASTER "$HOST" "$@"; }

# One file to HiPerGator: written under a temporary name and renamed only if every byte arrived.
# The storage root is never created here: the launch session makes it, with owner-only permissions.
send() {
    local size
    size=$(wc -c < "$1" | tr -d ' ')
    with_timeout 300 via_master "test -d $(q "$REMOTE_ROOT") || exit 44; mkdir -p $(q "$(dirname "$2")") && cat > $(q "$2.part") && test \"\$(wc -c < $(q "$2.part") | tr -d ' ')\" = $size && mv $(q "$2.part") $(q "$2")" < "$1"
}

# A control command that contacts no server. No master: say so and stop; never try to connect.
ssh -O check "$HOST" >/dev/null 2>&1 \
    || fail 3 "no HiPerGator connection is open. In Terminal run: ssh -fN $HOST"

commit=$(git -C "$REPO" rev-parse --verify --quiet "refs/heads/$BRANCH^{commit}") \
    || fail 1 "branch $BRANCH is not in the repository at TII_REPO"
pending=$(git -C "$REPO" --no-optional-locks status --porcelain | wc -l | tr -d ' ')
[ "$pending" -eq 0 ] || say "$pending uncommitted change(s) are not part of this deployment"

work=$(mktemp -d "${TMPDIR:-/tmp}/tii-deploy.XXXXXX") || fail 1 "could not make a temporary folder"
trap 'rm -rf "$work"' EXIT
bundle="$work/$commit.bundle"
git -C "$REPO" bundle create "$bundle" "refs/heads/$BRANCH" >/dev/null 2>&1 || fail 1 "git bundle failed"
# The deploy step as that commit has it, so it also works before any export exists on the cluster.
git -C "$REPO" show "$commit:$DEPLOY_STEP" > "$work/deploy.py" 2>/dev/null \
    || fail 1 "$DEPLOY_STEP is not in commit $commit"
digest=$(shasum -a 256 "$bundle" | cut -d' ' -f1)

there="$REMOTE_ROOT/deploy"
send "$bundle" "$there/$commit.bundle"
case $? in
    0) ;;
    44) fail 4 "the storage root does not exist on HiPerGator yet; create it first, owner-only: $REMOTE_ROOT" ;;
    *) fail 4 "the bundle did not reach HiPerGator whole" ;;
esac
send "$work/deploy.py" "$there/deploy.py" || fail 4 "the deploy step did not reach HiPerGator whole"

result=$(with_timeout 600 via_master "$(q "$REMOTE_PYTHON") $(q "$there/deploy.py") --root $(q "$REMOTE_ROOT") --checkout $(q "$CHECKOUT") --branch $(q "$BRANCH") --bundle $(q "$there/$commit.bundle") --sha256 $digest" </dev/null)
printf '%s\n' "$result"
case "$result" in
    *"\"commit\": \"$commit\""*) say "commit $commit is deployed; new stages will run from it" ;;
    *) fail 5 "HiPerGator did not deploy commit $commit" ;;
esac
