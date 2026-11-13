# Source on HiPerGator. The owner file pins shared-home state to one login node.
tsup() {
    local dir="$HOME/.local/share/tailscale" owner here attempt
    owner=$(cat "$dir/owner" 2>/dev/null)
    here=$(hostname -s)
    if [ -z "$owner" ] || [ "$owner" != "$here" ]; then
        echo "Tailscale belongs on ${owner:-an unrecorded node}; do not start it on $here." >&2
        echo "If that node is retired, verify its daemon is stopped before recording a new owner in $dir/owner." >&2
        return 1
    fi
    command tailscale --socket="$TS_SOCK" status >/dev/null 2>&1 && return 0
    if ! pgrep -u "$USER" -f "tailscaled.*$TS_SOCK" >/dev/null; then
        nohup tailscaled --tun=userspace-networking --statedir="$dir" \
            --socket="$TS_SOCK" </dev/null >"$dir/log" 2>&1 &
    fi
    for attempt in {1..20}; do
        if command tailscale --socket="$TS_SOCK" status --json >/dev/null 2>&1; then
            command tailscale --socket="$TS_SOCK" up --hostname=hipergator
            return $?
        fi
        sleep 0.5
    done
    echo "Tailscale did not become ready on $owner; inspect $dir/log." >&2
    return 1
}
