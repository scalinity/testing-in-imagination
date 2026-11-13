# HiPerGator connection keeper for macOS: install, use, uninstall

November 13, 2026. For Daniel's Mac, native Terminal. The five files sit beside this guide in `reports/research-advancement/keeper/`. They hold no personal values: the username is a placeholder.

## What it is

Five small files that hold your already authenticated HiPerGator SSH session open and tell you when you need to sign in again.

| File | Goes to | Purpose |
|---|---|---|
| `ssh_config_hpg.block` | appended to `~/.ssh/config` | Connection multiplexing and keepalives for the `hpg` host |
| `hpg_keeper.sh` | `~/.tii/keeper/` | Every 5 minutes: checks the master connection, sends `true` through it, relays study events, notifies you if the session dropped |
| `keeper.conf.example` | `~/.tii/keeper/keeper.conf` | Settings; no secrets |
| `com.tii.hpg-keeper.plist` | `~/Library/LaunchAgents/` | Runs the script every 300 seconds |
| `com.tii.caffeinate.plist` | `~/Library/LaunchAgents/` | Runs `caffeinate -i -s` so the Mac does not idle-sleep |

## What it never does

It never logs in, never opens a new connection, never reads, stores, or replays your password or a Duo code, and never touches the Duo prompt. If the master connection is gone, it only shows a notification. Every command it sends goes through the existing master with `BatchMode=yes`, `ControlMaster=no`, and `ProxyCommand=false`, so a missing master produces a local failure, not a login attempt against UF.

## What UF Research Computing's documentation says, and what it does not

Read on November 13, 2026:
- The SSH multiplexing page presents multiplexing as the way to avoid going through MFA on every connection, with `ControlMaster auto` and `ControlPersist 8h`, and suggests `ControlPath ~/.ssh/control/%C` on macOS when the socket path is too long. Another of their pages uses `ControlPersist 12h`.
- That page states no maximum session length and says nothing about keepalives or scripts.
- Their usage policy limits what runs on login nodes. The keeper sends one `true` every five minutes and, if you enable it, one `tail` of a small text file.

So the mechanisms are the documented ones. Holding the session open for longer than the 8 or 12 hours in their examples is not prohibited by anything I found, and is not explicitly permitted either. If you want that in writing, ask UF Research Computing: "Is it acceptable to keep one multiplexed SSH master connection open with ServerAliveInterval for several days, sending a trivial command every five minutes?" The keeper does not defeat any server-side limit: if HiPerGator closes the session, it stays closed until you sign in.

## Limits you should know

- `caffeinate` prevents idle sleep, and system sleep on AC power. It does not keep a MacBook awake with the lid closed unless it is on power with an external display. When the Mac sleeps, the connection will usually drop. That is fine: the watcher on HiPerGator keeps every job running, and you sign in once when you are back.
- These files were syntax-checked and the script's logic was exercised with stub commands on Linux. They have not been run on macOS.
- The first notification may ask you to allow notifications for Script Editor, which is how `osascript` shows them.

## If HiPerGator is already reached through an existing master connection

The repository's earlier work reaches HiPerGator through a shell alias, `sshhiper`, that runs `ssh daniel.escalante@hpg.rc.ufl.edu` over an SSH ControlMaster. Do not duplicate that setup.

1. Find the entry that already applies: `ssh -G hpg.rc.ufl.edu | grep -i -E '^(user|controlmaster|controlpath|controlpersist|serveraliveinterval) '`.
2. If a master is already running for the usual destination (`ssh -O check daniel.escalante@hpg.rc.ufl.edu`), keep that entry and its `ControlPath`. Add to it only the lines it lacks from `ssh_config_hpg.block`: `ServerAliveInterval 60`, `ServerAliveCountMax 5`, `TCPKeepAlive yes`, and `ControlPersist 12h` if no persist value is set. Skip step 3 below.
3. Set `TII_HPG_HOST` in `~/.tii/keeper/keeper.conf` to the destination that `ssh -O check` accepted, so the keeper and the existing master share one connection. Add a `Host hpg` alias only if it resolves to the same `ControlPath`; a second master would need a second Duo approval.

The keeper never starts a master. If none is running, the one command to run is `ssh -fN hpg`, or `sshhiper` where that alias is the configured one.

## Install

Run from `reports/research-advancement/keeper/`. The GatorLink username on record in this repository is `daniel.escalante`.

```bash
# 1. Folders
mkdir -p ~/.tii/keeper ~/.ssh/control ~/Library/LaunchAgents
chmod 700 ~/.ssh ~/.ssh/control

# 2. Script and settings
cp hpg_keeper.sh ~/.tii/keeper/hpg_keeper.sh
chmod 700 ~/.tii/keeper/hpg_keeper.sh
cp keeper.conf.example ~/.tii/keeper/keeper.conf

# 3. SSH config. Replace YOUR_GATORLINK with your username.
#    If ~/.ssh/config already has a "Host hpg" entry, do not run this line: open the file and add
#    the six option lines from ssh_config_hpg.block (ControlMaster ... TCPKeepAlive) to your entry.
sed 's/GATORLINK_USERNAME/YOUR_GATORLINK/' ssh_config_hpg.block >> ~/.ssh/config
chmod 600 ~/.ssh/config

# 4. launchd agents
sed "s|__HOME__|$HOME|g" com.tii.hpg-keeper.plist > ~/Library/LaunchAgents/com.tii.hpg-keeper.plist
cp com.tii.caffeinate.plist ~/Library/LaunchAgents/com.tii.caffeinate.plist
plutil -lint ~/Library/LaunchAgents/com.tii.hpg-keeper.plist ~/Library/LaunchAgents/com.tii.caffeinate.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tii.hpg-keeper.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tii.caffeinate.plist

# 5. Sign in once. Enter your password and approve Duo yourself; the command then returns
#    and the master connection stays in the background.
ssh -fN hpg
ssh -O check hpg          # should print "Master running"

# 6. Run the keeper once now and look at its log
launchctl kickstart -k gui/$(id -u)/com.tii.hpg-keeper
tail -n 5 ~/Library/Logs/tii-hpg-keeper.log     # should end with "ok"
```

Set `TII_REMOTE_EVENTS` in `~/.tii/keeper/keeper.conf` to `/blue/cai6734/daniel.escalante/private/tii-cee-1/state/events.tsv`, the study's event file on HiPerGator, to get a notification for each completed stage, gate decision, blocked item, failure, and completion.

## Daily use

- Open a HiPerGator shell any time with `ssh hpg`. No new Duo prompt while the master is up.
- If you see "HiPerGator sign-in needed", run `ssh -fN hpg` once.
- To close the session yourself: `ssh -O exit hpg`.
- To see what the keeper has done: `tail -n 20 ~/Library/Logs/tii-hpg-keeper.log`.

## Uninstall

```bash
launchctl bootout gui/$(id -u)/com.tii.hpg-keeper
launchctl bootout gui/$(id -u)/com.tii.caffeinate
rm -f ~/Library/LaunchAgents/com.tii.hpg-keeper.plist ~/Library/LaunchAgents/com.tii.caffeinate.plist
ssh -O exit hpg 2>/dev/null
rm -rf ~/.tii/keeper "$HOME/Library/Application Support/tii-keeper"
rm -f ~/Library/Logs/tii-hpg-keeper.log ~/Library/Logs/tii-hpg-keeper.out.log ~/Library/Logs/tii-hpg-keeper.err.log
# Remove the block this guide appended to ~/.ssh/config (skip if you merged the lines by hand):
sed -i '' '/# --- TII HiPerGator keeper (begin) ---/,/# --- TII HiPerGator keeper (end) ---/d' ~/.ssh/config
```
