# TII Monitor

A desktop window that shows, live, everything happening in study TII-CEE-1: the plan and its
gates, Slurm work on HiPerGator, the AI workers on this Mac, commits, and the connection between
the two machines. It only reads. Design and cadences: `DESIGN.md`. Cluster file formats:
`DATA_CONTRACT.md`.

## Run it

    open "monitor/src-tauri/target/release/bundle/macos/TII Monitor.app"

The first time, macOS may ask whether TII Monitor may access files in the Documents folder,
because the repository lives there. Choose Allow. The window says so while it waits; worker
activity and the connection are live meanwhile.

Demo mode, with fictional data and a DEMO banner:

    "monitor/src-tauri/target/release/bundle/macos/TII Monitor.app/Contents/MacOS/tii-monitor" --demo

Command-W closes the window and Command-Q quits; with one window, both end the app. Drag the window
by its top bar. It opens at 1168 by 733 points, the size of this Mac's display, and cannot be made
smaller than 1120 by 700, below which its panels no longer fit one screen.

## Build it

    cd monitor
    pnpm install
    pnpm tauri build          # writes src-tauri/target/release/bundle/macos/TII Monitor.app

The repository's path is compiled in from where the app was built, so rebuild after moving the
repository, or set `TII_MONITOR_REPO`.

## What it reads

| Source | How | Used for |
|---|---|---|
| `~/.claude/projects/<this repository>/*.jsonl` and `*/subagents/agent-*.jsonl` | Appended bytes only; tool names, descriptions and file paths, never tool outputs | AI workers |
| Git, branch `tii-cee-1/2026-11-13`, from commit `632bd18` | `git log`, `git diff --numstat`, `git ls-files --others`, with `GIT_OPTIONAL_LOCKS=0` | Commits and lines |
| `reports/research-advancement/` | Specification tables, authorization tables, STATUS.md, BLOCKED.md, the ledger's experiment register, experiment cards, GATE_DECISIONS/ | Plan board, blocked items, hover text, GPU limits and allowances |
| `tii-cee/results/figure1/*.json` | Three optional files | Figure 1 |
| `~/Library/Logs/tii-hpg-keeper.log` | Whole file | Connection cable |
| `ssh -O check daniel.escalante@hpg.rc.ufl.edu` | Local control command | SSH lamp; gate for every cluster read |
| `launchctl print gui/<uid>/<label>` | For the keeper, caffeinate and the local loop | Launch agents |
| HiPerGator, under `/blue/cai6734/daniel.escalante/private/tii-cee-1/` | See below | Jobs, GPUs, gates, progress, events |

## The rules for HiPerGator, and where they are enforced

All of it is in `src-tauri/src/remote.rs`, the only file that runs `ssh` with a remote command.

- The only commands are `cat`, `tail` and `ls`, on an allowlist: `state/` (list),
  `state/status.json`, `state/runlog.jsonl`, `state/events.tsv`, `state/progress/*.json`,
  `gates/*.json`, and `export/approved/` (list only). A read is built through a checked constructor;
  anything else is refused before a process is started. No Slurm command exists in the code.
- The transport is exactly `ssh -o BatchMode=yes -o ControlMaster=no -o ProxyCommand=false
  daniel.escalante@hpg.rc.ufl.edu <command>`, so it can only ride the existing connection.
- Before each read the master is checked with `ssh -O check`. If that fails, no ssh call is made and
  the last known state stays on screen with its age. The last known state is kept in
  `~/Library/Application Support/edu.ufl.tii-monitor/` so that it survives a restart.
- Reads are 120 seconds apart by default (the control beside "Read now" changes it) and never
  closer than 60 seconds, across restarts: the start time is written to disk before the first call.
- One read is at most: `ls state`, `cat status.json`, `tail -c` of the run log and the event queue
  from the last offset, `ls` and `tail` of new progress and gate files, `ls -R export/approved`.
  Hovering "next read" in the window lists the exact commands of the last read.

`cargo test` in `src-tauri/` covers the allowlist, the refusals, the 60-second floor, the
no-call-when-down rule, and the parsers.

## What the webview may do

Three commands: `get_state`, `set_remote_interval`, `read_cluster_now`. Its capability file grants
window dragging and zoom only: no shell, no filesystem, no network. The content security policy
allows nothing but the app's own files and its IPC channel. Fonts are bundled.

## Layout of this folder

    src/                 the window: Svelte 5, hand-drawn SVG, d3-scale for scales only
      lib/study.ts       what is fixed about the study: board layout, dependencies, glossary
      lib/derive.ts      pure functions from sources to what is drawn
      panels/            one file per region
    src-tauri/src/       all file, git and ssh access
    fixtures/            fictional data for demo mode; `pnpm fixtures` regenerates it

## Developing

    pnpm check                      # type-check the window
    (cd src-tauri && cargo test)    # backend tests
    pnpm snapshot && pnpm dev       # the window in a browser at http://localhost:1420/?mode=demo
                                    # (or ?mode=real), from one-off snapshots in .snapshot/
