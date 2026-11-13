# Demo fixtures

Everything under this folder is fictional. `generate.mjs` invents the jobs, the numbers, the worker
activity and the dates, in the formats of `../DATA_CONTRACT.md`, so that `TII Monitor --demo` can show
every panel in every state. None of it is study data, and the monitor reads it only in demo mode.

- `hpg/` mirrors the study root on HiPerGator: `state/`, `gates/`, `export/approved/`.
- `mac/` stands in for the Mac's own sources: transcripts, keeper log, launchd output, git output,
  the study documents that change, and the three Figure 1 files.
- `demo.json` holds the demo's clock.

Regenerate with `pnpm fixtures`.
