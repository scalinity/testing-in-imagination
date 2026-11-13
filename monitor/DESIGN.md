# TII monitor: design

One reader (Daniel), one Mac, open all day beside the work. Its job: answer "what is happening in
study TII-CEE-1 right now" without asking anyone. It is live: nothing on it needs a click to update.

## What the design takes from the subject

The study's own rule is that an imagined result is a prediction and never an observation, and that
"no panel is drawn with data that do not exist" (specification 1.7, 18). The monitor uses that rule
as its visual grammar, because it has the same problem: plans, estimates and stale readings sit
beside live facts and must never be mistaken for them.

| Mark | Means |
|---|---|
| Solid | Observed: a recorded job, a spent GPU-hour, a commit, a logged check |
| Hollow (outline only) | Planned, estimated or pending: a plan, an estimate, an allowance, time in the queue, uncommitted work |
| Hatched, 45 degrees | No data: a source that is missing or has not started, or a gap in a log |
| Faded, with an amber clock | Last known state, shown with its age |

The study also treats a gate that is not met as a pre-registered path and a missed date as
acceptable (specification 1.5, 17.3). So lateness and "gate not met" are drawn in plain ink, never
in an alarm color.

## Color: hue means state, quantity is ink

The tokens are the project site's own (`site/meeting/index.html`). The window follows the macOS
appearance, so both sets exist.

| Token | Dark | Light | Used for |
|---|---|---|---|
| ground | `#16181b` | `#f3f4f6` | Window |
| raised | `#1e2125` | `#ffffff` | Title bar, seam, tooltips |
| line / line-2 | `#2d3137` `#3e434b` | `#dfe2e6` `#c3c8cf` | Hairlines, grids |
| ink / ink-2 / ink-3 | `#eceef0` `#b4bac2` `#8f969f` | `#16181b` `#4a5059` `#6b727c` | Text and every quantity mark |
| done | `#4b515a` | `#c9ced5` | Finished work: solid, and quiet |
| live | `#3987e5` | `#2a78d6` | Running, alive, in flight, now |
| wait | `#fab219` | `#e09a00` | Waits on Daniel: blocked, sign-in needed, stale |
| retry | `#e87ba4` | `#e87ba4` | Ended early and resubmitted by the watcher |
| fail | `#d03b3b` | `#d03b3b` | Failed, refused, unreadable |

- There is no categorical palette. Accounts, code areas and tool kinds are told apart by position
  (mirrored bands, small multiples, the line a mark sits on), so hue has one meaning on the screen.
- There is no green. Finished work is solid and recedes, so attention goes to what is alive or
  needs a decision.
- Every state color ships with a glyph and a word; none carries meaning alone.
- The four state hues were run through the data-viz palette validator against each surface, all
  pairs. They clear the colorblind separation target and the normal-vision floor on both. Amber and
  pink sit below 3:1 contrast on the light surface, which is why they are always paired with a
  glyph and a label. Retry is pink rather than orange because orange sat too close to amber
  (normal-vision distance 13.6, below the floor of 15), and the two meet on the connection cable.

## Type

Atkinson Hyperlegible Next for everything, the face the project site already uses; it keeps 0/O and
1/l/I apart at small sizes, which a screen of job IDs and hashes needs. Atkinson Hyperlegible Mono
only for strings that are code: job IDs, stage names, paths, account names. Both are bundled; no
font is fetched. Scale 11 / 12 / 13 (base) / 15 px; weights 400, 600, 700. Sentence case, no
letter-spaced capitals. Tabular figures in aligned columns and axis ticks. Left aligned.

## Layout: two machines and the wire between them

```
+------------------------------------------------------------------------------------------+
| TII-CEE-1  M0 M1 M2 M3 M4 M5        sources: seven lamps   live   Fri, Nov 13, 4:17 AM EST |  drag bar
| Waits on you: BLOCKED-002 no independent cohort for E11 ...                               |  only when open
+------------------------------------------------------------------+-----------------------+
| plan rail | Nov 14-19 | Nov 20-26 (today v) | Nov 27-Dec 3 |... | Figure 1 of the paper |
|  M0 -<>G0- E0 ---------------<>G2a-- E2 =====                    |  A the regime map     |
|            |   +- E1 [ABCD] -<>G2c.. E7 =====                    |  B the accounting     |
|            |   |            -<>G2b----------+                    |  C the attribution    |
|  M1 -<>G1--+---+- E3                        |                    |  (hatched, with the   |
|                +- E8 ======+- E4 -<>G3- E5 -<>G4- E6 - M5 -<>G5  |   experiments each    |
|                +- E9 ======+                        E11 -+       |   panel waits for)    |
|                +- E10 =====+                                     |                       |
+------------------------------------------------------+---+-------+-----------------------+
| HiPerGator  read 4 s ago  next read in 1 min  [Read] | S | This Mac  keeper caffeinate loop|
| watcher heartbeat | authorization | stages by status | S | AI workers: one trace per worker|
| Jobs: a row per stage, a bar per attempt | GPUs now  | H | lead      . : .|. :   53 calls  |
|                                          | GPU-hours |   | cee-lib   .:|..:.     63 calls  |
| GPUs in use, mirrored: cai6734 up,       | progress  | c | Commits and lines, a row per    |
| cai5724 down, limit lines at 8 and 8     | events    | a | area: tii-cee/ reports/ monitor/|
+------------------------------------------------------+ b +--------------------------------+
```

- Top: the plan board. Its columns are the plan's own windows from specification 17.2, with a
  calendar rail above and a today pointer. Experiments are chips, gates are diamonds, and each gate
  is drawn as a bus bar that releases what sits to its right. It runs into Figure 1, because every
  experiment feeds that figure.
- Bottom left: HiPerGator. Bottom right: this Mac. The narrow seam between them holds the SSH
  master's lamp and the keeper's log as a vertical cable, newest at the top. A hatched stretch of
  cable is time with no logged check; an amber stretch is time the connection was down.
- Open BLOCKED items sit directly under the drag bar, full width, and take no room when none is open.
- Regions are separated by hairlines on one surface. No cards, no shadows, no row of big numbers.

### Two sizes

The Mac's display is 1168 by 733 points, so that is the primary size and the window opens at it.
Below 1400 by 860 the geometry is compact: 22-pixel rows on the board, a 300-pixel figure column
with thumbnail panels, 31-pixel worker rows, shorter labels. Everything stays on one screen; only
lists scroll inside their own panel. Above that size the same layout breathes. Figure 1 opens
large on click, since its thumbnail is too small to read values from. The window cannot be made
smaller than 1120 by 700: below that the panels stop fitting one screen, and a monitor that needs
scrolling is no longer read at a glance.

### Why the board's columns are plan windows and not a time axis

The dates in 17.2 are deadlines per window. Gate G2a is due Nov 25 yet releases E2, which the plan
has running from Nov 20. On a linear axis the release would be drawn after the work it releases.
So columns are the plan's windows, nodes inside a column sit in dependency order, and each column
carries day ticks so that its own scale is visible. Each column's label equals the date cell of the
live specification; if the specification's table changes, the board says so.

## Charts, form by job

| Question | Form | Why |
|---|---|---|
| Where is the study against the plan | Plan board, columns by plan window | See above |
| What ran, retried, failed | Gantt: a row per stage, a bar per attempt, an end letter per end state | Attempts are intervals |
| GPUs in use against 8 + 8 | Mirrored step area, one account each side of zero, limit lines | Two series told apart by position; one axis |
| GPU-hours against estimate and allowance | Bullet bar: solid spent, tick at the estimate, hollow track to the allowance | One measure against two references |
| Stage progress | Thin meter with done and total | A fraction |
| What each worker is doing | Trace: 180 time slices, four lines (tell, write, run, read), mark height by count | Rhythm and gaps read at a glance; hover gives the calls |
| Commits and code growth | Small multiples on one time axis and one scale: step area of net lines, a dot per commit, insertions and deletions as numbers, a hollow bar for uncommitted lines | Three areas without three hues |
| Connection health | Vertical cable of keeper checks | Fits the seam; continuity is the message |
| Figure 1 | Scatter with arrows; scatter with the diagonal; segmented bars | Fixed by specification 1.7 |

No chart has two vertical scales. Limits and thresholds are dashed; grids are solid hairlines.

## Hover

One tooltip component. Every mark answers three things: what this is in plain words, taken from the
study's documents; the value; and where it came from and how old it is ("state/status.json on
HiPerGator, written by the watcher 3:15 AM EST, read through the SSH master 2 min ago"). Tooltip
content is derived reactively, so it stays current while the pointer rests. Keyboard focus shows
the same as hover. Clicking a worker's name opens its latest tool calls and files touched.

## Liveness

| Source | Read | Stale after |
|---|---|---|
| Claude Code transcripts | every 1 s: stat, then only the appended bytes | 10 s |
| Git on the task branch | every 2 s: three read-only commands (about 20 ms each) | 10 s |
| Study documents and Figure 1 files | every 2 s: stat fingerprint, re-read on change | 10 s |
| Keeper log | every 5 s | 30 s |
| SSH master (`ssh -O check`) | every 20 s, and before every cluster read | 60 s |
| launchd jobs | every 20 s | 60 s |
| HiPerGator files | every 120 s by default; 60 s is a hard floor | 2.5 intervals |

- The window asks its backend for a snapshot once a second. A part is re-sent, and re-rendered,
  only when its content changed, so nothing flickers and hover and scroll positions hold.
- Every panel carries a small age that ticks every second and turns amber with a clock when its
  source is older than the table allows.
- The title bar has the app's own heartbeat: a dot that blinks on each answer, and "no update for
  N s" in amber if the backend stops answering.
- The watcher's own heartbeat is a meter that fills over one 15-minute cycle and turns amber once
  it is 5 minutes overdue.

## Motion

One: running things breathe slowly in the live color. Off under reduced motion.

## What a generic answer would have been, and where this differs

A generic "research job dashboard" is a dark grid of rounded cards, a row of big-number tiles, green
and red pills, and a jobs table. This design has none of those. It keeps one thing that resembles a
default, a dark ground with a blue accent, on purpose: it is the project's existing identity, the
window also follows the light appearance, and blue here is a state, not decoration.

## Demo mode

Started only with `--demo`. Every source is replaced by fictional files in `fixtures/`, the clock
is the fixtures' own, and a full-width striped banner says nothing on screen is real. The default
mode reads only real sources.
