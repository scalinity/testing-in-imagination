# BLOCKED items, study TII-CEE-1

Each item is something only Daniel can resolve. Each names what waits on it, the exact one-step
request, and the pre-registered path taken meanwhile. An item is closed by recording its
resolution here with the date; it is never deleted.

## BLOCKED-001: sign the compute addendum (opened November 13, 2026)

- What waits: GPU jobs on account `cai5724` and CPU jobs on the burst and CPU QOS of `cai5720`,
  `cai5724` and `cai5731`. Nothing else waits; every experiment can run on the `cai6734`
  allocation alone, only more slowly (8 GPUs at once instead of 16).
- Why it is blocked: Daniel's instruction of November 13, 2026 ("You can use any compute
  available on HiPerGator") widens AUTHORIZATION.md section 3, which lists `cai5724` as not
  authorized. The agent cannot extend that file, and the watcher and later sessions check only
  signed, hashed files, so the instruction is drafted as
  `reports/research-advancement/AUTHORIZATION_ADDENDUM_1.md` for signature.
- Request, one step: paste this paragraph in the session, or add it yourself to SIGNATURE.md:

  > I, Daniel Escalante, have reviewed and authorize AUTHORIZATION_ADDENDUM_1.md at reports/research-advancement/AUTHORIZATION_ADDENDUM_1.md, commit c5902ea4eb2e7d739bf73b83366e877f05dd32f2, SHA-256 f59cd6fa638d00b2d1bc9bdc4848ba9ae412f9abecb7cdfcfa83cb4475e05658. Record this paragraph verbatim in SIGNATURE.md as my signature, commit it, and bind the addendum's hash into every manifest.

- Meanwhile: every job runs on `cai6734` and `cai6734-b` as AUTHORIZATION.md section 3 states.
- Status: CLOSED November 13, 2026, 2:55 AM EST. Daniel confirmed the instruction directly in the
  session; both statements are recorded verbatim in SIGNATURE.md and bound to the addendum's
  SHA-256. GPU work may use 16 GPUs at once, 8 on each of `cai6734` and `cai5724`.

## BLOCKED-002: no independent credentialed cohort for E11 (opened November 13, 2026)

- What waits: E11 only (audit and replay on an independent clinical cohort, specification 12).
- Why it is blocked: AUTHORIZATION.md section 7 lists no other credentialed cohort, and launch
  check 7.4f found only `mimic-iv-ext-cdm` among the dataset directories in approved private
  storage (names only were read). The agent never requests or acquires credentialed data.
- Request, one step: if you hold credentialed access to another diagnostic cohort that already sits
  in approved private storage on HiPerGator, add it in a new authorization naming the dataset,
  version, path and its data use agreement. E11 would then run audit and replay only, with its card
  frozen before any access. If you hold none, no action is needed; the paper states the gap.
- Meanwhile: E7 (four public complete-record datasets) and the E2 reader panel are the study's
  evidence beyond its own agent and environment, as specification 12 states; they are breadth, not
  clinical validation.
- Status: OPEN.
