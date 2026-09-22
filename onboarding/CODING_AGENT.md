# Coding-agent onboarding — Testing in Imagination

## Do

- Treat `REPRODUCTION.md` as the source of truth for baseline claims; never invent Table 1 numbers.
- Prefer synthetic env work under `specs/env-and-tuples.md` until MIMIC access is confirmed.
- Log every run: Hydra overrides, seed, git SHA, metrics JSON path.
- Keep method changes as diffs against the locked research question (world-model planning vs model-free).

## Don’t

- Claim “we reproduced LA-CDM” because the repo cloned.
- Mix `mimic_imputed` observations into evaluation without labeling.
- Email teammates/advisor or push to shared remotes unless Daniel explicitly asks.
- Expand the disease set or action space without updating the spec.

## First code milestone

`SyntheticCDMEnv` + JSONL tuple dump matching §4 of `specs/env-and-tuples.md`.
