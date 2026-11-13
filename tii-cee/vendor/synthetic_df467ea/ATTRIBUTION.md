# Vendored snapshot: synthetic diagnosis study

**Author:** Edwin Salcedo
**Origin:** pull request #1 of this repository, branch `edwin-salcedo/synthetic-world-model`
**Commit:** `df467eadd07fe2bc761b618dafce8ed9f198641a`

This directory is a read-only snapshot. The files are byte-exact copies taken with
`git show df467ea:<path>` and are never modified here; repairs live in the `tiicee/twin` adapters.
The branch itself is not edited.

| File (path at the commit) | Git blob id at the commit |
|---|---|
| `extension/envs/synthetic_diagnosis/actions.py` | `f59f0ddca6b86d7a3ed499bd8d34f9de449139d2` |
| `extension/envs/synthetic_diagnosis/profiles.py` | `d89fc8d7359b7999700fa9e5689225bb4a57770a` |
| `extension/envs/synthetic_diagnosis/generators.py` | `3d81a2ff893f7d72d268e3c1af75b1a22b6ddeb8` |

`SHA256SUMS` lists the SHA-256 of each file. `tiicee/twin/source.py` verifies it on import and
refuses to load a snapshot that does not match. The SHA-256 of `SHA256SUMS` itself is the snapshot
digest recorded in result manifests (REVIEW-011).

What the study takes from these files: the class list and prior, the twelve tests and their costs,
the per-test finding likelihoods given the presenting phenotype, the atypical-presentation rate,
the availability of each test, the step limit, and the finding text used by one of the three
render templates. Every one of those numbers is Edwin Salcedo's design.
