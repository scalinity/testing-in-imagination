# Gate G1, synthetic part: repairs needed by E0 and E1

Evaluated November 13, 2026, about 4:00 AM EST. Immutable once committed. G1's other parts (the
REVIEW-001 repair for E3; the preflight, manifest binding and information-boundary tests for the
training track) are evaluated in their own records; a failed repair blocks only its dependents
(specification 8, 14.1).

## Rule

The repairs an experiment needs are verified or specifically adjudicated, legacy outputs preserved,
affected claims qualified.

## Evidence for E0 and E1

| Repair | Verified by | Commit |
|---|---|---|
| REVIEW-002: streams namespaced by purpose and split; fingerprints independent of display ids; identical latent cases across arms | tests/test_twin_streams.py, test_twin_env.py | 9a10c23 |
| REVIEW-006: expected charged cost = p_available × cost, checked at p = 0, p = 1 and intermediate p against enumeration | tests/test_twin_env.py | 9a10c23 |
| REVIEW-011: byte-exact snapshot of df467ea with SHA256SUMS verified at import; snapshot digest available to manifests | tests/test_twin_source.py; git blob identity checked by the lead | 9a10c23 |
| Section 8.1 additions: counterfactual interface, exact predictive, posterior and information, reduced six-test twin with exact dynamic programming, three renderings, four variants with informative missingness held out by code | tests/test_twin_*.py (154 tests); conformance with the twelve published information gains to 1e-12 | 9a10c23 |
| Section 8.4 boundaries for the twin: unrevealed results and the hidden diagnosis cannot change an ordinary reader or selector; imagined outcomes never enter a real history | tests/test_twin_boundaries.py | 9a10c23 |
| E0, the instrument | results/E0/manifest.json: 166 fixtures pass at ef7c4ac | ef7c4ac |

Legacy outputs: Edwin Salcedo's branch and results are untouched; his historical matrix.json stays
recorded as incompletely bound (REVIEW-011).

## Result

MET for E0 and E1.

## Path taken

E1 proceeds on the committed twin and library.
