# Module boundaries, study TII-CEE-1

Specification section 16, written down in M0. Each module has conformance fixtures and changes only
with a recorded decision in RESEARCH_LEDGER.md. Code lives in `tii-cee/` (package `tiicee`, Python
3.11 to 3.12, numpy 1.26.4 only, put on `sys.path` by jobs and never installed on the cluster).

| Module | Path | Contract | Conformance fixtures | Depends on |
|---|---|---|---|---|
| Interfaces | tiicee/interfaces.py | Reader.belief(h), Reader.read(h, a, o) → probability vector over the class list; WorldModel.outcomes(h, a, k, rng) → Predictive with weights summing to one, UNAVAILABLE included; ClassConditionalWorldModel.budget(h, a) | Exercised by every module below | numpy |
| CEE library | tiicee/cee/ | project, gain_terms, realized_gain, scale, tau_budget, tau_ratio, coherent_voi, select, commit, gate, both tracks, multi-step accounting, and the audit command (section 9.1) | E0 (tests/test_cee_*.py) | interfaces |
| Counterfactual environment | tiicee/twin/ over vendor/synthetic_df467ea/ | all_outcomes, true_predictive, true_posterior, true_information, exact_value, render, fingerprints; readers R0 and R4; world models W0, W1, W2, W2-cc and controlled errors | tests/test_twin_*.py | interfaces; Edwin Salcedo's snapshot (read-only, SHA256SUMS verified at import) |
| Statistics | tiicee/stats/ | paired_interval(records, metric, unit), t5_contrast, holm, mde, hierarchical_interval; every experiment uses these and computes no interval of its own | Planned fixtures against tim/analysis.py's recorded implementations | numpy |
| Record schema | tiicee/records.py (planned) | One row per decision point per arm with the fields of section 16; clinical rows stay private; only schema-approved aggregates leave | Planned | interfaces |
| Pipeline | tiicee/pipeline/, gates/, local/ | Stage contract, manifests with the authorization digests, watcher cycle, export guard, surface check, local loop | tests/test_pipeline_*.py, tests/test_local_loop*.py against a fake Slurm | standard library only (watcher on Python 3.9) |
| Clinical adapters | tiicee/clinical/ (planned, M1 and M3) | R1 probe over the replication runtime; W3 retrieval and W4 generative sampler on HiPerGator; R2 and the REVIEW-001 matched scoring over PR #9's content model at acf5254 | Probe conformance (section 10.1); REVIEW-001 fixtures | tim, the replication, the pinned upstream, on the cluster only |

Teammates' code is reached only through these adapters. Matching function names is not
conformance: each adapter carries fixtures for observation, cost and stopping semantics. Edwin
Salcedo's synthetic study (PR #1) and the content model of PR #9 are used as they stand, never
edited, and credited wherever their code or results are used.

## Output namespace

- Synthetic and public results: `tii-cee/results/<experiment>/` in the repository, aggregates and
  manifests only.
- Credentialed work: everything under `/blue/cai6734/daniel.escalante/private/tii-cee-1/private/`
  on HiPerGator; only export-guarded aggregates reach `…/tii-cee-1/export/approved/` and, through
  the local loop, the Mac.
- Legacy study outputs (cdm-extension, the replication, PR #1, PR #9) are never rewritten;
  corrections are errata.
