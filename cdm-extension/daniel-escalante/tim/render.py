"""REPORT.md for Phase A, rendered only from metrics.json (SPEC 7); no number is typed by hand."""
from tim.enums import DISEASES

LADDER = (("official", "Official primary rows (comparator)"), ("V0-rerun", "V0-rerun (guard off)"),
          ("V1", "V1 = official + G3"), ("V2-raw", "V2 raw (G2)"), ("V2", "V2 = G2 + G3"),
          ("V3-raw", "V3 raw (G1 + G2)"), ("V3", "V3 = G1 + G2 + G3 (method)"))
NAMES = {"V3-official": "V3 − official", "V1-official": "V1 − official", "V3-V1": "V3 − V1", "V2-V1": "V2 − V1",
         "V3-V2": "V3 − V2", "V2raw-official": "V2 raw − official", "V3raw-official": "V3 raw − official",
         "V3-V0rerun": "V3 − V0-rerun", "V1-V0": "V1 − V0", "V3-V0": "V3 − V0"}
BANNER = ("> **SYNTHETIC** — invented records from the CPU fixture at fixture scale. These numbers exercise "
          "the pipeline and say nothing about the method.\n\n")


def _n(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def _signed(value):
    return "—" if value is None else f"{value:+.2f}"


def _interval(c):
    return f"{_signed(c['delta_t5'])} [{_n(c['ci95'][0])}, {_n(c['ci95'][1])}]"


def _p(value):
    return "—" if value is None else (f"{value:.4f}" if value >= 0.0001 else "< 0.0001")


def _p_clause(value):
    return "p < 0.0001" if value is not None and value < 0.0001 else f"p = {_p(value)}"


def render(metrics: dict, disclosure: str) -> str:
    variants, primary = metrics["variants"], metrics["primary"]
    lines = [BANNER if metrics["source"] == "synthetic" else "",
             "# Phase A: commit guard on the LA-CDM baseline\n",
             "## Framing\n",
             "Phase A is termination recovery and action validation under the frozen metric. Its fallback (G3) "
             "reuses the hypothesis agent's answer, so it is not a new diagnostic predictor, and it supports no "
             "clinical-safety claim. The comparator is the replication's trained Nemotron Nano 8B policy (run "
             "7fcd9e5, seed 269, greedy); these results are not a reproduction of the paper.\n",
             f"Test-set status: a prospectively frozen follow-up analysis on a previously examined benchmark. "
             f"{disclosure}\n",
             "## Baseline recomputation\n",
             f"T5 recomputed from the official rows: {_n(variants['official']['metrics']['T5'], 4)} "
             f"(reference {_n(metrics['t5_reference'], 4)}). V0-rerun reproduced "
             f"{metrics['v0_rerun']['identical']}/{metrics['v0_rerun']['total']} official predictions"
             + (", so every Phase-A claim below is qualified and V0-rerun is reported as a second comparator.\n"
                if metrics["v0_rerun"]["qualified"] else ".\n"),
             "## Ladder\n",
             "| Arm | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 ($) | Nonanswers |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, label in LADDER:
        m, block = variants[key]["metrics"], variants[key]
        lines.append(f"| {label} | " + " | ".join(_n(m[f"T{i}"]) for i in range(1, 9)) + f" | {block['nonanswers']} |")
    controls = metrics["controls"]
    lines += ["", "Analytic references on the official nonanswers (expected T5 gains, not results): "
              f"uniform random fallback {_signed(controls['uniform_random'])}, training-majority fallback "
              f"{_signed(controls['training_majority'])}, perfect recovery {_signed(controls['perfect_recovery'])}.\n",
              "## Recovery and regressions\n",
              "| Arm | Recovery yield | Newly answered | Accuracy among newly answered | Regressions |",
              "|---|---|---|---|---|"]
    for key in ("V1", "V2-raw", "V2", "V3-raw", "V3"):
        r = metrics["recovery"][key]
        lines.append(f"| {dict(LADDER)[key]} | {_n(r['recovery_yield'], 3)} ({r['correct_recoveries']}/"
                     f"{r['baseline_nonanswers']}) | {r['newly_answered']} | {_n(r['newly_answered_accuracy'], 3)} | "
                     f"{r['regressions']} |")
    lines += ["", "## Termination, requests and cost\n",
              "| Arm | Diagnosis | Step limit | Malformed | Completion overflow | Observation overflow | "
              "Requested | Unavailable | Repeated | Guard replacements | Scorer errors |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in ("V0-rerun", "V2", "V3"):
        block = variants[key]
        t, q, e = block["termination"], block["requests"], block["events"]
        lines.append(f"| {dict(LADDER)[key]} | {t['diagnosis']} | {t['step_limit']} | {t['malformed']} | "
                     f"{t['length_completion']} | {t['length_observation']} | {q['requested']} | {q['unavailable']} | "
                     f"{q['repeated']} | {e['g1_replace'] + e['g2_commit']} | {block['scorer_errors']} |")
    lines += ["", "## Primary result\n",
              f"ΔT5 (V3 − official) = {_interval(primary)} percentage points over {variants['official']['n']} test "
              f"admissions; two-sided paired sign-flip randomization {_p_clause(primary['p_value'])} "
              f"({primary['method']}, {primary['discordant']} discordant). The difference is "
              f"**{primary['significance']}**. The test is exact for the exchangeability null, not for the weaker "
              "null ΔT5 = 0 when diseases move in opposite directions.\n",
              "## Family A (Holm, α = 0.05)\n", "| Comparison | ΔT5 [95% CI] | p | Holm p | Statement |",
              "|---|---|---|---|---|"]
    for key, c in metrics["family_a"].items():
        lines.append(f"| {NAMES[key]} | {_interval(c)} | {_p(c['p_value'])} | {_p(c['p_holm'])} | {c['significance']} |")
    lines += ["", "## Descriptive comparisons (intervals only)\n", "| Comparison | ΔT5 [95% CI] |", "|---|---|"]
    for key, c in {**metrics["descriptive"], **metrics["second_comparator"]}.items():
        lines.append(f"| {NAMES[key]} | {_interval(c)} |")
    lines += ["", "## Per-class recall (%)\n", "| Arm | " + " | ".join(DISEASES) + " |", "|---|---|---|---|---|"]
    for key in ("official", "V1", "V3"):
        m = variants[key]["metrics"]
        lines.append(f"| {dict(LADDER)[key]} | " + " | ".join(_n(m[f"T{i}"]) for i in range(1, 5)) + " |")
    lines += ["", "## Sensitivity (validation-based simulation at the test class sizes)\n"]
    for key, s in metrics["sensitivity"].items():
        lines.append(f"- {NAMES[key]}: validation effect {_signed(s['val_effect'])} points, power at that effect "
                     f"{_n(s['power_at_val_effect'], 3)}, minimum detectable effect at 80% power {_n(s['mde_80'])} points.")
    lines += ["", "A null result is read against these figures: an effect smaller than the minimum detectable effect "
              "would usually go undetected at this sample size.\n", "## Interpretation (SPEC §11)\n"]
    significant = {key: c["significance"] == "statistically significant" for key, c in metrics["family_a"].items()}
    primary_significant = primary["significance"] == "statistically significant"
    if significant["V1-official"] and not significant["V3-V1"]:
        lines.append("V1 − official is statistically significant but V3 − V1 is not: the hypothesis agent already held "
                     "the missing answers, and the online guards added nothing measurable.")
    if not primary_significant:
        lines.append("V3 − official is not statistically significant. Read it against the ladder, recovery yield "
                     "versus the analytic references, regressions, termination changes and the sensitivity figures: "
                     "G3 is the V1 row, G2 the V2 rows and G1 the step from V2 to V3.")
    if primary_significant and not (significant["V1-official"] and not significant["V3-V1"]):
        lines.append("The primary comparison is statistically significant; the Family-A rows separate the "
                     "fallback (V1) from the online guards (V3 − V1).")
    lines += ["", "## Protocol notes\n",
              "- `prereg/amendments/A1-code-fixes.md`: code fixes made before test access after an independent "
              "review, including the disclosure that the first smoke job's loader parsed the train and test files.",
              "- `prereg/amendments/A2-reference-scoring.md`: decisions use the reference scoring path after "
              "bench-a found the fast path outside its tolerance.", "",
              "## Limitations\n",
              "- Previously examined benchmark (see the disclosure above).",
              "- One policy seed: between-seed variation is unassessable.",
              "- The scorer is the policy's own likelihood, not a calibrated posterior; verbalizer and length effects remain.",
              "- No clinical-safety inference.",
              "- Person-level overlap is unassessable: the release ships no subject_id.",
              "- Baseline deviations D-1, D-3, D-5 and D-6 (REPRODUCTION.md) and the Blackwell runtime carry over.", ""]
    return "\n".join(lines)


B_NAMES = {"B-learned-s0-B-uniform": "B-learned-s0 − B-uniform (information beyond the reranking mechanism)",
           "B-learned-s0-B-marginal": "B-learned-s0 − B-marginal (patient-specific beyond test prevalence)",
           "B-oracle-B-uniform": "B-oracle − B-uniform (privileged availability, same mechanism)"}
PROBE_NAMES = {"hpi": "HPI TF-IDF", "hpi_bits": "HPI TF-IDF + 12 availability bits", "bits": "Availability bits only (descriptive)"}


def render_b(metrics: dict, disclosure: str) -> str:
    variants, family = metrics["variants"], metrics["family_b"]
    lines = [BANNER if metrics["source"] == "synthetic" else "",
             "# Phase B: retrospective chart-availability reranking\n",
             "## Framing\n",
             "Phase B estimates p(test present in chart | HPI) and reweights the policy's test requests on top of V3 "
             "(P0). It does not predict result content, does not update from revealed findings and does not evaluate "
             "imagined outcomes: it is retrospective chart-availability reranking, not full world-model planning. "
             "The oracle is a privileged chart-availability oracle; chart presence encodes clinicians' completed "
             "workups. These results are not a reproduction of the paper.\n",
             f"Test-set status: a prospectively frozen follow-up analysis on a previously examined benchmark. {disclosure}\n",
             "## Arms\n",
             "| Arm | T1 | T2 | T3 | T4 | T5 | T8 ($) | Nonanswers | Reranks | Stops |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    from tim.enums import PHASE_B_ARMS
    for arm in ("P0", *PHASE_B_ARMS):
        block = variants[arm]
        m, e = block["metrics"], block.get("events", {})
        lines.append(f"| {arm} | " + " | ".join(_n(m[f"T{i}"]) for i in (1, 2, 3, 4, 5, 8)) +
                     f" | {block['nonanswers']} | {e.get('wm_rerank', '—')} | {e.get('wm_stop', '—')} |")
    lines += ["", "## Family B (Holm, α = 0.05)\n", "| Contrast | ΔT5 [95% CI] | p | Holm p | Statement |",
              "|---|---|---|---|---|"]
    for key, c in family.items():
        lines.append(f"| {B_NAMES[key]} | {_interval(c)} | {_p(c['p_value'])} | {_p(c['p_holm'])} | {c['significance']} |")
    lines += ["", "## Every arm against P0 (descriptive, intervals only)\n", "| Arm − P0 | ΔT5 [95% CI] |", "|---|---|"]
    for arm in PHASE_B_ARMS:
        lines.append(f"| {arm} − P0 | {_interval(metrics['descriptive'][f'{arm}-P0'])} |")
    lines += ["", "## World-model quality\n",
              "| Split | Arm | α | ECE | AUROC | Brier | Brier skill vs marginal |", "|---|---|---|---|---|---|---|"]
    for split, arms in metrics["quality"].items():
        for arm, q in arms.items():
            lines.append(f"| {split} | {arm} | {_n(q['alpha'], 1)} | {_n(q['ece'], 3)} | {_n(q['auroc'], 3)} | "
                         f"{_n(q['brier'], 3)} | {_n(q['brier_skill'], 3)} |")
    lines += ["", "Brier skill is a descriptive summary only; calibration is measured, not assumed.\n",
              "## Leakage probes\n", "| Split | Probe | T5 |", "|---|---|---|"]
    for split, probes in metrics["leakage"].items():
        for probe, r in probes.items():
            lines.append(f"| {split} | {PROBE_NAMES[probe]} | {_n(r['T5'])} |")
    lines += ["", "A weak probe cannot certify that leakage is absent.\n", "## Interpretation (SPEC §4.5, §11)\n"]
    if not any(c["significance"] == "statistically significant" for c in family.values()):
        lines.append("No Family-B contrast is statistically significant. This is a finding about this fixed reranker at "
                     "the measured world-model precision, with the corruption arms and the oracle contrast above, and "
                     "no general claim about availability.")
    lines += ["Conclusions are limited to this fixed planner at the measured precision. Perfect information does not "
              "make the heuristic optimal, so a non-positive oracle contrast does not show that availability has no "
              "value. Learned seeds 1 and 2 and the corruption arms are descriptive.", "",
              "The inventory card and HPI audit are published beside this report (`inventory_card.json`, `hpi_audit.json`).\n",
              "## Limitations\n",
              "- Previously examined benchmark; one policy seed, so between-seed variation is unassessable.",
              "- Retrospective: availability in the chart reflects completed workups, not what a test would show.",
              "- No clinical-safety inference.", ""]
    return "\n".join(lines)
