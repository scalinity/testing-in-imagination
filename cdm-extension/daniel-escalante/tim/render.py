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
              "would usually go undetected at this sample size.\n",
              "## Limitations\n",
              "- Previously examined benchmark (see the disclosure above).",
              "- One policy seed: between-seed variation is unassessable.",
              "- The scorer is the policy's own likelihood, not a calibrated posterior; verbalizer and length effects remain.",
              "- No clinical-safety inference.",
              "- Person-level overlap is unassessable: the release ships no subject_id.",
              "- Baseline deviations D-1, D-3, D-5 and D-6 (REPRODUCTION.md) and the Blackwell runtime carry over.", ""]
    return "\n".join(lines)
