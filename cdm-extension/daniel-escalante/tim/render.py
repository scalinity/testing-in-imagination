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


def _money(value):
    return "—" if value is None else f"${value:,.2f}"


def _holm(c):
    return f"Holm-adjusted {_p_clause(c['p_holm'])}"


def _significant(c):
    return c["significance"] == "statistically significant"


def _headline(official, v3, primary, n):
    if not _significant(primary):
        return (f"On the {n} test admissions, the commit guard (V3) scores mean per-class recall (T5) "
                f"{_n(v3['T5'])} against {_n(official['T5'])} for the trained policy: ΔT5 = {_interval(primary)} "
                f"percentage points, and the difference is not statistically significant "
                f"({_p_clause(primary['p_value'])}).")
    verb = "raises" if primary["delta_t5"] > 0 else "lowers"
    text = (f"On the {n} test admissions, the commit guard (V3) {verb} mean per-class recall (T5) from "
            f"{_n(official['T5'])} to {_n(v3['T5'])}: ΔT5 = {_interval(primary)} percentage points, statistically "
            f"significant (paired sign-flip randomization, {_p_clause(primary['p_value'])}).")
    if primary["delta_t5"] > 0:
        missed = (100 - official["T5"], 100 - v3["T5"])
        text += (f" That is a {_n(100 * primary['delta_t5'] / official['T5'], 1)}% relative increase, and "
                 f"the average share of each disease's patients that are missed falls from {_n(missed[0])}% to "
                 f"{_n(missed[1])}%, {_n(100 * (missed[0] - missed[1]) / missed[0], 1)}% fewer.")
    return text


PHASE_B_SHOWN = (("P0", "P0 = V3, no reranking"), ("B-learned-s0", "Learned availability (seed 0)"),
                 ("B-marginal", "Test prevalence only"), ("B-uniform", "Uniform availability"),
                 ("B-oracle", "Privileged oracle (true chart availability)"))


def render_writeup(a: dict, b: dict, paper: dict, disclosure: str) -> str:
    """WRITEUP.md: both phases in one short account (SPEC §7, M9); every number comes from the arguments."""
    va, primary, family_a = a["variants"], a["primary"], a["family_a"]
    official, v1, v3 = (va[k]["metrics"] for k in ("official", "V1", "V3"))
    n, rec = va["official"]["n"], a["recovery"]["V3"]
    paper_t1 = paper["table1"]
    paper_t5, gap = paper_t1["T5"]["value"], v3["T5"] - paper_t1["T5"]["value"]
    v1_off, v3_v1 = family_a["V1-official"], family_a["V3-V1"]
    t, rerun = va["V0-rerun"]["termination"], a["v0_rerun"]
    lines = [BANNER if a["source"] == "synthetic" else "",
             "# Commit guard and chart-availability reranking on LA-CDM\n",
             f"**Summary.** {_headline(official, v3, primary, n)} The comparator is the replication's trained "
             "Nemotron Nano 8B policy; these results are not a reproduction of the paper.\n",
             "## The question\n",
             f"The trained LA-CDM policy ends {va['official']['nonanswers']} of {n} test episodes without a "
             "diagnosis, and the frozen metric counts each of those as wrong. In the guard-off re-run "
             f"({rerun['identical']}/{rerun['total']} predictions identical to the official rows), {t['step_limit']} "
             f"hit the step limit, {t['length_completion'] + t['length_observation']} ran out of context "
             f"({t['length_completion']} while generating, {t['length_observation']} after an observation) and "
             f"{t['malformed']} ended on a malformed action. Phase A asks whether guarding the "
             "policy's actions and committing to an answer recovers those admissions. Phase B asks whether "
             "reweighting the policy's test requests by how likely each test is to be in the chart, given the "
             "history of present illness (HPI), helps further.\n",
             "## What was built\n",
             "- **G1 validity guard.** When the policy proposes a malformed, unknown or repeated action, the guard "
             "replaces it with the candidate the policy itself scores highest (teacher-forced log-probability).",
             "- **G2 budget commit.** At the last allowed step, or when the trajectory would overflow, the guard "
             "makes the policy commit to its highest-scoring diagnosis.",
             "- **G3 hypothesis fallback.** If an episode still ends without a diagnosis, the hypothesis agent's "
             "latest valid hypothesis becomes the answer. It reuses an answer the system already produced, so it "
             "is not a new diagnostic predictor.",
             "- **Phase B reranking.** A TF-IDF logistic regression per test estimates p(test present in chart | "
             "HPI); a fixed planner reweights the policy's test requests on top of V3. The prevalence-only, uniform, "
             "oracle and corrupted arms isolate where any change comes from.\n",
             "## How it was tested\n",
             f"- The protocol (`prereg/protocol.json`) was frozen before any test access; amendments A1–A3 record "
             f"code fixes made before the affected test runs. Test-set status: a prospectively frozen follow-up "
             f"analysis on a previously examined benchmark. {disclosure}",
             f"- Re-running the policy with every guard off reproduced {rerun['identical']}/{rerun['total']} of the "
             "official test predictions.",
             "- ΔT5 is tested with a two-sided paired sign-flip randomization test; intervals are 95% "
             "class-stratified paired bootstrap; comparisons within Family A and Family B are Holm-adjusted.\n",
             "## Phase A: commit guard\n",
             "| Arm | Mean recall (T5) | Micro F1 (T6) | Macro F1 (T7) | Avg test cost (T8) | Nonanswers |",
             "|---|---|---|---|---|---|"]
    for key, label in (("official", "Trained policy, official rows (comparator)"), ("V1", "V1: fallback only"),
                       ("V3", "V3: G1 + G2 + G3 (method)")):
        m = va[key]["metrics"]
        lines.append(f"| {label} | {_n(m['T5'])} | {_n(m['T6'])} | {_n(m['T7'])} | {_money(m['T8'])} | "
                     f"{va[key]['nonanswers']}/{va[key]['n']} |")
    lines += [f"| Paper, as reported | {_n(paper_t5)} | {_n(paper_t1['T6']['value'])} | "
              f"{_n(paper_t1['T7']['value'])} | {_money(paper_t1['T8']['value'])} | — |", "",
              f"- **Recovery.** V3 answered {rec['newly_answered']} of the {rec['baseline_nonanswers']} admissions the "
              f"trained policy left without a diagnosis, {rec['correct_recoveries']} of them correctly; "
              f"{rec['regressions']} admissions the policy had right became wrong.",
              f"- **Where the gain comes from.** V1 − official (fallback only): {_interval(v1_off)}, "
              f"{v1_off['significance']} ({_holm(v1_off)}). V3 − V1 (adding the online guards): {_interval(v3_v1)}, "
              f"{v3_v1['significance']} ({_holm(v3_v1)}).",
              f"- **Cost.** Average provided-test cost is {_money(v3['T8'])} for V3, {_money(v1['T8'])} for V1 and "
              f"{_money(official['T8'])} for the trained policy.",
              f"- **The paper's figure.** The paper reports T5 {_n(paper_t5)}; V3 is {_n(abs(gap))} points "
              f"{'above' if gap > 0 else 'below'} it. That figure comes from the authors' own run and has no interval "
              "here, so this comparison is descriptive, not a test.", "",
              "| Arm | " + " | ".join(DISEASES) + " |", "|---|---|---|---|---|"]
    for key, label in (("official", "Trained policy"), ("V1", "V1"), ("V3", "V3")):
        m = va[key]["metrics"]
        lines.append(f"| {label} | " + " | ".join(_n(m[f"T{i}"]) for i in range(1, 5)) + " |")
    lines += ["", "Per-class recall (%) on the test admissions.\n"]
    if _significant(v1_off) and not _significant(v3_v1):
        lines.append(f"The fallback alone (V1) scores {_n(v1['T5'])}, and the online guards change that by an amount "
                     "that is not statistically significant: the hypothesis agent already held the missing answers, "
                     "and the online guards added nothing measurable.\n")
    vb, family_b = b["variants"], b["family_b"]
    lines += ["## Phase B: chart-availability reranking\n",
              "| Arm | Mean recall (T5) | ΔT5 vs P0 [95% CI], descriptive | Avg test cost (T8) | Nonanswers |",
              "|---|---|---|---|---|"]
    for key, label in PHASE_B_SHOWN:
        m = vb[key]["metrics"]
        delta = "—" if key == "P0" else _interval(b["descriptive"][f"{key}-P0"])
        lines.append(f"| {label} | {_n(m['T5'])} | {delta} | {_money(m['T8'])} | "
                     f"{vb[key]['nonanswers']}/{vb[key]['n']} |")
    lines += ["", "| Family B contrast | ΔT5 [95% CI] | Holm-adjusted p | Statement |", "|---|---|---|---|"]
    for key, c in family_b.items():
        lines.append(f"| {B_NAMES[key]} | {_interval(c)} | {_p(c['p_holm'])} | {c['significance']} |")
    significant_b = [B_NAMES[key] for key, c in family_b.items() if _significant(c)]
    bits = b["leakage"]["test"]["bits"]
    lines += ["", ("No Family-B contrast is statistically significant: at the measured world-model precision, "
                   "patient-specific availability estimates did not measurably beat prevalence-only or uniform "
                   "weights under this fixed reranker. That is a finding about this reranker, not a general claim "
                   "about chart availability.\n" if not significant_b else
                   "Statistically significant Family-B contrasts: " + "; ".join(significant_b) + ".\n"),
              f"The learned model's test AUROC for chart presence is "
              f"{_n(b['quality']['test']['B-learned-s0']['auroc'], 3)}. The oracle arm sees which tests are actually "
              "in the chart, so its difference from P0 is descriptive only. It is privileged: "
              f"the availability bits alone predict the diagnosis with T5 {_n(bits['T5'])} on test, against "
              f"{_n(100 / len(DISEASES))} by chance, because chart presence encodes the clinicians' completed "
              "workup.\n",
              "## Limitations\n",
              "- A previously examined benchmark (see the test-set status above) and one policy seed, so "
              "between-seed variation is unassessable.",
              "- The guard scores candidates with the policy's own likelihood, not a calibrated posterior; "
              "verbalizer and length effects remain.",
              "- Phase B is retrospective: chart availability reflects completed workups, not what a test would show.",
              "- No clinical-safety inference; person-level overlap is unassessable because the release ships no "
              "subject_id.\n",
              "## More detail\n",
              "- Full reports: [`phase-a/REPORT.md`](phase-a/REPORT.md), [`phase-b/REPORT.md`](phase-b/REPORT.md).",
              "- Specification, preregistration and decisions: [`SPEC.md`](../../SPEC.md), "
              "[`prereg/`](../../prereg/), [`DECISIONS.md`](../../DECISIONS.md).", ""]
    return "\n".join(lines)


def render_slide(a: dict, b: dict, paper: dict) -> str:
    """SLIDE.md: one slide of headline numbers (SPEC §7, M9); every number comes from the arguments."""
    va, vb, primary = a["variants"], b["variants"], a["primary"]
    v1_off, v3_v1 = a["family_a"]["V1-official"], a["family_a"]["V3-V1"]
    paper_t1 = paper["table1"]
    significant_b = sum(_significant(c) for c in b["family_b"].values())
    lines = [BANNER if a["source"] == "synthetic" else "",
             f"# Commit guard on LA-CDM: mean recall {_n(va['official']['metrics']['T5'])} → "
             f"{_n(va['V3']['metrics']['T5'])}\n",
             "| | Mean recall (T5) | Nonanswers | Avg test cost |", "|---|---|---|---|"]
    for block, label in ((va["official"], "Trained policy (comparator)"), (va["V1"], "+ hypothesis fallback (V1)"),
                         (va["V3"], "**V3: guards + fallback (method)**"),
                         (vb["B-learned-s0"], "V3 + learned availability reranking")):
        m = block["metrics"]
        lines.append(f"| {label} | {_n(m['T5'])} | {block['nonanswers']}/{block['n']} | {_money(m['T8'])} |")
    lines += [f"| Paper, as reported | {_n(paper_t1['T5']['value'])} | — | {_money(paper_t1['T8']['value'])} |", "",
              f"- **ΔT5 (V3 − comparator) = {_interval(primary)}** points, {primary['significance']} (paired "
              f"sign-flip {_p_clause(primary['p_value'])}, {va['official']['n']} test admissions).",
              f"- The fallback carries the gain: V1 − comparator {_interval(v1_off)}, {v1_off['significance']}; "
              f"V3 − V1 {_interval(v3_v1)}, {v3_v1['significance']} (Holm).",
              f"- Availability reranking: {significant_b} of {len(b['family_b'])} Family-B contrasts statistically "
              f"significant (Holm); privileged oracle − V3 {_interval(b['descriptive']['B-oracle-P0'])}, descriptive.",
              "",
              "These results are not a reproduction of the paper. One policy seed, a previously examined benchmark, "
              "no clinical-safety inference. Every number is rendered from `phase-a/metrics.json`, "
              "`phase-b/metrics.json` and the replication's `paper_values.json`.", ""]
    return "\n".join(lines)


# ---- DIAGNOSTICS.md (amendment A4, exploratory) -------------------------------------------------

D_NAMES = {"B-oracle-B-uniform": "B-oracle − B-uniform", "B-oracle-P0": "B-oracle − P0"}
D_ARMS = (("P0", "P0 = V3, no reranking"), ("B-uniform", "B-uniform (no availability information)"),
          ("B-learned-s0", "B-learned-s0 (learned availability)"), ("B-oracle", "B-oracle (true chart availability)"))
D_CATEGORIES = (("lab", "Lab panels"), ("imaging", "Imaging"), ("exam", "Physical exam"))


def _verdict(c, threshold, val_b) -> str:
    """The one-line verdict of amendment A4 (D4), from the B-oracle − B-uniform contrast."""
    where, holm, bar = f"B-oracle − B-uniform is {_interval(c)} points", _holm(c), _n(threshold)
    requests = {arm: val_b["variants"][arm]["requests"]["unavailable"] for arm in ("B-uniform", "B-oracle")}
    acts = (f" The planner does act on the signal: unavailable requests fall from {requests['B-uniform']} under "
            f"B-uniform to {requests['B-oracle']} under B-oracle.")
    untested = (" The corruption dose-response was not run (it needs GPU runs), so this rests on the oracle "
                "contrast alone.")
    if _significant(c) and c["delta_t5"] > 0:
        return (f"**Verdict: signal bottleneck.** {where}, statistically significant ({holm}): perfect chart "
                "availability raises mean class recall through this fixed planner, so the shortfall lies in the "
                f"learned availability signal.{untested}")
    if _significant(c):
        return (f"**Verdict: planner bottleneck.** {where}, statistically significant ({holm}) and negative, "
                f"so below the {bar}-point threshold: perfect chart availability makes this fixed planner worse, "
                f"and a better availability predictor cannot help it.{acts}{untested}")
    if c["ci95"][1] < threshold:
        return (f"**Verdict: planner bottleneck.** {where}, not statistically significant ({holm}), and the "
                f"interval's upper bound is below {bar} points, the smallest effect the test design detects with "
                "80% power: even perfect chart availability gives this fixed planner no detectable gain, so a "
                f"better availability predictor cannot help it.{acts}{untested}")
    return (f"**Verdict: inconclusive.** {where}, not statistically significant ({holm}), but the interval reaches "
            f"{bar} points, the smallest effect the test design detects with 80% power, so neither a useful oracle "
            f"gain nor its absence is established.{untested}")


def _relevant_to(by_disease) -> str:
    names = [d for d in DISEASES if by_disease[d]]
    return "all four" if len(names) == len(DISEASES) else ", ".join(names) or "none"


def render_diagnostics(d: dict, val_b: dict, threshold: float, dose_hours: float) -> str:
    """DIAGNOSTICS.md from diagnostics.json and analyze-val-b's validation_b.json; threshold and GPU-hours
    come from the committed addenda A and B. No number is typed by hand."""
    from tim.diagnostics import COUNTS
    from tim.enums import TEST_CODES
    family, variants = d["family_d"], val_b["variants"]
    published, n = val_b["descriptive"]["B-oracle-P0"]["ci95"], val_b["variants"]["P0"]["n"]
    lines = [BANNER if d["source"] == "synthetic" else "",
             "# Phase-B validation diagnostics: planner or signal?\n",
             _verdict(family["B-oracle-B-uniform"], threshold, val_b) + "\n",
             "## Scope\n",
             f"- Exploratory and validation only: the {n} validation admissions of val-a and val-b, analyzed under "
             "amendment A4 (`prereg/amendments/A4-validation-diagnostics.md`), committed before this stage read any "
             "row. No test admission was read; nothing was tuned or selected.",
             "- analyze-val-b had already published every arm's validation T5, so these contrasts are not blind, "
             "and the verdict rule and its threshold were written after those point estimates were visible.",
             "- Conclusions hold for this fixed planner (SPEC §4.4) and one policy seed. These results extend a "
             "replication attempt and are not a reproduction of the LA-CDM paper.\n",
             "## Does perfect chart availability help this planner? (Family D)\n",
             "| Contrast | ΔT5 [95% CI] | p | Holm p | Discordant | Result |", "|---|---|---|---|---|---|"]
    for key, label in D_NAMES.items():
        c = family[key]
        lines.append(f"| {label} | {_interval(c)} | {_p(c['p_value'])} | {_p(c['p_holm'])} | {c['discordant']} | "
                     f"{c['significance']} |")
    match = "matches the published" if family["B-oracle-P0"]["ci95"] == published else "does not match the published"
    lines += ["", "Two-sided paired sign-flip randomization tests, class-stratified paired bootstrap intervals, Holm "
              f"within Family D at α = 0.05. B-oracle − P0's interval {match} analyze-val-b interval for the same "
              f"seed ([{_n(published[0])}, {_n(published[1])}]).\n",
              "## Does the planner act on the signal?\n",
              "Per-arm validation summaries from analyze-val-b:\n",
              "| Arm | T5 | Requested | Provided | Unavailable | Reranks | Planner stops |",
              "|---|---|---|---|---|---|---|"]
    for arm, label in D_ARMS:
        v = variants[arm]
        r, events = v["requests"], v.get("events", {})
        lines.append(f"| {label} | {_n(v['metrics']['T5'])} | {r['requested']} | {r['provided']} | "
                     f"{r['unavailable']} | {events.get('wm_rerank', 0)} | {events.get('wm_stop', 0)} |")
    lines += ["", "## Which tests did the reranker add or drop?\n",
              "Each arm against P0, patient by patient, each test counted once. *Extra* tests are ones the arm "
              "requested and P0 did not; *dropped* tests the reverse. *Provided* means the result was in the "
              "chart. *Relevant* means diagnosis-relevant to the patient's labeled disease under amendment A4's "
              "rule: the guideline tests per pathology in Hager et al. (2024), the dataset's authors. Net relevant "
              "results = extra relevant and provided − dropped relevant and provided. Aggregate counts only.\n",
              "| Arm | Tests | Extra | provided | relevant | relevant, provided | Dropped | provided | relevant | "
              "relevant, provided | Net relevant results |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, _ in D_ARMS[1:]:
        changed = d["extra_tests"][arm]
        for key, label in (*D_CATEGORIES, (None, "All")):
            c = changed["total"] if key is None else changed["by_category"][key]
            net = c["extra_relevant_provided"] - c["dropped_relevant_provided"]
            lines.append(f"| {arm} | {label} | " + " | ".join(str(c[k]) for k in COUNTS) + f" | {net:+d} |")
    lines += ["", "By test (extra / dropped):\n",
              "| Test | Relevant to | " + " | ".join(arm for arm, _ in D_ARMS[1:]) + " |",
              "|---|---|" + "---|" * len(D_ARMS[1:])]
    for code in TEST_CODES.values():
        cells = [f"+{d['extra_tests'][arm]['by_test'][code]['extra']} / "
                 f"−{d['extra_tests'][arm]['by_test'][code]['dropped']}" for arm, _ in D_ARMS[1:]]
        lines.append(f"| {code} | {_relevant_to(d['relevance'][code])} | " + " | ".join(cells) + " |")
    lines += ["", "## Corruption dose-response: not run\n",
              "The eight corruption arms and B-marginal were run only on test (SPEC §9); validation has only α = 0 "
              "(B-oracle). Amendment A4 names the slope test (T5 on α, class-stratified bootstrap interval), but it "
              f"needs those 9 arms on validation: about {dose_hours:.1f} GPU-hours at addendum B's per-arm budget, "
              "an estimate rather than a measurement. It waits for an approval decision. Until then, whether recall "
              "falls as the availability signal degrades is untested on validation.\n",
              "## Limitations\n",
              f"- {n} validation admissions and one policy seed; between-seed variation is unassessable.",
              "- Relevance is guideline relevance of the requested action, not the value of the result it returned.",
              "- A null oracle contrast concerns this fixed planner only: perfect information does not make the "
              "heuristic optimal (SPEC §4.5).",
              "- Every number here is rendered from `diagnose-val-b/diagnostics.json`, "
              "`analyze-val-b/validation_b.json` and the committed addenda A and B.", ""]
    return "\n".join(lines)
