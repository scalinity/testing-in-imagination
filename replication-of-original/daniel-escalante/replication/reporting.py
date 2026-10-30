"""Render aggregate-only Markdown reports with explicit source and honest gaps.

render_report input contract (all fields optional; absence is reported):
  arms: {primary|zero_shot|decision_only|no_cost|incorrect_minus1: stats.summary}
  intervals: {arm: stats.single_arm_intervals result}
  paired: {primary_vs_<other arm>: stats.paired_bootstrap result}
  seed_differences: {comparison: stats.seed_difference_ci result}
  leakage: {admission_overlap: int|None, subject_overlap: int|None,
            grouped_sensitivity: str|None}
  implementation: {I1|I2|I3|I4|I5: status string}
  run: {seed: int, seeds: list[int], backbone: str, revision: str,
        upstream_sha: str, training_completed: bool}
  gaps: list[str]

Only stats-produced exact McNemar p-values in paired are rendered. Holm uses
the planned primary-versus-zero-shot/three-ablation pooled-accuracy family;
unrun comparisons remain in the family. Other metrics receive intervals, no
invented p-values. Reports never turn a paper delta into reproduction success.
The template is fully substituted; patient rows are never serialized here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from string import Template

from .stats import holm_adjust


RESULTS = Path(__file__).resolve().parents[1] / "results"
LABEL = "re-implementation on a different backbone"
COMPARISONS = ("primary_vs_zero_shot", "primary_vs_decision_only", "primary_vs_no_cost", "primary_vs_incorrect_minus1")


def _text(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _value(value, digits=2):
    if value is None:
        return "unassessable"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("report metric must be finite or None")
    return f"{value:.{digits}f}"


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(_text(v) for v in row) + " |" for row in rows)])


def _interval(result):
    if not result or result.get("ci95") is None:
        return "unassessable"
    lo, hi = result["ci95"]
    return f"[{_value(lo)}, {_value(hi)}]"


def render_report(summary_dict, output_path, source):
    """Write an aggregate report for source='synthetic' or source='mimic'.

    All runs and comparisons carry a different-backbone label. Synthetic rows
    remain orchestration demonstrations even if run.training_completed is true;
    this flag never turns invented data or mock updates into training evidence.
    Paper constants are read solely from the checked-in REPRODUCTION ledger.
    """
    if source not in ("synthetic", "mimic"):
        raise ValueError("source must be synthetic or mimic")
    arms = summary_dict.get("arms", {})
    unknown = set(arms) - {"primary", "zero_shot", "decision_only", "no_cost", "incorrect_minus1"}
    if unknown:
        raise ValueError("unknown arm names: " + ", ".join(sorted(unknown)))
    paper = json.loads((RESULTS / "paper_values.json").read_text())
    intervals = summary_dict.get("intervals", {})
    paired = summary_dict.get("paired", {})
    run = summary_dict.get("run", {})
    primary = arms.get("primary", {})
    zs = arms.get("zero_shot", {})
    gap = "unassessable: no logged run supplied"
    result_label = f"{source}; {LABEL}"
    banner = ("**SYNTHETIC — INVENTED DATA; MOCK UPDATES ARE NOT LEARNING. "
              "NO MIMIC RESULT OR PAPER REPRODUCTION IS ESTABLISHED.**" if source == "synthetic"
              else "**MIMIC source — credentialed-run results require manifests and completed CUDA checks.**")
    scope = (f"Every observed result below is a **{LABEL}** ({source}). "
             "Paper references come only from `REPRODUCTION.md`; no matching tolerance or reproduction verdict is inferred. "
             "Reported values alone do not verify data access, model revision, training completion or CUDA acceptance.")
    rows = []
    for claim, reference in paper["table1"].items():
        value = primary.get("claims", {}).get(claim)
        ci = intervals.get("primary", {}).get("metrics", {}).get(claim)
        delta = None if value is None else value - reference["value"]
        delta_ci = {"ci95": [bound - reference["value"] for bound in ci["ci95"]]} if ci and ci.get("ci95") is not None else None
        rows.append((claim, reference["label"], f"{reference['value']} {reference['unit']}",
                     _value(value), _interval(ci), _value(delta), _interval(delta_ci), "observed; " + result_label if value is not None else gap))
    main_table = _table(("Claim", "Metric", "Paper", "Observed", "95% interval", "Observed − paper", "95% delta interval", "Status"), rows)
    metric_notes = ("T1–T4 are class recalls (%); T5 is their unweighted mean (%), rather than pooled accuracy. "
                    "T6/T7 are upstream all/micro_f1 and all/macro_f1 (%); None diagnoses count as wrong. "
                    "T8 is mean USD for tests that actually returned data.\n\n"
                    f"Admissions: {_value(primary.get('n'), 0)}. Pooled accuracy: {_value(primary.get('pooled_accuracy'))}% ({result_label}). "
                    "Missing classes and missing hypothesis observations are unassessable. "
                    "Raw upstream empty-triplet zeros do not establish zero ECE.")
    baseline_value = zs.get("claims", {}).get("T5")
    baselines = _table(("Claim", "Method", "Paper mean recall (%)", "Observed mean recall (%)", "95% interval", "Observed − paper", "Status"), [
        ("B1", "Zero-shot", paper["baselines"]["B1"]["value"], _value(baseline_value), _interval(intervals.get("zero_shot", {}).get("metrics", {}).get("T5")), _value(None if baseline_value is None else baseline_value - paper["baselines"]["B1"]["value"]), result_label if zs else gap),
        ("B2", "ReAct", paper["baselines"]["B2"]["value"], "unassessable", "unassessable", "unassessable", "gap: ReAct implementation absent at pinned upstream")])
    before = zs.get("hypothesis", {})
    after = primary.get("hypothesis", {})

    def percent(value):
        return None if value is None else 100 * value

    no_cost = arms.get("no_cost", {})
    decision = arms.get("decision_only", {})
    c1, c2, c3 = (paper["dynamics"][key] for key in ("C1", "C2", "C3"))
    dynamics = _table(("Claim", "Paper", "Observed comparison", "Status"), [
        ("C1", f"{c1['before']} → {c1['after']}%", f"zero-shot → primary hypothesis accuracy: {_value(percent(before.get('accuracy')))} → {_value(percent(after.get('accuracy')))}%", result_label),
        ("C2", f"{c2['before']} → {c2['after']}", f"zero-shot → primary ECE: {_value(before.get('ece'), 4)} → {_value(after.get('ece'), 4)}", result_label),
        ("C3", f"${c3['before_cost_usd']} → ${c3['after_cost_usd']}; recall {c3['before_mean_recall']} → {c3['after_mean_recall']}%", f"no_cost → primary: ${_value(no_cost.get('claims', {}).get('T8'))} → ${_value(primary.get('claims', {}).get('T8'))}; recall {_value(no_cost.get('claims', {}).get('T5'))} → {_value(primary.get('claims', {}).get('T5'))}%", result_label),
        ("C4", "Table 3: exact values absent from ledger", f"decision_only → primary mean recall: {_value(decision.get('claims', {}).get('T5'))} → {_value(primary.get('claims', {}).get('T5'))}%", result_label)])
    dynamics += ("\n\nHypothesis statistics use actual trajectory triplets, weighted by observation count. "
                 "Unknown hypotheses contribute to hypothesis accuracy but are excluded from ECE by upstream. "
                 "These held-out endpoints are not a measured training curve. "
                 "C3 needs paired accuracy and cost evidence; a similar point accuracy does not establish equivalence.")
    dynamics += (f"\n\nPrimary hypothesis observations: {_value(after.get('triplet_count'), 0)} actual triplets; "
                 f"{_value(after.get('ece_triplet_count'), 0)} eligible for ECE ({result_label}).")
    if source == "synthetic":
        dynamics += " Synthetic comparisons demonstrate wiring only; they do not establish training improvement or a scientific mechanism."
    ablation_rows = []
    for arm, meaning in (("primary", "incorrect reward 0 (shipped primary)"), ("decision_only", "hypothesis/confidence agents disabled"),
                         ("no_cost", "cost reward disabled; USD costs still measured"), ("incorrect_minus1", "incorrect reward −1 (ablation)")):
        values = arms.get(arm, {})
        ablation_rows.append((arm, meaning, _value(values.get("claims", {}).get("T5")), _value(values.get("claims", {}).get("T8")), result_label if values else gap))
    ablations = _table(("Arm", "Definition", "Mean recall (%)", "Provided cost (USD)", "Status"), ablation_rows)
    ablations += ("\n\nBefore test cost, primary correct/wrong well-formed/undiagnosed-invalid totals are 2/1/0; "
                  "the −1 ablation gives 2/0/0. Do not silently replace primary reward 0 with −1. "
                  "No inter-arm performance ordering or oracle baseline is assumed.")
    pvalues = {name: paired.get(name, {}).get("mcnemar", {}).get("p_value") for name in COMPARISONS}
    adjusted = holm_adjust(pvalues)
    sig_rows = []
    for name in COMPARISONS:
        comparison = paired.get(name, {})
        for metric in ("T5", "T6", "T7", "T8", "pooled_accuracy"):
            result = comparison.get("metrics", {}).get(metric, {})
            sig_rows.append((name, metric, _value(result.get("estimate")), _interval(result), result_label if result else gap))
    significance = ("Differences are primary minus the comparison arm. T1–T7 and pooled-accuracy differences are percentage points; "
                    "T8 differences are USD. The paired bootstrap reuses identical patient draws in both arms. "
                    "Intervals concern held-out sampling conditional on fitted models, not training-seed uncertainty.\n\n" +
                    _table(("Comparison", "Metric", "Difference", "95% paired interval", "Status"), sig_rows) + "\n\n" +
                    _table(("Planned pooled-accuracy test", "Exact McNemar p", "Holm p (family = 4)", "Adjusted rejection"),
                           [(name, _value(v["p_value"], 6), _value(v["holm_p_value"], 6), str(v["reject"]) if v["p_value"] is not None else "unassessable") for name, v in adjusted.items()]))
    significance += ("\n\nMcNemar tests paired correctness discordances only, not T5, F1, ECE or cost. "
                     "No p-values are inferred from per-arm standard deviations or bootstrap interval exclusion. "
                     "Other metrics have intervals only; adding metric tests requires declaring and correcting the entire family. "
                     "Subject clustering is assessable only when subject IDs are supplied. "
                     "Repeated subjects make the admission-level McNemar test unassessable because independent pairs are required. "
                     "Class-stratified intervals condition on class composition; subject-cluster intervals condition on class-presence strata, "
                     "which can be degenerate when strata contain few subjects.")
    significance += "\n\n" + ("At least one planned pooled-accuracy difference is significant after Holm adjustment."
                                               if any(v["reject"] for v in adjusted.values())
                                               else "No planned pooled-accuracy difference is established as significant after Holm adjustment; unassessable tests cannot support a significance claim.")
    for name, comparison in paired.items():
        significance += f"\n\n{_text(name)} ({result_label}): {_text(comparison.get('method', 'method unreported'))}; "
        significance += f"resamples {_text(comparison.get('n_resamples', 'unreported'))}; seed {_text(comparison.get('seed', 'unreported'))}."
    seeds = summary_dict.get("seed_differences", {})
    significance += "\n\nBetween-training-seed uncertainty: "
    if seeds:
        significance += _table(("Comparison", "Completed/declared seeds", "Mean paired difference", "Student-t 95% interval", "Status"),
                               [(name, f"{v.get('n', 0)}/{v.get('declared_n', 0)}", _value(v.get("estimate")), _interval(v), f"{v.get('status', 'unassessable')}; {result_label}") for name, v in seeds.items()])
    else:
        significance += "unassessable; no complete paired seed-difference series supplied. One seed is insufficient; failed or unfinished declared seeds must remain visible."
    leakage_data = summary_dict.get("leakage", {})
    leakage = _table(("Audit", "Observed"), [
        ("Admission-ID overlap across train/val/test", _value(leakage_data.get("admission_overlap"), 0)),
        ("Subject overlap across train/val/test", _value(leakage_data.get("subject_overlap"), 0)),
        ("Subject-grouped sensitivity", leakage_data.get("grouped_sensitivity") or "unassessable: no grouped sensitivity result supplied")])
    leakage += ("\n\nAbsent subject IDs mean unassessable subject leakage, never zero. "
                "The executable pinned splitter hardcodes seed 269. D-4 says no canonical seed: this ledger/code discrepancy "
                "is retained; use 269 for parity and declare additional split seeds before comparing them. "
                "Checkpoint selection must use validation only; final test evaluation follows model locking. "
                "Leakage audits here are supplied aggregate checks, not inferred from metric values.")
    cost_rows = []
    for arm, values in arms.items():
        c = values.get("cost", {})
        cost_rows.append((arm, _value(c.get("provided_mean_usd")), _value(c.get("requested_mean_usd")),
                          _value(c.get("unavailable_requested_mean_usd")), _value(c.get("repeated_available_request_mean_usd")), _value(c.get("missing_count"), 0), result_label))
    cost = _table(("Arm", "Provided USD/admission", "Requested USD/admission", "Unavailable-request USD/admission", "Repeated available request USD/admission", "Missing requests", "Source"), cost_rows)
    cost += ("\n\nProvided cost is the T8 definition. Requested cost prices every recognized request at the same configured USD schedule; "
             "the difference includes unavailable and repeated available requests, separated above, and is not an observed charge. "
             "Repeated requests remain counted. Expected provided cost for one request is availability × cost; "
             "an unavailable request is free under the shipped provided-tests reward.")
    environment_notes = ("- F-1: unavailable tests share the trajectory error flag with malformed actions and diagnoses.\n"
                         "- F-2: only provided tests incur reward cost; unavailable requests are free.\n"
                         "- F-3: unavailable requests skip hypothesis updates and consume an action; max_steps 13 allows 14 actions.\n"
                         "- F-4: stale confidence already divided by 10 is reused in a prompt otherwise showing a raw 0–10 value.\n\n"
                         "These pinned behaviors remain intact. Unit-level confirmation is recorded in the ledger; this report does not claim a new CUDA smoke episode.")
    impl = summary_dict.get("implementation", {})
    implementation = _table(("Claim", "Check", "Observed status"), [
        ("I1", "Approved different backbone + LoRA; immutable revision", impl.get("I1", "unverified")),
        ("I2", "Cyclic action GRPO / hypothesis SFT / confidence GRPO, 100/100/100", impl.get("I2", "unverified")),
        ("I3", "Adam lr 1e-5, batch 2, seed 269", impl.get("I3", "unverified")),
        ("I4", "Configured USD costs including physical examination", impl.get("I4", "unverified")),
        ("I5", "Normalized costs sum to 1 including physical examination", impl.get("I5", "unverified"))])
    implementation += f"\n\nRun seed: {_text(run.get('seed', 'unreported'))}; declared seeds: {_text(run.get('seeds', 'unreported'))}. "
    implementation += f"Upstream SHA: {_text(run.get('upstream_sha', 'unreported'))}; immutable backbone revision: {_text(run.get('revision', 'unreported'))}. "
    implementation += "Training completion: " + ("not established by synthetic/mock execution." if source == "synthetic" else _text(run.get("training_completed", "unreported")) + "; verify against the run ledger.")
    gaps_list = ["B2 ReAct is absent at the upstream pin; no substitute is labeled ReAct.",
                 "Exact C4/Table 3 values are absent from REPRODUCTION.md and remain null in paper_values.json.",
                 "Different backbone and summarizer prevent a bit-for-bit paper comparison; no numerical acceptance is implied."]
    if source == "synthetic":
        gaps_list.append("MIMIC results, real learning, CUDA execution and paper reproduction remain unverified.")
    if not primary:
        gaps_list.append("T1–T8 have no primary logged-run summary.")
    gaps_list.extend(summary_dict.get("gaps", []))
    gaps = "\n".join("- " + _text(item) for item in gaps_list)
    text = Template((RESULTS / "REPORT_TEMPLATE.md").read_text()).substitute(
        banner=banner, scope=scope, main_table=main_table, metric_notes=metric_notes, baselines=baselines,
        dynamics=dynamics, ablations=ablations, significance=significance, leakage=leakage,
        cost=cost, environment_notes=environment_notes, implementation=implementation, gaps=gaps)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path
