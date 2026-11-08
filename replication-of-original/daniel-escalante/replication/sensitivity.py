"""Supplementary aggregates computed after S6 from the saved episodes.

Confusion counts come from the pre-specified greedy test episodes. The decoding
check re-evaluates each locked model once on test with upstream's evaluation
temperature: configure_grpo(eval_only=True) keeps the training temperature, 1.0,
where this replication decodes greedily. Greedy results stay the primary analysis;
the check measures how much of each gap the decoding choice explains. Nothing
here selects or tunes a model, and only counts and metrics leave the work directory.
"""

import json
from pathlib import Path

from .config import ARMS, OWN, stream_seed
from .manifest import assert_aggregate, eastern_now, file_digest, write_json
from .stats import CLASSES, confusion, holm_adjust, paired_bootstrap, single_arm_intervals, summary

TEMPERATURE = 1.0
ALL_ARMS = ("zero_shot", *ARMS)
MARKER = "<!-- supplementary aggregates -->"
METRICS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "pooled_accuracy")


def sampled_path(settings, protocol, seed, arm):
    return settings.work_dir / "sensitivity" / "decoding-t1" / protocol / str(seed) / f"{arm}-test.json"


def _admission(path):
    return Path(path).with_name(Path(path).stem + ".admission.json")


def admit(path):
    """Reserve one test evaluation before any generation; a reservation is never released."""
    path, admission = Path(path), _admission(path)
    if path.exists() or admission.exists():
        raise RuntimeError("the decoding check already admitted this arm on test; it runs once")
    admission.parent.mkdir(parents=True, exist_ok=True)
    try:
        with admission.open("x") as handle:  # exclusive create: concurrent jobs cannot both pass
            json.dump({"admitted_at": eastern_now()}, handle)
    except FileExistsError:
        raise RuntimeError("the decoding check already admitted this arm on test; it runs once") from None


def evaluate_arm(settings, arm):
    """GPU: evaluate one locked model (or the untrained policy) once on test at upstream's temperature.

    Completed protocol/seed units are skipped; an admitted unit without results was
    interrupted after test admission and is refused rather than evaluated again.
    """
    from . import pipeline
    if arm not in ALL_ARMS:
        raise ValueError(f"unknown arm: {arm}")
    pipeline._assert_runtime(settings)
    pipeline._receipt(settings, "s6")
    for protocol in pipeline.protocols(settings):
        for seed in settings.seeds:
            path = sampled_path(settings, protocol, seed, arm)
            if path.exists():
                continue
            admit(path)
            adapter_dir = None
            if arm != "zero_shot":
                selection = pipeline.read_json(settings.work_dir / "models" / protocol / str(seed) / arm / "selection.json")
                for output, expected in selection["run_outputs"].items():
                    if not Path(output).is_file() or file_digest(Path(output)) != expected:
                        raise RuntimeError("selected model changed after S6")
                adapter_dir = selection["adapter_dir"]
            pipeline._require_cuda()
            from .gpu import evaluate
            rows = evaluate(pipeline.cfg_for(settings, arm, protocol, seed), pipeline.prepared(settings, protocol),
                            split="test", seed=stream_seed(seed, "test", "policy"), adapter_dir=adapter_dir,
                            temperature=TEMPERATURE)
            path.parent.mkdir(parents=True, exist_ok=True)
            pipeline.private_write(path, rows, settings.source)


def _counts(rows):
    return {"undiagnosed": sum(r["prediction"] is None for r in rows),
            "requests": sum(len(r["requested_tests"]) for r in rows),
            "repeated_requests": sum(len(r["requested_tests"]) - len(set(r["requested_tests"])) for r in rows)}


FAMILY = ("primary_vs_decision_only", "primary_vs_no_cost", "primary_vs_incorrect_minus1", "primary_vs_zero_shot")


def build(greedy, cfg, seed, n_resamples, sampled=None, cluster_ids=None):
    """Aggregate-only supplement; ``sampled`` holds whichever arms finished the decoding check.

    ``cluster_ids`` are the subject identifiers S6 uses, aligned to patient order (None when absent).
    The Holm family is always the four planned comparisons; unfinished ones count as unrun tests.
    """
    bootstrap_seed = stream_seed(seed, "test", "bootstrap")
    options = {"seed": bootstrap_seed, "n_resamples": n_resamples, "cluster_ids": cluster_ids}
    result = {"greedy": {arm: {"confusion": confusion(rows), **_counts(rows)} for arm, rows in greedy.items()}}
    if sampled:
        result["upstream_decoding"] = {}
        for arm, rows in sampled.items():
            point = summary(rows, cfg)
            result["upstream_decoding"][arm] = {
                "temperature": TEMPERATURE, "claims": point["claims"], "pooled_accuracy": point["pooled_accuracy"],
                "hypothesis_accuracy": point["hypothesis"]["accuracy"], "confusion": confusion(rows), **_counts(rows),
                "intervals": single_arm_intervals(rows, cfg, **options)["metrics"],
                "minus_greedy": paired_bootstrap(rows, greedy[arm], cfg, **options)["metrics"],
            }
        if "primary" in sampled:
            comparisons = {name: paired_bootstrap(sampled["primary"], sampled[name.removeprefix("primary_vs_")], cfg, **options)
                           for name in FAMILY if name.removeprefix("primary_vs_") in sampled}
            holm = holm_adjust({name: comparisons[name]["mcnemar"]["p_value"] if name in comparisons else None
                                for name in FAMILY})
            for name, value in comparisons.items():
                value["holm"] = holm[name]
            result["upstream_decoding_comparisons"] = comparisons
            result["upstream_decoding_planned_family"] = list(FAMILY)
    assert_aggregate(result)
    return result


def _number(value, digits=2):
    return "unassessable" if value is None else f"{value:,.{digits}f}"


def _interval(metric):
    return "unassessable" if metric.get("ci95") is None else f"[{_number(metric['ci95'][0])}, {_number(metric['ci95'][1])}]"


def render(result, protocol="parity", seed=269):
    lines = [MARKER, "", "## Supplementary aggregates (added after S6)", "",
             f"Protocol: {protocol}; training seed: {seed}.", "",
             "Counts and metrics computed from the saved test episodes after S6; no patient row leaves the work directory. "
             "Rows are true diagnoses; columns are the episode's final diagnosis, or *undiagnosed* when the episode "
             "ended without a valid diagnosis (malformed action, step budget or trajectory length).", ""]
    for name, block in (("greedy decoding, pre-specified", result["greedy"]),
                        (f"temperature {TEMPERATURE}, upstream decoding", result.get("upstream_decoding", {}))):
        for arm, values in block.items():
            lines += [f"### Confusion: {arm} ({name})", "",
                      "| True \\ final | " + " | ".join(CLASSES) + " | undiagnosed |",
                      "| --- |" + " --- |" * (len(CLASSES) + 1)]
            lines += [f"| {label} | " + " | ".join(str(values["confusion"][label][p]) for p in (*CLASSES, "undiagnosed")) + " |"
                      for label in CLASSES]
            lines.append("")
    lines += ["### Episodes without a valid diagnosis", "",
              "| Arm | Decoding | Undiagnosed | Valid test requests | Repeated requests |", "| --- | --- | --- | --- | --- |"]
    for name, block in (("greedy", result["greedy"]), (f"temperature {TEMPERATURE}", result.get("upstream_decoding", {}))):
        lines += [f"| {arm} | {name} | {v['undiagnosed']} | {v['requests']} | {v['repeated_requests']} |" for arm, v in block.items()]
    lines.append("")
    if "upstream_decoding" in result:
        lines += [f"### Decoding check: upstream evaluation at temperature {TEMPERATURE}", "",
                  "Upstream evaluates with its training temperature (1.0); this replication pre-specified greedy decoding. "
                  "Each locked model ran once more on the same 240 test patients, sampling with independent request seeds. "
                  "Greedy remains the primary result. Intervals: class-stratified patient bootstrap; the difference is paired.", "",
                  "| Arm | Metric | Greedy | Temperature 1.0 | 95% interval | Difference | 95% paired interval |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for arm, values in result["upstream_decoding"].items():
            for metric in METRICS:
                sampled_value = values["intervals"][metric]["estimate"]
                delta = values["minus_greedy"][metric]
                greedy_value = None if sampled_value is None or delta["estimate"] is None else sampled_value - delta["estimate"]
                lines.append(f"| {arm} | {metric} | {_number(greedy_value)} | {_number(sampled_value)} | "
                             f"{_interval(values['intervals'][metric])} | {_number(delta['estimate'])} | {_interval(delta)} |")
        lines.append("")
        if "upstream_decoding_comparisons" in result:
            lines += ["| Comparison at temperature 1.0 | T5 difference | 95% paired interval | Exact McNemar p | Holm p (family = 4) |",
                      "| --- | --- | --- | --- | --- |"]
            for name in result["upstream_decoding_planned_family"]:
                value = result["upstream_decoding_comparisons"].get(name)
                if value is None:
                    lines.append(f"| {name} | not run | not run | not run | not run |")
                    continue
                lines.append(f"| {name} | {_number(value['metrics']['T5']['estimate'])} | {_interval(value['metrics']['T5'])} | "
                             f"{_number(value['mcnemar']['p_value'], 6)} | {_number(value['holm']['holm_p_value'], 6)} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def report(settings):
    """CPU: build the supplement from saved episodes and add it to the exported results."""
    import pandas as pd
    from . import pipeline
    pipeline._receipt(settings, "s6")
    supplements, texts = {}, []
    for protocol in pipeline.protocols(settings):
        # The same subject clusters S6 uses; v1.1 ships no subject_id, so this is usually None.
        frame = pd.read_csv(pipeline.prepared(settings, protocol) / "test.csv")
        subjects = dict(zip(frame["Patient ID"].map(str), frame["subject_id"])) if "subject_id" in frame else None
        for seed in settings.seeds:
            greedy = {arm: pipeline.private_read(pipeline._rows_path(settings, protocol, seed, arm), settings.source)
                      for arm in ALL_ARMS}
            sampled = {arm: pipeline.private_read(sampled_path(settings, protocol, seed, arm), settings.source)
                       for arm in ALL_ARMS if sampled_path(settings, protocol, seed, arm).exists()}
            clusters = [int(subjects[str(r["patient_id"])]) for r in greedy["primary"]] if subjects else None
            value = build(greedy, pipeline.cfg_for(settings, protocol=protocol, seed=seed), seed,
                          settings.bootstrap_resamples, sampled or None, cluster_ids=clusters)
            supplements[f"{protocol}/{seed}"] = value
            texts.append(render(value, protocol, seed))
    payload = {"source": settings.source, "label": "re-implementation on a different backbone",
               "written_at": eastern_now(), "supplements": supplements}
    assert_aggregate(payload)
    write_json(settings.work_dir / "sensitivity" / "supplement.json", payload)
    destination = OWN / "results" / settings.source
    write_json(destination / "supplementary.json", payload)
    report_path = destination / "REPORT.md"
    base = report_path.read_text().split(MARKER)[0].rstrip()
    report_path.write_text(base + "\n\n" + "\n".join(texts))
    print(f"supplementary aggregates written: {destination}")
