"""Resumable S0–S6 stages. All patient-level artifacts stay in private work storage."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import urllib.request

from omegaconf import OmegaConf

from .config import ARMS, MODEL, OWN, REVISION, ROOT, Settings, UPSTREAM, compose, stream_seed
from .manifest import (assert_aggregate, canonical, digest, eastern_now, file_digest,
                       make_manifest, source_fingerprint, stage_lock, stamp_metrics,
                       tracked_data_violations, tree_fingerprint, write_json)

STAGES = ("s0", "s1", "s2", "s3", "s4", "s5", "s6")


def read_json(path):
    return json.loads(Path(path).read_text())


def private_write(path, value, source):
    return write_json(path, {"source": source, "payload": value})


def private_read(path, source):
    value = read_json(path)
    if value["source"] != source:
        raise ValueError("artifact data source does not match the active configuration")
    return value["payload"]


def prepared(settings, protocol="parity"):
    base = settings.work_dir / "prepared"
    return base if protocol == "parity" else base / "grouped"


def protocols(settings):
    return ("parity", "grouped") if (prepared(settings, "grouped") / "train.csv").exists() else ("parity",)


def cfg_for(settings, arm="primary", protocol="parity", seed=269):
    return compose(arm, prepared(settings, protocol), settings.work_dir / "models" / protocol / str(seed) / arm, seed)


def _receipt(settings, stage):
    path = settings.work_dir / "receipts" / f"{stage}.json"
    if not path.exists():
        raise RuntimeError(f"stage {stage} must complete first")
    receipt = read_json(path)
    if receipt["source"] != settings.source:
        raise RuntimeError("work directory contains a different data source; choose a fresh work directory")
    index = STAGES.index(stage)
    if index:
        previous = _receipt(settings, STAGES[index - 1])
        if receipt.get("previous_receipt_id") != previous["receipt_id"]:
            raise RuntimeError("stage dependency receipt changed")
    for output, expected in receipt["outputs"].items():
        if not Path(output).is_file() or file_digest(Path(output)) != expected:
            raise RuntimeError(f"completed stage {stage} has a missing or changed artifact")
    return receipt


def _require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("real execution requires CUDA on HiPerGator")


def s0(settings):
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        raise RuntimeError("Python 3.11–3.12 is required")
    from .data import validate_raw
    audit = validate_raw(settings.data_dir, source="mimic") if not settings.mock else {"status": "synthetic generation occurs in S1"}
    if os.environ.get("VLLM_USE_V1") != "0":
        raise RuntimeError("VLLM_USE_V1 must be 0")
    upstream_dependencies = tomllib.loads((UPSTREAM / "pyproject.toml").read_text())["project"]["dependencies"]
    runtime_dependencies = tomllib.loads((OWN / "runtime/pyproject.toml").read_text())["project"]["dependencies"]
    if runtime_dependencies != upstream_dependencies:
        raise RuntimeError("runtime dependencies have drifted from the upstream pin")
    if not settings.mock:
        if sys.platform != "linux":
            raise RuntimeError("full upstream environment requires Linux/CUDA; use HiPerGator")
        env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(settings.work_dir / "runtime-env"))
        subprocess.run(["uv", "sync", "--frozen", "--project", str(OWN / "runtime"), "--python", sys.executable], env=env, check=True)
        request = urllib.request.Request(f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}")
        with urllib.request.urlopen(request, timeout=45) as response:
            metadata = json.load(response)
        if metadata["sha"] != REVISION:
            raise RuntimeError("published model revision does not resolve to the pinned commit")
    result = {"source": settings.source, "data_validation": audit, "python": sys.version.split()[0],
              "model_revision": REVISION, "model_revision_verification": "immutable pin recorded; live lookup deferred" if settings.mock else "PASS",
              "upstream_lockfile": "absent at pin; local lock preserves identical dependency declarations",
              "runtime_lock_sha256": file_digest(OWN / "runtime/uv.lock"),
              "dependency_source_overrides": {"xgrammar": {"version": "0.1.16", "commit": "989222175c2a30fb7987d8bcce35bec1bf6817f2", "reason": "release unavailable on package index; same-version publisher source"}},
              "cuda_stack_sync": "NOT_RUN (synthetic CPU)" if settings.mock else "PASS",
              "synthetic_n": settings.synthetic_n, "seeds": list(settings.seeds),
              "bootstrap_resamples": settings.bootstrap_resamples}
    out = write_json(settings.work_dir / "environment.json", result)
    return [out]


def s1(settings):
    from .data import generate_synthetic, prepare_splits, validate_raw
    if settings.mock:
        if settings.data_dir.exists() and any(settings.data_dir.iterdir()):
            marker = settings.data_dir / ".synthetic-generator.json"
            if not marker.exists() or read_json(marker) != {"source": "synthetic", "seed": settings.seed, "n": settings.synthetic_n}:
                raise RuntimeError("refusing to overwrite an unrecognized nonempty synthetic data directory")
        generate_synthetic(settings.data_dir, seed=settings.seed, n=settings.synthetic_n)
        write_json(settings.data_dir / ".synthetic-generator.json", {"source": "synthetic", "seed": settings.seed, "n": settings.synthetic_n})
    audit = validate_raw(settings.data_dir, source=settings.source)
    summarizer = None
    if not settings.mock:
        _require_cuda()
        from .gpu import make_summarizer
        summarizer = make_summarizer(model=MODEL, revision=REVISION, seed=settings.seed)
    split_audit = prepare_splits(settings.data_dir, prepared(settings), seed=settings.seed,
                                 fake_summarizer=settings.mock, summarizer=summarizer, source=settings.source)
    out = private_write(settings.work_dir / "data-audit.json", {"schema": audit, "split": split_audit["audit"]}, settings.source)
    return [out, *sorted(prepared(settings).rglob("*.csv")), *sorted(settings.data_dir.glob("*.pkl")), *sorted(settings.data_dir.glob("*.csv"))]


def s2(settings):
    from .shim import prepare_environment
    configs = {}
    outputs = []
    for protocol in protocols(settings):
        for seed in settings.seeds:
            for arm in (*ARMS, "zero_shot"):
                cfg = cfg_for(settings, arm, protocol, seed)
                environment = prepare_environment(cfg)
                prompt = environment.construct_prompt([{"patient_history": "invented smoke history", "test_results": {}, "condition": "appendicitis", "patient_id": 0}])[0]["prompt"]
                if prompt[0] != {"role": "system", "content": "detailed thinking off"}:
                    raise RuntimeError("system prompt shim was not applied")
                key = f"{protocol}/{seed}/{arm}"
                configs[key] = OmegaConf.to_container(cfg, resolve=True)
    outputs.append(private_write(settings.work_dir / "composed-configs.json", configs, settings.source))
    return outputs


def _evaluate(settings, arm, protocol, seed, split, adapter_dir=None):
    from .data import assert_disjoint, load_dataset
    assert_disjoint(prepared(settings, protocol))
    cfg = cfg_for(settings, arm, protocol, seed)
    if settings.mock:
        from .shim import evaluate_dataset
        dataset = load_dataset(prepared(settings, protocol), split, cfg)
        rows = evaluate_dataset(cfg, dataset, split=split, seed=stream_seed(seed, split, "policy"), trained=arm != "zero_shot")
    else:
        _require_cuda()
        from .gpu import evaluate
        rows = evaluate(cfg, prepared(settings, protocol), split=split, seed=stream_seed(seed, split, "policy"), adapter_dir=adapter_dir)
    if not rows:
        raise RuntimeError("evaluation produced no patients")
    return rows


def _saved_evaluation(settings, arm, protocol, seed, split="test", adapter_dir=None):
    """Resume completed arms without evaluating a final model on test again."""
    path = _rows_path(settings, protocol, seed, arm, split)
    manifest_path = path.with_suffix(".manifest.json")
    receipt_path = path.with_suffix(".receipt.json")
    inputs = {"data_files": tree_fingerprint(prepared(settings, protocol)), "split": split,
              "adapter_dir": adapter_dir, "seed": seed, "protocol": protocol, "arm": arm}
    if adapter_dir:
        inputs["adapter_files"] = tree_fingerprint(Path(adapter_dir), ("*.safetensors", "*.bin", "adapter_config.json"))
    manifest = make_manifest(settings, f"evaluation-{protocol}-{seed}-{arm}-{split}", cfg_for(settings, arm, protocol, seed), inputs)
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["manifest_id"] != manifest["manifest_id"] or not path.exists() or file_digest(path) != receipt["sha256"]:
            raise RuntimeError("evaluation receipt or final model changed; preserve this run and use a fresh work directory")
        return [path, manifest_path, receipt_path]
    if path.exists() or manifest_path.exists():
        raise RuntimeError("evaluation was interrupted after admission; inspect its artifacts before any test rerun")
    write_json(manifest_path, manifest)
    rows = _evaluate(settings, arm, protocol, seed, split, adapter_dir)
    private_write(path, rows, settings.source)
    write_json(receipt_path, {"source": settings.source, "manifest_id": manifest["manifest_id"], "sha256": file_digest(path)})
    return [path, manifest_path, receipt_path]


def _rows_path(settings, protocol, seed, arm, split="test"):
    return settings.work_dir / "episodes" / protocol / str(seed) / f"{arm}-{split}.json"


def s3(settings):
    outputs = []
    for protocol in protocols(settings):
        for seed in settings.seeds:
            outputs.extend(_saved_evaluation(settings, "zero_shot", protocol, seed))
    outputs.append(write_json(settings.work_dir / "baseline-gap.json", {"source": settings.source, "B2": "NOT_IMPLEMENTED upstream ReAct absent"}))
    return outputs


def s4(settings):
    outputs = []
    ledger_path = settings.work_dir / "training-ledger.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {"source": settings.source, "runs": []}
    for protocol in protocols(settings):
        for seed in settings.seeds:
            for arm in ARMS:
                directory = settings.work_dir / "models" / protocol / str(seed) / arm
                selection = directory / "selection.json"
                cfg = cfg_for(settings, arm, protocol, seed)
                run_manifest = make_manifest(settings, f"training-{protocol}-{seed}-{arm}", cfg,
                                             {"data_files": tree_fingerprint(prepared(settings, protocol)), "training_seed": seed})
                if selection.exists():
                    saved = read_json(selection)
                    if saved.get("training_manifest_id") != run_manifest["manifest_id"]:
                        raise RuntimeError("training config or data changed after checkpoint selection")
                    for path, expected in saved["run_outputs"].items():
                        if not Path(path).is_file() or file_digest(Path(path)) != expected:
                            raise RuntimeError("selected model or cached training artifact changed")
                    outputs.extend([selection, *map(Path, saved["run_outputs"])])
                    continue
                entry = {"protocol": protocol, "seed": seed, "arm": arm, "status": "RUNNING", "started_at": eastern_now()}
                ledger["runs"].append(entry)
                write_json(ledger_path, ledger)
                try:
                    manifest_path = directory / "manifest.json"
                    if manifest_path.exists() and read_json(manifest_path)["manifest_id"] != run_manifest["manifest_id"]:
                        raise RuntimeError("training inputs changed since the interrupted checkpoint")
                    arm_outputs = [write_json(manifest_path, run_manifest)]
                    if settings.mock:
                        train_rows = _evaluate(settings, arm, protocol, seed, "train")
                        val_rows = _evaluate(settings, arm, protocol, seed, "val")
                        arm_outputs.append(private_write(_rows_path(settings, protocol, seed, arm, "train"), train_rows, settings.source))
                        arm_outputs.append(private_write(_rows_path(settings, protocol, seed, arm, "val"), val_rows, settings.source))
                        info = {"adapter_dir": None, "execution": "mock", "learning": "NOT_RUN",
                                "selection": "fixed mock policy; validation exercised; no test access",
                                "wandb_run_ids": [], "training_patients": len(train_rows), "validation_patients": len(val_rows)}
                    else:
                        _require_cuda()
                        from .gpu import train
                        info = train(cfg, output_dir=directory, data_dir=prepared(settings, protocol), seed=seed)
                        adapter = Path(info["adapter_dir"])
                        selected = sorted(path for path in adapter.rglob("*") if path.is_file())
                        if not selected or not any(path.suffix in (".safetensors", ".bin") for path in selected):
                            raise RuntimeError("selected adapter has no weight files")
                        arm_outputs.extend(selected)
                    run_manifest["wandb_run_ids"] = info.get("wandb_run_ids", [])
                    write_json(manifest_path, run_manifest)
                    info.update({"source": settings.source, "arm": arm, "seed": seed,
                                 "config_hash": digest(OmegaConf.to_container(cfg, resolve=True)), "selection_split": "val",
                                 "training_manifest_id": run_manifest["manifest_id"],
                                 "run_outputs": {str(path): file_digest(path) for path in arm_outputs}})
                    outputs.append(write_json(selection, info))
                    outputs.extend(arm_outputs)
                    entry["status"] = "PASS"
                    entry["selection"] = str(selection)
                except BaseException as error:
                    entry["status"] = "FAILED"
                    entry["error_type"] = type(error).__name__
                    raise
                finally:
                    entry["finished_at"] = eastern_now()
                    write_json(ledger_path, ledger)
    outputs.append(ledger_path)
    return outputs


def s5(settings):
    outputs = []
    for protocol in protocols(settings):
        for seed in settings.seeds:
            for arm in ARMS:
                selection = read_json(settings.work_dir / "models" / protocol / str(seed) / arm / "selection.json")
                for path, expected in selection["run_outputs"].items():
                    if not Path(path).is_file() or file_digest(Path(path)) != expected:
                        raise RuntimeError("selected model changed before test admission")
                outputs.extend(_saved_evaluation(settings, arm, protocol, seed, adapter_dir=selection["adapter_dir"]))
    return outputs


def s6(settings):
    from .data import assert_disjoint
    from .stats import summary, paired_bootstrap, single_arm_intervals, seed_difference_ci, upper_bound_invariant
    from .reporting import render_report
    from .gpu import COMPATIBILITY_DEVIATIONS
    import pandas as pd
    results = {"source": settings.source, "execution": "mock" if settings.mock else "cuda",
               "label": "re-implementation on a different backbone", "model_revision": REVISION,
               "upstream_sha": read_json(settings.work_dir / "manifests/s0.json")["upstream_sha"],
               "protocols": {}, "baseline_gaps": {"B2": "ReAct not shipped upstream"},
               "training": read_json(settings.work_dir / "training-ledger.json"),
               "data_audit": private_read(settings.work_dir / "data-audit.json", settings.source),
               "bootstrap_resamples": settings.bootstrap_resamples, "training_seeds": list(settings.seeds)}
    outputs, report_parts = [], []
    for protocol in protocols(settings):
        current = {"seeds": {}, "leakage": assert_disjoint(prepared(settings, protocol))}
        frame = pd.read_csv(prepared(settings, protocol) / "test.csv")
        subjects = dict(zip(frame["Patient ID"].map(str), frame["subject_id"])) if "subject_id" in frame else None
        for seed in settings.seeds:
            cfg = cfg_for(settings, protocol=protocol, seed=seed)
            rows = {arm: private_read(_rows_path(settings, protocol, seed, arm), settings.source) for arm in (*ARMS, "zero_shot")}
            clusters = [int(subjects[str(r["patient_id"])]) for r in rows["primary"]] if subjects else None
            arms = {arm: {"summary": summary(data, cfg), "intervals": single_arm_intervals(data, cfg, seed=stream_seed(seed, "test", "bootstrap"), n_resamples=settings.bootstrap_resamples, cluster_ids=clusters)} for arm, data in rows.items()}
            for arm in ARMS:
                if not upper_bound_invariant([r["reward"] for r in rows[arm]], cfg_for(settings, arm, protocol, seed), arm)["holds"]:
                    raise RuntimeError("episode reward exceeded its mathematically justified ceiling")
            comparisons = {}
            for arm in ("zero_shot", "decision_only", "no_cost", "incorrect_minus1"):
                paired = paired_bootstrap(rows["primary"], rows[arm], cfg, seed=stream_seed(seed, "test", "bootstrap"), n_resamples=settings.bootstrap_resamples, cluster_ids=clusters)
                comparisons[arm] = paired
            current["seeds"][str(seed)] = {"arms": arms, "comparisons": comparisons}
        current["seed_differences"] = {
            f"primary_vs_{arm}/{metric}": seed_difference_ci([current["seeds"][str(seed)]["comparisons"][arm]["metrics"][metric]["estimate"] for seed in settings.seeds])
            for arm in ("zero_shot", "decision_only", "no_cost", "incorrect_minus1") for metric in ("T5", "T6", "T7", "T8", "pooled_accuracy")}
        results["protocols"][protocol] = current
        for seed in settings.seeds:
            run = current["seeds"][str(seed)]
            selection = read_json(settings.work_dir / "models" / protocol / str(seed) / "primary/selection.json")
            report_input = {
                "arms": {arm: values["summary"] for arm, values in run["arms"].items()},
                "intervals": {arm: values["intervals"] for arm, values in run["arms"].items()},
                "paired": {f"primary_vs_{arm}": value for arm, value in run["comparisons"].items()},
                "seed_differences": current["seed_differences"],
                "leakage": {"admission_overlap": sum(current["leakage"]["admission_overlap"].values()),
                            "subject_overlap": current["leakage"]["subjects"]["any_overlap"],
                            "grouped_sensitivity": "full arm matrix reported separately below" if len(protocols(settings)) == 2 else "unassessable: subject IDs absent"},
                "implementation": {"I1": "mock; no weights trained" if settings.mock else "pinned LoRA runtime completed",
                                   "I2": "configured; learning NOT_RUN" if settings.mock else "100/100/100 optimizer-update phases",
                                   "I3": "configured; Adam NOT_RUN" if settings.mock else "Adam, lr 1e-5, batch 2, declared seed",
                                   "I4": "configured upstream USD schedule; provenance taken from paper ledger",
                                   "I5": "twelve normalized costs sum to one (tested)"},
                "run": {"seed": seed, "seeds": list(settings.seeds), "backbone": MODEL, "revision": REVISION,
                        "upstream_sha": results["upstream_sha"], "training_completed": not settings.mock,
                        "git_sha": read_json(settings.work_dir / "manifests/s0.json")["git_sha"]},
                "validation_dynamics": selection.get("validation_history", []),
                "gaps": ["Local compatibility changes: " + text for text in COMPATIBILITY_DEVIATIONS] +
                         ["Packaging: no upstream lockfile; same-version xgrammar source build is pinned in the local lock.",
                          "A paper delta treats the published value as fixed; its interval does not incorporate unknown paper-run uncertainty.",
                          "Empty physical-examination strings become the existing unavailable sentinel before CSV parsing, avoiding non-string NaN observations.",
                          "Primary parity admissions may share subjects with training; the grouped protocol tests sensitivity to that leakage."]}
            path = render_report(report_input, settings.work_dir / "reports" / f"{protocol}-{seed}.md", source=settings.source)
            outputs.append(path)
            report_parts.append(f"# Protocol: {protocol}; training seed: {seed}\n\n" + path.read_text())
    # Patient resampling quantifies held-out uncertainty; seeds quantify training variation.
    results["between_seed_uncertainty"] = {"status": "not estimable from one training seed" if len(settings.seeds) == 1 else "paired t intervals reported per comparison"}
    private = settings.work_dir / "aggregates.json"
    # Ledger selections include private paths: aggregate export records outcomes, not paths.
    results["training"] = [{key: entry[key] for key in ("protocol", "seed", "arm", "status")}
                           for entry in results["training"]["runs"]]
    raw_audit = results["data_audit"]["schema"]
    results["data_audit"] = {"source": settings.source, "raw_admissions": raw_audit["total"],
                             "class_counts": raw_audit["class_counts"], "schema_valid": raw_audit["schema"]["valid"],
                             "subject_audit": raw_audit["subjects"], "mapping_rows": raw_audit["mapping"]["rows"]}
    launch = read_json(settings.work_dir / "manifests/s0.json")
    results["provenance"] = {"git_sha": launch["git_sha"], "source_fingerprint": launch["source_fingerprint"],
                             "stage_manifest_ids": {stage: read_json(settings.work_dir / "manifests" / f"{stage}.json")["manifest_id"] for stage in STAGES},
                             "runtime_lock_sha256": file_digest(OWN / "runtime/uv.lock"),
                             "wandb_run_ids": sorted({run_id for protocol in protocols(settings) for seed in settings.seeds for arm in ARMS
                                                      for run_id in read_json(settings.work_dir / "models" / protocol / str(seed) / arm / "selection.json").get("wandb_run_ids", [])})}
    private_write(private, results, settings.source)
    report = settings.work_dir / "REPORT.md"
    report.write_text("\n\n---\n\n".join(report_parts))
    return [private, report, *outputs]


def run_stage(settings, stage, dry_run=False):
    os.environ["VLLM_USE_V1"] = "0"
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    index = STAGES.index(stage)
    inputs = {"settings": {"source": settings.source, "data_dir": str(settings.data_dir),
                           "work_dir": str(settings.work_dir), "seeds": list(settings.seeds),
                           "synthetic_n": settings.synthetic_n, "bootstrap_resamples": settings.bootstrap_resamples}}
    if index and not dry_run:
        inputs["previous_stage"] = _receipt(settings, STAGES[index - 1])["receipt_id"]
    if stage == "s0" and settings.source == "mimic":
        inputs["raw_files"] = tree_fingerprint(settings.data_dir)
    cfg = cfg_for(settings)
    manifest = make_manifest(settings, stage, cfg, inputs, dry_run=dry_run)
    receipt_path = settings.work_dir / "receipts" / f"{stage}.json"
    manifest_path = settings.work_dir / "manifests" / f"{'dry-run-' if dry_run else ''}{stage}.json"
    with stage_lock(settings.work_dir, stage):
        if receipt_path.exists() and not dry_run:
            receipt = _receipt(settings, stage)
            if receipt["manifest_id"] != manifest["manifest_id"]:
                raise RuntimeError("completed stage manifest changed; use a new work directory to preserve the previous run")
            print(f"{stage.upper()}: RESUMED (verified artifact hashes)")
            return receipt
        write_json(manifest_path, manifest)
        if dry_run:
            print(f"{stage.upper()}: DRY_RUN; manifest saved; no stage computation")
            return manifest
        started_at = eastern_now()
        try:
            outputs = globals()[stage](settings)
            receipt = {"source": settings.source, "stage": stage, "manifest_id": manifest["manifest_id"],
                       "status": "PASS", "started_at": started_at, "finished_at": eastern_now(),
                       "previous_receipt_id": inputs.get("previous_stage"),
                       "outputs": {str(p): file_digest(p) for p in outputs}}
            receipt["receipt_id"] = digest(receipt)
            write_json(receipt_path, receipt)
        except BaseException as error:
            write_json(settings.work_dir / "failures" / f"{stage}-{started_at.replace(':', '')}.json",
                       {"source": settings.source, "manifest_id": manifest["manifest_id"], "status": "FAILED", "error_type": type(error).__name__})
            raise
    print(f"{stage.upper()}: PASS ({settings.source})")
    return receipt


def export_aggregates(settings):
    _receipt(settings, "s6")
    value = private_read(settings.work_dir / "aggregates.json", settings.source)
    allowed = {"source", "execution", "label", "model_revision", "upstream_sha", "protocols", "baseline_gaps", "training", "data_audit", "bootstrap_resamples", "training_seeds", "between_seed_uncertainty", "provenance"}
    if set(value) != allowed or value["source"] != settings.source:
        raise ValueError("aggregate export does not match the expected result structure")
    assert_aggregate(value)
    destination = OWN / "results" / settings.source
    stamp_metrics("metrics", value, settings.source, destination)
    shutil.copyfile(settings.work_dir / "REPORT.md", destination / "REPORT.md")
    print(f"aggregate-only results exported: {destination}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=(*STAGES, "all", "validate", "export", "check-git", "python"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fake-summarizer", action="store_true")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args(argv)
    settings = Settings.load(args.env_file)
    if args.fake_summarizer and not settings.mock:
        parser.error("--fake-summarizer is only permitted with synthetic input")
    if args.stage == "python":
        runtime = settings.work_dir / "runtime-env/bin/python"
        print(runtime if not settings.mock and runtime.exists() else sys.executable)
    elif args.stage == "validate":
        from .data import validate_raw
        print(json.dumps(validate_raw(settings.data_dir, source=settings.source), indent=2))
    elif args.stage == "check-git":
        violations = tracked_data_violations()
        if violations:
            raise RuntimeError("\n".join(violations))
        print("tracked-file check: PASS")
    elif args.stage == "export":
        export_aggregates(settings)
    else:
        for stage in STAGES if args.stage == "all" else (args.stage,):
            run_stage(settings, stage, args.dry_run)


if __name__ == "__main__":
    main()
