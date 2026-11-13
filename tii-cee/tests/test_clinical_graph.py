"""The training track's graphs (graph_training_m1.json, then graph_training_m2.json), checked against
the pipeline's own loader and against what each clinical stage reads."""
import importlib
import json

import pytest

from test_clinical_support import context, prepared
from tiicee.clinical import ROOT, backbone, bench, folds
from tiicee.pipeline import graph as graphs
from tiicee.pipeline import stages as registry
from tiicee.pipeline.paths import Paths

FRAGMENT = ROOT / "tii-cee" / "tiicee" / "clinical" / "graph_training_m1.json"
SECOND = FRAGMENT.with_name("graph_training_m2.json")
ROLES = {  # the roles and seeds each stage takes from its manifest
    "clinical_data": (set(), set()), "clinical_folds": (set(), set()),
    "clinical_bench": ({"data"}, {"training"}), "clinical_train": ({"data"}, {"training"}),
    "clinical_rollout": ({"split", "lab_test_mapping", "adapter", "train_csv"}, {"policy"}),
    "clinical_acceptance": ({"stored", "regenerated"}, set()),
}
ENVIRONMENT = {"clinical_bench": backbone.TRAINING_ENVIRONMENT, "clinical_train": backbone.TRAINING_ENVIRONMENT,
               "clinical_rollout": backbone.ROLLOUT_ENVIRONMENT}


def load(path):
    return graphs.Graph(json.loads(path.read_text()), graphs.load_allowances(Paths(code=ROOT)))


@pytest.fixture(scope="module")
def graph():
    return load(FRAGMENT)


@pytest.fixture(scope="module", params=[FRAGMENT, SECOND], ids=["m1", "m2"])
def either(request):
    return load(request.param)


def test_the_fragment_is_a_graph_the_pipeline_loads(graph):
    assert (graph.study, graph.milestone) == ("TII-CEE-1", "M1")
    assert [stage["id"] for stage in graph.stages] == [
        "e8-data", "e10-folds", "bench-nemotron", "e9-bench-granite", "p-rollout-val", "launch-check-7-4a"]
    for module in graph.stage_modules:
        importlib.import_module(module)
    assert {stage["command"][0] for stage in graph.stages} <= set(registry.REGISTRY)  # every stage name is registered


def test_each_stage_gets_the_roles_seeds_and_environment_it_reads(either):
    graph = either
    for stage in graph.stages:
        name, granite = stage["command"][0], "granite" in stage["command"]
        roles, seeds = ROLES[name]
        assert set(stage["bind"]) == roles | ({"backbone_overlay"} if granite else set()), stage["id"]
        assert set(stage["seeds"]) == seeds, stage["id"]
        assert stage["env"] == ENVIRONMENT.get(name, {}), stage["id"]       # exactly what the stage verifies at start
        assert stage["resources"]["gpus"] == (1 if name in ENVIRONMENT else 0), stage["id"]
        assert stage["kind"] == "normal" and stage["pairing_unit"] == "admission"
    overlay = graph.by_id["e9-bench-granite"]["bind"]["backbone_overlay"]
    assert (ROOT / overlay).resolve() == backbone.BACKBONES["granite"].overlay.resolve()


def test_inputs_come_from_stages_the_stage_waits_for(either, tmp_path, monkeypatch):
    graph = either
    for stage in graph.stages:
        for path in stage["bind"].values():
            if path.startswith("$TII/"):
                parts = path.split("/")
                assert parts[1:3] == ["private", "stages"] and parts[4] == "out", path  # a completed stage's output
                assert parts[3] in stage["depends_on"], (stage["id"], path)
    monkeypatch.setenv("CDM_WORK_DIR", str(prepared(tmp_path).parent))
    data = context(tmp_path, "e8-data")
    folds.run_data(data)  # the files the fragment names under e8-data/out are the ones the stage writes
    named = {path.split("/out", 1)[1].lstrip("/") for stage in graph.stages for path in stage["bind"].values()
             if path.startswith("$TII/private/stages/e8-data/out")}
    assert named == {"", "train.csv", "val.csv", "lab_test_mapping.csv"}
    assert all((data.out_dir / name).exists() for name in named)
    assert graph.by_id["launch-check-7-4a"]["bind"]["regenerated"].endswith("/p-rollout-val/out/rows.jsonl")


def test_the_two_paths_outside_the_repository_are_resolved_inside_private_storage(graph):
    absolute = {(stage["id"], role): path for stage in graph.stages for role, path in stage["bind"].items()
                if path.startswith("/")}
    assert set(absolute) == {("p-rollout-val", "adapter"), ("launch-check-7-4a", "stored")}
    assert all(path.startswith("/blue/cai6734/daniel.escalante/private/") for path in absolute.values())
    assert not any("FILL-IN" in path for path in absolute.values())


def test_benches_use_their_defaults_and_stay_within_three_gpu_hours(graph):
    nemotron, granite = graph.by_id["bench-nemotron"], graph.by_id["e9-bench-granite"]
    assert bench.options_of(nemotron["command"][1:]).reference is True
    assert bench.options_of(granite["command"][1:]).reference is False
    assert (nemotron["allowance_key"], nemotron["experiment"]) == ("benches", "benches")
    assert (granite["allowance_key"], granite["experiment"]) == ("E9", "E9")
    assert nemotron["estimate_gpu_hours"] == 2.4 and granite["estimate_gpu_hours"] == 2.1  # bench.py's estimates
    assert "bench-nemotron" in granite["depends_on"]  # the resume mechanics are benched first
    spent = {}
    for stage in graph.stages:
        spent[stage["allowance_key"]] = spent.get(stage["allowance_key"], 0) + stage["estimate_gpu_hours"]
        assert stage["estimate_gpu_hours"] <= 3
    assert all(total <= graph.allowances[key]["estimate_gpu_hours"] for key, total in spent.items() if total)


# ---- graph_training_m2.json: the E8 and E10 trainings after gate G_resume ----

E8_SEEDS, FOLDS = (270, 271, 272, 273), range(5)  # EXPERIMENT_CARDS/E8.md and E10.md


def test_the_second_graph_keeps_the_first_unchanged_and_adds_the_trainings():
    first, second = json.loads(FRAGMENT.read_text()), json.loads(SECOND.read_text())
    assert second["stages"][:len(first["stages"])] == first["stages"]  # the watcher keeps their records by id
    added = [stage["id"] for stage in second["stages"][len(first["stages"]):]]
    assert added == [name for seed in E8_SEEDS for name in (f"e8-train-s{seed}", f"e8-rollout-s{seed}")] + \
        [name for fold in FOLDS for name in (f"e10-train-f{fold}", f"e10-rollout-f{fold}")]
    assert second["stage_modules"] == first["stage_modules"]


def test_every_training_waits_on_g_resume_and_the_gate_reads_the_bench_report():
    graph = load(SECOND)
    (gate,) = graph.gates
    assert gate == {"id": "G_resume", "rule": "resume_gate.py", "inputs": ["bench-nemotron"],
                    "reads": ["export/approved/bench-nemotron.json"]}
    assert (ROOT / "tii-cee" / "gates" / gate["rule"]).is_file()
    for stage in graph.stages:
        name = stage["command"][0]
        assert (stage["gate"] == "G_resume") == (name == "clinical_train"), stage["id"]
        if name == "clinical_rollout" and stage["id"] != "p-rollout-val":
            trained = [d for d in stage["depends_on"] if graph.by_id[d]["command"][0] == "clinical_train"]
            assert len(trained) == 1 and graph.by_id[trained[0]]["gate"] == "G_resume"  # skipped with its training


def test_e8_trains_the_four_new_seeds_and_rolls_each_out_on_validation():
    graph = load(SECOND)
    data = "$TII/private/stages/e8-data/out"
    for seed in E8_SEEDS:
        train, rollout = graph.by_id[f"e8-train-s{seed}"], graph.by_id[f"e8-rollout-s{seed}"]
        assert train["bind"] == {"data": data} and train["seeds"] == {"training": seed}
        assert rollout["seeds"] == {"policy": seed}
        assert rollout["bind"] == {"split": f"{data}/val.csv", "lab_test_mapping": f"{data}/lab_test_mapping.csv",
                                   "train_csv": f"{data}/train.csv",
                                   "adapter": f"$TII/private/stages/e8-train-s{seed}/out/model/adapter"}
        assert train["experiment"] == rollout["experiment"] == train["allowance_key"] == rollout["allowance_key"] == "E8"


def test_each_e10_fold_policy_trains_off_its_fold_and_rolls_out_only_on_it():
    graph = load(SECOND)
    assert folds.PARTS == ("train", "val", "heldout")  # the files clinical_folds writes in each fold directory
    for fold in FOLDS:
        data = f"$TII/private/stages/e10-folds/out/fold-{fold}"
        train, rollout = graph.by_id[f"e10-train-f{fold}"], graph.by_id[f"e10-rollout-f{fold}"]
        assert train["bind"] == {"data": data} and train["seeds"] == {"training": 269}
        assert rollout["seeds"] == {"policy": 269}
        assert rollout["bind"] == {"split": f"{data}/heldout.csv", "lab_test_mapping": f"{data}/{folds.MAPPING}",
                                   "train_csv": f"{data}/train.csv",
                                   "adapter": f"$TII/private/stages/e10-train-f{fold}/out/model/adapter"}
        assert train["experiment"] == rollout["experiment"] == train["allowance_key"] == rollout["allowance_key"] == "E10"


def test_the_second_graph_stays_within_each_experiments_estimate():
    graph = load(SECOND)
    spent = {}
    for stage in graph.stages:
        spent[stage["allowance_key"]] = spent.get(stage["allowance_key"], 0) + stage["estimate_gpu_hours"]
        if stage["command"][0] == "clinical_train":
            assert stage["resources"]["time_hours"] == 24 and stage["env"] == backbone.TRAINING_ENVIRONMENT
    assert all(total <= graph.allowances[key]["estimate_gpu_hours"] for key, total in spent.items())
