"""The E1 pipeline end to end on a configuration small enough for a test: tables, rules, part A, Figure 1 inputs, manifest."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tiicee.experiments.e1 import cells, config, figure1, manifest, part_a, points, rules, run, tables
from tiicee.experiments.e1.context import Context

SMALL = config.tiny(seeds=(0, 1), variants=("base",), readers=("r0", "independence", "entrench", "fixed", "over", "under"),
                    horizon_reduced=2, horizon_full=2, n_train=250, n_eval=12, n_gate=30, n_boot=100, lam_frontier=(1.0,))


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    root = tmp_path_factory.mktemp("e1")
    out, scratch = root / "out", root / "scratch"
    for seed in SMALL.seeds:
        cells.run_cell(SMALL, seed, "base", out, scratch)
    intervals = run.c_intervals(SMALL, scratch)
    (out / "c_intervals.json").write_text(json.dumps(intervals))
    return root, out, scratch, intervals


def test_every_table_of_parts_b_c_and_d_is_written_for_every_cell(finished):
    _, out, scratch, _ = finished
    n_readers, n_r4 = len(SMALL.readers), len(SMALL.readers) - 1
    cell = 2 * n_readers * 3                                    # acquisition rules x readers x world models
    expected = {
        "B_reduced_horizon": cell * 3 * 3,                      # arms x horizons 0..2
        "B_reduced_commit": cell * (3 * 3 + 3 + 2 * 1),         # thresholds x arms, the step limit x arms, CEE-Commit x tracks x lambda
        "B_reduced_fallback": cell,
        "B_points": cell,
        "B_sandbox": 5,                                         # the sandbox's five reported cells
        "C_summary": 2 * n_readers * (7 + 1),                   # controlled errors, and the row across them
        "D_horizon": 2 * n_r4 * 2 * (3 + 2 * 2 * 7) * 3,        # arms: raw, two projections, tracks x variants x budgets
        "D_final": 2 * n_r4 * 2 * (3 + 2 * 2 * 7),              # the same arms at the final horizon only
        "D_commit": 2 * n_r4 * 2 * (3 + 28) * 2,                # threshold 0.8 and the step limit
        "D_tau": 2 * n_r4 * 2 * 28,
        "D_heldout": 3 * 6,                                     # tests x strata
    }
    for name, per_seed in expected.items():
        rows = tables.read_table(out, name)
        assert len(rows) == per_seed * len(SMALL.seeds), name
        assert {r["seed"] for r in rows} == set(SMALL.seeds) and {r["variant"] for r in rows} <= {"base", "strong_correlation"}
    assert len(list((scratch / "C_rows").glob("*.npz"))) == len(SMALL.seeds) * 2 * n_readers
    d = tables.read_table(out, "D_horizon")
    assert max(abs(r["residual"]) for r in d) < 1e-8            # the identity of 5.12 (iii) holds for every arm at every horizon
    # layout: the large table is kept apart, everything else sits under tables/, and all of it is gzipped
    assert sorted(p.name for p in (out / "large").iterdir()) == ["D_horizon"]
    assert "D_horizon" not in {p.name for p in (out / "tables").iterdir()} and "D_final" in {p.name for p in (out / "tables").iterdir()}
    assert all(p.name.endswith(".csv.gz") for p in tables.files(out)) and len(tables.files(out)) == len(expected) * len(SMALL.seeds)
    final = [r for r in d if r["horizon"] == SMALL.horizon_reduced]
    assert tables.read_table(out, "D_final") == final          # D_final is D_horizon at the final horizon, row for row
    b = tables.read_table(out, "B_points")
    assert max(r["identity_error"] for r in b) < 1e-9           # and the gain identity at every decision point


def test_tables_round_trip_and_cells_are_reproducible(finished, tmp_path):
    _, out, _, _ = finished
    again = tmp_path / "again"
    cells.run_cell(SMALL, 1, "base", again, None, parts=("B_reduced", "B_points"))
    for name in ("B_reduced_horizon", "B_points"):
        first, second = tables.cell_file(out, name, 1, "base"), tables.cell_file(again, name, 1, "base")
        assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(second.read_bytes()).hexdigest()
    rows = [{"a": 1, "b": 0.1 + 0.2, "c": None, "d": "text", "e": True}, {"a": 2, "b": 1e-300, "c": 3.5, "d": "x", "e": False, "f": -1}]
    for name in ("t.csv", "t.csv.gz"):
        tables.write(tmp_path / name, rows)
        back = tables.read(tmp_path / name)
        assert back[0] == {"a": 1, "b": 0.1 + 0.2, "c": None, "d": "text", "e": True, "f": None} and back[1]["b"] == 1e-300 and back[1]["f"] == -1
    import gzip
    import time

    packed = (tmp_path / "t.csv.gz").read_bytes()
    assert gzip.decompress(packed) == (tmp_path / "t.csv").read_bytes()
    time.sleep(1.1)                                             # a gzip header normally carries the time; these do not
    tables.write(tmp_path / "later.csv.gz", rows)
    assert (tmp_path / "later.csv.gz").read_bytes() == packed
    assert tables.cell_file(tmp_path, "B_points", 3, "base") == tmp_path / "tables" / "B_points" / "seed03_base.csv.gz"
    assert tables.cell_file(tmp_path, "D_horizon", 3, "base") == tmp_path / "large" / "D_horizon" / "seed03_base.csv.gz"


def test_part_c_intervals_use_every_row_and_resample_patients_within_seeds(finished):
    _, out, scratch, found = finished
    n_rows = len(SMALL.seeds) * 2 * len(SMALL.readers) * SMALL.horizon_full * SMALL.n_eval * 7
    assert found["rows"] == n_rows
    gap, pair = found["spearman_difference"], found["matched_pair"]
    assert gap["n_units"] == pair["n_units"] == len(SMALL.seeds) * SMALL.n_eval and gap["n_rows"] == n_rows
    assert "within strata" in gap["method"] and "within strata" in pair["method"]
    data = [np.load(p) for p in sorted((scratch / "C_rows").glob("*.npz"))]
    regret = np.concatenate([d["regret"].ravel() for d in data])
    e_w, kl = (np.concatenate([d[name].ravel() for d in data]) for name in ("e_w", "kl"))
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)
    assert np.isfinite(regret).all() and (regret >= -1e-12).all() and (kl >= 0).all() and (e_w >= 0).all()
    from tiicee import stats

    assert gap["estimate"] == pytest.approx(stats.spearman(regret, e_w) - stats.spearman(regret, kl), abs=1e-12)
    errors = list(data[0]["errors"])
    difference = np.concatenate([(d["regret"][:, :, errors.index("matched_responsive")] - d["regret"][:, :, errors.index("matched_ignored")]).ravel() for d in data])
    assert pair["estimate"] == pytest.approx(difference.mean(), abs=1e-12)


def test_rules_part_a_figure_and_manifest_run_from_the_tables(finished, monkeypatch):
    _, out, _, intervals = finished
    verdicts = run.evaluate_rules(out, SMALL)
    assert verdicts.pop("other_variants")["verdicts"] == {}     # one variant in this run
    assert verdicts == rules.evaluate(run.load_rule_tables(out), SMALL, c_intervals=intervals)
    assert all(v.get("status", "evaluated") == "evaluated" for v in verdicts.values())
    assert {"track_selection", "repair_present", "negative_controls", "no_harm", "gate_valid", "sandbox_cross_check", "e1c_supported",
            "variant_selection", "scale_present", "usable_outside", "selected_layer"} == set(verdicts)
    assert verdicts["no_harm"]["verdict"] is True               # exact, whatever the size of the run
    (out / "verdicts.json").write_text(json.dumps(verdicts))

    layer = verdicts["selected_layer"]
    for seed in SMALL.seeds:
        run._part_a_job((SMALL, seed, "base", str(out), (layer["track"], layer["scale"], layer["budget"])))
    a = tables.read_table(out, "A_values")
    assert len(a) == len(SMALL.seeds) * 5 * 2
    for row in a:
        assert row["v_***"] == pytest.approx(row["v_star"], abs=1e-10)
        assert all(row[part_a.key(c)] <= row["v_star"] + 1e-10 for c in part_a.COMBINATIONS)
        assert row["layer"] == (layer["label"] if row["agent"] == "repaired" else "none")

    monkeypatch.setattr(figure1, "POWERS", (0.5, 1.0, 2.0))
    monkeypatch.setattr(figure1, "DELTAS", (0.0, 0.4))
    for seed in SMALL.seeds:
        tables.write(tables.cell_file(out, "F_harmful_grid", seed, "base"), figure1.harm_cell(SMALL, seed, "base"))
    grid = tables.read_table(out, "F_harmful_grid")
    assert len(grid) == len(SMALL.seeds) * 3 * 2 * 2
    plain = next(r for r in grid if r["power"] == 1.0 and r["delta"] == 0.0 and r["seed"] == 0)
    marker_source = [r for r in tables.read_table(out, "B_points") if r["seed"] == 0 and r["reader"] == "independence" and r["world_model"] == "W0"]
    assert plain["log_tau_b"] == pytest.approx(np.mean([r["log_tau_b"] for r in marker_source]), abs=1e-12)   # same coordinates as a marker
    assert plain["kl_inc_at_tau_b"] == pytest.approx(np.mean([r["kl_inc_at_tau_b"] for r in marker_source]), abs=1e-12)

    verdicts = run.evaluate_rules(out, SMALL)
    assert verdicts["attribution"]["status"] == "evaluated"
    (out / "verdicts.json").write_text(json.dumps(verdicts))
    written = figure1.write(out, out.parent / "figure1", SMALL)
    assert [p.name for p in written] == ["regime_map.json", "regret_attribution.json"]
    regime = json.loads(written[0].read_text())
    assert regime["schema"] == "tii-fig1-regime/1" and regime["generated_at"].endswith(("-05:00", "-04:00"))
    assert [m["id"] for m in regime["markers"]] == [f"E1:{r}" for r in SMALL.readers]
    for marker in regime["markers"]:
        assert set(marker) == {"id", "label", "experiment", "kind", "tau_b", "tau_b_lo", "tau_b_hi", "kl_inc", "kl_inc_lo", "kl_inc_hi", "delta_log_loss"}
        assert marker["experiment"] == "E1" and marker["kind"] == "simulated"
        assert marker["tau_b_lo"] <= marker["tau_b"] <= marker["tau_b_hi"] and marker["kl_inc_lo"] <= marker["kl_inc"] <= marker["kl_inc_hi"]
    assert all(set(v) == {"tau_b", "kl_inc"} for v in regime["harmful_region"])
    notes = json.loads((out / "figure1_notes.json").read_text())["harmful_region"]
    assert notes["track"] == verdicts["selected_layer"]["track"] and notes["enclosed"] <= notes["harmful"] <= notes["grid_readers"]
    r0 = regime["markers"][0]
    # R0 under the truth: at the origin up to the 1% bisection tolerance of tau_B, which leaves about 1e-8 nats of incoherence
    assert r0["tau_b"] == pytest.approx(1.0, abs=0.01) and abs(r0["kl_inc"]) < 1e-6
    regret = json.loads(written[1].read_text())
    assert regret["schema"] == "tii-fig1-regret/1"
    for agent in regret["agents"]:
        assert set(agent) == {"id", "label", "variant", "selection", "reading", "commitment"} and agent["variant"] in ("raw", "repaired")
        assert agent["selection"] + agent["reading"] + agent["commitment"] == pytest.approx(1.0, abs=1e-9)

    record = manifest.build(SMALL, out, "test", "pytest")
    assert record["experiment_id"] == "E1" and record["snapshot"]["digest"].startswith("sha256:")
    assert record["card"]["sha256"] == hashlib.sha256(manifest.CARD.read_bytes()).hexdigest()
    assert set(record["authorization"]) == {"AUTHORIZATION.md", "AUTHORIZATION_ADDENDUM_1.md"}
    assert record["config"] == SMALL.as_dict() and record["seeds_by_purpose"]["user_seeds"] == [0, 1]
    assert {"generation", "fitting", "behavior", "corruption", "montecarlo"} <= set(record["seeds_by_purpose"])
    assert "tiicee/experiments/e1/rules.py" in record["bound_files"] and "tiicee/cee/core.py" in record["bound_files"]
    assert "vendor/synthetic_df467ea/SHA256SUMS" in record["bound_files"]
    name = "tables/B_points/seed00_base.csv.gz"
    assert record["tables"][name]["sha256"] == hashlib.sha256((out / name).read_bytes()).hexdigest() and record["tables"][name]["rows"] == 36
    assert record["tables"][name]["committed"] is True and record["tables"][name]["bytes"] == (out / name).stat().st_size
    large = "large/D_horizon/seed01_base.csv.gz"
    assert record["tables"][large]["committed"] is False
    assert record["tables"][large]["sha256"] == hashlib.sha256((out / large).read_bytes()).hexdigest()
    assert set(record["tables"]) == {p.relative_to(out).as_posix() for p in tables.files(out)}
    assert record["code"]["commit"] is None or len(record["code"]["commit"]) == 40
    assert record["code"]["bound_paths_commit"] is None or len(record["code"]["bound_paths_commit"]) == 40
    run.main(["manifest", "--config", "tiny", "--out", str(out), "--figure", str(out.parent / "figure1"), "--phase", "test"])
    written_manifest = json.loads((out / "manifest.json").read_text())
    assert set(written_manifest["files"]) == {"verdicts.json", "c_intervals.json", "figure1_notes.json", "regime_map.json", "regret_attribution.json"}
    assert written_manifest["files"]["verdicts.json"] == hashlib.sha256((out / "verdicts.json").read_bytes()).hexdigest()
    assert written_manifest["pending"] == [] and set(written_manifest["cell_seconds"]) == set()


def test_rules_are_also_evaluated_on_the_other_variants_and_part_c_is_marked_pending(tmp_path):
    """Reading R5 of the card: the other variants are reported, and a verdict that differs there is named."""
    cfg = config.tiny(seeds=(0, 1), variants=("base", "duplicated_panel"))
    for seed in cfg.seeds:
        for variant in cfg.variants:
            rows = []
            for selector in ("rand", "ig"):
                for reader in cfg.readers:
                    for world in ("W0", "W1", "W2"):
                        for arm, loss in (("raw", 0.60), ("stateless", 0.55), ("stateful", 0.50 if variant == "base" else 0.58)):
                            rows.append({"seed": seed, "variant": variant, "selector": selector, "reader": reader, "world_model": world,
                                         "arm": arm, "horizon": 3, "log_loss": loss + 0.001 * seed, "accuracy": 0.5})
            tables.write(tables.cell_file(tmp_path, "B_reduced_horizon", seed, variant), rows)
    verdicts = run.evaluate_rules(tmp_path, cfg)
    assert verdicts["track_selection"]["verdict"] == "stateful"
    other = verdicts["other_variants"]
    assert other["verdicts"]["duplicated_panel"]["track_selection"]["verdict"] == "stateless"       # stateful is worse there
    assert other["differs_from_primary"] == {"duplicated_panel": ["selected_layer", "track_selection"]}
    assert "e1c_supported" not in other["verdicts"]["duplicated_panel"]
    assert verdicts["e1c_supported"]["status"] == "incomplete" and verdicts["e1c_supported"]["note"] == run.PENDING
    assert "pending interval" in run.PENDING


def test_harmful_polygon_traces_the_boundary_of_the_harmful_grid_readers():
    def row(power, delta, drift, harm):
        return {"seed": 0, "power": power, "delta": delta, "drift": drift, "log_tau_b": float(np.log(power)), "kl_inc_at_tau_b": delta / 10,
                "harm_stateless": harm, "harm_stateful": -1.0}

    # projection alone hurts readers with power 3 and 4 at every drift, power 2 only without drift, and no under-reactive reader
    rows = [row(p, d, k, 1.0 if (p >= 3 or (p == 2 and d == 0)) else -1.0) for p, d, k in figure1.grid_readers()]
    polygon = figure1.harmful_polygon(rows, "stateless")
    assert polygon[0] == polygon[-1] == {"tau_b": 1.0, "kl_inc": 0.0}
    corners = {(round(v["tau_b"], 6), round(v["kl_inc"], 6)) for v in polygon}
    assert (2.0, 0.0) in corners and (3.0, 0.08) in corners and (4.0, 0.0) in corners and (4.0, 0.08) in corners
    assert all(v["tau_b"] >= 1.0 for v in polygon)                         # nothing on the under-reactive side
    assert (2.0, 0.01) not in corners                                      # power 2 with a drift is not harmful
    assert figure1.harmful_polygon(rows, "stateful") == []                 # no harmful reader, no region
    # a reader that is harmful with a harmless one beyond it is not swept in: the outline starts where harm is unbroken
    gap = [row(p, d, k, 1.0 if p in (1.5, 4.0) else -1.0) for p, d, k in figure1.grid_readers()]
    assert {round(v["tau_b"], 6) for v in figure1.harmful_polygon(gap, "stateless")} == {1.0, 4.0}
    # nine distinct readers per power (no drift, and four sizes of two kinds): 18 harmful, the 9 at power 4 enclosed
    summary = figure1.harmful_summary(gap, "stateless")
    assert (summary["grid_readers"], summary["harmful"], summary["enclosed"], summary["harmful_at_power_one"]) == (81, 18, 9, 0)
    summary = figure1.harmful_summary(rows, "stateless")
    assert (summary["harmful"], summary["enclosed"]) == (2 * 9 + 1, 2 * 9 + 1)
    assert figure1.harmful_summary(rows, "stateful")["harmful"] == 0


def test_context_gives_each_seed_its_own_order_fits_and_noise():
    a, b = Context(SMALL, 0, "base"), Context(SMALL, 1, "base")
    assert sorted(a.order("full").tolist()) == list(range(12)) and a.order("full").tolist() != b.order("full").tolist()
    assert a.order("full").tolist() == Context(SMALL, 0, "base").order("full").tolist()
    assert a.twin("reduced").model.tests == ("CT", "Liver Function Panel", "Ultrasound") and a.horizon("reduced") == 2
    assert Context(SMALL, 0, "duplicated_panel").twin("reduced").model.tests[-1] == "Liver Function Panel (repeat)"
    assert not np.allclose(a.fitted("reduced", "W1").prior, b.fitted("reduced", "W1").prior)
    assert a.noise(5, 0).tolist() != a.noise(5, 1).tolist() and a.noise(5, 0).tolist() == a.noise(5, 0).tolist()
    with pytest.raises(ValueError):
        a.world_filter("reduced", "W9")
    full = config.card()
    assert full.seeds == tuple(range(20)) and full.n_train == 5000 and full.n_eval == 2000 and full.horizon_full == 8
    assert full.variants == ("base", "no_correlation", "strong_correlation", "duplicated_panel") and "informative_missingness" not in full.variants
    assert full.lam_frontier == (0.0, 0.5, 1.0, 2.0) and full.threshold == 0.8 and full.threshold_sensitivity == (0.7, 0.9)
    assert full.mirror == (0.1, 0.2, 0.4) and full.temperatures == (0.5, 2.0) and full.n_boot == 10000
    assert points.error_names(full) == ("mirror(0.1)", "mirror(0.2)", "mirror(0.4)", "temperature(0.5)", "temperature(2)", "matched_ignored", "matched_responsive")
