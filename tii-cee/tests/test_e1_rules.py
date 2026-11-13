"""The decision rules of the E1 card on hand-made tables with hand-made verdicts."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from tiicee.experiments.e1 import rules
from tiicee.experiments.e1.config import Config

SEEDS = (0, 1, 2)
CFG = Config(name="rules-test", seeds=SEEDS, variants=("base",), n_boot=2000)
READERS = CFG.readers
R4 = READERS[1:]


def spread(differences, key):
    """A hand-checkable stand-in for the bootstrap interval: the mean, the smallest and the largest per-seed difference."""
    return {"estimate": float(np.mean(differences)), "lo": float(min(differences)), "hi": float(max(differences))}


def evaluator():
    return rules.Evaluator(CFG, spread)


# ------------------------------------------------------------------------------------------- part B
def b_table(log_loss):
    """log_loss(seed, reader, world, arm) at the final horizon; horizon 0 holds a decoy the rules must ignore."""
    rows = []
    for seed, selector, reader, world, arm in itertools.product(SEEDS, ("rand", "ig"), READERS, ("W0", "W1", "W2"), ("raw", "stateless", "stateful")):
        shift = 0.01 if selector == "rand" else -0.01                       # the two acquisition rules average out
        rows.append({"seed": seed, "variant": "base", "selector": selector, "reader": reader, "world_model": world, "arm": arm,
                     "horizon": 0, "log_loss": 9.0, "accuracy": 0.1})
        rows.append({"seed": seed, "variant": "base", "selector": selector, "reader": reader, "world_model": world, "arm": arm,
                     "horizon": 3, "log_loss": log_loss(seed, reader, world, arm) + shift, "accuracy": 0.5})
    return rows


def textbook(seed, reader, world, arm):
    """Drift readers are repaired by both tracks, the stateful one more; scale readers are harmed; R0 under W0 is untouched."""
    raw = 0.60 + 0.01 * seed
    if arm == "raw":
        return raw
    if reader in ("entrench", "fixed"):
        return raw - (0.10 if arm == "stateful" else 0.05)
    if reader in ("over", "under"):
        return raw + 0.04
    if reader == "r0" and world == "W0":
        return raw + 5e-10
    return raw - 0.01


def test_track_selection_prefers_the_lower_track_and_breaks_ties_toward_stateless():
    found = evaluator().track_selection(b_table(textbook))
    assert found["verdict"] == "stateful" and not found["tie"]
    assert found["stateful_minus_stateless"] == pytest.approx({"estimate": -0.05, "lo": -0.05, "hi": -0.05})

    def mixed(seed, reader, world, arm):            # stateful better in seeds 0 and 1, worse in seed 2: the interval straddles 0
        if reader in ("entrench", "fixed") and arm == "stateful":
            return textbook(seed, reader, world, "stateless") + (0.02 if seed == 2 else -0.02)
        return textbook(seed, reader, world, arm)

    found = evaluator().track_selection(b_table(mixed))
    assert found["verdict"] == "stateless" and found["tie"]
    assert found["stateful_minus_stateless"]["lo"] == pytest.approx(-0.02) and found["stateful_minus_stateless"]["hi"] == pytest.approx(0.02)

    worse = lambda seed, reader, world, arm: textbook(seed, reader, world, arm) + (0.2 if arm == "stateful" else 0.0)
    assert evaluator().track_selection(b_table(worse))["verdict"] == "stateless"

    def only_w0(seed, reader, world, arm):          # the rule reads W2; a stateful advantage under W0 alone decides nothing
        if arm == "stateful" and world != "W0":
            return textbook(seed, reader, world, "stateless")
        return textbook(seed, reader, world, arm)

    assert evaluator().track_selection(b_table(only_w0))["verdict"] == "stateless"


def test_repair_present_needs_both_drift_families_under_both_world_models():
    found = evaluator().repair_present(b_table(textbook), "stateful")
    assert found["verdict"] is True and found["killed"] is False
    assert found["raw_minus_corrected"]["entrench|W0"] == pytest.approx({"estimate": 0.10, "lo": 0.10, "hi": 0.10, "improves": True})
    assert set(found["raw_minus_corrected"]) == {"entrench|W0", "fixed|W0", "entrench|W2", "fixed|W2"}
    assert evaluator().repair_present(b_table(textbook), "stateless")["raw_minus_corrected"]["fixed|W2"]["estimate"] == pytest.approx(0.05)

    def straddles(seed, reader, world, arm):        # fixed drift under W2 gains in two seeds and loses in one
        if reader == "fixed" and world == "W2" and arm == "stateful" and seed == 1:
            return textbook(seed, reader, world, "raw") + 0.01
        return textbook(seed, reader, world, arm)

    found = evaluator().repair_present(b_table(straddles), "stateful")
    assert found["verdict"] is False and found["killed"] is False          # the truth still shows the repair
    assert found["raw_minus_corrected"]["fixed|W2"]["improves"] is False

    def fails_under_truth(seed, reader, world, arm):
        if reader == "entrench" and world == "W0" and arm == "stateful":
            return textbook(seed, reader, world, "raw") + 0.02
        return textbook(seed, reader, world, arm)

    found = evaluator().repair_present(b_table(fails_under_truth), "stateful")
    assert found["verdict"] is False and found["killed"] is True


def test_no_harm_is_the_largest_change_of_r0_under_the_truth():
    found = evaluator().no_harm(b_table(textbook))
    assert found["verdict"] is True and found["largest_change"] == pytest.approx(5e-10, rel=1e-3)

    def nudged(seed, reader, world, arm):
        if reader == "r0" and world == "W0" and arm == "stateless" and seed == 2:
            return textbook(seed, reader, world, "raw") - 3e-9             # an improvement is still a change
        return textbook(seed, reader, world, arm)

    found = evaluator().no_harm(b_table(nudged))
    assert found["verdict"] is False and found["largest_change"] == pytest.approx(3e-9, rel=1e-3)


def test_negative_controls_need_both_scale_families_to_worsen_under_the_truth():
    found = evaluator().negative_controls(b_table(textbook), "stateful")
    assert found["verdict"] is True
    assert found["corrected_minus_raw"]["over|W0"]["estimate"] == pytest.approx(0.04)

    def under_helped(seed, reader, world, arm):
        if reader == "under" and arm == "stateful":
            return textbook(seed, reader, world, "raw") - 0.01
        return textbook(seed, reader, world, arm)

    found = evaluator().negative_controls(b_table(under_helped), "stateful")
    assert found["verdict"] is False and found["corrected_minus_raw"]["under|W0"]["worsens"] is False
    assert evaluator().negative_controls(b_table(under_helped), "stateless")["verdict"] is True     # the track asked for is the one read


def test_negative_controls_are_decided_under_the_truth_not_under_w2():
    def worsens_only_under(world_that_worsens):
        def log_loss(seed, reader, world, arm):
            if reader in ("over", "under") and arm != "raw":
                return textbook(seed, reader, world, "raw") + (0.04 if world == world_that_worsens else -0.04)
            return textbook(seed, reader, world, arm)
        return log_loss

    assert evaluator().negative_controls(b_table(worsens_only_under("W0")), "stateful")["verdict"] is True
    found = evaluator().negative_controls(b_table(worsens_only_under("W2")), "stateful")
    assert found["verdict"] is False and found["corrected_minus_raw"]["over|W2"]["worsens"] is True


def test_rules_refuse_a_table_with_a_missing_seed():
    rows = [r for r in b_table(textbook) if r["seed"] != 1]
    ev = evaluator()
    assert ev.track_selection(rows) == {"status": "incomplete", "verdict": None}
    assert ev.repair_present(rows, "stateful")["status"] == "incomplete"
    assert ev.no_harm(rows)["status"] == "incomplete"
    assert ev.negative_controls(rows, "stateless")["status"] == "incomplete"


def points_table(gain_allowed, allowed):
    rows = []
    for seed, selector, reader, world in itertools.product(SEEDS, ("rand", "ig"), READERS, ("W0", "W1", "W2")):
        share = allowed(reader, world)
        rows.append({"seed": seed, "variant": "base", "selector": selector, "reader": reader, "world_model": world, "n_points": 100,
                     "gate_allowed": share, "gate_gain": gain_allowed(seed, reader, world) if share else None, "gain": 0.01,
                     "gate_harmful": 0.0, "ungated_harmful": 0.2})
    return rows


def test_gate_valid_pools_the_gain_of_allowed_applications_per_family_and_world_model():
    ev = evaluator()
    found = ev.gate_valid(points_table(lambda seed, reader, world: 0.02, lambda reader, world: 0.5))
    assert found["verdict"] is True and len(found["cells"]) == len(R4) * 3
    assert found["cells"]["over|W1"] == pytest.approx({"allowed_share": 0.5, "mean_gain_allowed": 0.02, "mean_gain_ungated": 0.01,
                                                       "harmful_share_gated": 0.0, "harmful_share_ungated": 0.2})
    # one family loses on average where the gate lets it through: seeds at +0.01, +0.01, -0.05
    losing = lambda seed, reader, world: (-0.05 if seed == 2 else 0.01) if (reader, world) == ("under", "W1") else 0.02
    found = ev.gate_valid(points_table(losing, lambda reader, world: 0.5))
    assert found["verdict"] is False and found["cells"]["under|W1"]["mean_gain_allowed"] == pytest.approx(-0.01)
    # a gate that never opens for a cell is reported, and decides nothing
    closed = lambda reader, world: 0.0 if (reader, world) == ("under", "W1") else 0.5
    found = ev.gate_valid(points_table(losing, closed))
    assert found["verdict"] is True and found["cells"]["under|W1"]["mean_gain_allowed"] is None


def test_sandbox_cross_check_compares_with_the_reported_numbers_within_monte_carlo_error():
    def table(shift):
        rows = []
        for (variant, reader), (raw, projected, raw_acc, projected_acc) in rules.SANDBOX.items():
            for seed, wobble in zip(SEEDS, (-0.01, 0.0, 0.01)):         # per-seed means spread by 0.01: se 0.01 / sqrt(3)
                rows.append({"seed": seed, "variant": variant, "reader": reader, "n_patients": 2000, "horizon": 8,
                             "mc_log_loss_raw": raw + wobble + shift(variant, reader), "mc_log_loss_stateful": projected + wobble,
                             "mc_accuracy_raw": raw_acc + wobble, "mc_accuracy_stateful": projected_acc + wobble})
        return rows

    found = evaluator().sandbox_cross_check(table(lambda variant, reader: 0.0))
    assert found["verdict"] is True and found["label"] == "Monte Carlo" and len(found["cells"]) == 5
    assert found["cells"]["base|entrench"]["mc_log_loss_raw"] == pytest.approx({"ours": 0.761, "sandbox": 0.761, "z": 0.0, "agrees": True}, abs=1e-9)
    # standard error: ours 0.01 / sqrt(3), the sandbox's 0.01 x sqrt(2000 / 4000); a shift of 0.03 is 3.29 of them
    se = np.hypot(0.01 / np.sqrt(3), 0.01 * np.sqrt(0.5))
    off = evaluator().sandbox_cross_check(table(lambda variant, reader: 0.03 if reader == "over" else 0.0))
    assert off["cells"]["base|over"]["mc_log_loss_raw"]["z"] == pytest.approx(0.03 / se) and off["verdict"] is False
    near = evaluator().sandbox_cross_check(table(lambda variant, reader: 0.02 if reader == "over" else 0.0))
    assert near["verdict"] is True and near["cells"]["base|over"]["mc_log_loss_raw"]["agrees"] is True
    assert rules.SANDBOX[("strong_correlation", "independence")] == (1.038, 0.903, 0.756, 0.723)


def test_gate_valid_weights_each_cell_by_the_applications_it_allows():
    # under S_rand the gate opens at 90% of the points and loses 0.01 on them; under S_ig at 10% and gains 0.05:
    # 0.9 x (-0.01) + 0.1 x 0.05 = -0.004 per point over 1.0 allowed, although the two means average to +0.02
    rows = points_table(lambda seed, reader, world: 0.02, lambda reader, world: 0.5)
    for r in rows:
        if (r["reader"], r["world_model"]) == ("fixed", "W2"):
            r["gate_allowed"], r["gate_gain"] = (0.9, -0.01) if r["selector"] == "rand" else (0.1, 0.05)
    found = evaluator().gate_valid(rows)
    assert found["cells"]["fixed|W2"]["mean_gain_allowed"] == pytest.approx(-0.004)
    assert found["cells"]["fixed|W2"]["allowed_share"] == pytest.approx(0.5) and found["verdict"] is False


# ------------------------------------------------------------------------------------------- part C
def c_result(gap, gap_ci, pair_ci):
    return {"spearman_difference": {"estimate": gap, "ci": list(gap_ci)}, "matched_pair": {"estimate": float(np.mean(pair_ci)), "ci": list(pair_ci)}}


@pytest.mark.parametrize("gap,gap_ci,pair_ci,verdict,killed", [
    (0.25, (0.21, 0.29), (0.001, 0.003), True, False),
    (0.20, (0.01, 0.39), (-0.004, -0.001), True, False),          # exactly the margin; the pair differs in either direction
    (0.19, (0.15, 0.23), (0.001, 0.003), False, False),           # margin missed, pair differs: unsupported, not killed
    (0.25, (-0.01, 0.51), (0.001, 0.003), False, False),          # the interval of the difference includes 0
    (0.25, (0.21, 0.29), (-0.001, 0.003), False, False),          # margin met, pair does not differ
    (0.10, (0.05, 0.15), (-0.001, 0.003), False, True),           # both fail: the claim is killed
])
def test_e1c_supported(gap, gap_ci, pair_ci, verdict, killed):
    found = evaluator().e1c_supported(c_result(gap, gap_ci, pair_ci))
    assert found["verdict"] is verdict and found["killed"] is killed
    assert evaluator().e1c_supported(None)["status"] == "incomplete"


# ------------------------------------------------------------------------------------------- part D
def d_table(log_loss):
    """log_loss(seed, reader, world, track, scale, budget) at the final horizon."""
    arms = [("raw", "none", "none")] + [(t, "none", "none") for t in ("stateless", "stateful")]
    arms += [(t, s, b) for t in ("stateless", "stateful") for s in ("shrink-only", "two-sided")
             for b in ("exact", "exact_x0.5", "exact_x2", "exact_noise", "w1", "w2cc", "heldout")]
    rows = []
    for seed, selector, reader, world, (track, scale, budget) in itertools.product(SEEDS, ("rand", "ig"), R4, ("W0", "W2"), arms):
        for horizon, value in ((1, 7.0), (4, log_loss(seed, reader, world, track, scale, budget))):
            rows.append({"seed": seed, "variant": "base", "selector": selector, "reader": reader, "world_model": world, "track": track,
                         "scale": scale, "budget": budget, "horizon": horizon, "log_loss": value})
    return rows


STRESS = {"exact": 0.50, "exact_x0.5": 0.52, "exact_x2": 0.55, "exact_noise": 0.51}


def scale_textbook(seed, reader, world, track, scale, budget):
    """Raw 0.70, projection 0.65, scaled 0.50 with the exact budget; two-sided is hurt badly by a doubled budget."""
    if track == "raw":
        return 0.70
    if scale == "none":
        return 0.65
    if budget in STRESS:
        return STRESS[budget] + (0.15 if (scale, budget) == ("two-sided", "exact_x2") else 0.0)
    return 0.60


def test_variant_selection_compares_worst_cases_over_the_stress_set():
    found = evaluator().variant_selection(d_table(scale_textbook), "stateful")
    # shrink-only: worst of (0.50, 0.52, 0.55, 0.51) = 0.55; two-sided: its doubled budget costs 0.70
    assert found["verdict"] == "shrink-only" and found["two_sided_minus_shrink_only"]["estimate"] == pytest.approx(0.15)

    def two_sided_safer(seed, reader, world, track, scale, budget):
        if scale == "two-sided" and budget in STRESS:
            return STRESS[budget] - 0.05
        return scale_textbook(seed, reader, world, track, "shrink-only" if scale == "two-sided" else scale, budget)

    found = evaluator().variant_selection(d_table(two_sided_safer), "stateful")
    assert found["verdict"] == "two-sided" and found["two_sided_minus_shrink_only"] == pytest.approx({"estimate": -0.05, "lo": -0.05, "hi": -0.05})

    def tie(seed, reader, world, track, scale, budget):          # two-sided better in two seeds, worse in one
        if scale == "two-sided" and budget in STRESS:
            return STRESS[budget] + (0.03 if seed == 0 else -0.03)
        return scale_textbook(seed, reader, world, track, "shrink-only" if scale == "two-sided" else scale, budget)

    found = evaluator().variant_selection(d_table(tie), "stateful")
    assert found["verdict"] == "shrink-only" and found["tie"] is True

    def only_better_on_average(seed, reader, world, track, scale, budget):
        # two-sided is far better at three stress points and worse at its worst one: the rule reads the worst case
        if scale == "two-sided" and budget in STRESS:
            return 0.60 if budget == "exact_x2" else 0.10
        return scale_textbook(seed, reader, world, track, "shrink-only" if scale == "two-sided" else scale, budget)

    assert evaluator().variant_selection(d_table(only_better_on_average), "stateful")["verdict"] == "shrink-only"


def test_scale_present_is_decided_on_the_two_sided_variant_under_the_truth():
    found = evaluator().scale_present(d_table(scale_textbook), "stateful")
    assert found["verdict"] is True and found["killed"] is False
    cells = found["comparator_minus_scaled"]["two-sided"]
    assert set(cells) == {f"{family}|{c}" for family in ("over", "under", "independence") for c in ("raw", "projection")}
    assert cells["under|raw"]["estimate"] == pytest.approx(0.20) and cells["under|projection"]["estimate"] == pytest.approx(0.15)

    def under_not_repaired(seed, reader, world, track, scale, budget):
        if reader == "under" and scale != "none" and world == "W0":
            return 0.66                                             # better than raw, worse than projection alone
        return scale_textbook(seed, reader, world, track, scale, budget)

    found = evaluator().scale_present(d_table(under_not_repaired), "stateful")
    assert found["verdict"] is False and found["killed"] is False
    assert found["comparator_minus_scaled"]["two-sided"]["under|raw"]["improves"] is True
    assert found["comparator_minus_scaled"]["two-sided"]["under|projection"]["improves"] is False

    nothing = lambda seed, reader, world, track, scale, budget: 0.70 if track == "raw" else 0.65 if scale == "none" else 0.80
    found = evaluator().scale_present(d_table(nothing), "stateful")
    assert found["verdict"] is False and found["killed"] is True

    def shrink_cannot_amplify(seed, reader, world, track, scale, budget):       # shrink-only leaves the under-reactive reader at projection
        if reader == "under" and scale == "shrink-only":
            return 0.65
        return scale_textbook(seed, reader, world, track, scale, budget)

    found = evaluator().scale_present(d_table(shrink_cannot_amplify), "stateful")
    assert found["verdict"] is True                                              # decided on two-sided
    assert found["comparator_minus_scaled"]["shrink-only"]["under|projection"]["improves"] is False

    def only_shrink_works(seed, reader, world, track, scale, budget):
        if scale == "two-sided" and budget == "exact":
            return 0.69
        return scale_textbook(seed, reader, world, track, scale, budget)

    found = evaluator().scale_present(d_table(only_shrink_works), "stateful")
    assert found["verdict"] is False and found["killed"] is False               # better than raw, worse than projection


def heldout_table(cases):
    """cases: per cell, (budget, exact mean, mean on the fold kept apart). Stress budgets are x1, x0.5, x2 and x1.5 (the noise draw)."""
    rows = []
    for seed in SEEDS:
        for i, (budget, exact, check) in enumerate(cases):
            row = {"seed": seed, "variant": "base", "test": f"t{i}", "stratum": 0, "bound": budget, "budget": budget,
                   "exact_mean": exact, "mean_check": check, "eligible": budget <= check}
            for name, factor in (("x1", 1.0), ("x0.5", 0.5), ("x2", 2.0), ("noise", 1.5)):
                row[f"overestimated_{name}"] = factor * budget > exact
                row[f"admitted_{name}"] = factor * budget <= check
            rows.append(row)
        rows.append({"seed": seed, "variant": "base", "test": "empty", "stratum": 5, "bound": None, "budget": 0.0, "eligible": False})
    return rows


def test_admission_rate_counts_overestimated_budgets_that_the_eligibility_test_lets_through():
    # cell A: budget 0.10, truth 0.12, check 0.25. Stress 0.10, 0.05, 0.20, 0.15: two overestimate (0.20, 0.15), both admitted.
    # cell B: budget 0.10, truth 0.30, check 0.11. Nothing overestimates.
    found = evaluator().admission(heldout_table([(0.10, 0.12, 0.25), (0.10, 0.30, 0.11)]))
    assert found["cases"] == 3 * 8 and found["overestimated"] == 3 * 2 and found["admitted_and_overestimated"] == 3 * 2
    assert found["rate"] == pytest.approx(0.25) and found["verdict"] is False and found["rate_among_overestimated"] == 1.0
    # cell C: budget 0.10, truth 0.12, check 0.16: 0.20 is refused, 0.15 is admitted -> one of eight cases per seed... with a clean second cell
    found = evaluator().admission(heldout_table([(0.10, 0.12, 0.16), (0.10, 0.30, 0.11), (0.10, 0.30, 0.11)]))
    assert found["rate"] == pytest.approx(1 / 12) and found["verdict"] is True and found["rate_among_overestimated"] == pytest.approx(0.5)
    assert found["eligible_share"] == 1.0


def test_usable_outside_needs_the_held_out_scale_arm_to_improve_under_w2_and_a_low_admission_rate():
    clean = heldout_table([(0.10, 0.30, 0.11)])
    leaky = heldout_table([(0.10, 0.12, 0.25), (0.10, 0.30, 0.11)])

    def heldout_works(seed, reader, world, track, scale, budget):
        return 0.55 if (world, budget) == ("W2", "heldout") else scale_textbook(seed, reader, world, track, scale, budget)

    found = evaluator().usable_outside(d_table(heldout_works), clean, "stateful", "shrink-only")
    assert found["verdict"] is True and found["improves"] is True and found["admission"]["rate"] == 0.0
    assert evaluator().usable_outside(d_table(heldout_works), leaky, "stateful", "shrink-only")["verdict"] is False
    found = evaluator().usable_outside(d_table(scale_textbook), clean, "stateful", "shrink-only")      # held-out arm at 0.60 < projection 0.65
    assert found["verdict"] is True

    def heldout_fails(seed, reader, world, track, scale, budget):
        return 0.68 if (world, budget) == ("W2", "heldout") else scale_textbook(seed, reader, world, track, scale, budget)

    found = evaluator().usable_outside(d_table(heldout_fails), clean, "stateful", "shrink-only")
    assert found["verdict"] is False and found["comparator_minus_scaled"]["over|raw"]["improves"] is True
    assert found["comparator_minus_scaled"]["over|projection"]["improves"] is False


# ------------------------------------------------------------------------------------------- part A
VALUES = {"v_SRC": 0.50, "v_*RC": 0.56, "v_S*C": 0.52, "v_SR*": 0.53, "v_**C": 0.60, "v_*R*": 0.62, "v_S**": 0.56, "v_***": 0.66, "v_star": 0.66}


def a_table(values_of):
    return [{"seed": seed, "variant": "base", "lam": lam, "reader": reader, "agent": agent, **values_of(seed, reader, agent)}
            for seed in SEEDS for lam in (0.0, 1.0) for reader in R4 for agent in ("raw", "repaired")]


def test_attribution_shares_and_order_spread_by_hand():
    found = evaluator().attribution(a_table(lambda seed, reader, agent: {k: v + 0.01 * (seed - 1) for k, v in VALUES.items()}))
    entry = found["agents"]["entrench|raw"]
    assert entry["regret"] == pytest.approx(0.16)
    # declared order: selection 0.06, reading 0.04, commitment 0.06 of 0.16
    assert entry["shares"] == pytest.approx({"selection": 0.375, "reading": 0.25, "commitment": 0.375})
    # over the six orders: selection 0.06, 0.06, 0.08, 0.10, 0.09, 0.10; reading 0.04, 0.04, 0.02, 0.02, 0.04, 0.03;
    # commitment 0.06, 0.06, 0.06, 0.04, 0.03, 0.03
    assert entry["spread"] == pytest.approx({"selection": 0.25, "reading": 0.125, "commitment": 0.1875})
    assert entry["unstable_components"] == [] and found["verdict"] is True and found["killed_for"] == []
    assert entry["orders"]["reading > commitment > selection"] == pytest.approx({"reading": 0.125, "commitment": 0.25, "selection": 0.625})
    assert len(entry["orders"]) == 6 and len(found["agents"]) == 2 * len(R4)


def test_attribution_is_killed_where_the_order_spread_exceeds_the_declared_share():
    def values_of(seed, reader, agent):
        if (reader, agent) == ("over", "repaired"):
            # reading is worth 0.01 after selection is fixed, and 0.05 when it is fixed first: spread 0.25 against a share of 0.0625
            return {**VALUES, "v_**C": 0.57, "v_S*C": 0.55}
        return VALUES

    found = evaluator().attribution(a_table(values_of))
    assert found["verdict"] is False and found["killed_for"] == ["over|repaired"]
    entry = found["agents"]["over|repaired"]
    assert entry["shares"]["reading"] == pytest.approx(0.01 / 0.16) and entry["spread"]["reading"] == pytest.approx(0.04 / 0.16)
    assert "reading" in entry["unstable_components"]
    nothing = evaluator().attribution(a_table(lambda seed, reader, agent: {k: 0.66 for k in VALUES}))
    assert nothing["verdict"] is True and "shares" not in nothing["agents"]["entrench|raw"]         # no regret, nothing to attribute


# ------------------------------------------------------------------------------------------- everything
def test_evaluate_chains_the_rules_and_derives_the_layer():
    assert rules.SANDBOX == {("base" if p == 0.07 else "strong_correlation", "independence" if r == "nb" else r): v
                             for (p, r), v in __import__("test_twin_support").load_cee_check()["REFERENCE"].items()}
    table = {"B_reduced_horizon": b_table(textbook), "B_points": points_table(lambda *a: 0.02, lambda *a: 0.5),
             "D_horizon": d_table(scale_textbook), "D_heldout": heldout_table([(0.10, 0.30, 0.11)]),
             "A_values": a_table(lambda seed, reader, agent: VALUES)}
    out = rules.evaluate(table, CFG, interval=spread, c_intervals=c_result(0.25, (0.21, 0.29), (0.001, 0.003)))
    verdicts = {name: found["verdict"] for name, found in out.items() if "verdict" in found}
    assert verdicts == {"track_selection": "stateful", "repair_present": True, "negative_controls": True, "no_harm": True,
                        "gate_valid": True, "e1c_supported": True, "variant_selection": "shrink-only", "scale_present": True,
                        "usable_outside": True, "attribution": True}
    assert out["selected_layer"]["label"] == "stateful+shrink-only(heldout)"
    # the coverage caveat travels with every verdict that rests on an interval over seeds, and with no other
    carrying = {name for name, found in out.items() if "interval_caveat" in found}
    assert carrying == {"track_selection", "repair_present", "negative_controls", "variant_selection", "scale_present", "usable_outside"}
    assert all("92%" in out[name]["interval_caveat"] and "liberal" in out[name]["interval_caveat"] for name in carrying)
    out = rules.evaluate({**table, "D_heldout": heldout_table([(0.10, 0.12, 0.25)])}, CFG, interval=spread)
    assert out["usable_outside"]["verdict"] is False and out["selected_layer"]["label"] == "stateful"      # no usable scale: projection only
    assert out["e1c_supported"]["status"] == "incomplete"
    partial = rules.evaluate({"B_reduced_horizon": b_table(textbook)}, CFG, interval=spread)
    assert set(partial) == {"track_selection", "repair_present", "negative_controls", "no_harm", "e1c_supported", "selected_layer"}
    # review F003: without G2c the track is known but the layer is not decided, never a silent projection-only layer
    assert partial["selected_layer"]["status"] == "incomplete" and partial["selected_layer"]["label"] is None
    assert partial["selected_layer"]["track"] == "stateful" and partial["selected_layer"]["scale"] is None
    assert out["selected_layer"]["status"] == "evaluated"
    no_heldout = rules.evaluate({key: table[key] for key in ("B_reduced_horizon", "D_horizon")}, CFG, interval=spread)
    assert "usable_outside" not in no_heldout and no_heldout["selected_layer"]["status"] == "incomplete"


def test_part_a_refuses_a_layer_that_waits_on_g2c(tmp_path):
    from tiicee.experiments.e1 import run
    partial = rules.evaluate({"B_reduced_horizon": b_table(textbook)}, CFG, interval=spread)
    (tmp_path / "verdicts.json").write_text(__import__("json").dumps(partial))
    with pytest.raises(SystemExit, match="waits on G2c"):
        run.main(["part-a", "--config", "tiny", "--out", str(tmp_path)])


def test_default_intervals_come_from_the_statistics_module():
    stats = pytest.importorskip("tiicee.stats")
    table = b_table(lambda seed, reader, world, arm: textbook(seed, reader, world, arm) + (0.001 * seed if arm == "stateful" else 0.0))
    found = rules.Evaluator(CFG).track_selection(table)
    differences = [-0.05 + 0.001 * seed for seed in SEEDS]
    from tiicee.twin.env import stream

    expected = stats.seed_interval(differences, n_boot=CFG.n_boot, rng=stream(CFG.analysis_seed, "montecarlo", CFG.eval_split, 0))
    assert found["stateful_minus_stateless"]["estimate"] == pytest.approx(expected["estimate"])
    assert [found["stateful_minus_stateless"]["lo"], found["stateful_minus_stateless"]["hi"]] == pytest.approx(expected["ci"])
    assert found["verdict"] == "stateful"


# ------------------------------------------------------------------- incomplete tables (review F001)
INCOMPLETE = {"status": "incomplete", "verdict": None}


def test_a_seed_with_one_acquisition_rule_is_missing_not_averaged():
    # seed 1 keeps only S_ig rows: before, its value was the S_ig value alone; now the seed counts as missing
    rows = [r for r in b_table(textbook) if not (r["seed"] == 1 and r["selector"] == "rand")]
    ev = evaluator()
    assert ev.track_selection(rows) == INCOMPLETE
    assert ev.repair_present(rows, "stateful") == INCOMPLETE
    assert ev.negative_controls(rows, "stateful") == INCOMPLETE


def test_a_missing_variant_is_incomplete_not_an_error():
    assert evaluator().track_selection(b_table(textbook), variant="no_correlation") == INCOMPLETE


def test_no_harm_refuses_unpaired_rows_instead_of_raising():
    rows = b_table(textbook)
    no_raw = [r for r in rows if not (r["reader"] == "r0" and r["arm"] == "raw" and r["seed"] == 0 and r["selector"] == "ig")]
    assert evaluator().no_harm(no_raw) == INCOMPLETE
    no_stateful = [r for r in rows if not (r["reader"] == "r0" and r["arm"] == "stateful" and r["seed"] == 2)]
    assert evaluator().no_harm(no_stateful) == INCOMPLETE
    one_selector = [r for r in rows if not (r["reader"] == "r0" and r["selector"] == "rand")]
    assert evaluator().no_harm(one_selector) == INCOMPLETE


def test_gate_valid_needs_every_family_world_model_seed_and_acquisition_rule():
    good = points_table(lambda seed, reader, world: 0.02, lambda reader, world: 0.5)
    ev = evaluator()
    assert ev.gate_valid(good)["verdict"] is True
    assert ev.gate_valid([r for r in good if not (r["reader"] == "over" and r["world_model"] == "W1")]) == INCOMPLETE
    assert ev.gate_valid([r for r in good if r["reader"] != "under"]) == INCOMPLETE
    assert ev.gate_valid([r for r in good if not (r["seed"] == 0 and r["selector"] == "rand" and r["reader"] == "fixed")]) == INCOMPLETE


def test_a_gate_that_never_opens_is_vacuous_not_valid():
    found = evaluator().gate_valid(points_table(lambda seed, reader, world: 0.02, lambda reader, world: 0.0))
    assert found["status"] == "evaluated" and found["verdict"] is None and found["vacuous"] is True


def test_sandbox_cross_check_needs_every_cell():
    rows = []
    for (variant, reader), (raw, projected, raw_acc, projected_acc) in rules.SANDBOX.items():
        for seed in SEEDS:
            rows.append({"seed": seed, "variant": variant, "reader": reader, "n_patients": 2000, "horizon": 8,
                         "mc_log_loss_raw": raw, "mc_log_loss_stateful": projected, "mc_accuracy_raw": raw_acc,
                         "mc_accuracy_stateful": projected_acc})
    ev = evaluator()
    assert ev.sandbox_cross_check(rows)["status"] == "evaluated"
    assert ev.sandbox_cross_check([r for r in rows if r["reader"] != "over"]) == INCOMPLETE
    assert ev.sandbox_cross_check([r for r in rows if not (r["reader"] == "fixed" and r["seed"] == 2)]) == INCOMPLETE


def test_attribution_needs_every_family_and_both_agents():
    rows = [{"seed": seed, "variant": "base", "lam": CFG.lam, "reader": family, "agent": agent}
            for seed in SEEDS for family in R4 for agent in ("raw", "repaired")]
    missing = [r for r in rows if not (r["reader"] == "under" and r["agent"] == "repaired")]
    assert evaluator().attribution(missing) == INCOMPLETE
    short = [r for r in rows if not (r["reader"] == "over" and r["seed"] == 1)]
    assert evaluator().attribution(short) == INCOMPLETE


def test_each_rule_draws_from_its_own_fixed_block_of_bootstrap_keys():
    # review F005: one counter ran across every rule, so a rule that stopped early shifted every later rule's stream
    keys = []

    def recording(differences, key):
        keys.append(key)
        return spread(differences, key)

    table = {"B_reduced_horizon": b_table(textbook), "D_horizon": d_table(scale_textbook), "D_heldout": heldout_table([(0.10, 0.30, 0.11)])}
    rules.evaluate(table, CFG, interval=recording)
    assert sorted(keys) == list(range(28))        # a complete evaluation: every interval once, in the committed numbering
    complete = list(keys)
    keys.clear()
    broken = [r for r in table["B_reduced_horizon"] if not (r["reader"] == "over" and r["world_model"] == "W0")]
    out = rules.evaluate({**table, "B_reduced_horizon": broken}, CFG, interval=recording)
    assert out["negative_controls"]["status"] == "incomplete"
    assert keys == [k for k in complete if not rules.KEY_BLOCKS["negative_controls"] <= k < rules.KEY_BLOCKS["variant_selection"]]
