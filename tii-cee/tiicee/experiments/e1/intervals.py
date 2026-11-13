"""Every interval and rank statistic of E1, taken from tiicee.stats and from nowhere else.

The bootstrap generator of each interval is a stream of the analysis seed, numbered by the caller,
so a verdict is reproducible from the per-seed tables alone.
"""
from __future__ import annotations

from tiicee.experiments.e1.config import Config
from tiicee.twin.env import stream


def _rng(cfg: Config, key: int):
    return stream(cfg.analysis_seed, "montecarlo", cfg.eval_split, key)


def spearman(x, y) -> float:
    from tiicee import stats

    return stats.spearman(x, y)


def seed_interval(differences, cfg: Config, key: int) -> dict:
    """Mean of the paired per-seed differences with its 95% bootstrap interval over seeds: {estimate, lo, hi, n}."""
    from tiicee import stats

    out = stats.seed_interval(list(differences), n_boot=cfg.n_boot, rng=_rng(cfg, key))
    return {"estimate": out["estimate"], "lo": out["ci"][0], "hi": out["ci"][1], "n": out["n_seeds"]}


def spearman_difference(target, x_a, x_b, units, within, cfg: Config, key: int) -> dict:
    """rho(target, x_a) - rho(target, x_b) with its 95% interval, resampling units within strata."""
    from tiicee import stats

    return stats.spearman_difference_interval(target, x_a, x_b, units, n_boot=cfg.n_boot, rng=_rng(cfg, key), within=within)


def paired(values_a, values_b, units, within, cfg: Config, key: int) -> dict:
    """Mean paired difference a - b over units with its 95% interval, resampling units within strata."""
    from tiicee import stats

    return stats.paired_interval(values_a, values_b, units, n_boot=cfg.n_boot, rng=_rng(cfg, key), within=within)
