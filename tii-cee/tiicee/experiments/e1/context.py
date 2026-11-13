"""What one (seed, variant) cell of E1 needs: the twins, the fitted world models, the random order."""
from __future__ import annotations

import functools

import numpy as np

from tiicee.experiments.e1 import filters
from tiicee.experiments.e1.config import Config
from tiicee.twin import model as tm
from tiicee.twin import world_models as twin_wm
from tiicee.twin.env import Twin, stream


class Context:
    def __init__(self, cfg: Config, seed: int, variant: str, missingness: bool = True):
        """missingness=False puts every test in every chart: the setting of the sandbox, used only for its cross-check."""
        self.cfg, self.seed, self.variant = cfg, int(seed), variant
        reduced = tm.build(variant, reduced=True, missingness=missingness)
        if cfg.reduced_tests is not None:
            extra = tuple(t for t in reduced.tests if t not in tm.REDUCED_TESTS)     # the repeat of the duplicated panel
            keep = tuple(cfg.reduced_tests) + tuple(t for t in extra if t.split(" (")[0] in cfg.reduced_tests)
            reduced = tm.restrict(reduced, keep, name=f"{reduced.name}[{len(keep)} tests]")
        self.twins = {"reduced": Twin(reduced), "full": Twin(tm.build(variant, reduced=False, missingness=missingness))}

    def twin(self, size: str) -> Twin:
        return self.twins[size]

    def horizon(self, size: str) -> int:
        limit = self.cfg.horizon_reduced if size == "reduced" else self.cfg.horizon_full
        return min(limit, len(self.twin(size).model.tests))

    @functools.lru_cache(maxsize=None)
    def fitted(self, size: str, name: str):
        """W1 or W2 of this seed, fitted on the training patients of the fitting stream."""
        fit = {"W1": twin_wm.fit_w1, "W2": twin_wm.fit_w2}[name]
        return fit(self.twin(size), seed=self.seed, n_train=self.cfg.n_train)

    def world_filter(self, size: str, name: str):
        """The vectorized form of a world model: W0, W1, W2, or mirror(alpha)."""
        twin = self.twin(size)
        if name == "W0":
            return filters.LatentFilter(twin.model)
        if name == "W1":
            return filters.CountFilter(self.fitted(size, "W1"))
        if name == "W2":
            return filters.LatentFilter(self.fitted(size, "W2").twin.model)
        if name.startswith("mirror"):
            return filters.LatentFilter(twin_wm.mirror_corrupt(twin, float(name[len("mirror("):-1])).twin.model)
        raise ValueError(f"unknown world model {name!r}")

    @functools.lru_cache(maxsize=None)
    def order(self, size: str) -> np.ndarray:
        """S_rand: one random order of the tests per seed, from the behavior stream."""
        n_tests = len(self.twin(size).model.tests)
        return stream(self.seed, "behavior", self.cfg.eval_split, n_tests).permutation(n_tests)

    def noise(self, n: int, level: int) -> np.ndarray:
        """Independent lognormal budget factors for the n decision points of one level (corruption stream)."""
        return np.exp(self.cfg.noise_sigma * stream(self.seed, "corruption", self.cfg.eval_split, level).normal(size=n))
