"""Synthetic CDM environment (always-available tests, Stage 1)."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .actions import (
    A_DIAG,
    A_TEST,
    A_TEST_SET,
    COST_NORM,
    COST_USD,
    is_diagnosis,
    is_test,
    is_valid_action,
)
from .generators import _stable_noise_seed, generate, render_finding
from .profiles import (
    AVAILABILITY,
    LIKELIHOOD,
    P_ATYPICAL,
    PRIOR_MIMIC_CDM,
    V1_MAX_STEPS,
    phenotype_given_y,
    sample_categorical,
)

# Reward shaping (sensible LA-CDM-like baseline; not paper-exact GRPO scale)
R_CORRECT = 1.0
R_INCORRECT = -1.0
R_INVALID = -0.5
# Per-test cost penalty uses normalized costs (sum to 1 over full inventory)
COST_SCALE = 1.0


class SyntheticCDMEnv:
    """POMDP-style synthetic abdominal diagnosis env.

    Latent y ~ categorical over 4 diagnoses. Every test always returns
    observation text via g_a(y, eps). Diagnosis actions terminate the episode.

    ``version="v1"`` (see ``profiles.py``) instead samples y from the
    MIMIC-CDM class prior, a presenting phenotype e, and one discrete finding
    per test from a test-specific likelihood, so tests differ in how much
    they reveal. ``step`` then reports the finding in ``info["finding"]``.
    With ``missingness=True`` each test is in the chart with probability
    ``AVAILABILITY[test]``; requesting a missing one returns UNAVAILABLE,
    costs nothing and burns a step, as in LA-CDM (F-2, F-3). v1 episodes end
    undiagnosed after ``max_steps`` actions (default 14, F-3).
    """

    def __init__(
        self,
        seed: int = 0,
        split: str = "train",
        max_tests: int = 12,
        cost_scale: float = COST_SCALE,
        version: str = "v0",
        missingness: bool = False,
        p_atypical: float = P_ATYPICAL,
        max_steps: Optional[int] = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train|val|test, got {split!r}")
        if version not in ("v0", "v1"):
            raise ValueError(f"version must be v0|v1, got {version!r}")
        if missingness and version != "v1":
            raise ValueError("missingness requires version='v1'")
        self.version = version
        self.missingness = bool(missingness)
        self.p_atypical = float(p_atypical)
        if max_steps is None and version == "v1":
            max_steps = V1_MAX_STEPS
        self.max_steps = max_steps
        self._phenotype_given_y = phenotype_given_y(self.p_atypical)
        self.seed = int(seed)
        self.split = split
        self.max_tests = int(max_tests)
        self.cost_scale = float(cost_scale)

        self._rng = random.Random(self.seed)
        self._episode_rng: Optional[random.Random] = None
        self._episode_id: Optional[str] = None
        self._episode_counter = 0

        self.y_true: Optional[str] = None
        self.patient_seed: Optional[int] = None
        self._eps_seed: Optional[int] = None
        self._history_lines: List[str] = []
        self._tests_taken: List[str] = []
        self._unavailable_requested: List[str] = []
        self.phenotype: Optional[str] = None
        self._in_chart: frozenset = frozenset(A_TEST)
        self._t: int = 0
        self._done: bool = False
        self._cumulative_cost_norm: float = 0.0

    # ------------------------------------------------------------------ API

    def reset(self, *, y: Optional[str] = None) -> str:
        """Sample latent disease (or fix y) and return initial history string."""
        self._episode_counter += 1
        self._episode_rng = random.Random(self._rng.randint(0, 2**31 - 1))
        self.patient_seed = self._episode_rng.randint(0, 2**31 - 1)
        self._eps_seed = self.patient_seed
        if y is None:
            if self.version == "v1":
                self.y_true = sample_categorical(PRIOR_MIMIC_CDM, self._episode_rng)
            else:
                self.y_true = self._episode_rng.choice(list(A_DIAG))
        else:
            if y not in A_DIAG:
                raise ValueError(f"Unknown y: {y!r}")
            self.y_true = y
        if self.version == "v1":
            # Drawn whether or not y was fixed, so the rng stream (and thus
            # the patient) does not depend on how reset was called.
            self.phenotype = sample_categorical(
                self._phenotype_given_y[self.y_true], self._episode_rng
            )
            chart_draws = {a: self._episode_rng.random() for a in A_TEST}
            if self.missingness:
                self._in_chart = frozenset(
                    a for a in A_TEST if chart_draws[a] < AVAILABILITY[a]
                )
            else:
                self._in_chart = frozenset(A_TEST)

        self._episode_id = (
            f"syn-{self.split}-s{self.seed}-e{self._episode_counter:04d}"
            f"-p{self.patient_seed}"
        )
        self._history_lines = [
            f"Patient encounter (synthetic id={self._episode_id}, "
            f"seed={self.patient_seed}, split={self.split}). "
            "Chief complaint: acute abdominal pain. "
            "Differential under consideration: appendicitis, cholecystitis, "
            "diverticulitis, pancreatitis."
        ]
        self._tests_taken = []
        self._unavailable_requested = []
        self._t = 0
        self._done = False
        self._cumulative_cost_norm = 0.0
        return self.history

    @property
    def history(self) -> str:
        return "\n".join(self._history_lines)

    @property
    def episode_id(self) -> Optional[str]:
        return self._episode_id

    @property
    def t(self) -> int:
        return self._t

    @property
    def done(self) -> bool:
        return self._done

    @property
    def tests_taken(self) -> List[str]:
        return list(self._tests_taken)

    @property
    def available_tests(self) -> List[str]:
        """Tests not yet requested (the agent does not see chart contents)."""
        requested = set(self._tests_taken) | set(self._unavailable_requested)
        return [a for a in A_TEST if a not in requested]

    @property
    def unavailable_requested(self) -> List[str]:
        return list(self._unavailable_requested)

    def step(
        self, action: str
    ) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Apply test or diagnosis action.

        Returns (observation_or_terminal_msg, reward, done, info).
        """
        obs, reward, done, info = self._step(action)
        if (
            not done
            and self.max_steps is not None
            and self._t >= self.max_steps
        ):
            # Budget exhausted without a diagnosis: LA-CDM's invalid outcome.
            self._done = done = True
            reward += R_INVALID
            info.update(
                {
                    "undiagnosed": True,
                    "y_true": self.y_true,
                    "correct": False,
                    "cumulative_cost_norm": self._cumulative_cost_norm,
                    "n_tests": len(self._tests_taken),
                }
            )
        return obs, reward, done, info

    def _step(
        self, action: str
    ) -> Tuple[str, float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode is done; call reset().")
        if self.y_true is None or self._eps_seed is None:
            raise RuntimeError("Call reset() before step().")

        info: Dict[str, Any] = {
            "episode_id": self._episode_id,
            "t": self._t,
            "action": action,
            "split": self.split,
            "available_tests_at_t": self.available_tests,
            "seed": self.seed,
            "patient_seed": self.patient_seed,
        }

        if not is_valid_action(action):
            obs = f"INVALID_ACTION: {action!r} is not a valid test or diagnosis."
            reward = R_INVALID
            self._append_obs(action, obs)
            self._t += 1
            info["invalid"] = True
            return obs, reward, False, info

        if is_test(action):
            if action in self._tests_taken or action in self._unavailable_requested:
                obs = (
                    f"[{action}] Already obtained earlier in this encounter; "
                    "repeating the same order yields no new information."
                )
                reward = -0.05  # small waste penalty
                self._append_obs(action, obs)
                self._t += 1
                info["duplicate_test"] = True
                return obs, reward, False, info

            if action not in self._in_chart:
                obs = (
                    f"[{action}] UNAVAILABLE: this test was not performed "
                    "for this patient."
                )
                self._unavailable_requested.append(action)
                self._append_obs(action, obs)
                self._t += 1
                # Free, and no information (F-2): cost counts provided tests only.
                info.update({"unavailable": True, "finding": None})
                return obs, 0.0, False, info

            if self.version == "v1":
                finding, obs = self._v1_observe(action)
                info["finding"] = finding
            else:
                obs = generate(action, self.y_true, self._eps_seed)
            cost_n = COST_NORM[action]
            cost_usd = COST_USD[action]
            reward = -self.cost_scale * cost_n
            self._cumulative_cost_norm += cost_n
            self._tests_taken.append(action)
            self._append_obs(action, obs)
            self._t += 1
            info.update(
                {
                    "cost_norm": cost_n,
                    "cost_usd": cost_usd,
                    "cumulative_cost_norm": self._cumulative_cost_norm,
                }
            )
            # Auto-force diagnosis opportunity if max tests hit — episode
            # continues until diagnose; we only flag the limit.
            if len(self._tests_taken) >= self.max_tests:
                info["max_tests_reached"] = True
            return obs, reward, False, info

        # Diagnosis action
        assert is_diagnosis(action)
        correct = action == self.y_true
        reward = R_CORRECT if correct else R_INCORRECT
        # mild cost of delay already paid via tests; optional unused-test
        # credit is omitted for simplicity
        obs = (
            f"DIAGNOSIS_COMMITTED: predicted={action}, "
            f"{'CORRECT' if correct else 'INCORRECT'}."
        )
        self._append_obs(action, obs)
        self._t += 1
        self._done = True
        info.update(
            {
                "y_true": self.y_true,
                "correct": correct,
                "cumulative_cost_norm": self._cumulative_cost_norm,
                "n_tests": len(self._tests_taken),
            }
        )
        return obs, reward, True, info

    def oracle_observe(self, test_name: str) -> str:
        """Counterfactual observation without advancing the episode.

        Uses the same g_a(y, eps) as step would for this patient/test.
        Does not require the test to be unused (true counterfactual).
        """
        if self.y_true is None or self._eps_seed is None:
            raise RuntimeError("Call reset() before oracle_observe().")
        if test_name not in A_TEST_SET:
            raise ValueError(f"Not a test action: {test_name!r}")
        if self.version == "v1":
            return self._v1_observe(test_name)[1]
        return generate(test_name, self.y_true, self._eps_seed)

    def oracle_finding(self, test_name: str) -> str:
        """v1 counterfactual finding, even for tests missing from the chart."""
        if self.version != "v1":
            raise RuntimeError("oracle_finding requires version='v1'")
        if self.y_true is None or self._eps_seed is None:
            raise RuntimeError("Call reset() before oracle_finding().")
        if test_name not in A_TEST_SET:
            raise ValueError(f"Not a test action: {test_name!r}")
        return self._v1_observe(test_name)[0]

    # --------------------------------------------------------------- helpers

    def _v1_observe(self, test_name: str) -> Tuple[str, str]:
        """Deterministic (finding, text) for this patient and test."""
        assert self.phenotype is not None and self._eps_seed is not None
        rng = random.Random(
            _stable_noise_seed(self.phenotype, test_name, self._eps_seed)
        )
        finding = sample_categorical(LIKELIHOOD[test_name][self.phenotype], rng)
        return finding, render_finding(test_name, finding, rng)

    def _append_obs(self, action: str, obs: str) -> None:
        self._history_lines.append(f"Action: {action}")
        self._history_lines.append(f"Observation: {obs}")
