"""Canonical actions and proposal classification with upstream's own parser (SPEC §3.1, §3.4)."""
from dataclasses import dataclass
import re

# Upstream's pattern, verbatim (src/environment/environment.py at 3f435a10).
PATTERN = r"Thought: (.*)\n{1,2}Action: (.*)\n{1,2}Action Input: (.*)"


@dataclass(frozen=True)
class Action:
    kind: str  # "test" | "diagnosis"
    name: str  # a test exactly as the prompt writes it, or a lowercase disease

    def completion(self, thought: str) -> str:
        """The text vLLM would return: a test's stop string is excluded, a diagnosis ends at EOS."""
        if self.kind == "test":
            return f"Thought: {thought}\nAction: Test\nAction Input: {self.name}\n"
        return f"Thought: {thought}\nAction: Diagnosis\nAction Input: {self.name}"

    def continuation(self, eos_token: str) -> str:
        """The scored continuation after "Action:", terminator included."""
        if self.kind == "test":
            return f" Test\nAction Input: {self.name}\nObservation:"
        return f" Diagnosis\nAction Input: {self.name}{eos_token}"


@dataclass(frozen=True)
class Proposal:
    kind: str  # an ACTION_KINDS member other than "overflow"
    thought: str | None
    action: Action | None


class Vocabulary:
    """Configured tests (configured order) and diseases; ties resolve in this order."""

    def __init__(self, tests, diseases):
        self.tests = [Action("test", name) for name in tests]
        self.diseases = [Action("diagnosis", name.lower()) for name in diseases]

    def untried(self, requested) -> list[Action]:
        return [a for a in self.tests if a.name.lower() not in requested]

    def classify(self, completion: str, requested) -> Proposal:
        match = re.search(PATTERN, completion)
        if not match:
            return Proposal("malformed", None, None)
        thought, action, value = match.group(1), match.group(2).lower(), match.group(3).lower()
        if action == "test":
            found = next((a for a in self.tests if a.name.lower() == value), None)
            if found is None:
                return Proposal("unknown_test", thought, None)
            if value in requested:
                return Proposal("repeated_test", thought, None)
            return Proposal("test", thought, found)
        if action == "diagnosis":
            found = next((a for a in self.diseases if a.name == value), None)
            return Proposal("diagnosis", thought, found) if found else Proposal("unknown_diagnosis", thought, None)
        return Proposal("unknown_action", thought, None)
