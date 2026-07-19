"""Fail-closed mutation-score policy for Mutmut CI/CD statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class MutationScore:
    """Validated Mutmut counts and their enforceable score."""

    killed: int
    survived: int
    no_tests: int
    suspicious: int
    timeout: int
    segfault: int
    skipped: int
    interrupted: int
    total: int

    @property
    def eligible(self) -> int:
        """Return every non-skipped mutant, including omitted status buckets."""
        return self.total - self.skipped

    @property
    def percent(self) -> float:
        """Return the killed percentage over eligible mutants."""
        return 100.0 * self.killed / self.eligible if self.eligible else 0.0

    def report(self, threshold: float) -> dict[str, Any]:
        """Return a JSON-safe threshold report."""
        passed = (
            self.interrupted == 0 and self.eligible > 0 and self.percent >= threshold
        )
        return {
            **asdict(self),
            "eligible": self.eligible,
            "score_percent": round(self.percent, 3),
            "threshold_percent": threshold,
            "passed": passed,
        }


_FIELDS = {
    "killed": "killed",
    "survived": "survived",
    "no_tests": "no_tests",
    "suspicious": "suspicious",
    "timeout": "timeout",
    "segfault": "segfault",
    "skipped": "skipped",
    "interrupted": "check_was_interrupted_by_user",
    "total": "total",
}


def mutation_score_from_mapping(raw: Mapping[str, object]) -> MutationScore:
    """Validate Mutmut 3.6 ``export-cicd-stats`` JSON."""
    values: dict[str, int] = {}
    for field, source in _FIELDS.items():
        value = raw.get(source)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"mutation statistic {source} must be a non-negative integer"
            )
        values[field] = value
    score = MutationScore(**values)
    accounted = (
        score.killed
        + score.survived
        + score.no_tests
        + score.suspicious
        + score.timeout
        + score.segfault
        + score.skipped
        + score.interrupted
    )
    if score.total < accounted:
        raise ValueError("mutation total is smaller than its reported status counts")
    return score


def validate_threshold(threshold: float) -> float:
    """Require a meaningful percentage threshold."""
    if not 0.0 < threshold <= 100.0:
        raise ValueError("mutation threshold must be greater than 0 and at most 100")
    return threshold


__all__ = ["MutationScore", "mutation_score_from_mapping", "validate_threshold"]
