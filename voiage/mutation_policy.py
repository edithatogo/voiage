"""Fail-closed mutation-score policy for Mutmut CI/CD statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

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

    def report(
        self, threshold: float, *, baseline: MutationScore | None = None
    ) -> dict[str, object]:
        """Return a JSON-safe threshold report."""
        non_decreasing = baseline is None or (
            self.eligible > 0
            and baseline.eligible > 0
            and self.killed * baseline.eligible >= baseline.killed * self.eligible
        )
        unresolved = self.eligible - self.killed
        baseline_unresolved = (
            None if baseline is None else baseline.eligible - baseline.killed
        )
        debt_non_increasing = baseline is None or (
            baseline_unresolved is not None and unresolved <= baseline_unresolved
        )
        passed = all(
            (
                self.interrupted == 0,
                self.eligible > 0,
                self.percent >= threshold,
                non_decreasing,
                debt_non_increasing,
            )
        )
        report: dict[str, object] = {
            **asdict(self),
            "eligible": self.eligible,
            "unresolved": unresolved,
            "score_percent": round(self.percent, 3),
            "threshold_percent": threshold,
            "non_decreasing": non_decreasing,
            "debt_non_increasing": debt_non_increasing,
            "passed": passed,
        }
        if baseline is not None:
            report.update(
                baseline_killed=baseline.killed,
                baseline_eligible=baseline.eligible,
                baseline_unresolved=baseline_unresolved,
                baseline_score_percent=round(baseline.percent, 3),
            )
        return report


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
