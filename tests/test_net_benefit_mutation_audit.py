"""Contract tests for the bounded Rust net-benefit mutation audit."""

from pathlib import Path

from scripts.run_net_benefit_mutation_audit import MUTANTS

ROOT = Path(__file__).resolve().parents[1]


def test_mutation_audit_covers_critical_formula_and_shape_semantics() -> None:
    assert set(MUTANTS) == {
        "subtract-cost-to-add-cost",
        "finite-result-guard-removed",
        "scalar-threshold-zeroed",
        "sample-threshold-ownership-removed",
    }
    assert len({mutation[0] for mutation in MUTANTS.values()}) == len(MUTANTS)


def test_ci_requires_and_archives_the_net_benefit_mutation_audit() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "id: net-benefit-mutation" in workflow
    assert "NET_BENEFIT_OUTCOME" in workflow
    assert "mutation-net-benefit.json" in workflow
