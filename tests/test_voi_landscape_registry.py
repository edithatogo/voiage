"""Contracts for the reproducible VOI/VOP software and method census."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import subprocess
import sys

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).parents[1]
LANDSCAPE = ROOT / "specs" / "software-landscape"
REGISTRY = LANDSCAPE / "registry.json"
SCHEMA = LANDSCAPE / "schema.json"
METHODS = LANDSCAPE / "methods.json"
METHOD_EVIDENCE = LANDSCAPE / "method-evidence.json"
METHOD_EVIDENCE_SCHEMA = LANDSCAPE / "method-evidence.schema.json"
ADJACENT_METHODS = LANDSCAPE / "adjacent-method-dispositions.json"
ADJACENT_METHODS_SCHEMA = LANDSCAPE / "adjacent-method-dispositions.schema.json"
GAP_REPORT = LANDSCAPE / "gap-report.json"
PARITY_FIXTURES = LANDSCAPE / "parity-fixtures.json"
IMPLEMENTATION_EVIDENCE = LANDSCAPE / "implementation-evidence.json"
UPSTREAM_EVIDENCE = LANDSCAPE / "upstream-feature-evidence.json"
UPSTREAM_EVIDENCE_SCHEMA = LANDSCAPE / "upstream-feature-evidence.schema.json"
DECISION_PROBLEM_SCHEMA = (
    ROOT / "specs" / "core-api" / "schemas" / "v2" / "decision-problem.schema.json"
)
DECISION_PROBLEM_EXAMPLE = (
    ROOT / "specs" / "core-api" / "examples" / "v2" / "decision-problem.example.json"
)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def test_software_landscape_validates_against_schema() -> None:
    """The checked-in census must obey its versioned public contract."""
    registry = _read_json(REGISTRY)
    schema = _read_json(SCHEMA)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(registry)


def test_required_ecosystems_and_seed_tools_are_present() -> None:
    """The initial census must span every requested discovery channel."""
    registry = _read_json(REGISTRY)
    assert isinstance(registry, dict)
    tools = registry["tools"]
    ecosystems = {tool["ecosystem"] for tool in tools}
    ids = {tool["id"] for tool in tools}

    assert {
        "r",
        "python",
        "rust",
        "julia",
        "mojo",
        "web",
        "commercial",
    } <= ecosystems
    assert {
        "r-voi",
        "r-bcea",
        "r-dampack",
        "savi",
        "analytica",
        "pyro-oed",
        "botorch",
        "decision-security",
        "r-surveyvoi",
        "r-predtools",
        "r-metanb",
        "gaussian-voi-supplement",
        "bayescal-voi",
        "metavoi",
        "nrel-geothermal-voi",
    } <= ids
    assert len(ids) == len(tools)


def test_method_and_feature_references_are_complete() -> None:
    """Every observed feature must map to a canonical method or disposition."""
    registry = _read_json(REGISTRY)
    methods = _read_json(METHODS)
    assert isinstance(registry, dict)
    assert isinstance(methods, dict)
    method_ids = {method["id"] for method in methods["methods"]}
    assert len(method_ids) == len(methods["methods"])

    for tool in registry["tools"]:
        assert tool["features"], tool["id"]
        for feature in tool["features"]:
            assert set(feature["method_ids"]) <= method_ids
            assert feature["parity_state"] in {
                "native",
                "equivalent",
                "adapter",
                "planned",
                "excluded",
                "not-reproducible",
            }
            assert feature["evidence"]


def test_method_taxonomy_covers_core_vop_and_ml_families() -> None:
    """Core decision VOI, perspective, and information design stay distinct."""
    methods = _read_json(METHODS)
    assert isinstance(methods, dict)
    by_id = {method["id"]: method for method in methods["methods"]}

    assert {
        "evpi",
        "evppi",
        "evsi",
        "enbs",
        "directional-evop",
        "perspective-frontier",
        "perspective-sample-information",
        "expected-information-gain",
        "robust-expected-information-gain",
        "bayesian-oed",
        "active-learning",
        "knowledge-gradient",
        "llm-routing-voi",
        "rag-acquisition-voi",
        "agent-information-voi",
    } <= by_id.keys()
    for method in by_id.values():
        assert method["class"] in {
            "estimand",
            "estimator",
            "workflow",
            "visualization",
            "related-analysis",
            "application",
        }
        assert method["moscow"] in {"must", "should", "could", "wont"}
        assert method["voiage_state"] in {
            "native",
            "equivalent",
            "planned",
            "excluded",
        }
        assert method["maturity"] in {
            "stable",
            "experimental",
            "planned",
        }

    assert by_id["expected-information-gain"]["class"] == "estimand"
    assert by_id["llm-routing-voi"]["class"] == "application"


def test_every_method_has_reviewed_source_coverage() -> None:
    """No method may enter the taxonomy without an auditable evidence state."""
    methods = _read_json(METHODS)
    evidence = _read_json(METHOD_EVIDENCE)
    schema = _read_json(METHOD_EVIDENCE_SCHEMA)
    assert isinstance(methods, dict)
    assert isinstance(evidence, dict)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(evidence)

    method_ids = {method["id"] for method in methods["methods"]}
    source_ids = {source["id"] for source in evidence["sources"]}
    assert len(source_ids) == len(evidence["sources"])
    for source in evidence["sources"]:
        identifier = source["identifier"]
        if identifier.startswith("doi:"):
            doi = identifier.removeprefix("doi:")
            assert source["url"] == f"https://doi.org/{doi}"
            assert "/" in doi
            assert " " not in doi
        elif source["kind"] == "repository-contract":
            assert identifier.startswith("repo:")
        else:
            assert identifier.startswith("url:")

    covered: set[str] = set()
    for method_evidence in evidence["coverage"]:
        assert method_evidence["method_id"] in method_ids
        assert method_evidence["method_id"] not in covered
        assert set(method_evidence["source_ids"]) <= source_ids
        if method_evidence["disposition"] in {
            "canonical-estimand",
            "canonical-estimator",
        }:
            assert {"uncertainty", "objective", "provenance"} <= set(
                method_evidence["required_decision_fields"]
            )
        if method_evidence["family"] == "value-of-perspective":
            assert "perspectives" in method_evidence["required_decision_fields"]
        covered.add(method_evidence["method_id"])

    assert covered == method_ids
    assert not {
        item["method_id"]
        for item in evidence["coverage"]
        if item["review_state"] == "triage-required"
    }


def test_adjacent_methods_have_explicit_non_duplicative_dispositions() -> None:
    """Named neighboring methods must be mapped, retained, added, or excluded."""
    methods = _read_json(METHODS)
    adjacent = _read_json(ADJACENT_METHODS)
    schema = _read_json(ADJACENT_METHODS_SCHEMA)
    assert isinstance(methods, dict)
    assert isinstance(adjacent, dict)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(adjacent)
    method_ids = {method["id"] for method in methods["methods"]}
    record_ids = {record["id"] for record in adjacent["records"]}
    assert len(record_ids) == len(adjacent["records"])
    assert {
        "buying-price-voi",
        "constructed-scale-voi",
        "robust-expected-information-gain",
        "validation-study-evsi",
        "blackwell-informativeness",
        "value-of-signals",
        "value-of-clairvoyance",
        "value-of-control",
        "value-of-flexibility",
        "rational-inattention",
        "bayesian-persuasion",
        "strategic-information-sharing",
        "causal-discovery-design",
        "model-discrimination-design",
        "measurement-test-accuracy-voi",
    } <= record_ids
    for record in adjacent["records"]:
        assert set(record["canonical_method_ids"]) <= method_ids
        if record["disposition"] == "exclude-from-voi-core":
            assert not record["canonical_method_ids"]


def test_search_snapshot_is_bounded_and_refreshable() -> None:
    """The snapshot records queries and expires before becoming silent lore."""
    registry = _read_json(REGISTRY)
    assert isinstance(registry, dict)
    searched_on = date.fromisoformat(registry["searched_on"])
    review_due = date.fromisoformat(registry["review_due"])

    assert searched_on < review_due
    assert (review_due - searched_on).days <= 93
    assert len(registry["searches"]) >= 9
    for search in registry["searches"]:
        assert search["channel"]
        assert search["query"]
        assert search["source_url"].startswith("https://")
        assert search["result"] in {"candidates-found", "no-direct-package-found"}


def test_landscape_freshness_validator_has_deterministic_boundary() -> None:
    """Scheduled enforcement must pass on the deadline and fail the next day."""
    validator = ROOT / "scripts" / "validate_voi_landscape_freshness.py"
    on_deadline = subprocess.run(
        [sys.executable, str(validator), "--as-of", "2026-10-23"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    overdue = subprocess.run(
        [sys.executable, str(validator), "--as-of", "2026-10-24"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert on_deadline.returncode == 0
    assert overdue.returncode == 1
    assert "review overdue" in overdue.stdout


def test_generated_feature_matrix_is_current() -> None:
    """The human matrix must be a deterministic projection of the registries."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_voi_feature_matrix.py"),
            "--check",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_generated_method_evidence_registry_is_current() -> None:
    """Method-level dispositions must be a deterministic reviewed projection."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_method_evidence_registry.py"),
            "--check",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_generated_method_implementation_evidence_is_current() -> None:
    """Every native claim must resolve to implementation and executable tests."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_method_implementation_evidence.py"),
            "--check",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    evidence = _read_json(IMPLEMENTATION_EVIDENCE)
    assert isinstance(evidence, dict)
    assert all(record["implementation_paths"] for record in evidence["records"])
    assert all(record["test_paths"] for record in evidence["records"])


def test_generated_upstream_feature_evidence_is_current_and_complete() -> None:
    """Every external feature must expose what upstream artifacts were reviewed."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_upstream_feature_evidence.py"),
            "--check",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    registry = _read_json(REGISTRY)
    evidence = _read_json(UPSTREAM_EVIDENCE)
    schema = _read_json(UPSTREAM_EVIDENCE_SCHEMA)
    assert isinstance(registry, dict)
    assert isinstance(evidence, dict)
    assert isinstance(schema, dict)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(evidence)

    expected = {
        (tool["id"], feature["id"])
        for tool in registry["tools"]
        if tool["scope"] == "external"
        for feature in tool["features"]
    }
    actual = {
        (record["tool_id"], record["feature_id"]) for record in evidence["records"]
    }
    assert actual == expected
    assert len(actual) == len(evidence["records"])
    for record in evidence["records"]:
        assert record["documentation_artifacts"]
        if (
            not record["source_artifacts"]
            or not record["test_artifacts"]
            or not record["example_artifacts"]
            or not record["schema_artifacts"]
        ):
            assert record["limitations"] != "None recorded."
        if record["source_artifacts"]:
            assert record["reviewed_revision"]
            assert all(
                record["reviewed_revision"] in artifact
                for artifact in record["source_artifacts"]
            )


def test_generated_gap_report_is_current_and_routed() -> None:
    """Every non-equivalent external feature must remain visible and owned."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_voi_gap_report.py"),
            "--check",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    report = _read_json(GAP_REPORT)
    assert isinstance(report, dict)
    assert report["summary"]["open_feature_gaps"] == len(report["feature_gaps"])
    assert report["summary"]["open_method_or_assurance_gaps"] == len(
        report["method_gaps"]
    )
    assert all(item["owner_track"] for item in report["method_gaps"])
    assert "expected-loss" not in {item["method_id"] for item in report["method_gaps"]}


def test_native_or_equivalent_external_claims_have_independent_fixtures() -> None:
    """No positive parity claim may rely only on competitor execution."""
    registry = _read_json(REGISTRY)
    assurance = _read_json(PARITY_FIXTURES)
    assert isinstance(registry, dict)
    assert isinstance(assurance, dict)

    positive_claims = {
        (tool["id"], feature["id"])
        for tool in registry["tools"]
        if tool["scope"] == "external"
        for feature in tool["features"]
        if feature["parity_state"] in {"native", "equivalent"}
    }
    records = {
        (record["tool_id"], record["feature_id"]): record
        for record in assurance["records"]
    }
    assert set(records) == positive_claims
    for record in records.values():
        assert record["assurance_state"] in {
            "independent-fixtures",
            "analytical-equivalence",
        }
        for relative_path in record["fixture_paths"] + record["test_paths"]:
            assert (ROOT / relative_path).is_file(), relative_path


def test_decision_problem_v2_is_backend_neutral_and_valid() -> None:
    """The frozen interchange model must represent decisions, information and value."""
    schema = _read_json(DECISION_PROBLEM_SCHEMA)
    example = _read_json(DECISION_PROBLEM_EXAMPLE)
    assert isinstance(schema, dict)
    assert isinstance(example, dict)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=None).validate(example)

    assert len(example["alternatives"]) >= 2
    assert example["uncertainty"]["state_variables"]
    assert example["information_actions"]
    assert example["objective"]["direction"] in {"maximize", "minimize"}
    assert example["perspectives"]
    assert example["population"]["scope"]
    assert example["time_horizon"]["value"] > 0
    assert example["provenance"]["input_artifact_ids"]


def test_decision_problem_v2_rejects_implicit_information_cost() -> None:
    """Every information action must carry an explicit cost, including no action."""
    schema = _read_json(DECISION_PROBLEM_SCHEMA)
    example = _read_json(DECISION_PROBLEM_EXAMPLE)
    assert isinstance(schema, dict)
    assert isinstance(example, dict)
    del example["information_actions"][0]["cost"]

    errors = list(Draft202012Validator(schema).iter_errors(example))
    assert any(error.validator == "required" for error in errors)
