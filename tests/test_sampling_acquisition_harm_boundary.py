"""Fail-closed contract tests for sampling-acquisition-harm research scope."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker, ValidationError
import pytest

import voiage
import voiage.experimental
import voiage.methods

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/sampling-acquisition-harm/v1"
PROHIBITED_SYMBOLS = {
    "SamplingAcquisitionHarmResult",
    "compute_sampling_acquisition_harm",
    "sampling_acquisition_harm_voi",
    "sampling_harm_voi",
    "value_of_sampling_acquisition_harm",
    "voiage_v1_sampling_acquisition_harm",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_fail_closed_capability_and_research_disposition_validate() -> None:
    schemas = CONTRACT / "schemas"
    pairs = (
        (schemas / "capability.schema.json", CONTRACT / "capabilities.json"),
        (
            schemas / "research-disposition.schema.json",
            CONTRACT / "research-disposition.json",
        ),
        (
            schemas / "scope-selection.schema.json",
            CONTRACT / "scope-selection.json",
        ),
        (
            schemas / "governance-snapshot.schema.json",
            CONTRACT / "governance-snapshot.json",
        ),
        (
            schemas / "review-candidate.schema.json",
            CONTRACT / "review-candidate.json",
        ),
        (
            schemas / "estimand-boundary.schema.json",
            CONTRACT / "estimand-boundary.json",
        ),
        (
            schemas / "prior-findings.schema.json",
            CONTRACT / "prior-findings.json",
        ),
        (
            schemas / "source-and-retrieval-register.schema.json",
            CONTRACT / "source-and-retrieval-register.json",
        ),
        (
            schemas / "review-preparation.schema.json",
            CONTRACT / "review-preparation.json",
        ),
    )
    for schema_path, artifact_path in pairs:
        schema = _json(schema_path)
        artifact = _json(artifact_path)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(artifact)


def test_discovery_is_explicitly_unsupported_on_every_execution_surface() -> None:
    capability = _json(CONTRACT / "capabilities.json")

    assert capability["maturity"] == "unsupported_research_scoping"
    assert capability["runtime_available"] is False
    assert capability["stable_claim_allowed"] is False
    assert capability["discovery"] == {
        "status": "unsupported",
        "executable": False,
        "callable": None,
        "reason": "No candidate-bound scientific approval or executable contract exists.",
    }
    for surface in (
        "stable_python_api",
        "experimental_python",
        "rust_kernel",
        "native_abi",
        "r",
        "julia",
    ):
        assert capability["surfaces"][surface]["status"] == "unsupported"
        assert capability["surfaces"][surface]["symbols"] == []
    assert capability["surfaces"]["mojo"]["status"] == "external_boundary"
    assert capability["surfaces"]["mojo"]["symbols"] == []


def test_research_disposition_prohibits_runtime_and_adjacent_method_aliases() -> None:
    disposition = _json(CONTRACT / "research-disposition.json")

    assert disposition["schema_version"] == "1.1.0"
    assert disposition["disposition"] == "unsupported_research_scoping"
    assert disposition["runtime_prohibited"] is True
    assert disposition["candidate_scope"] == "research_contract_only"
    assert disposition["scope_selection_ref"] == "scope-selection.json"
    assert disposition["approved_runtime_symbols"] == []
    adjacent = {item["issue"]: item for item in disposition["adjacent_methods"]}
    assert adjacent[570]["relationship"] == "not_sampling_acquisition_harm"
    assert adjacent[595]["relationship"] == "not_sampling_acquisition_harm"
    assert adjacent[570]["execution_reuse_allowed"] is False
    assert adjacent[595]["execution_reuse_allowed"] is False
    assert all(item["status"] != "satisfied" for item in disposition["gates"])


def test_h8a_selects_generic_kernel_exclusion_without_claiming_review() -> None:
    selection = _json(CONTRACT / "scope-selection.json")

    assert selection["selected_review_path"] == "generic_kernel_exclusion_review"
    assert selection["proposed_disposition"] == "reviewed_exclusion"
    assert selection["selection_status"] == "selected_for_candidate_bound_review"
    assert selection["scientific_disposition"] == "pending"
    assert selection["review_target"] == {
        "capability": "generic_automatic_scalar_or_authorizing_sampling_acquisition_harm_kernel",
        "domain": "unspecified",
        "jurisdiction": "unspecified",
        "population": "unspecified",
        "sampling_activity": "unspecified",
        "comparator": "unspecified",
        "affected_parties": "unspecified",
        "harm_categories": "unspecified",
        "excluded_semantics": "automatic_scalar_aggregation_or_study_authorization",
    }
    assert selection["runtime_disposition"] == "unsupported_research_scoping"
    assert selection["review_completed"] is False
    assert selection["human_confirmation_received"] is False
    assert selection["real_study_authorized"] is False
    assert selection["approved_runtime_symbols"] == []
    assert selection["next_tasks"] == [
        "H8-C",
        "H8-D",
        "H8-E",
        "H8-F",
        "H8-G",
        "H8-H",
    ]
    assert set(selection["reconsideration_entry_requirements"]) == {
        "narrow_domain_candidate",
        "applicable_jurisdiction_authority",
        "defined_population_and_affected_parties",
        "defined_sampling_action_and_comparator",
        "identified_harm_law_and_data",
        "defined_reporting_and_dropout_model",
        "declared_risk_ordering_and_component_ledger",
        "estimator_assurance_plan",
    }
    assert set(selection["downstream_review_gates"]) == {
        "candidate_bound_independent_review",
        "two_named_human_confirmations",
        "estimator_assurance",
        "maintainer_implementation_decision",
    }
    assert set(selection["not_authorized"]) == {
        "completed_reviewed_exclusion",
        "scientific_acceptance",
        "maintainer_disposition",
        "runtime_implementation",
        "polyglot_parity",
        "stable_promotion",
        "real_study_activity",
        "ethics_regulatory_authorization",
        "release",
        "publication",
        "registry_acceptance",
        "issue_closure",
    }


def test_h8a_source_and_evidence_bindings_resolve_exactly() -> None:
    selection = _json(CONTRACT / "scope-selection.json")
    commit = selection["reviewed_source_commit"]
    tree = selection["reviewed_source_tree"]
    assert commit["algorithm"] == tree["algorithm"] == "sha1"
    git = shutil.which("git")
    assert git is not None
    actual_tree = subprocess.check_output(
        [git, "-C", str(ROOT), "rev-parse", f"{commit['value']}^{{tree}}"],
        text=True,
    ).strip()
    assert actual_tree == tree["value"]

    for evidence_ref in selection["evidence_refs"]:
        path = Path(evidence_ref["path"])
        assert not path.is_absolute()
        assert ".." not in path.parts
        artifact = ROOT / path
        if not artifact.exists():
            artifact = ROOT / path.as_posix().replace(
                "conductor/tracks/", "conductor/archive/", 1
            )
        assert artifact.is_file()
        assert (
            hashlib.sha256(artifact.read_bytes()).hexdigest() == evidence_ref["sha256"]
        )


def test_scope_selection_schema_rejects_authority_or_integrity_relaxation() -> None:
    schema = _json(CONTRACT / "schemas/scope-selection.schema.json")
    valid = _json(CONTRACT / "scope-selection.json")
    invalid: list[dict[str, Any]] = []

    for field, value in (
        ("review_completed", True),
        ("human_confirmation_received", True),
        ("real_study_authorized", True),
        ("scientific_disposition", "reviewed_exclusion"),
        ("approved_runtime_symbols", ["compute_harm_adjusted_enbs"]),
    ):
        changed = deepcopy(valid)
        changed[field] = value
        invalid.append(changed)

    for field in (
        "next_tasks",
        "reconsideration_entry_requirements",
        "downstream_review_gates",
        "not_authorized",
    ):
        changed = deepcopy(valid)
        changed[field] = changed[field][:-1]
        invalid.append(changed)

    changed = deepcopy(valid)
    changed["evidence_refs"][0]["path"] = "../outside.json"
    invalid.append(changed)
    changed = deepcopy(valid)
    changed["selected_at"] = "2026-08-02T18:00:00+00:00"
    invalid.append(changed)

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for changed in invalid:
        with pytest.raises(ValidationError):
            validator.validate(changed)


def test_sampling_harm_method_family_has_no_runtime_declaration() -> None:
    capability = _json(CONTRACT / "capabilities.json")
    assert capability["method_family"] == "sampling_acquisition_harm_voi"
    assert all(surface["symbols"] == [] for surface in capability["surfaces"].values())

    family_tokens = (
        "sampling_acquisition_harm_voi",
        "sampling_acquisition_harm",
        "sampling-acquisition-harm",
    )
    source_roots = (
        (ROOT / "voiage", "*.py"),
        (ROOT / "rust", "*.rs"),
        (ROOT / "bindings", "*.jl"),
        (ROOT / "r-package", "*.R"),
    )
    for source_root, pattern in source_roots:
        for path in source_root.rglob(pattern):
            if path.name in {
                "sampling_harm_automated_challenge.py",
                "sampling_harm_human_commissioning.py",
                "sampling_harm_review_preparation.py",
                "sampling_harm_source_readiness.py",
            }:
                continue
            source = path.read_text(encoding="utf-8")
            assert all(token not in source for token in family_tokens), path

    governance_module = ROOT / "voiage/sampling_harm_review_preparation.py"
    governance_source = governance_module.read_text(encoding="utf-8")
    assert "compute_sampling_acquisition_harm" not in governance_source
    assert 'runtime_available"] is not False' in governance_source
    challenge_source = (ROOT / "voiage/sampling_harm_automated_challenge.py").read_text(
        encoding="utf-8"
    )
    assert "compute_sampling_acquisition_harm" not in challenge_source
    assert "all authority flags must remain false" in challenge_source

    schemas = {path.name for path in (CONTRACT / "schemas").glob("*.json")}
    assert schemas == {
        "adjacent-method-non-alias-delta.schema.json",
        "agent-assurance-review.schema.json",
        "automated-challenge-synthesis.schema.json",
        "candidate-context-decision.schema.json",
        "capability.schema.json",
        "estimand-boundary.schema.json",
        "governance-snapshot.schema.json",
        "governance-administrative-delta.schema.json",
        "human-commissioning-preflight.schema.json",
        "prior-findings.schema.json",
        "remediation-readiness-delta.schema.json",
        "remediation-register.schema.json",
        "review-candidate.schema.json",
        "review-preparation.schema.json",
        "reviewer-intake-readiness.schema.json",
        "research-disposition.schema.json",
        "scope-selection.schema.json",
        "source-review-intake-readiness.schema.json",
        "source-and-retrieval-register.schema.json",
        "source-observation-refresh.schema.json",
    }


def test_h8c_candidate_inputs_are_complete_but_never_claim_review() -> None:
    candidate = _json(CONTRACT / "review-candidate.json")
    boundary = _json(CONTRACT / "estimand-boundary.json")
    sources = _json(CONTRACT / "source-and-retrieval-register.json")
    findings = _json(CONTRACT / "prior-findings.json")
    snapshot = _json(CONTRACT / "governance-snapshot.json")

    assert candidate["scope"]["scientific_disposition"] == "pending"
    assert candidate["scope"]["runtime_available"] is False
    assert candidate["scope"]["approved_runtime_symbols"] == []
    assert candidate["adjacent_methods_not_aliased"] == [570, 571, 595, 598]
    assert candidate["required_independent_review_roles"] == [
        "estimand_domain",
        "estimator_assurance",
        "cross_language_api",
        "governance_publication",
        "domain_specialist",
    ]
    assert boundary["scientific_disposition"] == "pending"
    assert boundary["study_authorization_semantics"] == (
        "always_out_of_scope_for_software_output"
    )
    assert sources["retained_source_bytes"] == 0
    assert sources["exact_source_review_status"] == (
        "blocked_pending_independent_retrieval_and_drift_comparison"
    )
    assert sources["failed_retrievals"][0]["http_status"] == 403
    assert len(findings["findings"]) == 27
    assert len({item["id"] for item in findings["findings"]}) == 27
    assert all(item["disposition"] == "remediated" for item in findings["findings"])
    assert all(
        value is False
        for key, value in snapshot["authority_boundary"].items()
        if key != "preparation_only"
    )


def test_h8c_candidate_cross_artifact_bindings_and_freshness() -> None:
    git = shutil.which("git")
    assert git is not None
    preparation = _json(CONTRACT / "review-preparation.json")
    commit = preparation["candidate_commit"]["value"]

    def frozen_bytes(path: str) -> bytes:
        return subprocess.check_output([git, "show", f"{commit}:{path}"], cwd=ROOT)

    def frozen_json(path: str) -> dict[str, Any]:
        return json.loads(frozen_bytes(path))

    candidate = frozen_json(
        "specs/frontier/sampling-acquisition-harm/v1/review-candidate.json"
    )
    sources = frozen_json(candidate["source_register"])
    snapshot = frozen_json(candidate["governance_snapshot"])
    findings = frozen_json(candidate["prior_history"]["finding_inventory"])

    ledger_bytes = frozen_bytes(candidate["prior_history"]["evidence_ledger"])
    assert (
        hashlib.sha256(ledger_bytes).hexdigest()
        == candidate["prior_history"]["evidence_ledger_sha256"]
    )
    ledger = [json.loads(line) for line in ledger_bytes.splitlines()]
    ledger_entries = {entry["entry_sha256"] for entry in ledger}
    assert (
        ledger[-1]["entry_sha256"] == candidate["prior_history"]["latest_entry_sha256"]
    )
    assert {item["entry_sha256"] for item in findings["evidence_refs"]} <= (
        ledger_entries
    )
    assert {item["evidence_entry_sha256"] for item in findings["review_batches"]} <= (
        ledger_entries
    )
    for batch in findings["review_batches"]:
        subprocess.run(
            [git, "cat-file", "-e", f"{batch['reachable_merge_commit']}^{{commit}}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        merged_ledger = subprocess.check_output(
            [
                git,
                "show",
                f"{batch['reachable_merge_commit']}:{candidate['prior_history']['evidence_ledger']}",
            ],
            cwd=ROOT,
            text=True,
        )
        assert batch["evidence_entry_sha256"] in {
            json.loads(line)["entry_sha256"] for line in merged_ledger.splitlines()
        }

    source_manifest = frozen_bytes(sources["source_manifest"])
    assert (
        hashlib.sha256(source_manifest).hexdigest() == sources["source_manifest_sha256"]
    )
    for artifact in snapshot["canonical_projection"]["artifacts"]:
        assert (
            hashlib.sha256(frozen_bytes(artifact["path"])).hexdigest()
            == (artifact["sha256"])
        )
    for issue in snapshot["issues"]:
        assert (
            _canonical_sha256(issue["project_fields"]) == issue["project_fields_sha256"]
        )

    observed = datetime.fromisoformat(snapshot["observed_at"])
    expires = datetime.fromisoformat(snapshot["expires_at"])
    candidate_committed = datetime.fromisoformat(
        subprocess.check_output(
            [git, "show", "-s", "--format=%cI", commit], cwd=ROOT, text=True
        ).strip()
    )
    assert observed < expires
    assert observed <= candidate_committed <= expires


def test_h8c_candidate_schemas_reject_authority_scope_or_history_relaxation() -> None:
    cases = []
    for schema_name, artifact_name, mutate in (
        (
            "review-candidate.schema.json",
            "review-candidate.json",
            lambda item: item["authority_boundary"].__setitem__(
                "review_completed", True
            ),
        ),
        (
            "review-candidate.schema.json",
            "review-candidate.json",
            lambda item: item["scope"].__setitem__("runtime_available", True),
        ),
        (
            "estimand-boundary.schema.json",
            "estimand-boundary.json",
            lambda item: item.__setitem__(
                "scientific_disposition", "reviewed_exclusion"
            ),
        ),
        (
            "source-and-retrieval-register.schema.json",
            "source-and-retrieval-register.json",
            lambda item: item.__setitem__("retained_source_bytes", 1),
        ),
        (
            "prior-findings.schema.json",
            "prior-findings.json",
            lambda item: item["findings"].pop(),
        ),
        (
            "governance-snapshot.schema.json",
            "governance-snapshot.json",
            lambda item: item["authority_boundary"].__setitem__(
                "scientific_review_completed", True
            ),
        ),
    ):
        artifact = _json(CONTRACT / artifact_name)
        mutate(artifact)
        cases.append((_json(CONTRACT / "schemas" / schema_name), artifact))

    for schema, artifact in cases:
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate(artifact)


def test_no_python_native_or_binding_execution_symbol_exists() -> None:
    public_modules = (voiage, voiage.methods, voiage.experimental)
    for symbol in PROHIBITED_SYMBOLS:
        assert symbol not in voiage.__all__
        assert symbol not in voiage.methods.__all__
        assert all(not hasattr(module, symbol) for module in public_modules)

    native = importlib.import_module("voiage._core")
    for symbol in PROHIBITED_SYMBOLS:
        assert not hasattr(native, symbol)

    stable_api = _json(ROOT / "specs/v1/stable-api.json")
    declared_symbols = {
        symbol for category in stable_api["symbols"].values() for symbol in category
    }
    assert PROHIBITED_SYMBOLS.isdisjoint(declared_symbols)

    source_roots = (
        (ROOT / "voiage", "*.py"),
        (ROOT / "rust", "*.rs"),
        (ROOT / "bindings", "*.jl"),
        (ROOT / "r-package", "*.R"),
    )
    for source_root, pattern in source_roots:
        for path in source_root.rglob(pattern):
            source = path.read_text(encoding="utf-8")
            for symbol in PROHIBITED_SYMBOLS:
                assert symbol not in source, f"unexpected {symbol} in {path}"


def test_readme_never_claims_570_or_595_executes_sampling_harm() -> None:
    readme = (CONTRACT / "README.md").read_text(encoding="utf-8")

    assert "No runtime exists" in readme
    assert "#570" in readme
    for issue in ("#570", "#571", "#595", "#598"):
        assert issue in readme
    assert "None executes sampling-acquisition harm" in readme


def test_human_confirmation_plan_is_fail_closed_and_role_separated() -> None:
    track = ROOT / "conductor/archive/sampling_acquisition_harm_voi_20260802"
    plan = (track / "plan.md").read_text(encoding="utf-8")
    gates = (track / "human-confirmation-gates.md").read_text(encoding="utf-8")
    requirements = (track / "requirements.md").read_text(encoding="utf-8")
    normalized_gates = " ".join(gates.split())

    for task in ("H8-A", "H8-B", "H8-C", "H8-D", "H8-E", "H8-F", "H8-G", "H8-H"):
        assert f"**{task}:**" in plan

    assert "two distinct named" in requirements
    assert "orchestrating agent" in requirements
    assert "options, contingencies, rationale and" in requirements
    assert "`unsupported_research_scoping`" in gates
    assert "never provide the named human confirmation" in normalized_gates
    assert "partial mutation sets Sync State to `Conflict`" in gates
    assert "Issue closure is the final synchronized transition" in gates
