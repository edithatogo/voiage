"""Fail-closed contract tests for sampling-acquisition-harm research scope."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

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


def test_fail_closed_capability_and_research_disposition_validate() -> None:
    schemas = CONTRACT / "schemas"
    pairs = (
        (schemas / "capability.schema.json", CONTRACT / "capabilities.json"),
        (
            schemas / "research-disposition.schema.json",
            CONTRACT / "research-disposition.json",
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

    assert disposition["disposition"] == "unsupported_research_scoping"
    assert disposition["runtime_prohibited"] is True
    assert disposition["candidate_scope"] == "research_contract_only"
    assert disposition["approved_runtime_symbols"] == []
    adjacent = {item["issue"]: item for item in disposition["adjacent_methods"]}
    assert adjacent[570]["relationship"] == "not_sampling_acquisition_harm"
    assert adjacent[595]["relationship"] == "not_sampling_acquisition_harm"
    assert adjacent[570]["execution_reuse_allowed"] is False
    assert adjacent[595]["execution_reuse_allowed"] is False
    assert all(item["status"] != "satisfied" for item in disposition["gates"])


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
    assert "#595" in readme
    assert "does not execute sampling-acquisition harm" in readme


def test_human_confirmation_plan_is_fail_closed_and_role_separated() -> None:
    track = ROOT / "conductor/tracks/sampling_acquisition_harm_voi_20260802"
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
