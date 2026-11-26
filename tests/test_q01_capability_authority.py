"""Q01 capability, Rust-authority reconciliation, and Q03 source/rights preflight tests."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pytest

from voiage.contracts.capability_report import CliCapabilityReport

ROOT = Path(__file__).resolve().parents[1]
STABLE_API_PATH = ROOT / "specs" / "v2" / "stable-api.json"
CHECKLIST_PATH = ROOT / "specs" / "frontier" / "governance" / "promotion-checklist.json"
NUMERICS_LIB_PATH = ROOT / "rust" / "crates" / "voiage-numerics" / "src" / "lib.rs"
MATRIX_PATH = ROOT / "specs" / "v2" / "capability-authority-matrix.json"
PREFLIGHT_PATH = (
    ROOT / "specs" / "numerical-reference" / "v1" / "source-rights-preflight.json"
)

ALLOWED_AUTHORITY_MODES = frozenset(
    {"rust", "rust+python-orchestration", "python", "unsupported"}
)
ALLOWED_VERIFICATION_FAMILIES = frozenset(f"D{index:02d}" for index in range(1, 19))


def _exported_numerics_symbols() -> set[str]:
    """Extract publicly re-exported Rust symbols from `voiage-numerics/src/lib.rs`."""
    text = NUMERICS_LIB_PATH.read_text(encoding="utf-8")
    symbols: set[str] = set()
    for block in re.findall(r"pub use [^{;]+(?:\{([^}]+)\}|::(\w+));", text, re.DOTALL):
        group, single = block
        if single:
            symbols.add(single.strip())
        if group:
            for item in group.split(","):
                name = item.strip()
                if name:
                    symbols.add(name)
    return symbols


def validate_authority_matrix(
    matrix: dict[str, Any],
    *,
    exported_rust_symbols: set[str],
    stable_methods: set[str],
    frontier_families: set[str],
    cli_methods: set[str],
) -> None:
    """Fail closed on unbacked native claims or missing verification/exclusion contracts."""
    if matrix.get("schema_version") != "1.0.0":
        raise ValueError("capability-authority-matrix requires schema_version 1.0.0")
    entries = matrix.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("entries must be a non-empty list")

    seen_ids: set[str] = set()
    covered_stable: set[str] = set()
    covered_frontier: set[str] = set()
    covered_cli: set[str] = set()

    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("each entry must be a JSON object")
        entry_id = str(entry.get("id", "")).strip()
        if not entry_id or entry_id in seen_ids:
            raise ValueError(f"invalid or duplicate capability entry id: {entry_id!r}")
        seen_ids.add(entry_id)

        authority = entry.get("authoritative_implementation")
        if authority not in ALLOWED_AUTHORITY_MODES:
            raise ValueError(
                f"{entry_id}: invalid authoritative_implementation {authority!r}"
            )

        rust_symbols = entry.get("rust_symbols", [])
        if not isinstance(rust_symbols, list):
            raise TypeError(f"{entry_id}: rust_symbols must be a list")

        # Reject native claims that lack concrete exported Rust symbols.
        if authority in {"rust", "rust+python-orchestration"}:
            if not rust_symbols:
                raise ValueError(
                    f"{entry_id}: native authority {authority!r} requires at least one rust_symbol"
                )
            unknown = set(rust_symbols) - exported_rust_symbols
            if unknown:
                raise ValueError(
                    f"{entry_id}: declares non-exported rust_symbols {sorted(unknown)}"
                )

        # Require either D01-D18 verification family + reference artifact or explicit exclusion.
        families = entry.get("verification_families", [])
        ref_paths = entry.get("reference_artifacts", [])
        exclusion = entry.get("unsupported_or_excluded_reason")
        if families:
            if not isinstance(families, list) or not set(families).issubset(
                ALLOWED_VERIFICATION_FAMILIES
            ):
                raise ValueError(
                    f"{entry_id}: invalid verification_families {families!r}"
                )
            if not isinstance(ref_paths, list) or not ref_paths:
                raise ValueError(
                    f"{entry_id}: verification_families require reference_artifacts"
                )
            missing_refs = [
                rel
                for rel in ref_paths
                if not isinstance(rel, str) or not (ROOT / rel).exists()
            ]
            if missing_refs:
                raise ValueError(
                    f"{entry_id}: missing reference_artifacts on disk: {missing_refs}"
                )
        elif not isinstance(exclusion, str) or not exclusion.strip():
            raise ValueError(
                f"{entry_id}: missing both verification_families and unsupported_or_excluded_reason"
            )

        bindings = entry.get("bindings")
        if not isinstance(bindings, dict) or set(bindings) != {
            "python",
            "rust",
            "r",
            "julia",
        }:
            raise ValueError(
                f"{entry_id}: bindings must declare python, rust, r, and julia"
            )

        if entry.get("stable_api_method") in stable_methods:
            covered_stable.add(entry["stable_api_method"])
        if entry.get("frontier_family") in frontier_families:
            covered_frontier.add(entry["frontier_family"])
        if entry.get("cli_method") in cli_methods:
            covered_cli.add(entry["cli_method"])

    if missing_stable := stable_methods - covered_stable:
        raise ValueError(f"missing stable-api.json methods: {sorted(missing_stable)}")
    if missing_frontier := frontier_families - covered_frontier:
        raise ValueError(
            f"missing promotion-checklist families: {sorted(missing_frontier)}"
        )
    if missing_cli := cli_methods - covered_cli:
        raise ValueError(f"missing CLI capability methods: {sorted(missing_cli)}")


def validate_source_rights_preflight(preflight: dict[str, Any]) -> None:
    """Enforce Q03 source/rights preflight boundaries before corpus admission."""
    if preflight.get("schema_version") != "1.0.0":
        raise ValueError("source-rights preflight requires schema_version 1.0.0")
    owned = preflight.get("owned_synthetic_references", [])
    external = preflight.get("external_candidates", [])
    if not isinstance(owned, list) or not owned:
        raise ValueError("owned_synthetic_references must be non-empty")
    if not isinstance(external, list) or not external:
        raise ValueError("external_candidates must be non-empty")

    for ref in owned:
        rel_path = ref.get("path", "")
        if not isinstance(rel_path, str) or not (ROOT / rel_path).exists():
            raise ValueError(f"owned reference path missing: {rel_path!r}")
        if not ref.get("reference_implementation"):
            raise ValueError(f"{rel_path}: must name separate reference_implementation")
        if ref.get("reference_implementation") == "voiage-under-test":
            raise ValueError(
                f"{rel_path}: reference cannot be generated solely by code under test"
            )
        if not ref.get("license"):
            raise ValueError(f"{rel_path}: missing license declaration")

    for cand in external:
        cid = cand.get("id", "<unknown>")
        if cand.get("admission_state") != "candidate-source-only":
            raise ValueError(
                f"{cid}: unacquired external source cannot carry state {cand.get('admission_state')!r}"
            )
        if (
            cand.get("license_cleared") is not False
            or cand.get("checksum_verified") is not False
        ):
            raise ValueError(
                f"{cid}: unacquired external source must record license_cleared=false and checksum_verified=false"
            )
        if not cand.get("url"):
            raise ValueError(f"{cid}: candidate URL required")


def test_witnesses_reject_unbacked_native_claims_missing_exclusions_and_unverified_sources() -> (
    None
):
    """Distinguishing witnesses for Q01 and Q03 failure modes."""
    rust_exports = _exported_numerics_symbols()
    with pytest.raises(ValueError, match="requires at least one rust_symbol"):
        validate_authority_matrix(
            {
                "schema_version": "1.0.0",
                "entries": [
                    {
                        "id": "python_only_claiming_rust",
                        "stable_api_method": "evpi",
                        "cli_method": "evpi",
                        "authoritative_implementation": "rust",
                        "rust_symbols": [],
                        "verification_families": ["D01"],
                        "reference_artifacts": [
                            "specs/numerical-reference/v1/evpi-cases.json"
                        ],
                        "bindings": {
                            "python": "native",
                            "rust": "native",
                            "r": "c-abi",
                            "julia": "c-abi",
                        },
                    }
                ],
            },
            exported_rust_symbols=rust_exports,
            stable_methods={"evpi"},
            frontier_families=set(),
            cli_methods={"evpi"},
        )

    with pytest.raises(ValueError, match="unsupported_or_excluded_reason"):
        validate_authority_matrix(
            {
                "schema_version": "1.0.0",
                "entries": [
                    {
                        "id": "missing_verification_contract",
                        "stable_api_method": "evpi",
                        "authoritative_implementation": "python",
                        "rust_symbols": [],
                        "bindings": {
                            "python": "python",
                            "rust": "unsupported",
                            "r": "unsupported",
                            "julia": "unsupported",
                        },
                    }
                ],
            },
            exported_rust_symbols=rust_exports,
            stable_methods={"evpi"},
            frontier_families=set(),
            cli_methods=set(),
        )

    with pytest.raises(ValueError, match="unacquired external source"):
        validate_source_rights_preflight(
            {
                "schema_version": "1.0.0",
                "owned_synthetic_references": [
                    {
                        "path": "specs/numerical-reference/v1/evpi-cases.json",
                        "method": "evpi",
                        "license": "Apache-2.0",
                        "reference_implementation": "independent_hand_calculation",
                    }
                ],
                "external_candidates": [
                    {
                        "id": "S01",
                        "url": "https://example.invalid/voi",
                        "admission_state": "validated",
                        "license_cleared": False,
                        "checksum_verified": False,
                    }
                ],
            }
        )


def test_capability_authority_matrix_reconciles_all_surfaces() -> None:
    assert MATRIX_PATH.is_file(), f"Missing Q01 reconciliation artifact: {MATRIX_PATH}"
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    stable_api = json.loads(STABLE_API_PATH.read_text(encoding="utf-8"))
    checklist = json.loads(CHECKLIST_PATH.read_text(encoding="utf-8"))

    from typer.testing import CliRunner

    from voiage.cli import app

    cli_res = CliRunner().invoke(app, ["capabilities", "--json"])
    assert cli_res.exit_code == 0
    cli_report = CliCapabilityReport.model_validate_json(cli_res.stdout)

    validate_authority_matrix(
        matrix,
        exported_rust_symbols=_exported_numerics_symbols(),
        stable_methods=set(stable_api["methods"]),
        frontier_families={item["name"] for item in checklist["families"]},
        cli_methods=set(cli_report.methods),
    )

    current_matrix = json.loads(
        (ROOT / "specs" / "bindings" / "current-capability-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    current_by_method = {
        str(row["capability"]).lower(): row for row in current_matrix["capabilities"]
    }
    entries_by_id = {entry["id"]: entry for entry in matrix["entries"]}
    for method in (
        "evpi",
        "enbs",
        "evppi",
        "evsi",
        "normal_normal_two_arm_evsi",
        "ceaf",
        "dominance",
    ):
        current_row = current_by_method[method]
        bindings = entries_by_id[method]["bindings"]
        assert (bindings["r"] == "unsupported") == (current_row["r"] == "unavailable")
        assert (bindings["julia"] == "unsupported") == (
            current_row["julia"] == "unavailable"
        )


def test_q03_source_rights_preflight_manifest_is_valid() -> None:
    assert PREFLIGHT_PATH.is_file(), f"Missing Q03 preflight manifest: {PREFLIGHT_PATH}"
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    validate_source_rights_preflight(preflight)
    assert {cand["id"] for cand in preflight["external_candidates"]} >= {
        "S01",
        "S02",
        "S03",
        "S04",
        "S05",
        "S06",
    }
