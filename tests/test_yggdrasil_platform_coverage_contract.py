"""Fail-closed contracts for maximum-coverage Yggdrasil filtering."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from jsonschema import Draft202012Validator
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = ROOT / "specs" / "yggdrasil-platform-coverage" / "v1"
SCHEMA = CONTRACT_ROOT / "platform-coverage.schema.json"
MANIFEST = CONTRACT_ROOT / "voiage-ffi-platform-coverage.json"
VALIDATOR = ROOT / "scripts" / "validate_yggdrasil_platform_coverage.py"
RECIPE = ROOT / "packaging" / "yggdrasil" / "V" / "voiage_ffi" / "build_tarballs.jl"
EXPANDED_RUN = (
    ROOT
    / "conductor"
    / "archive"
    / "yggdrasil_maximum_platform_coverage_20260821"
    / "buildkite-31971-expanded-matrix.json"
)
TERMINAL_RUN = EXPANDED_RUN.with_name("buildkite-31972-terminal-matrix.json")
PRODUCT_EVIDENCE = EXPANDED_RUN.with_name("phase-4-product-abi-evidence-20260821.json")


def _payload() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _run_validator(
    tmp_path: Path, payload: dict[str, Any]
) -> subprocess.CompletedProcess[str]:
    candidate = tmp_path / "platform-coverage.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--manifest", str(candidate)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_committed_contract_is_schema_valid_and_semantically_reconciled(
    tmp_path: Path,
) -> None:
    """The canonical manifest must pass both structural and semantic checks."""
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    payload = _payload()

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    result = _run_validator(tmp_path, payload)

    assert result.returncode == 0, result.stderr
    assert "platform coverage contract valid" in result.stdout


def test_contract_pins_evidence_filtered_maximum_coverage_policy() -> None:
    """Only the two initial gaps and the evidenced Windows target are excluded."""
    payload = _payload()

    assert payload["catalogue"]["function"] == "supported_platforms()"
    assert payload["catalogue"]["binarybuilderbase_revision"] == (
        "76c4aab80ad5019af59af0f42e5669109cd5194b"
    )
    assert payload["candidate"]["source_revision"] == (
        "964a0fc334ece9509387cd07d43776adf38be240"
    )
    assert (
        payload["repository_recipe"]["sha256"]
        == hashlib.sha256(RECIPE.read_bytes()).hexdigest()
    )
    assert payload["policy"]["mode"] == "inclusive_negative_filter"
    assert payload["policy"]["stage"] == "evidence_filtered"
    assert [item["predicate"] for item in payload["policy"]["negative_filters"]] == [
        'Sys.isfreebsd(p) && arch(p) == "aarch64"',
        'arch(p) == "riscv64"',
        'Sys.iswindows(p) && arch(p) == "i686"',
    ]


def test_recipe_uses_inclusive_universe_and_narrow_evidenced_filters() -> None:
    """The repository recipe must attempt every non-excluded standard target."""
    recipe = RECIPE.read_text(encoding="utf-8")

    assert "platforms = supported_platforms()" in recipe
    assert "platforms = [" not in recipe
    assert recipe.count("filter!(") == 3
    assert (
        'filter!(p -> !(Sys.isfreebsd(p) && arch(p) == "aarch64"), platforms)' in recipe
    )
    assert 'filter!(p -> arch(p) != "riscv64", platforms)' in recipe
    assert 'filter!(p -> !(Sys.iswindows(p) && arch(p) == "i686"), platforms)' in recipe
    assert "Rust toolchain is not available on aarch64-unknown-freebsd" in recipe
    assert "Rust toolchain is not available on riscv64" in recipe
    assert "Rust toolchain cannot link i686-w64-mingw32" in recipe


def test_expanded_run_preserves_every_attempt_and_root_failure() -> None:
    """The superseded expanded run remains exact-head, per-target evidence."""
    receipt = json.loads(EXPANDED_RUN.read_text(encoding="utf-8"))

    assert receipt["candidate_head"] == ("70e6087dce9cd1e59f644e761c1eecf7d7f2fa58")
    assert receipt["build"]["attempted"] == 16
    assert receipt["build"]["passed"] == 15
    assert receipt["build"]["failed"] == 1
    assert len(receipt["platforms"]) == 18
    failed = [row for row in receipt["platforms"] if row["state"] == "failed"]
    assert [row["id"] for row in failed] == ["i686-w64-mingw32"]
    assert "_Unwind_Resume" in failed[0]["diagnostic"]


def test_terminal_run_reconciles_with_manifest_platform_evidence() -> None:
    """Every terminal target is represented once and bound to its exact job."""
    payload = _payload()
    receipt = json.loads(TERMINAL_RUN.read_text(encoding="utf-8"))
    receipt_by_id = {row["id"]: row for row in receipt["platforms"]}

    assert set(receipt_by_id) == set(payload["catalogue"]["platforms"])
    assert receipt["candidate_head"] == payload["candidate"]["head"]
    assert receipt["recipe_sha256"] == payload["candidate"]["recipe_sha256"]
    assert receipt["build"]["passed"] == payload["aggregates"]["included"] == 15
    for record in payload["platforms"]:
        terminal = receipt_by_id[record["id"]]
        if record["disposition"] == "included":
            assert terminal["state"] == "passed"
            assert terminal["job"] in record["evidence"]["locators"]
        else:
            assert terminal["state"] == record["lifecycle"]


def test_product_and_runtime_claims_reconcile_without_cross_target_overclaim() -> None:
    """Every included archive is hashed, while runtime claims remain host-bound."""
    payload = _payload()
    evidence = json.loads(PRODUCT_EVIDENCE.read_text(encoding="utf-8"))
    included = {
        row["id"] for row in payload["platforms"] if row["disposition"] == "included"
    }

    assert {row["platform"] for row in evidence["artifacts"]} == included
    assert evidence["product_archive_validation"] == {
        "downloaded": 15,
        "sha256_matched": 15,
        "product_path_matched": 15,
        "license_path_matched": 15,
        "failures": 0,
    }
    assert all(len(row["sha256"]) == 64 for row in evidence["artifacts"])
    runtime_platforms = {
        row["platform"]
        for row in evidence["native_smokes"]
        if row["status"] == "passed"
    }
    assert runtime_platforms == {"aarch64-apple-darwin", "x86_64-apple-darwin"}
    for record in payload["platforms"]:
        expected = "passed" if record["id"] in runtime_platforms else "not_run"
        if record["disposition"] == "included":
            assert record["evidence"]["runtime"] == expected


def test_recipe_preserves_release_product_and_shared_musl_build() -> None:
    """Coverage expansion must not weaken the submitted product recipe."""
    recipe = RECIPE.read_text(encoding="utf-8")

    assert 'version = v"2.1.0"' in recipe
    assert "964a0fc334ece9509387cd07d43776adf38be240" in recipe
    assert 'RUSTFLAGS="-C target-feature=-crt-static"' in recipe
    assert "cargo build \\" in recipe
    assert "--release \\" in recipe
    assert "--locked \\" in recipe
    assert "--package voiage-ffi \\" in recipe
    assert 'LibraryProduct("libvoiage_ffi", :libvoiage_ffi)' in recipe


def _remove_classification(payload: dict[str, Any]) -> None:
    payload["platforms"].pop()
    payload["aggregates"]["classified"] -= 1


def _duplicate_platform(payload: dict[str, Any]) -> None:
    payload["platforms"].append(deepcopy(payload["platforms"][0]))
    payload["aggregates"]["classified"] += 1


def _remove_reconsideration_trigger(payload: dict[str, Any]) -> None:
    excluded = next(
        row for row in payload["platforms"] if row["disposition"] == "excluded"
    )
    del excluded["exclusion"]["reconsideration_trigger"]


def _forge_totals(payload: dict[str, Any]) -> None:
    payload["aggregates"]["included"] += 1


def _forge_runtime_claim(payload: dict[str, Any]) -> None:
    included = next(
        row for row in payload["platforms"] if row["disposition"] == "included"
    )
    included["evidence"]["runtime"] = "passed"


def _add_uncontracted_filter(payload: dict[str, Any]) -> None:
    platform = next(
        row for row in payload["platforms"] if row["id"] == "i686-w64-mingw32"
    )
    filter_contract = {
        "predicate": "Sys.iswindows(p)",
        "specificity": "exact_arch_os",
        "evidence_kind": "hosted_failure",
        "reason_category": "project_architecture_limitation",
        "reason": "One Windows target failed in the expanded hosted build matrix.",
        "primary_evidence": "https://buildkite.com/julialang/yggdrasil/builds/1",
        "observed_at": "2026-08-21T00:00:00Z",
        "reconsideration_trigger": "Retest after the project architecture limitation is removed.",
        "matched_platform_ids": [platform["id"]],
    }
    payload["policy"]["stage"] = "evidence_filtered"
    payload["policy"]["negative_filters"].append(filter_contract)
    payload["candidate"]["hosted_run"] = {
        "state": "in_progress",
        "url": "https://buildkite.com/julialang/yggdrasil/builds/1",
        "platform_status_source": "buildkite",
    }
    platform["disposition"] = "excluded"
    platform["lifecycle"] = "excluded_evidenced"
    platform["evidence"] = {
        "build": "not_run",
        "product": "not_run",
        "abi_smoke": "not_run",
        "runtime": "not_run",
        "locators": [filter_contract["primary_evidence"]],
    }
    platform["exclusion"] = {
        key: value
        for key, value in filter_contract.items()
        if key != "matched_platform_ids"
    }
    payload["aggregates"]["included"] -= 1
    payload["aggregates"]["excluded"] += 1
    payload["aggregates"]["lifecycle_counts"]["pending"] -= 1
    payload["aggregates"]["lifecycle_counts"]["excluded_evidenced"] += 1


def _replace_with_placeholder_evidence(payload: dict[str, Any]) -> None:
    policy_filter = payload["policy"]["negative_filters"][0]
    platform_id = policy_filter["matched_platform_ids"][0]
    platform = next(row for row in payload["platforms"] if row["id"] == platform_id)
    placeholder = "https://example.invalid/not-authoritative"
    policy_filter["primary_evidence"] = placeholder
    platform["exclusion"]["primary_evidence"] = placeholder


def _mislabel_evidence_filtered_policy(payload: dict[str, Any]) -> None:
    payload["policy"]["stage"] = "initial_expansion"


def _diverge_candidate_recipe_digest(payload: dict[str, Any]) -> None:
    payload["candidate"]["recipe_sha256"] = "0" * 64


def _use_stale_terminal_job_locator(payload: dict[str, Any]) -> None:
    included = next(
        row for row in payload["platforms"] if row["disposition"] == "included"
    )
    included["evidence"]["locators"] = [
        "https://buildkite.com/julialang/yggdrasil/builds/31971#superseded"
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        _remove_classification,
        _duplicate_platform,
        _remove_reconsideration_trigger,
        _forge_totals,
        _forge_runtime_claim,
        _add_uncontracted_filter,
        _replace_with_placeholder_evidence,
        _mislabel_evidence_filtered_policy,
        _diverge_candidate_recipe_digest,
        _use_stale_terminal_job_locator,
    ],
    ids=[
        "unclassified-catalogue-platform",
        "duplicate-platform",
        "missing-reconsideration-trigger",
        "stale-aggregate-counts",
        "runtime-overclaim",
        "broad-uncontracted-filter",
        "placeholder-exclusion-evidence",
        "mislabel-evidence-filtered-policy",
        "candidate-recipe-digest-divergence",
        "stale-terminal-job-locator",
    ],
)
def test_validator_rejects_pathological_contract_mutations(
    tmp_path: Path,
    mutate: Any,
) -> None:
    """Structural, reconciliation, and evidence overclaim mutations fail closed."""
    payload = _payload()
    mutate(payload)

    result = _run_validator(tmp_path, payload)

    assert result.returncode != 0
    assert result.stderr.strip()


def test_every_catalogue_platform_is_required_by_reconciliation(tmp_path: Path) -> None:
    """Removing any single classification fails regardless of adjusted aggregates."""
    original = _payload()
    for platform_id in original["catalogue"]["platforms"]:
        payload = deepcopy(original)
        removed = next(row for row in payload["platforms"] if row["id"] == platform_id)
        payload["platforms"].remove(removed)
        payload["aggregates"]["classified"] -= 1
        payload["aggregates"][removed["disposition"]] -= 1
        payload["aggregates"]["lifecycle_counts"][removed["lifecycle"]] -= 1

        result = _run_validator(tmp_path, payload)

        assert result.returncode != 0, platform_id
        assert "classification mismatch" in result.stderr


def test_cli_rejects_malformed_json(tmp_path: Path) -> None:
    """Malformed evidence never falls through to a partial semantic check."""
    candidate = tmp_path / "malformed.json"
    candidate.write_text('{"schema_version":', encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--manifest", str(candidate)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "cannot load JSON" in result.stderr
