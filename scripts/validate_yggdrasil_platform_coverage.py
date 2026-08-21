#!/usr/bin/env python3
"""Validate the fail-closed Yggdrasil maximum-platform-coverage contract."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlparse

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as SchemaValidationError

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = ROOT / "specs" / "yggdrasil-platform-coverage" / "v1"
DEFAULT_SCHEMA = CONTRACT_ROOT / "platform-coverage.schema.json"
DEFAULT_MANIFEST = CONTRACT_ROOT / "voiage-ffi-platform-coverage.json"

INITIAL_FILTERS = {
    'Sys.isfreebsd(p) && arch(p) == "aarch64"': {"aarch64-unknown-freebsd"},
    'arch(p) == "riscv64"': {"riscv64-linux-gnu"},
}
LIFECYCLES = (
    "pending",
    "building",
    "passed",
    "failed_actionable",
    "failed_transient",
    "excluded_upstream",
    "excluded_evidenced",
)


class ContractError(ValueError):
    """Raised when structural or semantic platform evidence fails closed."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: cannot load JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{path}: root must be a JSON object")
    return payload


def _validate_schema(payload: dict[str, Any], schema_path: Path) -> None:
    schema = _load_object(schema_path)
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
        ).validate(payload)
    except (SchemaError, SchemaValidationError) as exc:
        raise ContractError(f"schema validation failed: {exc.message}") from exc


def _validate_platform_identity(payload: dict[str, Any], errors: list[str]) -> None:
    catalogue = payload["catalogue"]["platforms"]
    records = payload["platforms"]
    catalogue_ids = set(catalogue)
    record_ids = [record["id"] for record in records]

    if len(catalogue_ids) != len(catalogue):
        errors.append("catalogue contains duplicate platform identities")
    if len(set(record_ids)) != len(record_ids):
        errors.append("platform records contain duplicate identities")
    if catalogue_ids != set(record_ids):
        missing = sorted(catalogue_ids - set(record_ids))
        unexpected = sorted(set(record_ids) - catalogue_ids)
        errors.append(
            f"catalogue classification mismatch: missing={missing}, unexpected={unexpected}"
        )
    errors.extend(
        f"{record['id']}: id and BinaryBuilder triplet differ"
        for record in records
        if record["id"] != record["triplet"]
    )


def _validate_filters(payload: dict[str, Any], errors: list[str]) -> None:
    policy = payload["policy"]
    filters = policy["negative_filters"]
    predicates = [item["predicate"] for item in filters]
    if len(set(predicates)) != len(predicates):
        errors.append("negative filter predicates must be unique")

    by_predicate = {item["predicate"]: item for item in filters}
    if policy["stage"] == "initial_expansion" and set(predicates) != set(
        INITIAL_FILTERS
    ):
        errors.append("initial expansion must contain exactly the two approved filters")
    for predicate, expected_ids in INITIAL_FILTERS.items():
        item = by_predicate.get(predicate)
        if item is None:
            errors.append(f"required initial filter missing: {predicate}")
        elif set(item["matched_platform_ids"]) != expected_ids:
            errors.append(f"initial filter has unexpected platform scope: {predicate}")

    excluded = {
        record["id"]: record
        for record in payload["platforms"]
        if record["disposition"] == "excluded"
    }
    declared_matches: set[str] = set()
    for item in filters:
        matches = set(item["matched_platform_ids"])
        declared_matches.update(matches)
        if not matches:
            errors.append(f"negative filter matches no platform: {item['predicate']}")
        if not matches <= set(excluded):
            errors.append(
                f"negative filter includes a non-excluded platform: {item['predicate']}"
            )
        predicate = item["predicate"]
        specificity = item["specificity"]
        if "||" in predicate or predicate.strip() == "true":
            errors.append(f"negative filter predicate is over-broad: {predicate}")
        if specificity == "exact_platform" and (
            "triplet(p)" not in predicate or len(matches) != 1
        ):
            errors.append(
                f"exact-platform filter is not platform-specific: {predicate}"
            )
        if specificity == "exact_arch_os" and (
            "arch(p)" not in predicate or "Sys.is" not in predicate
        ):
            errors.append(f"architecture/OS filter lacks both constraints: {predicate}")
        if specificity == "exact_arch" and (
            "arch(p)" not in predicate
            or item["reason_category"] != "upstream_toolchain_unavailable"
        ):
            errors.append(
                f"architecture-wide filter requires an upstream toolchain gap: {predicate}"
            )
        evidence_host = urlparse(item["primary_evidence"]).hostname or ""
        if evidence_host.endswith(".invalid"):
            errors.append(f"negative filter uses placeholder evidence: {predicate}")
        if (
            item["evidence_kind"] == "hosted_failure"
            and payload["candidate"]["hosted_run"]["state"] == "not_started"
        ):
            errors.append(f"hosted-failure filter predates a hosted run: {predicate}")
        if (
            item["evidence_kind"] == "upstream_source"
            and item["reason_category"] != "upstream_toolchain_unavailable"
        ):
            errors.append(
                f"upstream-source filter must describe a toolchain gap: {predicate}"
            )
        for platform_id in matches & set(excluded):
            exclusion = excluded[platform_id]["exclusion"]
            fields = (
                "predicate",
                "specificity",
                "evidence_kind",
                "reason_category",
                "reason",
                "primary_evidence",
                "observed_at",
                "reconsideration_trigger",
            )
            errors.extend(
                f"{platform_id}: exclusion {field} differs from its policy filter"
                for field in fields
                if exclusion[field] != item[field]
            )
    if declared_matches != set(excluded):
        errors.append("excluded platform records and negative-filter matches differ")


def _validate_evidence_layers(payload: dict[str, Any], errors: list[str]) -> None:
    for record in payload["platforms"]:
        platform_id = record["id"]
        evidence = record["evidence"]
        if evidence["runtime"] == "passed" and evidence["abi_smoke"] != "passed":
            errors.append(f"{platform_id}: runtime pass requires an ABI-smoke pass")
        if evidence["abi_smoke"] == "passed" and evidence["product"] != "passed":
            errors.append(f"{platform_id}: ABI-smoke pass requires a product pass")
        if evidence["product"] == "passed" and evidence["build"] != "passed":
            errors.append(f"{platform_id}: product pass requires a build pass")
        if record["lifecycle"] == "passed" and (
            evidence["build"] != "passed" or evidence["product"] != "passed"
        ):
            errors.append(
                f"{platform_id}: passed lifecycle requires build and product passes"
            )
        if record["disposition"] == "excluded" and any(
            evidence[layer] == "passed"
            for layer in ("build", "product", "abi_smoke", "runtime")
        ):
            errors.append(
                f"{platform_id}: excluded platform cannot claim passed evidence"
            )


def _validate_aggregates(payload: dict[str, Any], errors: list[str]) -> None:
    records = payload["platforms"]
    aggregates = payload["aggregates"]
    lifecycle_counts = Counter(record["lifecycle"] for record in records)
    expected = {
        "catalogue": len(payload["catalogue"]["platforms"]),
        "classified": len(records),
        "included": sum(record["disposition"] == "included" for record in records),
        "excluded": sum(record["disposition"] == "excluded" for record in records),
    }
    for field, value in expected.items():
        if aggregates[field] != value:
            errors.append(f"aggregate {field} is {aggregates[field]}, expected {value}")
    for lifecycle in LIFECYCLES:
        actual = aggregates["lifecycle_counts"][lifecycle]
        expected_count = lifecycle_counts[lifecycle]
        if actual != expected_count:
            errors.append(
                f"aggregate lifecycle {lifecycle} is {actual}, expected {expected_count}"
            )


def _validate_hosted_state(payload: dict[str, Any], errors: list[str]) -> None:
    candidate = payload["candidate"]
    hosted = candidate["hosted_run"]
    if hosted["state"] == "not_started" and (
        hosted["url"] is not None or hosted["platform_status_source"] != "none"
    ):
        errors.append("not-started hosted run cannot claim a URL or status source")
    if hosted["state"] != "not_started" and (
        hosted["url"] is None or hosted["platform_status_source"] == "none"
    ):
        errors.append("started hosted run requires a URL and status source")
    if hosted["state"] == "terminal" and any(
        record["lifecycle"] in {"pending", "building", "failed_transient"}
        for record in payload["platforms"]
        if record["disposition"] == "included"
    ):
        errors.append("terminal hosted run contains non-terminal included platforms")
    if hosted["state"] == "terminal":
        job_prefix = f"{hosted['url']}#"
        errors.extend(
            f"{record['id']}: terminal pass lacks an exact hosted-run job locator"
            for record in payload["platforms"]
            if record["disposition"] == "included"
            and record["lifecycle"] == "passed"
            and not any(
                locator.startswith(job_prefix)
                for locator in record["evidence"]["locators"]
            )
        )


def _validate_repository_recipe(payload: dict[str, Any], errors: list[str]) -> None:
    recipe = payload["repository_recipe"]
    path = ROOT / recipe["path"]
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        errors.append(f"repository recipe cannot be read: {exc}")
        return
    if digest != recipe["sha256"]:
        errors.append(
            f"repository recipe digest is {digest}, expected {recipe['sha256']}"
        )
    if recipe["source_revision"] != payload["candidate"]["source_revision"]:
        errors.append("repository and external candidate source revisions differ")
    if recipe["sha256"] != payload["candidate"]["recipe_sha256"]:
        errors.append("repository and external candidate recipe digests differ")


def validate_contract(
    payload: dict[str, Any], schema_path: Path = DEFAULT_SCHEMA
) -> None:
    """Validate schema, platform reconciliation, filters, and evidence strength."""
    _validate_schema(payload, schema_path)
    errors: list[str] = []
    _validate_platform_identity(payload, errors)
    _validate_filters(payload, errors)
    _validate_evidence_layers(payload, errors)
    _validate_aggregates(payload, errors)
    _validate_hosted_state(payload, errors)
    _validate_repository_recipe(payload, errors)
    if errors:
        raise ContractError("; ".join(errors))


def build_parser() -> argparse.ArgumentParser:
    """Build the non-interactive validator command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    return parser


def main() -> int:
    """Validate one manifest and report a concise deterministic result."""
    args = build_parser().parse_args()
    try:
        payload = _load_object(args.manifest)
        validate_contract(payload, args.schema)
    except ContractError as exc:
        print(f"platform coverage contract invalid: {exc}", file=sys.stderr)
        return 1
    print(
        "platform coverage contract valid: "
        f"{payload['aggregates']['classified']} classified, "
        f"{payload['aggregates']['included']} included, "
        f"{payload['aggregates']['excluded']} excluded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
