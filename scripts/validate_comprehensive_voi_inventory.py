#!/usr/bin/env python3
"""Validate semantic invariants in the comprehensive VOI software inventory."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).parents[1]
LANDSCAPE = ROOT / "specs" / "software-landscape"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _evidence_references(version: dict[str, Any]) -> set[str]:
    references: set[str] = set()
    for surface in version["schema_surfaces"]:
        references.update(surface["evidence_ids"])
    for capability in version["capabilities"]:
        references.update(capability["evidence_ids"])
        for child in capability["subfeatures"]:
            references.update(child["evidence_ids"])
        for child in capability["options"]:
            references.update(child["evidence_ids"])
        for child in capability["defaults"]:
            references.update(child["evidence_ids"])
    for lesson in version["adoption_lessons"]:
        references.update(lesson["evidence_ids"])
    for coverage in version["extraction_coverage"]:
        references.update(coverage["evidence_ids"])
    return references


def validate(
    inventory: dict[str, Any],
    protocol: dict[str, Any],
    *,
    as_of: date,
) -> list[str]:
    """Return deterministic semantic validation errors."""
    errors: list[str] = []
    review_age = (as_of - date.fromisoformat(inventory["reviewed_on"])).days
    maximum_age = protocol["freshness"]["maximum_age_days"]
    if review_age < 0 or review_age > maximum_age:
        errors.append(
            f"inventory review age {review_age} is outside 0..{maximum_age} days"
        )
    if date.fromisoformat(inventory["review_due"]) < as_of:
        errors.append("inventory review_due is before the validation date")

    search_ids = {item["id"] for item in inventory["search_observations"]}
    product_ids = {item["id"] for item in inventory["products"]}
    required_dimensions = set(protocol["extraction"]["required_dimensions"])
    commercial_availability = {
        "public-documentation",
        "public-demonstration",
        "paid-or-private",
        "inaccessible",
    }
    commercial_strengths = {
        "version-pinned-documentation",
        "public-observation",
        "vendor-claim",
        "inaccessible",
    }
    strength_observability = {
        "executable-version-pinned-source-and-tests": "inspectable",
        "version-pinned-source": "inspectable",
        "version-pinned-documentation": "documented",
        "public-observation": "observed",
        "vendor-claim": "claimed",
        "inaccessible": "inaccessible",
    }

    for observation in inventory["search_observations"]:
        unknown = set(observation["candidate_product_ids"]) - product_ids
        if unknown:
            errors.append(
                f"{observation['id']}: unknown candidate products {sorted(unknown)}"
            )

    for product in inventory["products"]:
        product_id = product["id"]
        unknown_searches = (
            set(product["inclusion"]["search_observation_ids"]) - search_ids
        )
        if unknown_searches:
            errors.append(
                f"{product_id}: unknown search observations {sorted(unknown_searches)}"
            )
        duplicate = product["duplicate_resolution"]
        canonical_id = duplicate["canonical_product_id"]
        if duplicate["relation"] == "canonical" and canonical_id != product_id:
            errors.append(f"{product_id}: canonical relation must point to itself")
        if duplicate["relation"] != "canonical" and canonical_id not in product_ids:
            errors.append(f"{product_id}: duplicate relation lacks canonical product")

        rights = product["rights"]
        if (
            rights["review_state"] in {"unknown", "no-license"}
            and rights["source_reuse"] != "prohibited"
        ):
            errors.append(f"{product_id}: unknown/no-license source reuse prohibited")
        if product["category"] in {"commercial", "hosted"}:
            if product["availability"] not in commercial_availability:
                errors.append(f"{product_id}: commercial availability is over-claimed")
            if rights["source_reuse"] == "permitted":
                errors.append(f"{product_id}: commercial source reuse is over-claimed")

        for version in product["versions"]:
            version_id = version["id"]
            evidence = {
                observation["id"]: observation
                for observation in version["evidence_observations"]
            }
            unknown_evidence = _evidence_references(version) - evidence.keys()
            if unknown_evidence:
                errors.append(
                    f"{product_id}/{version_id}: unknown evidence "
                    f"{sorted(unknown_evidence)}"
                )
            for observation in evidence.values():
                expected_observability = strength_observability[observation["strength"]]
                if observation["observability"] != expected_observability:
                    errors.append(
                        f"{product_id}/{version_id}/{observation['id']}: "
                        "evidence strength and observability disagree"
                    )
                if observation["rights_record_id"] != product_id:
                    errors.append(
                        f"{product_id}/{version_id}/{observation['id']}: "
                        "rights record does not match product"
                    )
                if (
                    product["category"] in {"commercial", "hosted"}
                    and observation["strength"] not in commercial_strengths
                ):
                    errors.append(
                        f"{product_id}/{version_id}/{observation['id']}: "
                        "commercial evidence exceeds observability ceiling"
                    )

            coverage = {
                item["dimension"]: item for item in version["extraction_coverage"]
            }
            missing = required_dimensions - coverage.keys()
            extra = coverage.keys() - required_dimensions
            if missing or extra:
                errors.append(
                    f"{product_id}/{version_id}: extraction coverage mismatch; "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
            for dimension, item in coverage.items():
                if item["status"] == "observed" and not item["evidence_ids"]:
                    errors.append(
                        f"{product_id}/{version_id}/{dimension}: "
                        "observed coverage requires evidence"
                    )
    return sorted(errors)


def main() -> int:
    """Run schema and semantic validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        type=Path,
        default=LANDSCAPE / "comprehensive-inventory.json",
    )
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()

    inventory = _read_json(args.inventory)
    schema = _read_json(LANDSCAPE / "comprehensive-inventory.schema.json")
    protocol = _read_json(LANDSCAPE / "review-protocol.json")
    schema_errors = sorted(
        Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
        ).iter_errors(inventory),
        key=lambda error: list(error.absolute_path),
    )
    if schema_errors:
        for error in schema_errors:
            print(f"ERROR schema {list(error.absolute_path)}: {error.message}")
        return 1

    errors = validate(inventory, protocol, as_of=args.as_of)
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 1
    print(
        "Comprehensive VOI inventory validation passed: "
        f"{len(inventory['products'])} products, "
        f"{len(inventory['search_observations'])} search observations."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
