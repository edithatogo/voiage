#!/usr/bin/env python3
"""Generate the Phase 3 capability/adoption map from the frozen inventory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
LANDSCAPE = ROOT / "specs" / "software-landscape"
INVENTORY = LANDSCAPE / "comprehensive-inventory.json"
OUTPUT = LANDSCAPE / "capability-adoption-map.json"


def generate() -> dict[str, object]:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    records = []
    for product in sorted(inventory["products"], key=lambda item: item["id"]):
        for version in sorted(product["versions"], key=lambda item: item["id"]):
            lessons = sorted(item["id"] for item in version["adoption_lessons"])
            for capability in sorted(version["capabilities"], key=lambda item: item["id"]):
                records.append({
                    "product_id": product["id"], "version_id": version["id"],
                    "capability_id": capability["id"], "capability_kind": capability["kind"],
                    "canonical_ids": capability["canonical_ids"],
                    "parity_state": capability["parity_state"],
                    "evidence_ids": sorted(capability["evidence_ids"]),
                    "adoption_lessons": lessons,
                })
    return {"schema_version": "1.0.0", "source_inventory": "comprehensive-inventory.json",
            "reviewed_on": inventory["reviewed_on"],
            "parity_states": ["native", "equivalent", "adapter", "planned", "excluded", "not-reproducible"],
            "records": records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(generate(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        return 0 if OUTPUT.read_text(encoding="utf-8") == rendered else 1
    OUTPUT.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
