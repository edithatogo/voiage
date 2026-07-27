#!/usr/bin/env python3
"""Generate deterministic Phase 3 capability-map summary views."""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parents[1]
LANDSCAPE = ROOT / "specs" / "software-landscape"
SOURCE = LANDSCAPE / "capability-adoption-map.json"
OUTPUT = LANDSCAPE / "capability-adoption-views.json"

def generate() -> dict[str, object]:
    data = json.loads(SOURCE.read_text())
    records = data["records"]
    by_product: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_product[record["product_id"]].append(record["capability_id"])
    return {"schema_version": "1.0.0", "source_map": SOURCE.name,
            "record_count": len(records),
            "by_parity_state": dict(sorted(Counter(r["parity_state"] for r in records).items())),
            "by_capability_kind": dict(sorted(Counter(r["capability_kind"] for r in records).items())),
            "by_product": {key: sorted(value) for key, value in sorted(by_product.items())}}

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--check", action="store_true"); args=parser.parse_args()
    rendered=json.dumps(generate(), indent=2)+"\n"
    if args.check: return 0 if OUTPUT.read_text()==rendered else 1
    OUTPUT.write_text(rendered); return 0
if __name__ == "__main__": raise SystemExit(main())
