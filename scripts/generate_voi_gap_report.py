#!/usr/bin/env python3
"""Generate the machine-readable VOI software and method gap report."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
LANDSCAPE = ROOT / "specs" / "software-landscape"
REGISTRY = LANDSCAPE / "registry.json"
METHODS = LANDSCAPE / "methods.json"
EVIDENCE = LANDSCAPE / "method-evidence.json"
IMPLEMENTATION_EVIDENCE = LANDSCAPE / "implementation-evidence.json"
OUTPUT = LANDSCAPE / "gap-report.json"

IMPLEMENTATION_TRACKS = {
    "stable": "stable_voi_rust_core_completion_20260723",
    "experimental": "supported_frontier_method_completion_20260723",
    "planned": "supported_frontier_method_completion_20260723",
}


def render() -> str:
    """Render routed feature and method gaps from canonical registries."""
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    methods = json.loads(METHODS.read_text(encoding="utf-8"))["methods"]
    evidence = {
        item["method_id"]: item
        for item in json.loads(EVIDENCE.read_text(encoding="utf-8"))["coverage"]
    }
    implementation_evidence = {
        item["method_id"]: item
        for item in json.loads(IMPLEMENTATION_EVIDENCE.read_text(encoding="utf-8"))[
            "records"
        ]
    }
    feature_states: Counter[str] = Counter()
    affected_tools: defaultdict[str, set[str]] = defaultdict(set)
    feature_gaps: list[dict[str, object]] = []
    for tool in registry["tools"]:
        if tool["scope"] != "external":
            continue
        for feature in tool["features"]:
            feature_states[feature["parity_state"]] += 1
            if feature["parity_state"] in {"native", "equivalent"}:
                continue
            for method_id in feature["method_ids"]:
                affected_tools[method_id].add(tool["id"])
            feature_gaps.append(
                {
                    "tool_id": tool["id"],
                    "feature_id": feature["id"],
                    "parity_state": feature["parity_state"],
                    "method_ids": feature["method_ids"],
                    "evidence": feature["evidence"],
                    "voiage_evidence": feature["voiage_evidence"],
                }
            )

    method_gaps = []
    for method in methods:
        method_evidence = evidence[method["id"]]
        implementation = implementation_evidence.get(method["id"])
        remaining_implementation_gate = (
            implementation["remaining_gate"] if implementation else "implementation"
        )
        if (
            method["voiage_state"] == "native"
            and method_evidence["promotion_gate"] == "none"
            and method["id"] not in affected_tools
            and remaining_implementation_gate == "none"
        ):
            continue
        method_gaps.append(
            {
                "method_id": method["id"],
                "maturity": method["maturity"],
                "voiage_state": method["voiage_state"],
                "review_state": method_evidence["review_state"],
                "promotion_gate": method_evidence["promotion_gate"],
                "authority_state": (
                    implementation["authority_state"]
                    if implementation
                    else "not-implemented"
                ),
                "remaining_implementation_gate": remaining_implementation_gate,
                "affected_tool_ids": sorted(affected_tools[method["id"]]),
                "owner_track": (
                    "ml_llm_agent_voi_20260723"
                    if method_evidence["family"]
                    in {
                        "information-theoretic-design-and-machine-learning",
                        "llm-rag-and-agent-applications",
                    }
                    else IMPLEMENTATION_TRACKS[method["maturity"]]
                ),
            }
        )

    payload = {
        "schema_version": "1.0.0",
        "generated_from": {
            "software_registry_schema_version": registry["schema_version"],
            "method_registry_schema_version": json.loads(
                METHODS.read_text(encoding="utf-8")
            )["schema_version"],
            "searched_on": registry["searched_on"],
            "review_due": registry["review_due"],
        },
        "summary": {
            "external_tools": sum(
                tool["scope"] == "external" for tool in registry["tools"]
            ),
            "canonical_methods": len(methods),
            "external_feature_states": dict(sorted(feature_states.items())),
            "open_feature_gaps": len(feature_gaps),
            "open_method_or_assurance_gaps": len(method_gaps),
        },
        "feature_gaps": sorted(
            feature_gaps, key=lambda item: (item["tool_id"], item["feature_id"])
        ),
        "method_gaps": sorted(method_gaps, key=lambda item: item["method_id"]),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    """Write the gap report or verify the checked-in projection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render()
    if args.check:
        return 0 if OUTPUT.read_text(encoding="utf-8") == rendered else 1
    OUTPUT.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
