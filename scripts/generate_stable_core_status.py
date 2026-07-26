#!/usr/bin/env python3
"""Generate the stable-core maturity and v1.1 assurance status registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "specs/v1/stable-core-status.json"
SOURCES = (
    "specs/v1/stable-estimator-assurance.json",
    "specs/v1/stable-estimator-statistical-assurance.json",
    "specs/v1/stable-core-validation-evidence.json",
)

OPEN_REPOSITORY_GATES = [
    "replication-and-convergence-assurance",
    "matched-cross-language-performance",
    "dependency-frontier-remediation",
]
HUMAN_GATES = [
    "phase-1-manual-verification",
    "phase-2-manual-verification",
    "phase-3-manual-verification",
    "phase-4-manual-verification",
]


def _load(relative_path: str) -> dict[str, Any]:
    with (ROOT / relative_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _runtime_state(method_id: str, reporting_class: str) -> str:
    if method_id in {"net-benefit", "dominance"}:
        return "not-applicable-deterministic"
    if method_id in {
        "evsi-nested-mc",
        "evsi-regression",
        "evsi-moment-matching",
    }:
        return "integrated-replication-capable"
    if method_id in {
        "expected-loss",
        "evpi",
        "evppi-regression",
        "evsi-nested-mc",
        "evsi-regression",
        "evsi-moment-matching",
        "ceaf",
        "structural-voi",
    }:
        return "integrated-incomplete-assurance"
    if method_id == "enbs":
        return "inherits-open-evsi-gate"
    if reporting_class == "deterministic":
        raise ValueError(f"unclassified deterministic assurance profile: {method_id}")
    return "contract-defined-runtime-not-integrated"


def generate_status() -> dict[str, Any]:
    """Build status from the three normative stable-core contracts."""
    estimator = _load(SOURCES[0])
    statistical = _load(SOURCES[1])
    validation = _load(SOURCES[2])

    statistical_by_id = {
        profile["method_id"]: profile for profile in statistical["profiles"]
    }
    validation_by_id = {method["method_id"]: method for method in validation["methods"]}
    estimator_ids = {profile["method_id"] for profile in estimator["profiles"]}
    if estimator_ids != set(statistical_by_id) or estimator_ids != set(
        validation_by_id
    ):
        raise ValueError("stable-core source contracts cover different methods")

    methods: list[dict[str, Any]] = []
    for profile in estimator["profiles"]:
        method_id = profile["method_id"]
        statistical_profile = statistical_by_id[method_id]
        validation_profile = validation_by_id[method_id]
        runtime_state = _runtime_state(
            method_id, statistical_profile["reporting_class"]
        )
        open_gates: list[str] = []
        if profile["implementation_state"] != "conformant":
            open_gates.append("estimator-assurance-evidence")
        if runtime_state == "inherits-open-evsi-gate":
            open_gates.append("upstream-evsi-assurance")
        elif runtime_state == "integrated-incomplete-assurance":
            open_gates.append("replication-and-convergence-assurance")
        elif runtime_state == "contract-defined-runtime-not-integrated":
            open_gates.append("runtime-statistical-envelope-integration")

        methods.append(
            {
                "api_maturity": profile["maturity"],
                "authority_boundary": validation_profile["implementation_boundary"],
                "implementation_state": profile["implementation_state"],
                "method_id": method_id,
                "open_gates": open_gates,
                "reporting_class": statistical_profile["reporting_class"],
                "runtime_assurance_state": runtime_state,
                "validation_state": "analytical-independent-metamorphic",
                "v1_1_assurance_state": ("eligible" if not open_gates else "open"),
            }
        )

    return {
        "aggregate": {
            "human_gates": HUMAN_GATES,
            "open_method_gates": [
                method["method_id"]
                for method in methods
                if method["v1_1_assurance_state"] == "open"
            ],
            "open_repository_gates": OPEN_REPOSITORY_GATES,
            "v1_1_release_ready": False,
        },
        "contract_version": "1.0.0",
        "generated_from": list(SOURCES),
        "methods": methods,
        "status": "generated-evidence",
    }


def _render(status: dict[str, Any]) -> str:
    return json.dumps(status, indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Write the registry or verify that the checked-in copy is current."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked-in generated registry is absent or stale",
    )
    args = parser.parse_args()
    rendered = _render(generate_status())

    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != rendered:
            print(
                f"{OUTPUT.relative_to(ROOT)} is missing or stale; "
                "run scripts/generate_stable_core_status.py",
                file=sys.stderr,
            )
            return 2
        return 0

    OUTPUT.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
