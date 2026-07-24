#!/usr/bin/env python3
"""Generate implementation evidence for every method claimed native."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
LANDSCAPE = ROOT / "specs" / "software-landscape"
METHODS = LANDSCAPE / "methods.json"
OUTPUT = LANDSCAPE / "implementation-evidence.json"

EVIDENCE = {
    "net-benefit": (
        "voiage/core/utils.py",
        "tests/test_utils_comprehensive.py",
        "python-stable-gap",
    ),
    "evpi": (
        "rust/crates/voiage-numerics/src/evpi.rs",
        "rust/crates/voiage-numerics/tests/foundational_kernels.rs",
        "rust-authoritative",
    ),
    "evppi": (
        "rust/crates/voiage-numerics/src/evppi.rs",
        "rust/crates/voiage-numerics/tests/evppi.rs",
        "rust-authoritative",
    ),
    "evsi": (
        "rust/crates/voiage-numerics/src/evsi.rs",
        "rust/crates/voiage-numerics/tests/evsi.rs",
        "rust-authoritative",
    ),
    "enbs": (
        "rust/crates/voiage-numerics/src/enbs.rs",
        "rust/crates/voiage-numerics/tests/foundational_kernels.rs",
        "rust-authoritative",
    ),
    "ceac": ("voiage/plot/ceac.py", "tests/test_plotting.py", "visualization-only"),
    "ceaf": (
        "rust/crates/voiage-numerics/src/ceaf.rs",
        "rust/crates/voiage-numerics/tests/ceaf.rs",
        "rust-authoritative",
    ),
    "voi-curve": (
        "voiage/plot/voi_curves.py",
        "tests/test_plotting.py",
        "visualization-only",
    ),
    "dominance": (
        "rust/crates/voiage-numerics/src/dominance.rs",
        "rust/crates/voiage-numerics/tests/dominance.rs",
        "rust-authoritative",
    ),
    "icer": (
        "voiage/methods/dominance.py",
        "tests/test_dominance.py",
        "related-analysis",
    ),
    "cea-plane": (
        "voiage/plot/dominance.py",
        "tests/test_plotting.py",
        "visualization-only",
    ),
    "evppi-regression": (
        "rust/crates/voiage-numerics/src/evppi.rs",
        "rust/crates/voiage-numerics/tests/evppi.rs",
        "rust-authoritative",
    ),
    "evsi-nested-mc": (
        "rust/crates/voiage-numerics/src/evsi.rs",
        "rust/crates/voiage-numerics/tests/evsi.rs",
        "rust-authoritative",
    ),
    "evsi-regression": (
        "rust/crates/voiage-numerics/src/evsi_regression.rs",
        "rust/crates/voiage-numerics/tests/evsi_regression.rs",
        "rust-authoritative",
    ),
    "evsi-moment-matching": (
        "rust/crates/voiage-numerics/src/evsi_moment.rs",
        "rust/crates/voiage-numerics/tests/evsi_moment.rs",
        "rust-authoritative",
    ),
    "directional-evop": (
        "voiage/methods/perspective.py",
        "tests/test_perspective.py",
        "python-experimental",
    ),
    "pairwise-evop": (
        "voiage/methods/perspective.py",
        "tests/test_perspective.py",
        "python-experimental",
    ),
    "perspective-optima": (
        "voiage/methods/perspective.py",
        "tests/test_perspective.py",
        "python-experimental",
    ),
    "perspective-switching": (
        "voiage/methods/perspective.py",
        "tests/test_perspective.py",
        "python-experimental",
    ),
    "structural-voi": (
        "rust/crates/voiage-numerics/src/structural.rs",
        "tests/test_structural.py",
        "rust-authoritative",
    ),
    "nma-voi": (
        "voiage/methods/network_meta_analysis.py",
        "tests/test_network_meta_analysis.py",
        "rust-backed-python-orchestration",
    ),
    "sequential-voi": (
        "voiage/methods/sequential.py",
        "tests/test_sequential.py",
        "python-experimental",
    ),
    "real-options-voi": (
        "voiage/methods/dynamic_real_options.py",
        "tests/test_dynamic_real_options.py",
        "python-experimental",
    ),
    "portfolio-voi": (
        "voiage/methods/portfolio.py",
        "tests/test_portfolio_comprehensive.py",
        "python-experimental",
    ),
    "heterogeneity-voi": (
        "voiage/methods/heterogeneity.py",
        "tests/test_heterogeneity.py",
        "python-experimental",
    ),
    "preference-voi": (
        "voiage/methods/preference.py",
        "tests/test_preference.py",
        "python-experimental",
    ),
    "equity-voi": (
        "voiage/methods/distributional.py",
        "tests/test_distributional.py",
        "python-experimental",
    ),
    "implementation-voi": (
        "voiage/methods/implementation.py",
        "tests/test_implementation.py",
        "python-experimental",
    ),
    "validation-voi": (
        "voiage/methods/validation.py",
        "tests/test_validation.py",
        "python-experimental",
    ),
    "causal-transportability-voi": (
        "voiage/methods/causal_transportability.py",
        "tests/test_causal_transportability.py",
        "python-experimental",
    ),
    "data-quality-voi": (
        "voiage/methods/data_quality.py",
        "tests/test_data_quality.py",
        "python-experimental",
    ),
    "monitoring-voi": (
        "voiage/methods/monitoring_surveillance.py",
        "tests/test_monitoring_surveillance.py",
        "python-experimental",
    ),
    "value-of-computation": (
        "voiage/methods/computational.py",
        "tests/test_computational.py",
        "python-experimental",
    ),
}


def render() -> str:
    """Render evidence records after checking the registry state."""
    methods = json.loads(METHODS.read_text(encoding="utf-8"))["methods"]
    native = {
        method["id"]: method for method in methods if method["voiage_state"] == "native"
    }
    if set(native) != set(EVIDENCE):
        missing = sorted(set(native) - set(EVIDENCE))
        stale = sorted(set(EVIDENCE) - set(native))
        raise ValueError(
            f"implementation evidence mismatch: missing={missing}, stale={stale}"
        )
    records = []
    for method_id in sorted(native):
        implementation_path, test_path, authority_state = EVIDENCE[method_id]
        for relative_path in (implementation_path, test_path):
            if not (ROOT / relative_path).is_file():
                raise FileNotFoundError(relative_path)
        records.append(
            {
                "method_id": method_id,
                "maturity": native[method_id]["maturity"],
                "authority_state": authority_state,
                "implementation_paths": [implementation_path],
                "test_paths": [test_path],
                "remaining_gate": (
                    "stable-rust-authority"
                    if authority_state == "python-stable-gap"
                    else "experimental-promotion"
                    if native[method_id]["maturity"] == "experimental"
                    else "none"
                ),
            }
        )
    payload = {"schema_version": "1.0.0", "records": records}
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    """Write the evidence registry or verify the checked-in projection."""
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
