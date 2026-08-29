"""Export installed MCDA-information schemas and refresh governed evidence pins."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from voiage.contracts.mcda_information import (
    MCDA_INFORMATION_INPUT_SCHEMA_V1,
    MCDA_INFORMATION_RESULT_SCHEMA_V1,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/mcda-information/v1"
SCHEMAS = CONTRACT / "schemas"
EVIDENCE_ARTIFACTS = (
    "scripts/export_mcda_information_contracts.py",
    "voiage/contracts/mcda_information.py",
    "voiage/methods/mcda_information.py",
    "voiage/plot/mcda_information.py",
    "tests/test_mcda_information_contract.py",
    "tests/test_mcda_information.py",
    "tests/test_mcda_information_surfaces.py",
    "specs/frontier/mcda-information/v1/README.md",
    "specs/frontier/mcda-information/v1/capabilities.json",
    "specs/frontier/mcda-information/v1/fixtures/manifest.json",
    "specs/frontier/mcda-information/v1/fixtures/normative/input.json",
    "specs/frontier/mcda-information/v1/fixtures/normative/expected.json",
    "specs/frontier/mcda-information/v1/fixtures/cases/probability-sum.json",
    "specs/frontier/mcda-information/v1/fixtures/cases/negative-weight.json",
    "specs/frontier/mcda-information/v1/fixtures/cases/unknown-partition.json",
    "specs/frontier/mcda-information/v1/fixtures/cases/post-information-normalization.json",
    "specs/frontier/mcda-information/v1/schemas/mcda-information-input.schema.json",
    "specs/frontier/mcda-information/v1/schemas/mcda-information-result.schema.json",
    "docs/astro-site/src/content/docs/examples/mcda-information-value.mdx",
    "conductor/archive/supported_frontier_method_completion_20260723/mcda-information-reference-review.md",
    "conductor/archive/supported_frontier_method_completion_20260723/mcda-information-implementation-review.md",
)


def main() -> None:
    """Write deterministic schema projections and immutable artifact digests."""
    SCHEMAS.mkdir(parents=True, exist_ok=True)
    schemas = {
        "mcda-information-input.schema.json": MCDA_INFORMATION_INPUT_SCHEMA_V1,
        "mcda-information-result.schema.json": MCDA_INFORMATION_RESULT_SCHEMA_V1,
    }
    for name, schema in schemas.items():
        (SCHEMAS / name).write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    evidence = {
        "artifacts": [
            {
                "path": path,
                "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
            }
            for path in EVIDENCE_ARTIFACTS
        ],
        "evidence_scope": (
            "Portable schemas, independently reviewed analytical fixture, pathology "
            "fixtures, exact Python evaluator, CLI, accessible plots and documentation."
        ),
        "execution_status": "experimental_python",
        "method_family": "finite_additive_mcda_perfect_information",
        "schema_version": "1.0.0",
        "stable_claim_allowed": False,
    }
    (CONTRACT / "fixtures/evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
