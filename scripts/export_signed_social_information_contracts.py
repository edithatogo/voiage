"""Export signed-social information schemas and deterministic fixtures."""

# pyright: reportAny=false, reportExplicitAny=false

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import typing

from voiage.contracts.signed_social_information import (
    SIGNED_SOCIAL_INFORMATION_INPUT_SCHEMA_V1,
    SIGNED_SOCIAL_INFORMATION_RESULT_SCHEMA_V1,
)
from voiage.methods.signed_social_information import signed_social_information_value

ROOT = Path(__file__).parents[1]
TARGET = ROOT / "specs/frontier/signed-social-information/v1"
SOURCE_FIXTURE = (
    ROOT
    / "tests/fixtures/signed_social_information"
    / "li_pozzi_harmful_private_positive_social.json"
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Write schemas, normative fixture, manifest and artifact hashes."""
    input_schema = TARGET / "schemas/input.schema.json"
    result_schema = TARGET / "schemas/result.schema.json"
    fixture_input = TARGET / "fixtures/normative/input.json"
    fixture_expected = TARGET / "fixtures/normative/expected.json"
    _write_json(input_schema, SIGNED_SOCIAL_INFORMATION_INPUT_SCHEMA_V1)
    _write_json(result_schema, SIGNED_SOCIAL_INFORMATION_RESULT_SCHEMA_V1)
    fixture_input.parent.mkdir(parents=True, exist_ok=True)
    _ = shutil.copyfile(SOURCE_FIXTURE, fixture_input)
    specification = typing.cast(
        "dict[str, typing.Any]", json.loads(fixture_input.read_text(encoding="utf-8"))
    )
    _write_json(
        fixture_expected,
        signed_social_information_value(specification).to_contract_dict(),
    )
    fixture_files = [
        ("normative/input.json", fixture_input),
        ("normative/expected.json", fixture_expected),
        ("../schemas/input.schema.json", input_schema),
        ("../schemas/result.schema.json", result_schema),
    ]
    manifest = {
        "version": "v1",
        "status": "fixture-backed",
        "method": "signed_social_information_value",
        "authority": "exact independent finite-world enumeration and pathology mutations",
        "normative": [
            {
                "name": "Li-Pozzi harmful-private positive-social construction",
                "input_artifact": "normative/input.json",
                "expected_output_artifact": "normative/expected.json",
            }
        ],
        "pathologies": [],
        "files": [
            {"path": relative, "sha256": _sha256(path)}
            for relative, path in fixture_files
        ],
    }
    manifest_path = TARGET / "fixtures/manifest.json"
    _write_json(manifest_path, manifest)
    artifacts = [
        TARGET / "README.md",
        TARGET / "capabilities.json",
        manifest_path,
        fixture_input,
        fixture_expected,
        input_schema,
        result_schema,
        ROOT
        / "docs/astro-site/src/content/docs/examples/signed-social-information-value.mdx",
    ]
    _write_json(
        TARGET / "fixtures/evidence.json",
        {
            "schema_version": "signed-social-information-evidence-v1",
            "artifacts": [
                {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
                for path in artifacts
            ],
        },
    )


if __name__ == "__main__":
    main()
