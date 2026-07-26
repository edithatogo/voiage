"""Contract checks for the supported public Rust facade."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib

from jsonschema import Draft202012Validator

ROOT = Path(__file__).parents[1]
CONTRACT_PATH = ROOT / "specs/v1/rust-facade.json"
SCHEMA_PATH = ROOT / "specs/v1/rust-facade.schema.json"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_rust_facade_contract_conforms_to_schema() -> None:
    Draft202012Validator(_load_json(SCHEMA_PATH)).validate(_load_json(CONTRACT_PATH))


def test_rust_facade_manifest_matches_contract() -> None:
    contract = _load_json(CONTRACT_PATH)
    manifest_path = ROOT / contract["package"]["manifest"]
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    dependencies = set(manifest["dependencies"])

    assert manifest["package"]["name"] == contract["package"]["name"]
    assert manifest["package"]["version"]["workspace"] is True
    assert manifest["package"]["rust-version"]["workspace"] is True
    assert dependencies.isdisjoint(contract["forbidden_dependencies"])
    assert dependencies == {
        f"voiage-{namespace}" for namespace in contract["public_namespaces"]
    }


def test_rust_facade_is_published_last_without_ffi_or_python() -> None:
    contract = _load_json(CONTRACT_PATH)
    workflow = (ROOT / ".github/workflows/rust-crates-release.yml").read_text(
        encoding="utf-8"
    )
    facade_publish = "cargo publish --locked --package voiage"

    assert contract["publication"]["repository_ready"] is True
    assert facade_publish in workflow
    assert workflow.rfind(facade_publish) > workflow.rfind(
        "cargo publish --locked --package voiage-serialization"
    )
    assert "cargo publish --locked --package voiage-ffi" not in workflow
    assert "cargo publish --locked --package voiage-python" not in workflow
