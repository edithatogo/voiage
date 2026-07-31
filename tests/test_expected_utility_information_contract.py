from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

ROOT = (
    Path(__file__).parents[1] / "specs/frontier/expected-utility-information-pricing/v1"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_normative_requests_validate_and_manifest_hashes_match() -> None:
    schema = _load(ROOT / "schemas/request.schema.json")
    manifest = _load(ROOT / "fixtures/manifest.json")
    assert manifest["method_maturity"] == "experimental"
    for record in manifest["fixtures"]:
        path = ROOT / "fixtures" / record["path"]
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        jsonschema.Draft202012Validator(schema).validate(json.loads(payload)["request"])


def test_request_contract_rejects_unknown_fields() -> None:
    schema = _load(ROOT / "schemas/request.schema.json")
    request = _load(ROOT / "fixtures/normative/affine-clairvoyant.json")["request"]
    request["undeclared"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(request)


def test_result_contract_has_no_duplicate_voc_scalar() -> None:
    schema = _load(ROOT / "schemas/result.schema.json")
    properties = schema["properties"]
    assert "voc" not in properties
    assert "presentation" in properties
    assert (
        properties["schema_version"]["const"]
        == "expected-utility-information-result-v1"
    )
