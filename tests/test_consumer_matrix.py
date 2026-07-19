from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from scripts.check_consumer_matrix import evaluate_matrix

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "specs/integration/vop-voiage/bundles/consumer-matrix.json"


def test_current_n_minus_one_and_incompatible_matrix_is_independent() -> None:
    evidence = evaluate_matrix(ROOT, MATRIX)
    assert evidence["passed"] is True
    assert [case["actual"] for case in evidence["cases"]] == [
        "backward_compatible",
        "identity_compatible",
        "incompatible",
    ]
    assert "vop" not in {name.split(".")[0] for name in sys.modules}


@pytest.mark.parametrize("field", ["pin_sha256", "descriptor_sha256"])
def test_matrix_content_addresses_every_external_input(
    tmp_path: Path, field: str
) -> None:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    section, key = (
        ("bundle", "pin_sha256") if field == "pin_sha256" else ("descriptors", "sha256")
    )
    matrix[section][key] = "0" * 64
    target = tmp_path / "matrix.json"
    target.write_text(json.dumps(matrix), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        evaluate_matrix(ROOT, target)
