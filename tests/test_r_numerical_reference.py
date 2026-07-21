"""Keep the packaged R numerical oracle identical to the canonical fixture."""

import json
from pathlib import Path


def test_r_package_carries_canonical_evpi_reference() -> None:
    root = Path(__file__).resolve().parents[1]
    canonical = root / "specs/numerical-reference/v1/evpi-cases.json"
    packaged = root / "r-package/voiageR/inst/extdata/evpi-cases.json"

    assert json.loads(packaged.read_text(encoding="utf-8")) == json.loads(
        canonical.read_text(encoding="utf-8")
    )
