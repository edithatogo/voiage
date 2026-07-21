"""Keep the Astro API reference aligned with the normative stable contract."""

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_astro_api_reference_covers_normative_contract() -> None:
    contract = json.loads(
        (ROOT / "specs/v1/stable-api.json").read_text(encoding="utf-8")
    )
    reference = (
        ROOT / "docs/astro-site/src/content/docs/api-reference/index.mdx"
    ).read_text(encoding="utf-8")

    assert "Rust owns numerical policy" in reference
    assert "absolute tolerance `1e-10`" in reference
    assert "relative tolerance" in reference
    assert "`1e-8`" in reference
    for symbol in contract["symbols"]["stable"]:
        assert f"`{symbol}`" in reference or symbol in reference
    for method in contract["methods"]:
        assert f"`{method}`" in reference
        assert contract["methods"][method]["implementation"] in reference.lower()
    for error in contract["errors"].values():
        assert f"`{error}`" in reference
