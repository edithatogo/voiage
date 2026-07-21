"""Keep the Astro C ABI reference aligned with the checked-in manifests."""

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_astro_c_abi_reference_covers_symbols_and_layouts() -> None:
    reference = (
        ROOT / "docs/astro-site/src/content/docs/reference/c-abi.mdx"
    ).read_text(encoding="utf-8")
    symbols = (ROOT / "specs/abi/v1/symbols.txt").read_text(encoding="utf-8")
    layouts = (ROOT / "specs/abi/v1/layouts.txt").read_text(encoding="utf-8")

    assert "header" in reference
    assert "authoritative" in reference
    for line in symbols.splitlines():
        if line and not line.startswith("#"):
            assert line in reference
    for line in layouts.splitlines():
        if line and not line.startswith("#"):
            type_name = line.split()[0].split(".")[0]
            assert type_name in reference
    for status in (
        "OK",
        "INVALID_ARGUMENT",
        "DIMENSION_MISMATCH",
        "BACKEND_UNAVAILABLE",
        "NUMERICAL_FAILURE",
        "SERIALIZATION_FAILURE",
        "BUFFER_TOO_SMALL",
        "PANIC",
        "INTERNAL_ERROR",
    ):
        assert status in reference
