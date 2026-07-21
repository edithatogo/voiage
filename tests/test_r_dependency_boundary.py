"""Ensure the R stable adapter does not require the Python bridge to install."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_rust_backed_r_adapter_keeps_reticulate_optional() -> None:
    description = (ROOT / "r-package/voiageR/DESCRIPTION").read_text(encoding="utf-8")
    imports = (
        description.split("Imports:", 1)[1].split("Suggests:", 1)[0]
        if "Imports:" in description
        else ""
    )
    assert "reticulate" not in imports
    assert "reticulate (>= 1.20)" in description
