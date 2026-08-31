"""Ensure the R stable adapter does not require the Python bridge to install."""

from pathlib import Path
import re

from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def test_rust_backed_r_adapter_keeps_reticulate_optional() -> None:
    description = (ROOT / "r-package/voiageR/DESCRIPTION").read_text(encoding="utf-8")
    imports = (
        description.split("Imports:", 1)[1].split("Suggests:", 1)[0]
        if "Imports:" in description
        else ""
    )
    assert "reticulate" not in imports
    suggests = description.split("Suggests:\n", 1)[1].split("\nVignetteBuilder:", 1)[0]
    requirement = re.search(r"reticulate \(>= (\d+(?:\.\d+)+)\)", suggests)
    assert requirement is not None
    assert Version(requirement[1]) >= Version("1.20")
