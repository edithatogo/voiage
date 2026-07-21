from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("readme", "required_phrases"),
    [
        (
            Path("bindings/julia/README.md"),
            (
                "## Setup",
                "## First workflow",
                "## Release and caveats",
                "using Voiage",
                "evpi(",
            ),
        ),
    ],
)
def test_binding_walkthrough_readmes_cover_the_core_tutorial_surface(
    readme: Path,
    required_phrases: tuple[str, ...],
) -> None:
    text = readme.read_text(encoding="utf-8")

    assert readme.exists(), f"expected {readme} to exist"
    for phrase in required_phrases:
        assert phrase in text, f"expected {readme} to mention {phrase!r}"


def test_binding_walkthrough_readmes_are_language_appropriate() -> None:
    expectations = {
        Path("bindings/julia/README.md"): ("using Voiage", "Pkg.test()"),
    }

    for readme, phrases in expectations.items():
        text = readme.read_text(encoding="utf-8")
        for phrase in phrases:
            assert phrase in text, f"expected {readme} to mention {phrase!r}"
