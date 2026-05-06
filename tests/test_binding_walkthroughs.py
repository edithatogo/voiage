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
        (
            Path("bindings/go/README.md"),
            (
                "## Setup",
                "## First workflow",
                "## Release and caveats",
                "go test ./...",
                "voiage.EVPI(",
            ),
        ),
        (
            Path("bindings/rust/README.md"),
            (
                "## Setup",
                "## First workflow",
                "## Release and caveats",
                "cargo test",
                "use voiage_core::evpi",
            ),
        ),
        (
            Path("bindings/typescript/README.md"),
            (
                "## Setup",
                "## First workflow",
                "## Release and caveats",
                "evpi(",
                "npm",
            ),
        ),
        (
            Path("bindings/dotnet/README.md"),
            (
                "## Setup",
                "## First workflow",
                "## Release and caveats",
                "InformationValue.Evpi",
                "net11.0",
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
        Path("bindings/go/README.md"): ("Go Binding", "go vet ./..."),
        Path("bindings/rust/README.md"): (
            "use voiage_core::evpi",
            "cargo package --locked --allow-dirty",
        ),
        Path("bindings/typescript/README.md"): ("@voiage/core", "npm run check"),
        Path("bindings/dotnet/README.md"): ("Voiage.Core", "dotnet pack"),
    }

    for readme, phrases in expectations.items():
        text = readme.read_text(encoding="utf-8")
        for phrase in phrases:
            assert phrase in text, f"expected {readme} to mention {phrase!r}"
