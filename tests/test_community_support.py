from __future__ import annotations

from pathlib import Path


def test_community_support_surface_exists() -> None:
    """The repository should expose a clear community support surface."""
    required_paths = [
        Path("SUPPORT.md"),
        Path("CODE_OF_CONDUCT.md"),
        Path("SECURITY.md"),
        Path("docs/developer_guide/community_support.rst"),
        Path(".github/ISSUE_TEMPLATE/question.md"),
    ]
    for path in required_paths:
        assert path.is_file(), f"expected community support document at {path}"


def test_community_support_links_are_referenced_from_main_docs() -> None:
    """The main docs should point at the support and governance surface."""
    readme = Path("README.md").read_text(encoding="utf-8")
    contributing = Path("CONTRIBUTING.md").read_text(encoding="utf-8")
    developer_index = Path("docs/developer_guide/index.rst").read_text(encoding="utf-8")

    for needle in (
        "SUPPORT.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "community_support",
    ):
        assert needle in readme or needle in contributing or needle in developer_index
