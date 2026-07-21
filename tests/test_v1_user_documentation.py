"""Ensure the Astro v1 readiness page covers the required user topics."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs/astro-site/src/content/docs/user-guide/v1-release-readiness.mdx"


def test_v1_readiness_page_covers_required_user_topics() -> None:
    text = PAGE.read_text(encoding="utf-8")
    for heading in (
        "## Installation",
        "## Concepts and first analysis",
        "## Compatibility and migration",
        "## Security and reproducibility",
        "## Extension maturity and support",
    ):
        assert heading in text


def test_v1_readiness_page_uses_astro_internal_links() -> None:
    text = PAGE.read_text(encoding="utf-8")
    assert ".rst" not in text
    for link in (
        "/getting-started/",
        "/introduction/",
        "/api-reference/",
        "/user-guide/migration-guide/",
        "/reference/bindings/",
    ):
        assert link in text
