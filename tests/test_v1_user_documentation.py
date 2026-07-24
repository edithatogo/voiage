"""Ensure the Astro v1 readiness page covers the required user topics."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs/astro-site/src/content/docs/user-guide/v1-release-readiness.mdx"
MIGRATION_PAGE = (
    ROOT / "docs/astro-site/src/content/docs/user-guide/migration-guide.mdx"
)
VERSION_POLICY_PAGE = (
    ROOT
    / "docs/astro-site/src/content/docs/developer-guide/versioning-and-release-policy.mdx"
)


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
        "/voiage/getting-started",
        "/voiage/introduction",
        "/voiage/api-reference",
        "/voiage/user-guide/migration-guide",
        "/voiage/reference/bindings",
    ):
        assert link in text


def test_v1_migration_guide_covers_the_0x_compatibility_boundary() -> None:
    text = MIGRATION_PAGE.read_text(encoding="utf-8")

    for required in (
        "## Migrating from 0.x to 1.0",
        "Python 3.12",
        "Rust core is required",
        "ParameterSet",
        "deprecated_raw_dict_parameter_samples",
        "FutureWarning",
        "BackendNotAvailableError",
        "## Rollback and support",
    ):
        assert required in text


def test_version_policy_describes_the_implemented_warning_bridge() -> None:
    text = VERSION_POLICY_PAGE.read_text(encoding="utf-8")

    assert "deprecated_raw_dict_parameter_samples" in text
    assert "FutureWarning" in text
    assert "planned compatibility-bridge phase" not in text
