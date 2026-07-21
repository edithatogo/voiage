"""Test the repository-owned Astro documentation link validator."""

from pathlib import Path

from scripts.validate_astro_docs import validate

ROOT = Path(__file__).resolve().parents[1]


def test_astro_documentation_links_are_valid() -> None:
    assert validate(ROOT) == []


def test_astro_documentation_validator_reports_missing_route(tmp_path: Path) -> None:
    content = tmp_path / "docs/astro-site/src/content/docs"
    content.mkdir(parents=True)
    (content / "index.mdx").write_text(
        "[missing](/does-not-exist/)\n", encoding="utf-8"
    )

    findings = validate(tmp_path)

    assert len(findings) == 1
    assert findings[0].message == "Astro route does not exist"
