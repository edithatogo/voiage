"""Contract checks for the Astro/Starlight documentation authority."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_astro_site_is_the_active_docs_build() -> None:
    workflow = (ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8")
    manifest = (ROOT / "docs/astro-site/package.json").read_text(encoding="utf-8")

    assert "docs/astro-site" in workflow
    assert "pnpm run build" in workflow
    assert '"astro"' in manifest
    assert '"@astrojs/starlight"' in manifest
    assert "sphinx" not in workflow.lower()


def test_astro_content_has_no_legacy_rst_links() -> None:
    content_root = ROOT / "docs/astro-site/src/content/docs"
    stale_links = [
        path.relative_to(ROOT).as_posix()
        for path in content_root.rglob("*.mdx")
        if ".rst" in path.read_text(encoding="utf-8")
    ]
    assert not stale_links, f"Astro content contains legacy RST links: {stale_links}"
