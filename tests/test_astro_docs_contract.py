"""Contract checks for the Astro/Starlight documentation authority."""

import json
from pathlib import Path
import re
from shutil import which
import subprocess

from packaging.version import Version
import yaml

ROOT = Path(__file__).parents[1]


def test_astro_site_is_the_active_docs_build() -> None:
    workflow = (ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8")
    manifest = (ROOT / "docs/astro-site/package.json").read_text(encoding="utf-8")

    assert "docs/astro-site" in workflow
    assert "pnpm run build" in workflow
    assert '"astro"' in manifest
    assert '"@astrojs/starlight"' in manifest
    assert "sphinx" not in workflow.lower()


def test_astro_site_uses_current_polyglot_stack() -> None:
    """The docs build pins the reviewed current stack and owner's plugin."""
    manifest = json.loads(
        (ROOT / "docs/astro-site/package.json").read_text(encoding="utf-8")
    )
    config = (ROOT / "docs/astro-site/astro.config.mjs").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8")
    modules = (ROOT / ".gitmodules").read_text(encoding="utf-8")

    dependencies = manifest["dependencies"]
    for package, floor, ceiling in (
        ("astro", "7.1.3", "8"),
        ("@astrojs/starlight", "0.41.4", "0.43"),
    ):
        pin = dependencies[package]
        assert re.fullmatch(r"\d+\.\d+\.\d+", pin), (package, pin)
        assert Version(floor) <= Version(pin) < Version(ceiling)
    lock = yaml.safe_load(
        (ROOT / "docs/astro-site/pnpm-lock.yaml").read_text(encoding="utf-8")
    )
    for package in ("astro", "@astrojs/starlight"):
        assert (
            lock["importers"]["."]["dependencies"][package]["specifier"]
            == dependencies[package]
        )
    assert dependencies["astro-polyglot"].startswith("link:")
    assert 'from "astro-polyglot"' in config
    assert "polyglot({" in config
    assert "submodules: recursive" in workflow
    assert ".repo-tools/astro-polyglot" in modules


def test_astro_release_build_has_no_package_age_delay() -> None:
    workspace = (ROOT / "docs/astro-site/pnpm-workspace.yaml").read_text(
        encoding="utf-8"
    )

    assert "minimumReleaseAge: 0" in workspace
    assert "allowBuilds:" in workspace
    assert "esbuild: true" in workspace
    assert "sharp: true" in workspace


def test_astro_content_has_no_legacy_rst_links() -> None:
    content_root = ROOT / "docs/astro-site/src/content/docs"
    stale_links = [
        path.relative_to(ROOT).as_posix()
        for path in content_root.rglob("*.mdx")
        if ".rst" in path.read_text(encoding="utf-8")
    ]
    assert not stale_links, f"Astro content contains legacy RST links: {stale_links}"


def test_user_documentation_exists_only_in_the_astro_content_tree() -> None:
    git = which("git")
    assert git is not None
    tracked = subprocess.run(
        [git, "ls-files", "docs/**/*.md", "docs/*.md"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.splitlines()
    legacy_user_docs = [
        path
        for path in tracked
        if (ROOT / path).exists()
        if not path.startswith("docs/astro-site/src/content/docs/")
        and not path.startswith("docs/release/")
        and not path.startswith("docs/reviews/")
    ]

    assert legacy_user_docs == []
