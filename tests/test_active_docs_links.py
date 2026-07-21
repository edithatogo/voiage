"""Keep public repository entry points on the Astro documentation site."""

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_public_entry_points_do_not_link_to_legacy_rst_docs() -> None:
    for relative_path in ("README.md", "CONTRIBUTING.md"):
        text = (ROOT / relative_path).read_text(encoding="utf-8").lower()
        assert "docs/" not in text or ".rst" not in text
