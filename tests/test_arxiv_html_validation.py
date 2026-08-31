"""LaTeXML's zero exit status must not hide an omitted manuscript figure."""

from pathlib import Path

import pytest

from scripts.validate_arxiv_html import main, validate_html


def _html(tmp_path: Path, image: str) -> Path:
    path = tmp_path / "main.html"
    path.write_text(
        f"<html><head><title>Paper</title></head><body><h1>Abstract</h1>"
        f"{image}<h2>References</h2></body></html>",
        encoding="utf-8",
    )
    return path


def test_complete_html_requires_a_present_local_graphic(tmp_path: Path) -> None:
    (tmp_path / "figure.png").write_bytes(b"fixture")
    path = _html(tmp_path, '<img class="ltx_graphics" src="figure.png">')
    assert validate_html(path) == []


@pytest.mark.parametrize(
    "image",
    [
        '<img src="" class="ltx_graphics ltx_missing ltx_missing_image">',
        '<img class="ltx_graphics">',
        '<img class="ltx_graphics" src="missing.png">',
        '<img class="ltx_graphics" src="https://example.org/image.png">',
        '<img class="ltx_graphics" src="../outside.png">',
        "",
    ],
)
def test_missing_or_nonportable_graphics_fail_closed(
    tmp_path: Path, image: str
) -> None:
    assert validate_html(_html(tmp_path, image))


def test_semantic_errors_are_not_hidden_by_a_valid_figure(tmp_path: Path) -> None:
    (tmp_path / "figure.png").write_bytes(b"fixture")
    path = tmp_path / "main.html"
    path.write_text('<img class="ltx_graphics" src="figure.png">ltx_ERROR')
    errors = validate_html(path)
    assert len(errors) == 6
    assert "LaTeXML emitted an error marker" in errors


@pytest.mark.parametrize("complete", [True, False])
def test_cli_reports_validation_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    complete: bool,
) -> None:
    (tmp_path / "figure.png").write_bytes(b"fixture" if complete else b"")
    path = _html(tmp_path, '<img class="ltx_graphics" src="figure.png">')
    monkeypatch.setattr("sys.argv", ["validate_arxiv_html.py", str(path)])
    if complete:
        main()
        assert "validation: pass" in capsys.readouterr().out
    else:
        with pytest.raises(SystemExit, match="nonempty local artifact"):
            main()
