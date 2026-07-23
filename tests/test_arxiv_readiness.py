import importlib.util
from pathlib import Path


def test_arxiv_readiness_pipeline_is_latex_first_and_non_submitting() -> None:
    root = Path.cwd()
    workflow = (root / ".github/workflows/arxiv-readiness.yml").read_text()
    manifest = (root / "paper/readiness-manifest.json").read_text()
    builder = (root / "scripts/build_arxiv_submission.py").read_text()
    readability = (root / "scripts/audit_arxiv_readability.py").read_text()
    main_tex = (root / "paper/main.tex").read_text()

    assert "\\documentclass" in main_tex
    assert '"canonical_source": "paper/main.tex"' in manifest
    assert '"submission_performed": false' in manifest
    assert '"sourceright": "submodule"' in manifest
    assert '"authentext": "submodule"' in manifest
    assert "workflow_dispatch:" in workflow
    assert "permissions: {}" in workflow
    assert "texlive: [2023, 2025]" in workflow
    assert "chktex lacheck latexmk" in workflow
    assert "prepare_arxiv_variants.py" in workflow
    assert "audit_arxiv_readability.py" in workflow
    assert "validate_arxiv_html.py" in workflow
    assert 'version: "0.11.29"' in workflow
    assert "submission_performed" in builder
    assert "pandoc" not in builder
    assert "arxiv.org" not in builder
    assert '"status": "review_only"' in readability
    assert "nltk.data.find" in readability
    assert "nltk.download" not in readability
    assert "fewer than 30 sentences" in readability


def test_arxiv_template_tools_are_pinned_and_registered() -> None:
    root = Path.cwd()
    requirements = (root / "requirements-arxiv.txt").read_text()
    modules = (root / ".gitmodules").read_text()

    assert "arxiv-collector==" in requirements
    assert "arxiv-latex-cleaner==" in requirements
    assert "textstat==0.7.13" in requirements
    assert "nltk==3.10.0" in requirements
    assert "pyphen==0.17.2" in requirements
    assert "edithatogo/sourceright.git" in modules
    assert "edithatogo/authentext.git" in modules


def test_readability_normalization_joins_pdf_line_break_hyphenation() -> None:
    root = Path.cwd()
    path = root / "scripts/audit_arxiv_readability.py"
    spec = importlib.util.spec_from_file_location("audit_arxiv_readability", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    normalized = module.normalize_text(
        "A reproducible inter-\nnational manuscript.\n\nSecond sentence."
    )
    assert normalized == "A reproducible international manuscript. Second sentence."
