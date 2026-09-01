import ast
import importlib.util
import json
from pathlib import Path
import re
import tarfile

from packaging.version import Version


def test_arxiv_readiness_pipeline_is_latex_first_and_non_submitting() -> None:
    root = Path.cwd()
    workflow = (root / ".github/workflows/arxiv-readiness.yml").read_text()
    manifest = (root / "paper/readiness-manifest.json").read_text()
    builder = (root / "scripts/build_arxiv_submission.py").read_text()
    readability = (root / "scripts/audit_arxiv_readability.py").read_text()
    main_tex = (root / "paper/main.tex").read_text()

    assert not (root / "paper/paper.tex").exists()
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
    disposition = json.loads(
        (
            root
            / "conductor/tracks/remaining_backlog_delivery_20260831"
            / "nltk-residual-advisory-20260902.json"
        ).read_text()
    )
    selected_version = Version(disposition["selected_version"])

    assert "arxiv-collector==" in requirements
    assert "arxiv-latex-cleaner==" in requirements
    assert "textstat==0.7.13" in requirements
    nltk_versions = []
    for filename in ("requirements-arxiv.txt", "requirements-joss.txt"):
        lines = (root / filename).read_text().splitlines()
        nltk_lines = [line for line in lines if line.lower().startswith("nltk")]
        assert len(nltk_lines) == 1
        pin = re.fullmatch(r"nltk==(3\.10\.\d+)", nltk_lines[0])
        assert pin is not None
        version = Version(pin[1])
        assert version == selected_version
        nltk_versions.append(version)
    assert nltk_versions[0] == nltk_versions[1]
    assert "pyphen==0.17.2" in requirements
    assert "edithatogo/sourceright.git" in modules
    assert "edithatogo/authentext.git" in modules


def test_readability_nltk_residual_advisory_is_bounded() -> None:
    root = Path.cwd()
    disposition = json.loads(
        (
            root
            / "conductor/tracks/remaining_backlog_delivery_20260831"
            / "nltk-residual-advisory-20260902.json"
        ).read_text()
    )

    assert disposition["advisory"] == "GHSA-8mgp-746c-j5xp"
    assert disposition["affected_versions"] == "<=3.10.3"
    assert disposition["patched_release"] is None
    assert disposition["selected_version"] == "3.10.3"
    assert disposition["scope"] == [
        "scripts/audit_arxiv_readability.py",
        "scripts/audit_joss_readability.py",
    ]

    expected_affected_calls = {
        "TransitionParser.train",
        "TransitionParser.parse",
        "AveragedPerceptron.save",
        "AveragedPerceptron.load",
        "PerceptronTagger.save_to_json",
        "save_maxent_params",
    }
    expected_affected_identifiers = {
        "TransitionParser",
        "AveragedPerceptron",
        "PerceptronTagger",
        "save_maxent_params",
    }
    assert set(disposition["affected_model_persistence_calls"]) == (
        expected_affected_calls
    )
    assert set(disposition["affected_api_identifiers"]) == (
        expected_affected_identifiers
    )

    observed_calls: set[str] = set()
    for filename in disposition["scope"]:
        source = (root / filename).read_text()
        for identifier in expected_affected_identifiers:
            assert identifier not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            parts: list[str] = []
            target = node.func
            while isinstance(target, ast.Attribute):
                parts.append(target.attr)
                target = target.value
            if isinstance(target, ast.Name):
                parts.append(target.id)
            if parts:
                observed_calls.add(".".join(reversed(parts)))
    assert expected_affected_calls.isdisjoint(observed_calls)
    assert disposition["upgrade_rule"] == (
        "Update both exact pins together to the first stable patched release, "
        "then refresh this disposition and its callsite audit."
    )


def test_manuscript_uses_citation_order_and_cross_referenced_end_matter() -> None:
    root = Path.cwd()
    main_tex = (root / "paper/main.tex").read_text()
    summary = (root / "paper/sections/summary.tex").read_text()
    glossary = (root / "paper/sections/glossary.tex").read_text()
    abbreviations = (root / "paper/sections/abbreviations.tex").read_text()

    assert "\\bibliographystyle{unsrt}" in main_tex
    assert "\\input{sections/glossary}" in main_tex
    assert "\\input{sections/abbreviations}" in main_tex
    assert "\\label{sec:glossary}" in glossary
    assert "\\label{sec:abbreviations}" in abbreviations
    assert "\\ref{sec:glossary}" in summary
    assert "\\ref{sec:abbreviations}" in summary
    normalized_summary = " ".join(summary.split())
    for expansion in (
        "Value of Information (VOI)",
        "expected value of perfect information (EVPI)",
        "expected value of partial perfect information (EVPPI)",
        "expected value of sample information (EVSI)",
        "expected net benefit of sampling (ENBS)",
        "cost-effectiveness acceptability frontier (CEAF)",
    ):
        assert expansion in normalized_summary


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


def test_arxiv_source_archive_contains_only_submission_sources() -> None:
    """Reject duplicate manuscripts and nested build output in the upload bundle."""
    archive = Path("build/arxiv/voiage-arxiv-source.tar.gz")
    if not archive.exists():
        return

    with tarfile.open(archive, "r:gz") as source:
        names = source.getnames()
        tex_sources = [
            member
            for member in source.getmembers()
            if member.isfile() and member.name.endswith(".tex")
        ]

        assert "main.tex" in names
        assert "paper.tex" not in names
        assert not any(name.startswith("build/") for name in names)
        assert {name for name in names if name.endswith(".pdf")} == {
            "figures/synthetic_health_example.pdf"
        }
        assert {
            name for name in names if name.startswith("data/") and name.endswith(".csv")
        } == {
            "data/synthetic_health_example_curve.csv",
            "data/synthetic_health_example_results.csv",
            "data/synthetic_health_example_sensitivity.csv",
            "data/synthetic_health_example_summary.csv",
        }
        documentclasses = 0
        for member in tex_sources:
            extracted = source.extractfile(member)
            assert extracted is not None
            documentclasses += extracted.read().decode().count("\\documentclass")
        assert documentclasses == 1
