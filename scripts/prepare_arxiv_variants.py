#!/usr/bin/env python3
"""Prepare optional cleaner and collector variants for explicit comparison."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
VARIANTS = ROOT / "build" / "arxiv-variants"
TOOLS_BIN = ROOT / ".venv-arxiv" / "bin"


def optional_tool(name: str) -> str | None:
    """Resolve a tool from the isolated manuscript environment or PATH."""
    isolated = TOOLS_BIN / name
    if isolated.is_file():
        return str(isolated)
    return shutil.which(name)


def copy_manuscript_tree(destination: Path) -> None:
    """Copy authored manuscript inputs without local or legacy build state."""
    shutil.copytree(
        PAPER,
        destination,
        ignore=shutil.ignore_patterns(
            ".quarto",
            "__pycache__",
            "build",
            "*.py",
            "*.qmd",
            "paper.tex",
        ),
    )


def run(command: list[str], cwd: Path) -> None:
    """Run a required variant preparation command."""
    # Commands are repository-owned fixed argument lists, never user shell text.
    subprocess.run(command, cwd=cwd, check=True)  # noqa: S603


def rebuild_and_validate(source: Path, build: Path) -> None:
    """Compile and audit a transformed source tree."""
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir(parents=True)
    command = ["latexmk"]
    if not (source / "latexmkrc").exists():
        command.extend(["-r", str(PAPER / "latexmkrc")])
    command.extend([f"-outdir={build}", "main.tex"])
    run(command, source)
    python = str(ROOT / ".venv-arxiv" / "bin" / "python")
    run(
        [
            python,
            str(ROOT / "scripts/audit_arxiv_pdf.py"),
            str(build / "main.pdf"),
            str(build / "main.log"),
        ],
        ROOT,
    )
    run([python, str(ROOT / "scripts/validate_arxiv.py"), str(source)], ROOT)


def main() -> None:
    """Build cleaner and collector variants without selecting one for upload."""
    VARIANTS.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    cleaner = optional_tool("arxiv_latex_cleaner")
    if cleaner:
        cleaner_input = VARIANTS / "cleaner-input"
        shutil.rmtree(cleaner_input, ignore_errors=True)
        copy_manuscript_tree(cleaner_input)
        run(
            [
                cleaner,
                str(cleaner_input),
                "--config",
                str(PAPER / "cleaner_config.yaml"),
                "--keep_bib",
            ],
            ROOT,
        )
        cleaner_output = Path(f"{cleaner_input}_arXiv")
        rebuild_and_validate(cleaner_output, VARIANTS / "cleaner-build")
        diff = shutil.which("diff")
        if diff is None:
            raise SystemExit("missing required comparison tool: diff")
        comparison = subprocess.run(  # noqa: S603
            [
                diff,
                "-ru",
                "--exclude=metadata.json",
                "--exclude=readiness-manifest.json",
                "--exclude=cleaner_config.yaml",
                str(cleaner_input),
                str(cleaner_output),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        (VARIANTS / "cleaner.diff").write_text(comparison.stdout)
    else:
        missing.append("arxiv_latex_cleaner")

    collector = optional_tool("arxiv-collector")
    if collector:
        collector_input = VARIANTS / "collector-input"
        shutil.rmtree(collector_input, ignore_errors=True)
        copy_manuscript_tree(collector_input)
        run([collector, "main.tex"], collector_input)
        collector_output = VARIANTS / "collector-output"
        shutil.rmtree(collector_output, ignore_errors=True)
        collector_output.mkdir()
        with tarfile.open(collector_input / "arxiv.tar.gz", "r:gz") as archive:
            archive.extractall(collector_output, filter="data")
        rebuild_and_validate(collector_output, VARIANTS / "collector-build")
    else:
        missing.append("arxiv-collector")
    if missing:
        raise SystemExit(
            "missing optional tools; sync requirements-arxiv.txt: " + ", ".join(missing)
        )
    print(f"variants prepared under {VARIANTS}")


if __name__ == "__main__":
    main()
