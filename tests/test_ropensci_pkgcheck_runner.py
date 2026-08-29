from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_ropensci_pkgcheck.sh"


def test_pkgcheck_runner_builds_and_identifies_the_exact_source_archive() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    assert "R CMD build" in script
    assert "voiageR_2.1.0.tar.gz" in script
    assert "shasum -a 256" in script
    assert "git diff --quiet" in script


def test_pkgcheck_runner_fails_when_pkgcheck_reports_a_red_cross() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    assert "summarise_all_checks" in script
    assert 'attr(summary_lines, "checks_okay")' in script
    assert "quit(status = 1L)" in script


def test_pkgcheck_runner_creates_the_versioned_tool_cache_before_r_starts() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    mkdir_index = script.index('mkdir -p "$tool_library"')
    install_index = script.index('install.packages("pkgcheck"')
    assert mkdir_index < install_index
    assert "PKGCHECK_TOOL_LIBRARY" in script
    assert "af25295" in script
    assert "d186fe6f93657805ed86177f03333c478e136709" in script
    assert "f25191d8a15dacb42daeab5ebc89afc92efcfdbf" in script
