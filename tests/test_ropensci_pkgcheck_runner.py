from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_ropensci_pkgcheck.sh"


def test_pkgcheck_runner_builds_and_identifies_the_exact_source_archive() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    assert "R CMD build" in script
    assert "voiageR_2.2.0.tar.gz" in script
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
    assert "af25295aed5fbb0229c20a8f68e91ed9e53e8d19" in script
    assert "d186fe6f93657805ed86177f03333c478e136709" in script
    assert "2679f1e899a9e3777eaa9a9ac5566a2aeafc11d9" in script
    assert "e0679a1bf759d637dc779108818265c681bdbec77542433534ece0e3a1fec977" in script


def test_pkgcheck_runner_reuses_authenticated_gh_without_exposing_the_token() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert "command -v gh >/dev/null 2>&1" in script
    assert "GITHUB_TOKEN=$(gh auth token 2>/dev/null || true)" in script
    assert "export GITHUB_TOKEN" in script
    assert "echo $GITHUB_TOKEN" not in script
