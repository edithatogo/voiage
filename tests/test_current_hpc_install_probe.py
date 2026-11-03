"""Reject ambient Python/package evidence before installed acceptance."""

import json
import os
import subprocess
lazy from pathlib import Path

import pytest

from scripts import probe_current_hpc_install as probe


def test_probe_rejects_ambient_packages_outside_installed_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "probe.json"
    monkeypatch.setattr(
        probe.sys,
        "argv",
        [
            "probe",
            "--prefix",
            str(tmp_path),
            "--cell",
            "spack",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="selected prefix"):
        probe.main()
    assert not output.exists()


def test_probe_rejects_unqualified_python_before_imports(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "probe.json"
    monkeypatch.setattr(
        probe.sys,
        "argv",
        [
            "probe",
            "--prefix",
            str(tmp_path),
            "--cell",
            "spack",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(probe.sys, "version_info", (3, 15, 2))
    with pytest.raises(ValueError, match="Python must be 3.15.1"):
        probe.main()
    assert not output.exists()


@pytest.mark.parametrize(
    "name",
    ["libarrow.so.2600", "libparquet.so.2600", "libstdc++.so.6", "libgcc_s.so.1"],
)
def test_probe_rejects_scientific_and_compiler_libraries_from_os(name: str) -> None:
    assert not probe.permitted_link_target(
        Path("/usr/lib") / name, Path("/selected"), "spack"
    )


def test_probe_allows_only_profile_declared_system_ssl() -> None:
    path = Path("/usr/lib/libssl.so.3")
    assert not probe.permitted_link_target(path, Path("/selected"), "spack")
    assert probe.permitted_link_target(path, Path("/selected"), "easybuild-2023a")


def test_installed_driver_stops_at_failed_source_admission(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "qualify_linux_sources.sh").write_text("exit 23\n")
    output = tmp_path / "receipts"
    result = subprocess.run(
        [
            "/bin/bash",
            str(repository / "scripts/qualify_linux_installed.sh"),
            "spack",
            str(output),
        ],
        cwd=tmp_path,
        env=dict(os.environ, RUNNER_TEMP=str(tmp_path)),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 23
    assert (
        json.loads((output / "result.json").read_text())["installed_cell_accepted"]
        is False
    )
    assert not (output / "install-stage").exists()
