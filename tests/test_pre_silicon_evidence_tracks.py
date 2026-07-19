from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HARDWARE_ROOT = ROOT / "hardware" / "pre_silicon"


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_fpga_and_asic_tracks_iterate_free_pre_silicon_options() -> None:
    fpga_plan = _read("conductor/archive/fpga-implementation_20260511/plan.md")
    asic_plan = _read("conductor/archive/asic-implementation_20260511/plan.md")
    combined_notes = "\n".join(
        (
            _read("conductor/archive/fpga-implementation_20260511/working-notes.md"),
            _read("conductor/archive/asic-implementation_20260511/working-notes.md"),
        )
    )

    for needle in ("GitHub Actions", "Codespaces", "Google Cloud Shell"):
        assert needle in fpga_plan
        assert needle in asic_plan

    for needle in ("Verilator", "Yosys", "nextpnr", "OSS CAD Suite"):
        assert needle in fpga_plan

    for needle in ("OpenROAD", "OpenLane", "SKY130", "RTL-to-GDS"):
        assert needle in asic_plan

    for needle in ("physical FPGA board", "Tiny Tapeout", "SkyWater MPW", "Chrome"):
        assert needle in combined_notes


def test_pre_silicon_kernel_assets_are_repo_owned() -> None:
    expected_paths = (
        HARDWARE_ROOT / "README.md",
        HARDWARE_ROOT / "fixtures" / "evpi_fixed_point_fixture.json",
        HARDWARE_ROOT / "rtl" / "evpi_fixed_point_kernel.v",
        HARDWARE_ROOT / "tb" / "evpi_fixed_point_kernel_tb.v",
    )

    for path in expected_paths:
        assert path.exists(), f"missing pre-silicon asset: {path}"

    fixture = _load_json(HARDWARE_ROOT / "fixtures" / "evpi_fixed_point_fixture.json")
    assert fixture["kernel"] == "evpi_fixed_point"
    assert fixture["scale"] == 1000
    assert len(fixture["cases"]) >= 3

    testbench = (HARDWARE_ROOT / "tb" / "evpi_fixed_point_kernel_tb.v").read_text(
        encoding="utf-8"
    )
    for case in fixture["cases"]:
        pattern = (
            rf"check_case\(32'd{case['expected_with_information']},\s*"
            rf"32'd{case['expected_without_information']},\s*"
            rf"32'd{case['expected_evpi']}\)"
        )
        assert re.search(pattern, testbench), f"testbench missing fixture case: {case}"


def test_committed_pre_silicon_manifests_have_required_evidence_shape() -> None:
    for target, track in (
        ("fpga", "fpga-implementation_20260511"),
        ("asic", "asic-implementation_20260511"),
    ):
        manifest = _load_json(
            ROOT
            / "conductor"
            / "archive"
            / track
            / "handoff"
            / "pre_silicon_evidence_manifest.json"
        )

        assert manifest["target"] == target
        assert manifest["evidence_kind"] == "ci_pre_silicon"
        assert manifest["input_fixture"]["sha256"]
        assert manifest["source_files"]
        assert manifest["tools"]
        assert manifest["commands"]
        assert any(
            command["name"] == "verilator_testbench_simulation_run"
            for command in manifest["commands"]
        )
        if target == "asic":
            artifact_names = {
                artifact
                for command in manifest["commands"]
                for artifact in command["output_artifacts"]
            }
            assert "artifacts/asic/evpi_fixed_point.gds" in artifact_names
            assert "artifacts/asic/evpi_fixed_point.def" in artifact_names
            assert "reports/asic/openroad_timing.rpt" in artifact_names
            assert "reports/asic/openroad_area.rpt" in artifact_names
        for command in manifest["commands"]:
            assert command["status"]
            assert command["output_artifacts"]


def test_manifest_generator_writes_probe_manifests(tmp_path: Path) -> None:
    output_root = tmp_path / "evidence"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "pre_silicon_evidence.py"),
        "all",
        "--output-root",
        str(output_root),
        "--probe-only",
    ]

    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    for target in ("fpga", "asic"):
        manifest = _load_json(
            output_root / target / "pre_silicon_evidence_manifest.json"
        )
        assert manifest["target"] == target
        assert manifest["input_fixture"]["sha256"]
        assert {tool["name"] for tool in manifest["tools"]}
        assert all(
            command["status"] == "not_run_probe" for command in manifest["commands"]
        )
        artifact_argv = " ".join(
            " ".join(command["argv"]) for command in manifest["commands"]
        )
        expected_artifact_root = (output_root / target / "artifacts" / target).as_posix()
        assert expected_artifact_root in artifact_argv.replace("\\", "/")
