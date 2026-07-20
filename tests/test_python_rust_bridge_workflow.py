"""Contracts for hosted PyO3/maturin bridge validation."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / ".github/workflows/python-rust-bridge.yml"


def _workflow() -> dict[str, object]:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_bridge_workflow_has_minimal_permissions_and_pinned_actions() -> None:
    workflow = _workflow()
    text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert workflow["permissions"] == {}
    for job in workflow["jobs"].values():
        assert job["permissions"] == {"contents": "read"}
    assert "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0" in text
    assert "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1" in text
    assert "astral-sh/setup-uv@11f9893b081a58869d3b5fccaea48c9e9e46f990" in text
    assert "dtolnay/rust-toolchain@2c7215f132e9ebf062739d9130488b56d53c060c" in text
    assert "persist-credentials: false" in text
    assert text.count('      - "tests/packaging/**"') == 2


def test_bridge_workflow_covers_supported_python_and_abi3_platforms() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]

    python_matrix = jobs["python-compatibility"]["strategy"]["matrix"]
    assert python_matrix == {"python": ["3.10", "3.11", "3.12", "3.13", "3.14"]}
    assert jobs["python-compatibility"]["runs-on"] == "ubuntu-24.04"

    platform_matrix = jobs["abi3-platforms"]["strategy"]["matrix"]
    assert [entry["os"] for entry in platform_matrix["include"]] == [
        "ubuntu-24.04",
        "macos-14",
        "windows-2025",
    ]
    assert jobs["abi3-platforms"]["runs-on"] == "${{ matrix.os }}"


def test_bridge_workflow_uses_locked_maturin_builds_and_import_checks() -> None:
    workflow = _workflow()
    rendered = str(workflow["jobs"])

    assert "uv sync --locked --extra ci" in rendered
    assert "maturin develop" not in rendered
    assert "PyO3/maturin-action@e83996d129638aa358a18fbd1dfb82f0b0fb5d3b" in rendered
    assert (
        "--locked --release --compatibility pypi --interpreter python3.10" in rendered
    )
    assert "uv pip install" in rendered
    assert "uv pip check" in rendered
    assert "python -m zipfile -l" in rendered
    assert "cp310-abi3" in rendered
    assert "cargo metadata --manifest-path rust/Cargo.toml --locked" in rendered
    assert "tests/test_runtime_adapter.py" in rendered
    assert "tests/packaging/test_wheel_black_box.py" in rendered
    assert "--import-mode=importlib" in rendered
    assert "WHEEL_VENV" in rendered
    assert 'cp "$GITHUB_WORKSPACE/tests/packaging/test_wheel_black_box.py"' in rendered
    assert "continue-on-error" not in rendered


def test_bridge_builds_embed_checkout_and_platform_provenance() -> None:
    workflow = _workflow()
    for job_name in ("python-compatibility", "abi3-platforms"):
        job = workflow["jobs"][job_name]
        rendered = str(job)
        assert "VOIAGE_SOURCE_REVISION" in rendered
        assert "github.sha" in rendered
        assert "VOIAGE_SOURCE_CLEAN" in rendered
        assert "VOIAGE_SOURCE_TREE_GIT_OID" in rendered
        assert "EXPECTED_SOURCE_TREE_GIT_OID" in rendered
        assert "SOURCE_DATE_EPOCH" in rendered
        assert "git show -s --format=%ct HEAD" in rendered
        assert "EXPECTED_SOURCE_REVISION" in rendered
        assert "EXPECTED_PLATFORM_SUFFIX" in rendered


def test_bridge_workflow_installs_the_built_wheel_on_every_supported_runtime() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]

    compatibility = str(jobs["python-compatibility"])
    platforms = str(jobs["abi3-platforms"])
    compatibility_install = next(
        step
        for step in jobs["python-compatibility"]["steps"]
        if step["name"] == "Create clean locked test environment"
    )["run"]
    platform_install = next(
        step
        for step in jobs["abi3-platforms"]["steps"]
        if step["name"] == "Install actual wheel into clean environment"
    )["run"]
    assert "maturin-action" in compatibility
    assert "uv pip install" in compatibility
    assert "locked-requirements.txt" not in compatibility
    assert "--no-deps" not in compatibility_install
    assert "uv pip check" in compatibility
    assert "3.14" in compatibility
    assert "maturin-action" in platforms
    assert "uv pip install" in platforms
    assert "locked-requirements.txt" not in platforms
    assert "--no-deps" not in platform_install
    assert "uv pip check" in platforms


def test_bridge_platform_matrix_validates_actual_wheel_suffix() -> None:
    workflow = _workflow()
    platform_job = workflow["jobs"]["abi3-platforms"]

    assert platform_job["strategy"]["matrix"]["include"] == [
        {
            "os": "ubuntu-24.04",
            "platform-suffix": "manylinux_2_17_x86_64.manylinux2014_x86_64",
            "compatibility": "manylinux2014",
        },
        {
            "os": "macos-14",
            "platform-suffix": "macosx_11_0_arm64",
            "compatibility": "pypi",
        },
        {
            "os": "windows-2025",
            "platform-suffix": "win_amd64",
            "compatibility": "pypi",
        },
    ]
    native_build = next(
        step for step in platform_job["steps"] if step["name"] == "Build abi3 wheel"
    )
    assert native_build["if"] == "runner.os != 'Linux'"
    assert '--compatibility "${{ matrix.compatibility }}"' in native_build["run"]
    linux_build = next(
        step
        for step in platform_job["steps"]
        if step["name"] == "Build manylinux2014 abi3 wheel"
    )
    assert linux_build["if"] == "runner.os == 'Linux'"
    assert linux_build["with"]["manylinux"] == "2014"
    assert linux_build["with"]["maturin-version"] == "v1.14.1"
    inspect = next(
        step
        for step in platform_job["steps"]
        if step["name"] == "Inspect wheel tags and contents"
    )
    assert inspect["env"]["EXPECTED_PLATFORM_SUFFIX"] == "${{ matrix.platform-suffix }}"
    assert "endswith(expected)" in inspect["run"]
