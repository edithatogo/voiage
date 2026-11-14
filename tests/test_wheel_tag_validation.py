"""Behavioral checks for the release wheel filename contract."""

from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_wheel_tags.py"
LINUX_TAGS = "manylinux_2_17_x86_64.manylinux2014_x86_64"


@pytest.mark.parametrize(
    ("actual_abi", "actual_platform", "expected_platform", "valid"),
    [
        ("cp315-abi3", LINUX_TAGS, LINUX_TAGS, True),
        ("cp315-abi3", "manylinux2014_x86_64.manylinux_2_17_x86_64", LINUX_TAGS, True),
        ("cp315-abi3", "manylinux2014_x86_64", LINUX_TAGS, False),
        ("cp315-abi3", f"{LINUX_TAGS}.linux_x86_64", LINUX_TAGS, False),
        ("cp315-abi3", f"{LINUX_TAGS}.manylinux2014_x86_64", LINUX_TAGS, False),
        (
            "cp315-abi3",
            "manylinux2014_aarch64.manylinux_2_17_aarch64",
            LINUX_TAGS,
            False,
        ),
        ("cp314-abi3", LINUX_TAGS, LINUX_TAGS, False),
        ("cp315-cp315", LINUX_TAGS, LINUX_TAGS, False),
        ("cp315-abi3", "macosx_11_0_arm64", "macosx_11_0_arm64", True),
        ("cp315-abi3", "win_amd64", "win_amd64", True),
        ("cp315-abi3", "win_arm64", "win_amd64", False),
    ],
)
def test_wheel_tags_accept_only_the_exact_abi_and_platform_collection(
    tmp_path: Path,
    actual_abi: str,
    actual_platform: str,
    expected_platform: str,
    valid: bool,
) -> None:
    (tmp_path / f"voiage-3.1.0-{actual_abi}-{actual_platform}.whl").touch()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path),
            "--abi",
            "cp315-abi3",
            "--platform",
            expected_platform,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is valid, result.stderr


@pytest.mark.parametrize("wheel_count", [0, 2])
def test_wheel_tag_check_requires_exactly_one_wheel(
    tmp_path: Path, wheel_count: int
) -> None:
    for index in range(wheel_count):
        (tmp_path / f"voiage-{index}-cp315-abi3-win_amd64.whl").touch()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path),
            "--abi",
            "cp315-abi3",
            "--platform",
            "win_amd64",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "expected exactly one wheel" in result.stderr
    assert "Validated wheel tags:" not in result.stdout
