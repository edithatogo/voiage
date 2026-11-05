"""Reject incomplete or incomparable build-space receipts before budgeting."""

from configparser import ConfigParser
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from scripts.build_capacity_budget import build_budget as _build_budget

config = ConfigParser(interpolation=None)
config.read(Path(__file__).resolve().parents[1] / "tox.ini")
ENVIRONMENTS = tuple(config["tox"]["env_list"].split())


def tox_receipt() -> dict:
    return {
        "testenvs": {
            name: {
                "result": {"success": True, "exit_code": 0, "skipped": False},
                "installed_packages": ["numpy==2.5.4", "pytest==9.1.1"],
            }
            for name in ENVIRONMENTS
        }
    }


def build_budget(cold: dict, warm: dict, *, reserve_bytes: int) -> dict:
    return _build_budget(
        cold,
        warm,
        reserve_bytes=reserve_bytes,
        cold_tox=tox_receipt(),
        warm_tox=tox_receipt(),
    )


def receipt(state: str) -> dict:
    """Construct a small synthetic guard receipt with a complete identity."""
    return {
        "reason": "completed",
        "returncode": 0,
        "elapsed_seconds": 10,
        "workload": {
            "id": "full-tox",
            "cache_state": state,
            "source_head": "a" * 40,
            "source_tree": "b" * 40,
            "tracked_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "untracked_paths": [],
            "python": "3.15.1",
            "command": ["tox", "run-parallel", "-p", "2"],
            "configuration_sha256": {
                "uv.lock": "c" * 64,
                "tox.ini": hashlib.sha256(
                    (Path(__file__).resolve().parents[1] / "tox.ini").read_bytes()
                ).hexdigest(),
            },
            "declared_configuration": {"platform": "test", "cold_scope": "uv/tox only"},
        },
        "disk": {
            "sampling_interval_seconds": 2,
            "physical_capacity_qualified": False,
            "filesystems": [
                {
                    "device": 1,
                    "paths": ["/work"],
                    "samples": 3,
                    "start_free_bytes": 10000,
                    "end_free_bytes": 8000,
                    "min_free_bytes": 7000,
                    "total_bytes": 20000,
                }
            ],
            "artifacts_before": [],
            "artifacts_after": [],
        },
    }


def test_budget_includes_explicit_reserve_without_claiming_hard_limit() -> None:
    cold, warm = receipt("cold"), receipt("warm")
    warm["disk"]["filesystems"][0].update(
        start_free_bytes=8000, min_free_bytes=7500, end_free_bytes=7800
    )
    report = build_budget(cold, warm, reserve_bytes=2000)
    assert report["filesystems"][0]["observed_additional_bytes"] == 3000
    assert report["filesystems"][0]["planning_additional_free_bytes"] == 5000
    assert report["hard_limit_adopted"] is False
    assert report["physical_capacity_qualified"] is False


@pytest.mark.parametrize(
    "change",
    [
        "failed",
        "monitor_failed",
        "wrong_source",
        "wrong_config",
        "wrong_command",
        "dirty",
        "untracked",
        "wrong_cache",
        "missing_samples",
        "impossible_space",
        "duplicate_device",
    ],
)
def test_budget_rejects_unusable_pair(change: str) -> None:
    cold, warm = receipt("cold"), receipt("warm")
    if change == "failed":
        warm["returncode"] = 1
    elif change == "monitor_failed":
        warm["reason"] = "interrupted_or_monitor_failed"
    elif change == "wrong_source":
        warm["workload"]["source_head"] = "d" * 40
    elif change == "wrong_config":
        warm["workload"]["declared_configuration"] = {}
    elif change == "wrong_command":
        warm["workload"]["command"] = ["true"]
    elif change == "dirty":
        warm["workload"]["tracked_diff_sha256"] = "d" * 64
    elif change == "untracked":
        warm["workload"]["untracked_paths"] = ["new-test.py"]
    elif change == "wrong_cache":
        warm["workload"]["cache_state"] = "mixed"
    elif change == "missing_samples":
        warm["disk"]["filesystems"][0]["samples"] = 0
    elif change == "impossible_space":
        warm["disk"]["filesystems"][0]["min_free_bytes"] = 12000
    elif change == "duplicate_device":
        warm["disk"]["filesystems"].append(deepcopy(warm["disk"]["filesystems"][0]))
    with pytest.raises(ValueError):
        build_budget(cold, warm, reserve_bytes=2000)


def test_reserve_is_positive_and_pair_requires_same_filesystems() -> None:
    cold, warm = receipt("cold"), receipt("warm")
    with pytest.raises(ValueError):
        build_budget(cold, warm, reserve_bytes=0)
    warm["disk"]["filesystems"][0]["device"] = 2
    with pytest.raises(ValueError):
        build_budget(cold, warm, reserve_bytes=2000)


def test_cli_binds_original_bytes_and_refuses_to_replace_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import json
    import sys

    from scripts.build_capacity_budget import main

    cold, warm, output = (
        tmp_path / name for name in ("cold.json", "warm.json", "budget.json")
    )
    cold.write_text(json.dumps(receipt("cold")))
    warm.write_text(json.dumps(receipt("warm")))
    tox_cold, tox_warm = tmp_path / "tox-cold.json", tmp_path / "tox-warm.json"
    tox_cold.write_text(json.dumps(tox_receipt()))
    tox_warm.write_text(json.dumps(tox_receipt()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "budget",
            str(cold),
            str(warm),
            "--cold-tox-results",
            str(tox_cold),
            "--warm-tox-results",
            str(tox_warm),
            "--reserve-mib",
            "2048",
            "--reserve-rationale",
            "Preserve the 2 GiB watchdog reserve",
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    report = json.loads(output.read_text())
    assert report["input_sha256"] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in (cold, warm)
    ]
    assert report["reserve_rationale"] == "Preserve the 2 GiB watchdog reserve"
    with pytest.raises(FileExistsError):
        main()


def test_cli_failure_does_not_create_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import sys

    from scripts.build_capacity_budget import main

    cold, warm, output = (
        tmp_path / name for name in ("cold.json", "warm.json", "budget.json")
    )
    cold.write_text(json.dumps(receipt("cold")))
    failed = receipt("warm")
    failed["returncode"] = 1
    warm.write_text(json.dumps(failed))
    tox_cold, tox_warm = tmp_path / "tox-cold.json", tmp_path / "tox-warm.json"
    tox_cold.write_text(json.dumps(tox_receipt()))
    tox_warm.write_text(json.dumps(tox_receipt()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "budget",
            str(cold),
            str(warm),
            "--cold-tox-results",
            str(tox_cold),
            "--warm-tox-results",
            str(tox_warm),
            "--reserve-mib",
            "2048",
            "--reserve-rationale",
            "Preserve the 2 GiB watchdog reserve",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError):
        main()
    assert not output.exists()


def test_successful_matching_tox_packages_are_bound() -> None:
    cold, warm = tox_receipt(), tox_receipt()
    report = _build_budget(
        receipt("cold"),
        receipt("warm"),
        reserve_bytes=2048,
        cold_tox=cold,
        warm_tox=warm,
    )
    assert len(report["tox_inventory_sha256"]) == len(ENVIRONMENTS)


@pytest.mark.parametrize(
    "damage",
    ["changed_package", "failed_env", "skipped_env", "missing_env", "missing_packages"],
)
def test_mismatched_or_unsuccessful_tox_results_are_rejected(damage: str) -> None:
    cold, warm = tox_receipt(), deepcopy(tox_receipt())
    if damage == "changed_package":
        warm["testenvs"]["py315"]["installed_packages"][0] = "numpy==2.5.5"
    elif damage == "failed_env":
        warm["testenvs"]["py315"]["result"]["success"] = False
    elif damage == "skipped_env":
        warm["testenvs"]["py315"]["result"]["skipped"] = True
    elif damage == "missing_env":
        del warm["testenvs"]["py315"]
    else:
        del warm["testenvs"]["py315"]["installed_packages"]
    with pytest.raises(ValueError):
        _build_budget(
            receipt("cold"),
            receipt("warm"),
            reserve_bytes=2048,
            cold_tox=cold,
            warm_tox=warm,
        )


def test_same_local_wheel_path_without_content_identity_is_rejected() -> None:
    cold, warm = tox_receipt(), tox_receipt()
    for item in (cold, warm):
        item["testenvs"]["py315"]["installed_packages"].append(
            "voiage @ file:///tmp/package/1/voiage.whl"
        )
    with pytest.raises(ValueError, match="lacks retained content identity"):
        _build_budget(
            receipt("cold"),
            receipt("warm"),
            reserve_bytes=2048,
            cold_tox=cold,
            warm_tox=warm,
        )
