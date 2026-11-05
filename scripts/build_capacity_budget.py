"""Derive an observed planning allowance from a matched cold/warm guard pair.

This Voiage report is deliberately not an admission gate or a physical-space
certificate. Filesystem deltas include other writers and sampling misses spikes.
"""

import argparse
import configparser
import hashlib
import json
from pathlib import Path
from typing import Any

GIB = 1024**3
EMPTY_DIFF = hashlib.sha256(b"").hexdigest()
IDENTITY_FIELDS = (
    "id",
    "source_head",
    "source_tree",
    "python",
    "command",
    "configuration_sha256",
    "declared_configuration",
)


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    """Reject missing, boolean, negative and non-integral byte observations."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _filesystems(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Validate terminal filesystem arithmetic without claiming attribution."""
    disk = report.get("disk", {})
    if not isinstance(disk, dict):
        raise TypeError("Disk observations must be an object")
    records = disk.get("filesystems")
    if not isinstance(records, list) or not records:
        raise ValueError("Missing filesystem observations")
    result: dict[int, dict[str, Any]] = {}
    interval = disk.get("sampling_interval_seconds")
    if not isinstance(interval, (float, int)) or not 0 < interval < float("inf"):
        raise ValueError("Invalid sampling interval")
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Filesystem observation must be an object")
        device = _integer(record.get("device"), "device")
        if device in result:
            raise ValueError("Duplicate filesystem device")
        _integer(record.get("samples"), "samples", 2)
        values = {
            key: _integer(record.get(key), key)
            for key in (
                "start_free_bytes",
                "min_free_bytes",
                "end_free_bytes",
                "total_bytes",
            )
        }
        if not (
            values["min_free_bytes"]
            <= min(values["start_free_bytes"], values["end_free_bytes"])
            and max(values["start_free_bytes"], values["end_free_bytes"])
            <= values["total_bytes"]
        ):
            raise ValueError("Impossible free-space observations")
        if not record.get("paths"):
            raise ValueError("Filesystem paths are missing")
        result[device] = record
    return result


def _tox_inventory(result: dict[str, Any], expected: set[str]) -> dict[str, list[str]]:
    """Require the complete tox matrix and exact installed package inventories."""
    if not isinstance(result, dict) or not isinstance(result.get("testenvs"), dict):
        raise ValueError("Missing tox result environments")  # noqa: TRY004 - invalid receipt schema
    envs = result["testenvs"]
    inventory: dict[str, list[str]] = {}
    for name in expected:
        entry = envs.get(name)
        if not isinstance(entry, dict) or not isinstance(entry.get("result"), dict):
            raise ValueError(f"Missing tox result for {name}")  # noqa: TRY004 - invalid receipt schema
        state = entry["result"]
        if (
            state.get("success") is not True
            or type(state.get("exit_code")) is not int
            or state.get("exit_code") != 0
            or state.get("skipped") is not False
        ):
            raise ValueError(f"Tox environment {name} did not pass")
        packages = entry.get("installed_packages")
        if (
            not isinstance(packages, list)
            or not packages
            or any(not isinstance(p, str) or not p for p in packages)
        ):
            raise ValueError(f"Missing installed package inventory for {name}")
        if any(" @ file:" in package for package in packages):
            raise ValueError(
                f"Local package artifact for {name} lacks retained content identity"
            )
        if len(set(packages)) != len(packages):
            raise ValueError(f"Duplicate package inventory for {name}")
        inventory[name] = sorted(packages)
    unexpected = {
        name
        for name, entry in envs.items()
        if name not in expected and isinstance(entry, dict) and entry.get("result")
    }
    if unexpected:
        raise ValueError("Unexpected executed tox environments")
    return inventory


def _configured_tox_environments(workload: dict[str, Any]) -> set[str]:
    """Verify the actual tox configuration before deriving the required matrix."""
    path = Path(__file__).resolve().parents[1] / "tox.ini"
    payload = path.read_bytes()
    if (
        workload["configuration_sha256"].get("tox.ini")
        != hashlib.sha256(payload).hexdigest()
    ):
        raise ValueError("Current tox.ini does not match the measured source")
    config = configparser.ConfigParser(interpolation=None)
    config.read_string(payload.decode("utf-8"))
    names = config["tox"]["env_list"].split()
    if not names or len(names) != len(set(names)):
        raise ValueError("Invalid tox environment list")
    return set(names)


def build_budget(
    cold: dict[str, Any],
    warm: dict[str, Any],
    *,
    reserve_bytes: int,
    cold_tox: dict[str, Any],
    warm_tox: dict[str, Any],
) -> dict[str, Any]:
    """Validate successful comparable receipts and retain an explicit reserve.

    Artifact allocations remain individual snapshots. They are never summed:
    overlapping roots, hardlinks and filesystem clones can share storage.
    """
    _integer(reserve_bytes, "reserve_bytes", 1)
    for report, state in ((cold, "cold"), (warm, "warm")):
        if not isinstance(report, dict):
            raise TypeError("Measurement must be an object")
        if (
            report.get("reason") != "completed"
            or type(report.get("returncode")) is not int
            or report["returncode"] != 0
        ):
            raise ValueError("Both measurements must finish successfully")
        workload = report.get("workload", {})
        if not isinstance(workload, dict):
            raise TypeError("Workload identity must be an object")
        if workload.get("cache_state") != state:
            raise ValueError(f"Expected declared {state} cache state")
        if (
            workload.get("tracked_diff_sha256") != EMPTY_DIFF
            or workload.get("untracked_paths") != []
        ):
            raise ValueError(
                "Calibration requires a clean tracked and untracked source tree"
            )
        if any(not workload.get(field) for field in IDENTITY_FIELDS):
            raise ValueError("Incomplete source/environment/configuration identity")
        elapsed = report.get("elapsed_seconds")
        if not isinstance(elapsed, (float, int)) or not 0 < elapsed < float("inf"):
            raise ValueError("Elapsed observation is missing or invalid")
    if any(
        cold["workload"][field] != warm["workload"][field] for field in IDENTITY_FIELDS
    ):
        raise ValueError("Cold and warm identities or commands do not match")
    expected_envs = _configured_tox_environments(cold["workload"])
    cold_inventory = _tox_inventory(cold_tox, expected_envs)
    warm_inventory = _tox_inventory(warm_tox, expected_envs)
    if cold_inventory != warm_inventory:
        raise ValueError("Cold and warm installed tox packages differ")
    inventory_digests = {
        name: hashlib.sha256(
            json.dumps(packages, separators=(",", ":")).encode()
        ).hexdigest()
        for name, packages in sorted(cold_inventory.items())
    }
    cold_fs, warm_fs = _filesystems(cold), _filesystems(warm)
    if cold_fs.keys() != warm_fs.keys():
        raise ValueError("Cold and warm filesystems differ")
    filesystems = []
    for device, first in cold_fs.items():
        second = warm_fs[device]
        if (
            first["total_bytes"] != second["total_bytes"]
            or first["paths"] != second["paths"]
        ):
            raise ValueError("Filesystem identity changed")
        growth = max(
            first["start_free_bytes"] - first["min_free_bytes"],
            second["start_free_bytes"] - second["min_free_bytes"],
        )
        filesystems.append(
            {
                "device": device,
                "paths": first["paths"],
                "observed_additional_bytes": growth,
                "reserve_bytes": reserve_bytes,
                "planning_additional_free_bytes": growth + reserve_bytes,
                "observed_additional_gib": round(growth / GIB, 3),
                "planning_additional_free_gib": round(
                    (growth + reserve_bytes) / GIB, 3
                ),
            }
        )
    return {
        "schema_version": "voiage.build-capacity-budget.v1",
        "scope": cold["workload"]["id"],
        "identity": {key: cold["workload"][key] for key in IDENTITY_FIELDS},
        "tox_inventory_sha256": inventory_digests,
        "hard_limit_adopted": False,
        "physical_capacity_qualified": False,
        "filesystems": filesystems,
        "elapsed_seconds": {
            "cold": cold["elapsed_seconds"],
            "warm": warm["elapsed_seconds"],
        },
        "artifact_snapshots": {
            state: {
                "before": report["disk"].get("artifacts_before", []),
                "after": report["disk"].get("artifacts_after", []),
            }
            for state, report in (("cold", cold), ("warm", warm))
        },
        "limitations": [
            "Cache state and scope are operator declarations; OS caches need not be cold.",
            "Filesystem deltas include unrelated activity and samples can miss peaks.",
            "Artifacts are boundary allocations, not peaks or exclusive physical usage.",
            "The explicit reserve is a planning assumption, not a capacity guarantee.",
            "This workload cannot qualify a different compiler or complete HPC matrix.",
        ],
    }


def main() -> int:
    """Write a new budget only after both input receipts pass validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cold", type=Path)
    parser.add_argument("warm", type=Path)
    parser.add_argument("--cold-tox-results", type=Path, required=True)
    parser.add_argument("--warm-tox-results", type=Path, required=True)
    parser.add_argument("--reserve-mib", type=int, required=True)
    parser.add_argument("--reserve-rationale", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.reserve_rationale.strip():
        raise ValueError("Reserve rationale must not be blank")
    inputs = [path.read_bytes() for path in (args.cold, args.warm)]
    tox_inputs = [
        path.read_bytes() for path in (args.cold_tox_results, args.warm_tox_results)
    ]
    report = build_budget(
        *[json.loads(data) for data in inputs],
        reserve_bytes=args.reserve_mib * 1024**2,
        cold_tox=json.loads(tox_inputs[0]),
        warm_tox=json.loads(tox_inputs[1]),
    )
    report["input_sha256"] = [hashlib.sha256(data).hexdigest() for data in inputs]
    report["tox_input_sha256"] = [
        hashlib.sha256(data).hexdigest() for data in tox_inputs
    ]
    report["reserve_rationale"] = args.reserve_rationale
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
