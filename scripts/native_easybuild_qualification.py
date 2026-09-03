# ruff: noqa: S603, S607
"""Guard and record sequential native EasyBuild qualification runs."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import jsonschema

BASE = "9307d9ec7fcdc808ed7931afc298fa3bebac36e8"
CATALOGUE = "58e8b5a48767cbed1bf5669675d9638580d7259f"
CATALOGUE_TREE = "7c1cff95bf2f5365b4ecfe173c38548e5241d188"
CATALOGUE_MANIFEST_SHA256 = (
    "dca2a52184a172cf712941627819b53656ce4724fbee24174e452d12e8cf83a2"
)
ROOT_RECIPE_SHA256 = {
    "2023a": "653c6ad1f04c79f6e517f3d0336f4f4948459cc1d2de7e6056f37e1c1561fff0",
    "2024a": "7a3db9d93860b0fb5afefbaed3f1cf4b77fca47cddb5258fb381f6afb94bf71f",
}
GENERATIONS = {
    "2023a": [
        "packaging/easybuild",
        "packaging/easybuild-2023a-polars-overlay/2023a",
        "packaging/easybuild-2023a-arrow-overlay/2023a",
        "packaging/easybuild-2023a-rust-overlay/2023a",
        "packaging/easybuild-2023a-overlay/2023a",
    ],
    "2024a": [
        "packaging/easybuild",
        "packaging/easybuild-2024a-polars-overlay/2024a",
        "packaging/easybuild-2024a-arrow-overlay/2024a",
        "packaging/easybuild-2024a-rust-overlay/2024a",
        "packaging/easybuild-overlay/2024a",
    ],
}
REQUIRED_PROBES = {
    "module-load",
    "cli",
    "rust-engine",
    "numerical-evpi",
    "arrow-roundtrip",
    "polars-roundtrip",
    "linkage",
    "generation-isolation",
}
RECEIPT_KEYS = {
    "schema_version",
    "terminal",
    "outcome",
    "run_root",
    "identity",
    "environment",
    "resources",
    "preflight",
    "commands",
    "artifacts",
    "probes",
    "failure",
    "run",
}
IDENTITY_KEYS = {
    "generation",
    "source_commit",
    "source_tree",
    "source_relevant_tree_sha256",
    "catalogue_root",
    "catalogue_commit",
    "catalogue_tree",
    "catalogue_relevant_tree_sha256",
    "easybuild_version",
    "easyblocks_version",
    "root_recipe",
    "root_recipe_sha256",
    "robot_paths",
    "robot_manifest_sha256",
    "source_cache_manifest_sha256",
}
COMMAND_KEYS = {"stage", "argv", "exit_code", "signal", "log", "log_sha256", "parsed"}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regular_inside(run_root: Path, relative: str) -> Path:
    """Resolve a declared artifact without permitting traversal or symlinks."""
    candidate = run_root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("artifact path escapes declared run root")
    current = run_root
    if current.is_symlink() or not current.is_dir():
        raise ValueError("declared run root is not a regular directory")
    for part in Path(relative).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("symlink artifact is forbidden")
    try:
        candidate.resolve(strict=True).relative_to(run_root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("artifact path escapes declared run root") from exc
    if not candidate.is_file():
        raise ValueError("artifact must be a regular file")
    return candidate


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} has unexpected or missing fields")
    return value


def _manifest(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text())
    value = _exact_keys(value, {"root", "entries"}, "inventory")
    root = path.parent / value["root"]
    if root.is_symlink() or not root.is_dir():
        raise ValueError("inventory root is missing or a symlink")
    root = root.resolve(strict=True)
    entries = value["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("inventory is empty")
    for item in entries:
        relative = Path(item.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("inventory path escapes its root")
        artifact = root / relative
        if item.get("type") == "file":
            _exact_keys(item, {"path", "type", "size", "sha256"}, "inventory file")
            if (
                artifact.is_symlink()
                or not artifact.is_file()
                or artifact.stat().st_size != item["size"]
                or sha256(artifact) != item["sha256"]
            ):
                raise ValueError("inventory entry bytes do not match size and digest")
        elif item.get("type") == "symlink":
            _exact_keys(
                item,
                {"path", "type", "link_target", "resolved_path"},
                "inventory symlink",
            )
            if (
                not artifact.is_symlink()
                or os.readlink(artifact) != item["link_target"]
                or Path(item["link_target"]).is_absolute()
            ):
                raise ValueError("inventory symlink text is invalid")
            try:
                resolved = artifact.resolve(strict=True)
                resolved_relative = resolved.relative_to(root)
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                raise ValueError(
                    "inventory symlink is dangling, cyclic, or escapes root"
                ) from exc
            if str(resolved_relative) != item["resolved_path"]:
                raise ValueError("inventory symlink resolved target is invalid")
        else:
            raise ValueError("inventory entry type is invalid")
    return entries


def run(argv: list[str], log: Path, env: dict[str, str]) -> int:
    """Run a controlled command and retain its combined transcript."""
    with log.open("wb") as stream:
        result = subprocess.run(
            argv, stdout=stream, stderr=subprocess.STDOUT, env=env, check=False
        )
    return result.returncode


def clean_environment(home: Path) -> tuple[dict[str, str], list[str], str]:
    """Construct the complete allowlisted environment used by native commands."""
    allowed = {"PATH", "LANG", "LC_ALL", "TERM", "TMPDIR"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update({"HOME": str(home), "PYTHONNOUSERSITE": "1"})
    serialized = [f"{key}={env[key]}" for key in sorted(env)]
    digest = hashlib.sha256(
        json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return env, serialized, digest


def write_inventory(base: Path, output: Path) -> str:
    """Write a deterministic inventory with constrained symlink evidence."""
    entries = []
    for item in sorted(base.rglob("*")):
        if item.is_symlink():
            target = os.readlink(item)
            if Path(target).is_absolute():
                raise RuntimeError(f"inventory contains absolute symlink: {item}")
            try:
                resolved = item.resolve(strict=True)
                resolved_relative = resolved.relative_to(base.resolve(strict=True))
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    f"inventory contains unsafe symlink: {item}"
                ) from exc
            entries.append(
                {
                    "path": str(item.relative_to(base)),
                    "type": "symlink",
                    "link_target": target,
                    "resolved_path": str(resolved_relative),
                }
            )
        elif item.is_file():
            entries.append(
                {
                    "path": str(item.relative_to(base)),
                    "type": "file",
                    "size": item.stat().st_size,
                    "sha256": sha256(item),
                }
            )
    if not entries:
        raise RuntimeError(f"inventory is empty: {base}")
    output.write_text(
        json.dumps(
            {"root": str(base.relative_to(output.parent)), "entries": entries},
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    return sha256(output)


def parsed_failure(log: Path) -> str:
    """Return the last non-empty transcript line as bounded failure evidence."""
    lines = [
        line.strip()
        for line in log.read_text(errors="replace").splitlines()
        if line.strip()
    ]
    return lines[-1][-1000:] if lines else "command produced no diagnostic output"


def directory_digest(base: Path) -> str:
    """Hash the names, sizes, and bytes of a no-symlink source tree."""
    entries = []
    for item in sorted(base.rglob("*")):
        if item.is_symlink():
            raise RuntimeError("directory digest refuses symbolic links")
        if item.is_file():
            entries.append(
                [str(item.relative_to(base)), item.stat().st_size, sha256(item)]
            )
    if not entries:
        raise RuntimeError("directory digest refuses an empty source cache")
    return hashlib.sha256(
        json.dumps(entries, separators=(",", ":")).encode()
    ).hexdigest()


def parse_build_success(log: Path, module_name: str) -> bool:
    """Recognize EasyBuild 5.4's canonical terminal success line."""
    text = log.read_text(errors="replace")
    del module_name  # identity is independently derived from the installed modulefile
    return bool(
        re.search(
            r"(?m)^== COMPLETED: Installation ended successfully \(took [^)]+\)$", text
        )
    )


def git_manifest_digest(checkout: Path, revision: str) -> tuple[str, bool]:
    """Hash the complete tracked tree and reject tracked symbolic links."""
    listing = subprocess.check_output(
        ["git", "-C", str(checkout), "ls-tree", "-r", revision], text=True
    ).splitlines()
    if not listing:
        raise RuntimeError("tracked tree is empty")
    no_symlinks = all(not line.startswith("120000 ") for line in listing)
    return hashlib.sha256(("\n".join(listing) + "\n").encode()).hexdigest(), no_symlinks


def validate_receipt(path: Path, root: Path) -> dict[str, Any]:
    """Validate one terminal generation receipt and its artifacts."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("receipt must be a regular file")
    data: Any = json.loads(path.read_text())
    schema = json.loads(
        (root / "specs/native-easybuild-terminal-receipt-v1.schema.json").read_text()
    )
    try:
        jsonschema.Draft202012Validator(schema).validate(data)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"receipt schema validation failed: {exc.message}") from exc
    _exact_keys(data, RECEIPT_KEYS, "receipt")
    run_meta = _exact_keys(
        data["run"],
        {"run_id", "started_at", "ended_at", "predecessor_receipt_sha256"},
        "run metadata",
    )
    try:
        started = datetime.fromisoformat(run_meta["started_at"])
        ended = datetime.fromisoformat(run_meta["ended_at"])
    except (TypeError, ValueError) as exc:
        raise ValueError("run timestamps are invalid") from exc
    if (
        not run_meta["started_at"].endswith("Z")
        or not run_meta["ended_at"].endswith("Z")
        or started.utcoffset() != UTC.utcoffset(started)
        or ended.utcoffset() != UTC.utcoffset(ended)
        or ended < started
    ):
        raise ValueError("run timestamps are reversed")
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != "voiage.native-easybuild-terminal-receipt.v1"
    ):
        raise ValueError("invalid native EasyBuild receipt schema")
    if data.get("terminal") is not True or data.get("outcome") not in {
        "passed",
        "failed_terminal",
    }:
        raise ValueError("receipt is not terminal")
    identity = data.get("identity")
    _exact_keys(identity, IDENTITY_KEYS, "identity")
    if (identity["generation"] == "2023a") != (
        run_meta["predecessor_receipt_sha256"] is None
    ):
        raise ValueError("generation predecessor receipt binding is invalid")
    if (
        not isinstance(identity, dict)
        or identity.get("source_commit") != BASE
        or identity.get("catalogue_commit") != CATALOGUE
        or identity.get("generation") not in GENERATIONS
        or identity.get("easybuild_version") != "5.4.0"
        or identity.get("easyblocks_version") != "5.4.0"
        or identity.get("root_recipe_sha256")
        != ROOT_RECIPE_SHA256.get(identity.get("generation"))
    ):
        raise ValueError("receipt identity is stale")
    source_manifest, no_source_symlinks = git_manifest_digest(root, BASE)
    expected_tree = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", f"{BASE}^{{tree}}"], text=True
    ).strip()
    if (
        identity["source_tree"] != expected_tree
        or identity["source_relevant_tree_sha256"] != source_manifest
        or not no_source_symlinks
    ):
        raise ValueError("source tree or manifest is stale")
    expected_robot_hash = hashlib.sha256(
        json.dumps(identity["robot_paths"], separators=(",", ":")).encode()
    ).hexdigest()
    if identity["robot_manifest_sha256"] != expected_robot_hash:
        raise ValueError("robot manifest digest is invalid")
    run_root = Path(data["run_root"])
    if run_root != path.parent.resolve(strict=True) or run_root.is_symlink():
        raise ValueError("receipt is outside its exact declared run root")
    environment = _exact_keys(
        data["environment"],
        {
            "allowlist",
            "digest",
            "home",
            "python_no_user_site",
            "module_implementation",
            "module_version",
            "module_init",
        },
        "environment",
    )
    if environment["python_no_user_site"] != "1":
        raise ValueError("user Python environment is not disabled")
    pairs = dict(item.split("=", 1) for item in environment["allowlist"])
    if (
        set(pairs)
        - {"PATH", "LANG", "LC_ALL", "TERM", "TMPDIR", "HOME", "PYTHONNOUSERSITE"}
        or pairs.get("HOME") != str(run_root / "home")
        or pairs.get("PYTHONNOUSERSITE") != "1"
        or environment["home"] != "home"
        or environment["module_implementation"] != "EnvironmentModules"
        or environment["module_init"] != "modulecmd bash"
        or not environment["module_version"]
    ):
        raise ValueError("environment contains inherited or noncanonical values")
    encoded = json.dumps(
        environment["allowlist"], sort_keys=True, separators=(",", ":")
    ).encode()
    if environment["digest"] != hashlib.sha256(encoded).hexdigest():
        raise ValueError("environment digest is invalid")
    preflight = _exact_keys(
        data["preflight"],
        {
            "source_clean",
            "catalogue_clean",
            "source_no_symlinks",
            "catalogue_no_symlinks",
            "prefix_absent",
            "install_empty",
            "module_tree_empty",
            "preinstalled_voiage_absent",
        },
        "preflight",
    )
    if not all(value is True for value in preflight.values()):
        raise ValueError("preflight does not prove fresh clean inputs")
    commands = data.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError("receipt lacks command evidence")
    for item in commands:
        _exact_keys(item, COMMAND_KEYS, "command")
        parsed_keys = {
            "easybuild": {"easybuild_completed", "installed_module_full_name"},
            "module-probe": {"structured_probe_written"},
            "module-unload": {"voiage_absent_after_unload"},
        }[item["stage"]]
        _exact_keys(item["parsed"], parsed_keys, "parsed command evidence")
        artifact = _regular_inside(run_root, str(item.get("log", "")))
        if not artifact.is_file() or sha256(artifact) != item.get("log_sha256"):
            raise ValueError("command log is missing or changed")
    if data["outcome"] == "failed_terminal":
        failure = _exact_keys(
            data["failure"],
            {"stage", "command_index", "exit_code", "signal", "parsed_failure"},
            "failure",
        )
        index = failure["command_index"]
        if (
            not isinstance(index, int)
            or not 0 <= index < len(commands)
            or failure["stage"] != commands[index]["stage"]
            or failure["exit_code"] == 0
            or commands[index]["exit_code"] != failure["exit_code"]
            or not failure["parsed_failure"]
            or data["probes"]
        ):
            raise ValueError("failed terminal receipt lacks exact nonzero evidence")
        return data
    if data["failure"] is not None:
        raise ValueError("passed receipt contains failure evidence")
    if data["outcome"] == "passed":
        if [item["stage"] for item in commands] != [
            "easybuild",
            "module-probe",
            "module-unload",
        ] or any(item.get("exit_code") != 0 for item in commands):
            raise ValueError("passed receipt contains a failed command")
        probes = data.get("probes")
        if (
            not isinstance(probes, list)
            or {p.get("name") for p in probes} != REQUIRED_PROBES
            or any(p.get("status") != "passed" for p in probes)
        ):
            raise ValueError("passed receipt lacks the exact probe matrix")
        generation = identity["generation"]
        recipe = f"packaging/easybuild/voiage-2.2.0-foss-{generation}.eb"
        robot = GENERATIONS[generation] + ["CATALOGUE/easybuild/easyconfigs"]
        expected = [
            "eb",
            recipe,
            "--robot",
            f"--robot-paths={':'.join(robot)}",
            "--prefix=install",
            "--buildpath=build",
            "--sourcepath=sources",
            "--modules-tool=EnvironmentModules",
            "--module-syntax=Tcl",
            "--disable-download",
            "--disable-use-existing-modules",
            "--force",
        ]
        if (
            commands[0]["argv"] != expected
            or not commands[0]["parsed"].get("easybuild_completed")
            or not commands[0]["parsed"].get("installed_module_full_name")
        ):
            raise ValueError(
                "passed receipt does not prove an exact actual EasyBuild build"
            )
        module_name = commands[0]["parsed"]["installed_module_full_name"]
        expected_probe = [
            "bash",
            "--noprofile",
            "--norc",
            "module-probe.sh",
            "MODULE_ROOT",
            module_name,
            "native_easybuild_probe.py",
            "install",
            generation,
            "probe.json",
            "OPPOSITE_INSTALL",
        ]
        if commands[1]["argv"] != expected_probe or commands[2]["argv"] != [
            "bash",
            "--noprofile",
            "--norc",
            "module-unload.sh",
            "MODULE_ROOT",
            module_name,
        ]:
            raise ValueError("module command argv or discovered module name drifted")
        artifacts = _exact_keys(
            data["artifacts"],
            {
                "build_inventory",
                "build_inventory_sha256",
                "module_inventory",
                "module_inventory_sha256",
                "source_cache_inventory",
                "source_cache_inventory_sha256",
                "probe",
                "probe_sha256",
                "preflight",
                "preflight_sha256",
                "catalogue_evidence",
                "catalogue_evidence_sha256",
                "postflight",
                "postflight_sha256",
            },
            "artifacts",
        )
        preflight_path = _regular_inside(run_root, artifacts["preflight"])
        catalogue_path = _regular_inside(run_root, artifacts["catalogue_evidence"])
        postflight_path = _regular_inside(run_root, artifacts["postflight"])
        if (
            sha256(preflight_path) != artifacts["preflight_sha256"]
            or sha256(catalogue_path) != artifacts["catalogue_evidence_sha256"]
            or sha256(postflight_path) != artifacts["postflight_sha256"]
        ):
            raise ValueError("preflight or catalogue evidence is unbound")
        preflight_evidence = json.loads(preflight_path.read_text())
        postflight_evidence = json.loads(postflight_path.read_text())
        expected_preflight_keys = {
            "environment_digest",
            "source_head",
            "source_tree",
            "source_status",
            "catalogue_head",
            "catalogue_tree",
            "catalogue_status",
            "prefix_absent",
            "module_tree_empty",
            "preinstalled_voiage_absent",
            "input_cache_digest",
            "staged_cache_digest_before",
            "staged_cache_digest_after",
        }
        if (
            set(preflight_evidence) != expected_preflight_keys
            or preflight_evidence["environment_digest"] != environment["digest"]
            or preflight_evidence["source_head"] != BASE
            or preflight_evidence["source_tree"] != identity["source_tree"]
            or preflight_evidence["source_status"] != ""
            or preflight_evidence["catalogue_head"] != CATALOGUE
            or preflight_evidence["catalogue_tree"] != identity["catalogue_tree"]
            or preflight_evidence["catalogue_status"] != ""
            or not all(
                preflight_evidence[key] is True
                for key in (
                    "prefix_absent",
                    "module_tree_empty",
                    "preinstalled_voiage_absent",
                )
            )
            or len(
                {
                    preflight_evidence[key]
                    for key in (
                        "input_cache_digest",
                        "staged_cache_digest_before",
                        "staged_cache_digest_after",
                    )
                }
            )
            != 1
        ):
            raise ValueError("preflight transcript is invalid")
        if postflight_evidence != {
            key: preflight_evidence[key]
            for key in (
                "source_head",
                "source_tree",
                "source_status",
                "catalogue_head",
                "catalogue_tree",
                "catalogue_status",
            )
        }:
            raise ValueError("postflight checkout evidence drifted")
        catalogue_evidence = json.loads(catalogue_path.read_text())
        listing = catalogue_evidence.get("ls_tree")
        if (
            not isinstance(listing, list)
            or not listing
            or hashlib.sha256(("\n".join(listing) + "\n").encode()).hexdigest()
            != identity["catalogue_relevant_tree_sha256"]
            or catalogue_evidence
            != {
                "commit": identity["catalogue_commit"],
                "tree": identity["catalogue_tree"],
                "manifest_sha256": identity["catalogue_relevant_tree_sha256"],
                "ls_tree": listing,
            }
        ):
            raise ValueError("portable catalogue evidence is invalid")
        if (
            identity["catalogue_tree"] != CATALOGUE_TREE
            or identity["catalogue_relevant_tree_sha256"] != CATALOGUE_MANIFEST_SHA256
        ):
            raise ValueError(
                "catalogue object closure differs from the pinned contract"
            )
        inventories: dict[str, list[dict[str, Any]]] = {}
        for key in ("build_inventory", "module_inventory", "source_cache_inventory"):
            inventory = _regular_inside(run_root, artifacts[key])
            if sha256(inventory) != artifacts[key + "_sha256"]:
                raise ValueError("inventory is unbound")
            inventories[key] = _manifest(inventory)
        if (
            identity["source_cache_manifest_sha256"]
            != artifacts["source_cache_inventory_sha256"]
        ):
            raise ValueError("source cache identity binding is invalid")
        inventory_paths = {
            key: {item["path"] for item in value} for key, value in inventories.items()
        }
        if not any(
            "voiage/" in item.lower() for item in inventory_paths["module_inventory"]
        ):
            raise ValueError("module inventory lacks generated Voiage modulefile")
        required_tokens = ("voiage", "_core", "python", "pyarrow", "polars")
        if any(
            not any(
                token in item.lower() for item in inventory_paths["build_inventory"]
            )
            for token in required_tokens
        ):
            raise ValueError("install inventory lacks required native runtime entries")
        if not inventory_paths["source_cache_inventory"]:
            raise ValueError("source inventory lacks checksum-bound sources")
        probe_path = _regular_inside(run_root, artifacts["probe"])
        if sha256(probe_path) != artifacts["probe_sha256"]:
            raise ValueError("structured probe transcript is unbound or changed")
        probe: Any = json.loads(probe_path.read_text())
        _exact_keys(
            probe,
            {
                "schema_version",
                "generation",
                "paths",
                "evpi",
                "arrow",
                "polars",
                "linkage",
                "thread",
                "module",
            },
            "probe",
        )
        if (
            probe.get("generation") != identity["generation"]
            or probe.get("evpi")
            != {
                "input": [[0.0, 2.0], [2.0, 0.0]],
                "dtype": "float64",
                "value": 1.0,
                "tolerance": 0.0,
            }
            or probe.get("arrow")
            != {
                "version": "25.0.1",
                "schema": "value: int64",
                "values": [1, None, 3],
                "null_count": 1,
                "buffer_equal": True,
                "buffer_size_positive": True,
            }
            or probe.get("polars")
            != {
                "version": "1.42.1",
                "schema": {"value": "Int64"},
                "values": [1, 3],
                "null_count": 0,
                "lazy": True,
                "arrow_equal": True,
            }
            or probe.get("thread")
            != {"calls": 16, "imports_inside_worker": True, "engines": ["rust"] * 16}
        ):
            raise ValueError("structured native probe transcript is invalid")
        linkage = _exact_keys(
            probe["linkage"],
            {
                "objects",
                "tool",
                "targets",
                "transcripts",
            },
            "linkage",
        )
        if len(linkage["objects"]) < 3 or set(linkage["transcripts"]) != set(
            linkage["objects"]
        ):
            raise ValueError("native linkage evidence is incomplete")
        install_root = run_root / "install"
        for object_path in linkage["objects"]:
            candidate = Path(object_path)
            try:
                candidate.resolve(strict=True).relative_to(
                    install_root.resolve(strict=True)
                )
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError(
                    "linkage object escapes the exact generation prefix"
                ) from exc
        transcript_text = "\n".join(linkage["transcripts"].values())
        parsed_targets = sorted(set(re.findall(r"(?<!\S)(/[^\s(]+)", transcript_text)))
        if linkage["targets"] != parsed_targets or "not found" in transcript_text:
            raise ValueError("linkage targets do not match retained transcripts")
        opposite = (
            run_root.parent
            / f"voiage-easybuild-{'2024a' if identity['generation'] == '2023a' else '2023a'}"
            / "install"
        )
        system_roots = (
            Path("/lib"),
            Path("/lib64"),
            Path("/usr/lib"),
            Path("/usr/lib64"),
        )
        for target_text in parsed_targets:
            target = Path(target_text)
            if (
                target == opposite
                or opposite in target.parents
                or (
                    not (target == install_root or install_root in target.parents)
                    and not any(
                        target == base or base in target.parents
                        for base in system_roots
                    )
                )
            ):
                raise ValueError("linkage target escapes exact allowed roots")
        if probe["module"] != {
            "loaded_paths_introduced": True,
            "unload_paths_removed": True,
            "fresh_shell": True,
        }:
            raise ValueError("module isolation evidence is incomplete")
        if (
            not isinstance(probe["paths"], list)
            or len(probe["paths"]) != 6
            or any(
                not (Path(item) == install_root or install_root in Path(item).parents)
                for item in probe["paths"]
            )
        ):
            raise ValueError("installed runtime paths are incomplete")
        if (
            identity.get("source_tree")
            != subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", f"{BASE}^{{tree}}"],
                text=True,
            ).strip()
        ):
            raise ValueError("source tree is stale")
    return data


def validate_matrix(path: Path, root: Path) -> dict[str, Any]:
    """Require terminal passing receipts for both maintained generations."""
    data: Any = json.loads(path.read_text())
    if (
        not isinstance(data, dict)
        or set(data)
        != {"schema_version", "run_id", "shared_inputs", "sequence", "receipts"}
        or data["schema_version"] != "voiage.native-easybuild-matrix.v1"
    ):
        raise ValueError("invalid native EasyBuild matrix")
    if (
        data["sequence"] != ["2023a", "2024a"]
        or not isinstance(data["run_id"], str)
        or not data["run_id"]
    ):
        raise ValueError("matrix does not prove sequential generation order")
    shared = _exact_keys(
        data["shared_inputs"],
        {
            "source_commit",
            "source_tree",
            "catalogue_commit",
            "easybuild_version",
            "easyblocks_version",
        },
        "matrix shared inputs",
    )
    if shared != {
        "source_commit": BASE,
        "source_tree": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", f"{BASE}^{{tree}}"], text=True
        ).strip(),
        "catalogue_commit": CATALOGUE,
        "easybuild_version": "5.4.0",
        "easyblocks_version": "5.4.0",
    }:
        raise ValueError("matrix shared inputs are stale")
    receipts = data["receipts"]
    if not isinstance(receipts, dict) or set(receipts) != set(GENERATIONS):
        raise ValueError("both EasyBuild generations are required")
    if any(
        not isinstance(item, dict) or set(item) != {"path", "sha256"}
        for item in receipts.values()
    ):
        raise ValueError("matrix receipt bindings are invalid")
    if len({item["path"] for item in receipts.values()}) != 2:
        raise ValueError("generation receipts must be distinct")
    seen_roots: set[str] = set()
    validated: dict[str, tuple[dict[str, Any], Path]] = {}
    for generation, binding in receipts.items():
        receipt_path = _regular_inside(
            path.parent.resolve(strict=True), str(binding["path"])
        )
        if sha256(receipt_path) != binding["sha256"]:
            raise ValueError("matrix receipt digest is invalid")
        receipt = validate_receipt(receipt_path, root)
        if (
            receipt["outcome"] != "passed"
            or receipt["identity"]["generation"] != generation
        ):
            raise ValueError("both EasyBuild generations must pass exactly")
        if receipt["run_root"] in seen_roots:
            raise ValueError("generation run roots must be distinct")
        seen_roots.add(receipt["run_root"])
        validated[generation] = (receipt, receipt_path)
    earlier, earlier_path = validated["2023a"]
    later, _ = validated["2024a"]
    if (
        earlier["run"]["run_id"] != data["run_id"]
        or later["run"]["run_id"] != data["run_id"]
        or later["run"]["predecessor_receipt_sha256"] != sha256(earlier_path)
        or datetime.fromisoformat(earlier["run"]["ended_at"])
        > datetime.fromisoformat(later["run"]["started_at"])
    ):
        raise ValueError("matrix receipts do not form one ordered sequential run")
    return data


def main() -> int:
    """Run one guarded EasyBuild generation and capture terminal evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("generation", choices=GENERATIONS)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--predecessor-receipt", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    if args.generation == "2024a" and args.predecessor_receipt is None:
        raise SystemExit("2024a requires the terminal 2023a predecessor receipt")
    if args.generation == "2023a" and args.predecessor_receipt is not None:
        raise SystemExit("2023a must start the sequential run")
    predecessor_sha = (
        sha256(args.predecessor_receipt) if args.predecessor_receipt else None
    )
    if (
        args.root.absolute() != args.root.resolve()
        or args.workspace.absolute() != args.workspace.resolve()
    ):
        raise SystemExit("source and workspace roots must not be symlinks")
    root = args.root.resolve()
    workspace = args.workspace.resolve()
    source_cache = args.source_cache.resolve()
    if args.predecessor_receipt:
        predecessor = validate_receipt(args.predecessor_receipt.resolve(), root)
        if (
            predecessor["outcome"] != "passed"
            or predecessor["identity"]["generation"] != "2023a"
            or predecessor["run"]["run_id"] != args.run_id
            or predecessor["identity"]["source_commit"] != BASE
            or predecessor["identity"]["catalogue_commit"] != CATALOGUE
        ):
            raise SystemExit(
                "predecessor is not the passing 2023a receipt for this run"
            )
    if not args.execute:
        raise SystemExit("refusing native build without explicit --execute")
    if source_cache.is_symlink() or not source_cache.is_dir():
        raise SystemExit("prepared source cache must be a real directory")
    if any(item.is_symlink() for item in source_cache.rglob("*")):
        raise SystemExit("prepared source cache must not contain symlinks")
    input_cache_digest = directory_digest(source_cache)
    if (
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        != BASE
    ):
        raise SystemExit("source checkout is not the reviewed commit")
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip():
        raise SystemExit("source checkout must be clean")
    if (
        subprocess.check_output(
            ["git", "-C", str(args.catalogue), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        != CATALOGUE
    ):
        raise SystemExit("EasyBuild catalogue is not the reviewed commit")
    if (
        sys.platform != "linux"
        or shutil.which("eb") is None
        or shutil.which("modulecmd") is None
    ):
        raise SystemExit("Linux, EasyBuild, and Environment Modules are required")
    if "5.4.0" not in subprocess.check_output(["eb", "--version"], text=True):
        raise SystemExit("EasyBuild framework 5.4.0 is required")
    easyblocks_version = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import easybuild.easyblocks; print(easybuild.easyblocks.__version__)",
        ],
        text=True,
    ).strip()
    if easyblocks_version != "5.4.0":
        raise SystemExit("EasyBuild easyblocks 5.4.0 is required")
    if root == workspace or root in workspace.parents or workspace in root.parents:
        raise SystemExit("workspace and source must be separate, non-symlink trees")
    if os.environ.get("VOIAGE_VM_LOCAL_STORAGE") != "confirmed":
        raise SystemExit("workspace must be explicitly confirmed as VM-local storage")
    free = shutil.disk_usage(workspace).free
    if free < 250 * 1024**3 or (os.cpu_count() or 0) < 8:
        raise SystemExit("native workspace needs 250 GiB free and at least 8 CPUs")
    run_dir = workspace / f"voiage-easybuild-{args.generation}"
    run_dir.mkdir()  # deliberate refusal to overwrite evidence
    catalogue = args.catalogue.resolve()
    if subprocess.check_output(
        ["git", "-C", str(catalogue), "status", "--porcelain"], text=True
    ).strip():
        raise SystemExit("EasyBuild catalogue must be clean")
    robot_relative = GENERATIONS[args.generation] + ["CATALOGUE/easybuild/easyconfigs"]
    robot_actual = [str(root / item) for item in GENERATIONS[args.generation]] + [
        str(catalogue / "easybuild/easyconfigs")
    ]
    recipe_relative = f"packaging/easybuild/voiage-2.2.0-foss-{args.generation}.eb"
    recipe = root / recipe_relative
    prefix, build, sources, home = (
        run_dir / "install",
        run_dir / "build",
        run_dir / "sources",
        run_dir / "home",
    )
    if any(item.exists() for item in (prefix, build, sources, home)):
        raise SystemExit("generation prefix/build/source/home must start absent")
    for item in (build, home):
        item.mkdir()
    shutil.copytree(source_cache, sources, copy_function=shutil.copy2)
    for item in sources.rglob("*"):
        item.chmod(0o555 if item.is_dir() else 0o444)
    sources.chmod(0o555)
    staged_cache_digest_before = directory_digest(sources)
    if staged_cache_digest_before != input_cache_digest:
        raise SystemExit("staged source cache differs from prepared input")
    env, allowlist, env_digest = clean_environment(home)
    actual_argv = [
        "eb",
        str(recipe),
        "--robot",
        f"--robot-paths={':'.join(robot_actual)}",
        f"--prefix={prefix}",
        f"--buildpath={build}",
        f"--sourcepath={sources}",
        "--modules-tool=EnvironmentModules",
        "--module-syntax=Tcl",
        "--disable-download",
        "--disable-use-existing-modules",
        "--force",
    ]
    recorded_argv = [
        "eb",
        recipe_relative,
        "--robot",
        f"--robot-paths={':'.join(robot_relative)}",
        "--prefix=install",
        "--buildpath=build",
        "--sourcepath=sources",
        "--modules-tool=EnvironmentModules",
        "--module-syntax=Tcl",
        "--disable-download",
        "--disable-use-existing-modules",
        "--force",
    ]
    absent_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util; raise SystemExit(importlib.util.find_spec('voiage') is not None)",
        ],
        cwd=home,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    preflight_file = run_dir / "preflight.json"
    preflight_evidence = {
        "environment_digest": env_digest,
        "source_head": BASE,
        "source_tree": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True
        ).strip(),
        "source_status": subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        ).strip(),
        "catalogue_head": CATALOGUE,
        "catalogue_tree": subprocess.check_output(
            ["git", "-C", str(catalogue), "rev-parse", "HEAD^{tree}"], text=True
        ).strip(),
        "catalogue_status": subprocess.check_output(
            ["git", "-C", str(catalogue), "status", "--porcelain"], text=True
        ).strip(),
        "prefix_absent": not prefix.exists(),
        "module_tree_empty": not prefix.exists(),
        "preinstalled_voiage_absent": absent_probe.returncode == 0,
        "input_cache_digest": input_cache_digest,
        "staged_cache_digest_before": staged_cache_digest_before,
        "staged_cache_digest_after": None,
    }
    preflight_file.write_text(json.dumps(preflight_evidence, sort_keys=True) + "\n")
    if absent_probe.returncode != 0:
        raise SystemExit("clean preflight found a preinstalled Voiage")
    build_log = run_dir / "build.log"
    build_code = run(actual_argv, build_log, env)
    commands = [
        {
            "stage": "easybuild",
            "argv": recorded_argv,
            "exit_code": build_code,
            "signal": -build_code if build_code < 0 else None,
            "log": "build.log",
            "log_sha256": sha256(build_log),
            "parsed": {
                "easybuild_completed": False,
                "installed_module_full_name": None,
            },
        }
    ]
    probes: list[dict[str, str]] = []
    failure: dict[str, Any] | None = None
    probe_json = run_dir / "probe.json"
    module_files = list(prefix.rglob("voiage/2.2.0*")) if prefix.exists() else []
    if (
        build_code == 0
        and len(module_files) == 1
        and module_files[0].is_file()
        and not module_files[0].is_symlink()
    ):
        module_name = str(
            module_files[0].relative_to(module_files[0].parents[1])
        ).removesuffix(".lua")
        commands[0]["parsed"]["installed_module_full_name"] = module_name
        commands[0]["parsed"]["easybuild_completed"] = parse_build_success(
            build_log, module_name
        )
        script = run_dir / "module-probe.sh"
        script.write_text(
            '#!/bin/bash\nset -euo pipefail\neval "$(modulecmd bash purge)"\neval "$(modulecmd bash use "$1")"\neval "$(modulecmd bash load "$2")"\nexec python3 "$3" --prefix "$4" --generation "$5" --output "$6" --opposite-prefix "$7"\n'
        )
        script.chmod(0o700)
        probe_log = run_dir / "probe.log"
        probe_argv = [
            "bash",
            "--noprofile",
            "--norc",
            str(script),
            str(module_files[0].parents[1]),
            module_name,
            str(root / "scripts/native_easybuild_probe.py"),
            str(prefix),
            args.generation,
            str(probe_json),
            str(
                workspace
                / f"voiage-easybuild-{'2024a' if args.generation == '2023a' else '2023a'}"
                / "install"
            ),
        ]
        probe_code = run(probe_argv, probe_log, env)
        commands.append(
            {
                "stage": "module-probe",
                "argv": [
                    "bash",
                    "--noprofile",
                    "--norc",
                    "module-probe.sh",
                    "MODULE_ROOT",
                    module_name,
                    "native_easybuild_probe.py",
                    "install",
                    args.generation,
                    "probe.json",
                    "OPPOSITE_INSTALL",
                ],
                "exit_code": probe_code,
                "signal": -probe_code if probe_code < 0 else None,
                "log": "probe.log",
                "log_sha256": sha256(probe_log),
                "parsed": {"structured_probe_written": probe_json.is_file()},
            }
        )
        unload_log = run_dir / "unload.log"
        unload_script = run_dir / "module-unload.sh"
        unload_script.write_text(
            '#!/bin/bash\nset -euo pipefail\neval "$(modulecmd bash purge)"\neval "$(modulecmd bash use "$1")"\neval "$(modulecmd bash load "$2")"\neval "$(modulecmd bash unload "$2")"\npython3 -c \'import importlib.util; raise SystemExit(importlib.util.find_spec("voiage") is not None)\'\n'
        )
        unload_script.chmod(0o700)
        unload_code = run(
            [
                "bash",
                "--noprofile",
                "--norc",
                str(unload_script),
                str(module_files[0].parents[1]),
                module_name,
            ],
            unload_log,
            env,
        )
        commands.append(
            {
                "stage": "module-unload",
                "argv": [
                    "bash",
                    "--noprofile",
                    "--norc",
                    "module-unload.sh",
                    "MODULE_ROOT",
                    module_name,
                ],
                "exit_code": unload_code,
                "signal": -unload_code if unload_code < 0 else None,
                "log": "unload.log",
                "log_sha256": sha256(unload_log),
                "parsed": {"voiage_absent_after_unload": unload_code == 0},
            }
        )
        code = probe_code or unload_code
        if code == 0:
            probes = [
                {"name": name, "status": "passed"} for name in sorted(REQUIRED_PROBES)
            ]
    else:
        code = build_code or 1
    if code != 0:
        index = next(
            (i for i, item in enumerate(commands) if item["exit_code"] != 0), 0
        )
        commands[index]["exit_code"] = commands[index]["exit_code"] or 1
        failure = {
            "stage": commands[index]["stage"],
            "command_index": index,
            "exit_code": commands[index]["exit_code"],
            "signal": commands[index]["signal"],
            "parsed_failure": parsed_failure(run_dir / commands[index]["log"]),
        }
    artifacts: dict[str, Any] = {
        "build_inventory": None,
        "build_inventory_sha256": None,
        "module_inventory": None,
        "module_inventory_sha256": None,
        "source_cache_inventory": None,
        "source_cache_inventory_sha256": None,
        "probe": None,
        "probe_sha256": None,
        "preflight": None,
        "preflight_sha256": None,
        "catalogue_evidence": None,
        "catalogue_evidence_sha256": None,
        "postflight": None,
        "postflight_sha256": None,
    }
    if code == 0:
        preflight_evidence["staged_cache_digest_after"] = directory_digest(sources)
        preflight_file.write_text(json.dumps(preflight_evidence, sort_keys=True) + "\n")
        postflight_file = run_dir / "postflight.json"
        postflight = {
            "source_head": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_tree": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True
            ).strip(),
            "source_status": subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip(),
            "catalogue_head": subprocess.check_output(
                ["git", "-C", str(catalogue), "rev-parse", "HEAD"], text=True
            ).strip(),
            "catalogue_tree": subprocess.check_output(
                ["git", "-C", str(catalogue), "rev-parse", "HEAD^{tree}"], text=True
            ).strip(),
            "catalogue_status": subprocess.check_output(
                ["git", "-C", str(catalogue), "status", "--porcelain"], text=True
            ).strip(),
        }
        expected_postflight = {key: preflight_evidence[key] for key in postflight}
        if postflight != expected_postflight:
            raise SystemExit("source or catalogue checkout drifted during build")
        postflight_file.write_text(json.dumps(postflight, sort_keys=True) + "\n")
        artifacts.update(
            {
                "postflight": postflight_file.name,
                "postflight_sha256": sha256(postflight_file),
            }
        )
        catalogue_manifest, _ = git_manifest_digest(catalogue, "HEAD")
        catalogue_tree = subprocess.check_output(
            ["git", "-C", str(catalogue), "rev-parse", "HEAD^{tree}"], text=True
        ).strip()
        catalogue_file = run_dir / "catalogue-evidence.json"
        catalogue_listing = subprocess.check_output(
            ["git", "-C", str(catalogue), "ls-tree", "-r", "HEAD"], text=True
        ).splitlines()
        catalogue_file.write_text(
            json.dumps(
                {
                    "commit": CATALOGUE,
                    "tree": catalogue_tree,
                    "manifest_sha256": catalogue_manifest,
                    "ls_tree": catalogue_listing,
                },
                sort_keys=True,
            )
            + "\n"
        )
        artifacts.update(
            {
                "preflight": preflight_file.name,
                "preflight_sha256": sha256(preflight_file),
                "catalogue_evidence": catalogue_file.name,
                "catalogue_evidence_sha256": sha256(catalogue_file),
            }
        )
        for key, directory in (
            ("build_inventory", prefix),
            ("module_inventory", module_files[0].parents[1]),
            ("source_cache_inventory", sources),
        ):
            target = run_dir / f"{key}.json"
            artifacts[key] = target.name
            artifacts[key + "_sha256"] = write_inventory(directory, target)
        artifacts["probe"], artifacts["probe_sha256"] = "probe.json", sha256(probe_json)
    source_manifest, source_no_symlinks = git_manifest_digest(root, "HEAD")
    catalogue_manifest, catalogue_no_symlinks = git_manifest_digest(catalogue, "HEAD")
    source_tree = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    catalogue_tree = subprocess.check_output(
        ["git", "-C", str(catalogue), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    receipt = {
        "schema_version": "voiage.native-easybuild-terminal-receipt.v1",
        "terminal": True,
        "outcome": "passed" if code == 0 else "failed_terminal",
        "run_root": str(run_dir.resolve()),
        "identity": {
            "generation": args.generation,
            "source_commit": BASE,
            "source_tree": source_tree,
            "source_relevant_tree_sha256": source_manifest,
            "catalogue_root": str(catalogue),
            "catalogue_commit": CATALOGUE,
            "catalogue_tree": catalogue_tree,
            "catalogue_relevant_tree_sha256": catalogue_manifest,
            "easybuild_version": "5.4.0",
            "easyblocks_version": easyblocks_version,
            "root_recipe": recipe_relative,
            "root_recipe_sha256": sha256(recipe),
            "robot_paths": robot_relative,
            "robot_manifest_sha256": hashlib.sha256(
                json.dumps(robot_relative, separators=(",", ":")).encode()
            ).hexdigest(),
            "source_cache_manifest_sha256": artifacts["source_cache_inventory_sha256"]
            or "0" * 64,
        },
        "environment": {
            "allowlist": allowlist,
            "digest": env_digest,
            "home": "home",
            "python_no_user_site": "1",
            "module_implementation": "EnvironmentModules",
            "module_version": subprocess.check_output(
                ["modulecmd", "--version"], text=True, stderr=subprocess.STDOUT
            ).strip(),
            "module_init": "modulecmd bash",
        },
        "resources": {
            "cpu_count": os.cpu_count(),
            "free_bytes_before": free,
        },
        "preflight": {
            "source_clean": True,
            "catalogue_clean": True,
            "source_no_symlinks": source_no_symlinks,
            "catalogue_no_symlinks": catalogue_no_symlinks,
            "prefix_absent": True,
            "install_empty": True,
            "module_tree_empty": True,
            "preinstalled_voiage_absent": True,
        },
        "commands": commands,
        "artifacts": artifacts,
        "probes": probes,
        "failure": failure,
        "run": {
            "run_id": args.run_id,
            "started_at": started_at,
            "ended_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "predecessor_receipt_sha256": predecessor_sha,
        },
    }
    receipt_path = run_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    validate_receipt(receipt_path, root)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
