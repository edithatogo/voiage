from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any

import jsonschema
import pytest

import scripts.native_easybuild_qualification as qualification


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _inventory(run_root: Path, name: str) -> tuple[str, str]:
    roots = {
        "build-inventory": "install",
        "module-inventory": "install/modules/all",
        "source-cache-inventory": "sources",
    }
    names = {
        "build-inventory": [
            "install/bin/python",
            "install/lib/python/site-packages/voiage/__init__.py",
            "install/lib/python/site-packages/voiage/_core.so",
            "install/lib/python/site-packages/numpy/__init__.py",
            "install/lib/python/site-packages/pyarrow/__init__.py",
            "install/lib/python/site-packages/polars/__init__.py",
            "install/lib/voiage/_core.so",
            "install/lib/pyarrow/lib.so",
            "install/lib/polars/polars.so",
        ],
        "module-inventory": ["install/modules/all/voiage/2.2.0.lua"],
        "source-cache-inventory": ["sources/voiage-2.2.0.tar.gz"],
    }[name]
    payloads = []
    for relative in names:
        payload = run_root / relative
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_text(f"real {relative} payload\n")
        payloads.append(payload)
    inventory = run_root / f"{name}.json"
    inventory_root = run_root / roots[name]
    _write_json(
        inventory,
        {
            "root": roots[name],
            "entries": [
                {
                    "path": str(payload.relative_to(inventory_root)),
                    "type": "file",
                    "size": payload.stat().st_size,
                    "sha256": _digest(payload),
                }
                for payload in payloads
            ],
        },
    )
    return inventory.name, _digest(inventory)


def _probe(generation: str, prefix: Path | None = None) -> dict[str, Any]:
    prefix = prefix or Path(f"/opt/easybuild/{generation}")
    return {
        "schema_version": "voiage.native-easybuild-probe.v1",
        "generation": generation,
        "paths": [
            f"{prefix}/bin/python",
            f"{prefix}/lib/python/site-packages/voiage/__init__.py",
            f"{prefix}/lib/python/site-packages/voiage/_core.so",
            f"{prefix}/lib/python/site-packages/numpy/__init__.py",
            f"{prefix}/lib/python/site-packages/pyarrow/__init__.py",
            f"{prefix}/lib/python/site-packages/polars/__init__.py",
        ],
        "evpi": {
            "input": [[0.0, 2.0], [2.0, 0.0]],
            "dtype": "float64",
            "value": 1.0,
            "tolerance": 0.0,
        },
        "arrow": {
            "version": "25.0.1",
            "schema": "value: int64",
            "values": [1, None, 3],
            "null_count": 1,
            "buffer_equal": True,
            "buffer_size_positive": True,
        },
        "polars": {
            "version": "1.42.1",
            "schema": {"value": "Int64"},
            "values": [1, 3],
            "null_count": 0,
            "lazy": True,
            "arrow_equal": True,
        },
        "linkage": {
            "objects": [
                f"{prefix}/lib/voiage/_core.so",
                f"{prefix}/lib/pyarrow/lib.so",
                f"{prefix}/lib/polars/polars.so",
            ],
            "tool": "/usr/bin/ldd",
            "targets": ["/usr/lib/libc.so"],
            "transcripts": {
                f"{prefix}/lib/voiage/_core.so": "libc.so => /usr/lib/libc.so (0x1)\n",
                f"{prefix}/lib/pyarrow/lib.so": "libc.so => /usr/lib/libc.so (0x1)\n",
                f"{prefix}/lib/polars/polars.so": "libc.so => /usr/lib/libc.so (0x1)\n",
            },
        },
        "thread": {
            "calls": 16,
            "imports_inside_worker": True,
            "engines": ["rust"] * 16,
        },
        "module": {
            "loaded_paths_introduced": True,
            "unload_paths_removed": True,
            "fresh_shell": True,
        },
    }


def _receipt(tmp_path: Path, generation: str = "2023a") -> Path:
    run_root = tmp_path.resolve()
    logs = {name: run_root / f"{name}.log" for name in ("build", "probe", "unload")}
    logs["build"].write_text(
        "== COMPLETED: Installation ended successfully (took 1 min)\n"
    )
    logs["probe"].write_text("module use, load, CLI and native probes passed\n")
    logs["unload"].write_text("module unloaded and import absent\n")
    probe_path = run_root / "probe.json"
    _write_json(probe_path, _probe(generation, run_root / "install"))
    build_inventory, build_hash = _inventory(run_root, "build-inventory")
    module_inventory, module_hash = _inventory(run_root, "module-inventory")
    source_inventory, source_hash = _inventory(run_root, "source-cache-inventory")
    recipe = f"packaging/easybuild/voiage-2.2.0-foss-{generation}.eb"
    robot = qualification.GENERATIONS[generation] + ["CATALOGUE/easybuild/easyconfigs"]
    allowlist = [
        f"HOME={run_root / 'home'}",
        "LANG=C.UTF-8",
        "PATH=/usr/bin:/bin",
        "PYTHONNOUSERSITE=1",
    ]
    env_digest = hashlib.sha256(
        json.dumps(allowlist, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_tree = subprocess.check_output(
        ["/usr/bin/git", "rev-parse", f"{qualification.BASE}^{{tree}}"], text=True
    ).strip()
    source_manifest, _ = qualification.git_manifest_digest(
        Path.cwd(), qualification.BASE
    )
    tooling_root = Path(qualification.__file__).resolve().parents[1]
    tooling_commit = subprocess.check_output(
        ["/usr/bin/git", "-C", str(tooling_root), "rev-parse", "HEAD"], text=True
    ).strip()
    tooling_tree = subprocess.check_output(
        ["/usr/bin/git", "-C", str(tooling_root), "rev-parse", "HEAD^{tree}"],
        text=True,
    ).strip()
    preflight_path = run_root / "preflight.json"
    _write_json(
        preflight_path,
        {
            "environment_digest": env_digest,
            "source_head": qualification.BASE,
            "source_tree": source_tree,
            "source_status": "",
            "catalogue_head": qualification.BASE,
            "catalogue_tree": source_tree,
            "catalogue_status": "",
            "tooling_head": tooling_commit,
            "tooling_tree": tooling_tree,
            "tooling_status": "",
            "prefix_absent": True,
            "module_tree_empty": True,
            "preinstalled_voiage_absent": True,
            "input_cache_digest": "f" * 64,
            "staged_cache_digest_before": "f" * 64,
            "staged_cache_digest_after": "f" * 64,
        },
    )
    postflight_path = run_root / "postflight.json"
    _write_json(
        postflight_path,
        {
            "source_head": qualification.BASE,
            "source_tree": source_tree,
            "source_status": "",
            "catalogue_head": qualification.BASE,
            "catalogue_tree": source_tree,
            "catalogue_status": "",
            "tooling_head": tooling_commit,
            "tooling_tree": tooling_tree,
            "tooling_status": "",
        },
    )
    catalogue_path = run_root / "catalogue-evidence.json"
    catalogue_listing = subprocess.check_output(
        ["/usr/bin/git", "ls-tree", "-r", qualification.BASE], text=True
    ).splitlines()
    _write_json(
        catalogue_path,
        {
            "commit": qualification.BASE,
            "tree": source_tree,
            "manifest_sha256": source_manifest,
            "ls_tree": catalogue_listing,
        },
    )
    data = {
        "schema_version": "voiage.native-easybuild-terminal-receipt.v1",
        "terminal": True,
        "outcome": "passed",
        "run_root": str(run_root),
        "identity": {
            "generation": generation,
            "source_commit": qualification.BASE,
            "source_tree": source_tree,
            "source_relevant_tree_sha256": source_manifest,
            "catalogue_root": str(Path.cwd()),
            "catalogue_commit": qualification.BASE,
            "catalogue_tree": source_tree,
            "catalogue_relevant_tree_sha256": source_manifest,
            "easybuild_version": "5.4.0",
            "easyblocks_version": "5.4.0",
            "root_recipe": recipe,
            "root_recipe_sha256": qualification.ROOT_RECIPE_SHA256[generation],
            "robot_paths": robot,
            "robot_manifest_sha256": hashlib.sha256(
                json.dumps(robot, separators=(",", ":")).encode()
            ).hexdigest(),
            "source_cache_manifest_sha256": source_hash,
            "tooling_root": str(tooling_root),
            "tooling_commit": tooling_commit,
            "tooling_tree": tooling_tree,
            "tooling_driver_sha256": qualification.git_blob_sha256(
                tooling_root, tooling_commit, qualification.TOOLING_FILES[0]
            ),
            "tooling_probe_sha256": qualification.git_blob_sha256(
                tooling_root, tooling_commit, qualification.TOOLING_FILES[1]
            ),
        },
        "environment": {
            "allowlist": allowlist,
            "digest": env_digest,
            "home": "home",
            "python_no_user_site": "1",
            "module_implementation": "EnvironmentModules",
            "module_version": "5.4.0",
            "module_init": "modulecmd bash",
        },
        "resources": {"cpu_count": 8, "free_bytes_before": 300 * 1024**3},
        "preflight": {
            "source_clean": True,
            "catalogue_clean": True,
            "source_no_symlinks": True,
            "catalogue_no_symlinks": True,
            "prefix_absent": True,
            "install_empty": True,
            "module_tree_empty": True,
            "preinstalled_voiage_absent": True,
        },
        "commands": [
            {
                "stage": "easybuild",
                "argv": [
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
                ],
                "exit_code": 0,
                "signal": None,
                "log": logs["build"].name,
                "log_sha256": _digest(logs["build"]),
                "parsed": {
                    "easybuild_completed": True,
                    "installed_module_full_name": f"voiage/2.2.0-foss-{generation}",
                },
            },
            {
                "stage": "module-probe",
                "argv": [
                    "bash",
                    "--noprofile",
                    "--norc",
                    "module-probe.sh",
                    "MODULE_ROOT",
                    f"voiage/2.2.0-foss-{generation}",
                    "native_easybuild_probe.py",
                    "install",
                    generation,
                    "probe.json",
                    "OPPOSITE_INSTALL",
                ],
                "exit_code": 0,
                "signal": None,
                "log": logs["probe"].name,
                "log_sha256": _digest(logs["probe"]),
                "parsed": {"structured_probe_written": True},
            },
            {
                "stage": "module-unload",
                "argv": [
                    "bash",
                    "--noprofile",
                    "--norc",
                    "module-unload.sh",
                    "MODULE_ROOT",
                    f"voiage/2.2.0-foss-{generation}",
                    "home",
                ],
                "exit_code": 0,
                "signal": None,
                "log": logs["unload"].name,
                "log_sha256": _digest(logs["unload"]),
                "parsed": {"voiage_absent_after_unload": True},
            },
        ],
        "artifacts": {
            "build_inventory": build_inventory,
            "build_inventory_sha256": build_hash,
            "module_inventory": module_inventory,
            "module_inventory_sha256": module_hash,
            "source_cache_inventory": source_inventory,
            "source_cache_inventory_sha256": source_hash,
            "probe": probe_path.name,
            "probe_sha256": _digest(probe_path),
            "preflight": preflight_path.name,
            "preflight_sha256": _digest(preflight_path),
            "catalogue_evidence": catalogue_path.name,
            "catalogue_evidence_sha256": _digest(catalogue_path),
            "postflight": postflight_path.name,
            "postflight_sha256": _digest(postflight_path),
        },
        "probes": [
            {"name": name, "status": "passed"}
            for name in sorted(qualification.REQUIRED_PROBES)
        ],
        "failure": None,
        "run": {
            "run_id": "test-run",
            "started_at": "2026-09-04T00:00:00Z",
            "ended_at": "2026-09-04T01:00:00Z",
            "predecessor_receipt_sha256": None if generation == "2023a" else "a" * 64,
        },
    }
    path = run_root / "receipt.json"
    _write_json(path, data)
    return path


@pytest.fixture(autouse=True)
def _catalogue_at_test_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qualification, "CATALOGUE", qualification.BASE)
    tree = subprocess.check_output(
        ["/usr/bin/git", "rev-parse", f"{qualification.BASE}^{{tree}}"], text=True
    ).strip()
    manifest, _ = qualification.git_manifest_digest(Path.cwd(), qualification.BASE)
    monkeypatch.setattr(qualification, "CATALOGUE_TREE", tree)
    monkeypatch.setattr(qualification, "CATALOGUE_MANIFEST_SHA256", manifest)
    real = qualification.subprocess.check_output

    def clean_status(argv: list[str], **kwargs: Any) -> str:
        if argv[-2:] == ["status", "--porcelain"]:
            return ""
        return real(argv, **kwargs)

    monkeypatch.setattr(qualification.subprocess, "check_output", clean_status)


def _mutate(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    data = json.loads(path.read_text())
    change(data)
    _write_json(path, data)


def _schema_validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(
        Path("specs/native-easybuild-terminal-receipt-v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _receipt_data(tmp_path: Path) -> dict[str, Any]:
    return json.loads(_receipt(tmp_path).read_text())


def test_accepts_complete_bound_receipt(tmp_path: Path) -> None:
    assert (
        qualification.validate_receipt(_receipt(tmp_path), Path.cwd())["outcome"]
        == "passed"
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["resources"].__setitem__("cpu_count", 7),
        lambda d: d["resources"].__setitem__("free_bytes_before", 250 * 1024**3 - 1),
        lambda d: d["resources"].__setitem__("unbounded", True),
        lambda d: d["commands"][0]["parsed"].pop("easybuild_completed"),
        lambda d: d["commands"][0]["parsed"].__setitem__("unexpected", True),
        lambda d: d["commands"][1]["parsed"].__setitem__(
            "structured_probe_written", "yes"
        ),
        lambda d: d["commands"][2]["parsed"].__setitem__(
            "voiage_absent_after_unload", 1
        ),
        lambda d: d["probes"].__setitem__(0, d["probes"][1]),
        lambda d: d["probes"].append({"name": "invented", "status": "passed"}),
        lambda d: d.__setitem__("failure", {"stage": "easybuild"}),
        lambda d: d["artifacts"].__setitem__("probe", None),
        lambda d: d["commands"][0].__setitem__("exit_code", 1),
    ],
)
def test_terminal_receipt_schema_rejects_strict_passed_mutations(
    tmp_path: Path, change: Callable[[dict[str, Any]], None]
) -> None:
    data = _receipt_data(tmp_path)
    change(data)
    assert list(_schema_validator().iter_errors(data))


def test_terminal_receipt_schema_accepts_strict_failed_shape(tmp_path: Path) -> None:
    data = _receipt_data(tmp_path)
    data["outcome"] = "failed_terminal"
    data["commands"] = data["commands"][:1]
    data["commands"][0]["exit_code"] = 1
    data["commands"][0]["parsed"] = {
        "easybuild_completed": False,
        "installed_module_full_name": None,
    }
    data["artifacts"] = dict.fromkeys(data["artifacts"])
    data["probes"] = []
    data["failure"] = {
        "stage": "easybuild",
        "command_index": 0,
        "exit_code": 1,
        "signal": None,
        "parsed_failure": "build failed",
    }
    _schema_validator().validate(data)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["failure"].__setitem__("exit_code", 0),
        lambda d: d["failure"].__setitem__("parsed_failure", ""),
        lambda d: d["failure"].__setitem__("extra", "unbound"),
        lambda d: d["failure"].pop("command_index"),
        lambda d: d["probes"].append({"name": "cli", "status": "passed"}),
        lambda d: d.__setitem__("failure", None),
    ],
)
def test_terminal_receipt_schema_rejects_strict_failed_mutations(
    tmp_path: Path, change: Callable[[dict[str, Any]], None]
) -> None:
    data = _receipt_data(tmp_path)
    data["outcome"] = "failed_terminal"
    data["commands"] = data["commands"][:1]
    data["commands"][0]["exit_code"] = 1
    data["commands"][0]["parsed"] = {
        "easybuild_completed": False,
        "installed_module_full_name": None,
    }
    data["artifacts"] = dict.fromkeys(data["artifacts"])
    data["probes"] = []
    data["failure"] = {
        "stage": "easybuild",
        "command_index": 0,
        "exit_code": 1,
        "signal": None,
        "parsed_failure": "build failed",
    }
    change(data)
    assert list(_schema_validator().iter_errors(data))


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["preflight"].__setitem__("source_clean", False),
        lambda d: d["preflight"].__setitem__("catalogue_clean", False),
        lambda d: d["preflight"].__setitem__("prefix_absent", False),
        lambda d: d["environment"].__setitem__("digest", "0" * 64),
        lambda d: d["identity"].__setitem__("source_commit", "0" * 40),
        lambda d: d["identity"].__setitem__("root_recipe_sha256", "0" * 64),
        lambda d: d["commands"][0]["argv"].append("--dry-run"),
        lambda d: d["commands"][0]["parsed"].__setitem__("easybuild_completed", False),
        lambda d: d["probes"].pop(),
        lambda d: d.__setitem__("invented_success", True),
    ],
)
def test_rejects_receipt_contract_mutations(
    tmp_path: Path, change: Callable[[dict[str, Any]], None]
) -> None:
    path = _receipt(tmp_path)
    _mutate(path, change)
    with pytest.raises(ValueError):
        qualification.validate_receipt(path, Path.cwd())


def test_rejects_missing_or_changed_transcript(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    (tmp_path / "build.log").write_text("fabricated replacement\n")
    with pytest.raises(ValueError, match="missing or changed"):
        qualification.validate_receipt(path, Path.cwd())


def test_rejects_symlink_artifact(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    build_log = tmp_path / "build.log"
    real_log = tmp_path / "real-build.log"
    build_log.rename(real_log)
    build_log.symlink_to(real_log)
    with pytest.raises(ValueError, match="symlink"):
        qualification.validate_receipt(path, Path.cwd())


def test_rejects_artifact_outside_run_root(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    path = _receipt(run_root)
    outside = tmp_path / "outside.log"
    outside.write_text("outside\n")
    _mutate(
        path,
        lambda d: d["commands"][0].update(
            {"log": "../outside.log", "log_sha256": _digest(outside)}
        ),
    )
    with pytest.raises(ValueError, match="escapes"):
        qualification.validate_receipt(path, Path.cwd())


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("evpi", "value", 0.0),
        ("arrow", "buffer_equal", False),
        ("polars", "lazy", False),
        ("thread", "imports_inside_worker", False),
    ],
)
def test_rejects_scientific_probe_mutations(
    tmp_path: Path, section: str, field: str, value: Any
) -> None:
    path = _receipt(tmp_path)
    probe_path = tmp_path / "probe.json"
    probe = json.loads(probe_path.read_text())
    probe[section][field] = value
    _write_json(probe_path, probe)
    _mutate(
        path, lambda d: d["artifacts"].__setitem__("probe_sha256", _digest(probe_path))
    )
    with pytest.raises(ValueError, match="probe"):
        qualification.validate_receipt(path, Path.cwd())


def test_matrix_requires_distinct_receipts_for_both_generations(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    matrix = tmp_path / "matrix.json"
    _write_json(
        matrix,
        {
            "schema_version": "voiage.native-easybuild-matrix.v1",
            "receipts": {"2023a": path.name, "2024a": path.name},
        },
    )
    with pytest.raises(ValueError):
        qualification.validate_matrix(matrix, Path.cwd())


def test_matrix_requires_both_generations(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    _write_json(
        matrix,
        {
            "schema_version": "voiage.native-easybuild-matrix.v1",
            "receipts": {"2023a": "receipt.json"},
        },
    )
    with pytest.raises(ValueError):
        qualification.validate_matrix(matrix, Path.cwd())


def test_rejects_inherited_environment_name(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    _mutate(path, lambda d: d["environment"]["allowlist"].append("PYTHONPATH=x"))
    with pytest.raises(ValueError):
        qualification.validate_receipt(path, Path.cwd())


@pytest.mark.parametrize(
    "field", ["all_resolved", "allowed_roots_only", "opposite_prefix_absent"]
)
def test_rejects_linkage_or_cross_generation_failure(
    tmp_path: Path, field: str
) -> None:
    path = _receipt(tmp_path)
    probe_path = tmp_path / "probe.json"
    probe = json.loads(probe_path.read_text())
    probe["linkage"][field] = False
    _write_json(probe_path, probe)
    _mutate(
        path, lambda d: d["artifacts"].__setitem__("probe_sha256", _digest(probe_path))
    )
    with pytest.raises(ValueError, match="linkage"):
        qualification.validate_receipt(path, Path.cwd())


def test_rejects_zero_exit_failed_terminal(tmp_path: Path) -> None:
    path = _receipt(tmp_path)

    def failed(data: dict[str, Any]) -> None:
        data["outcome"] = "failed_terminal"
        data["probes"] = []
        data["failure"] = {
            "stage": "easybuild",
            "command_index": 0,
            "exit_code": 0,
            "signal": None,
            "parsed_failure": "claimed failure",
        }

    _mutate(path, failed)
    with pytest.raises(ValueError):
        qualification.validate_receipt(path, Path.cwd())


def test_rejects_retargeted_catalogue(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    _mutate(path, lambda d: d["identity"].__setitem__("catalogue_tree", "0" * 40))
    with pytest.raises(ValueError):
        qualification.validate_receipt(path, Path.cwd())


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["run"].__setitem__("run_id", ""),
        lambda d: d["run"].__setitem__("ended_at", "2025-01-01T00:00:00Z"),
        lambda d: d["run"].__setitem__("predecessor_receipt_sha256", "a" * 64),
        lambda d: d["artifacts"].__setitem__("preflight_sha256", "0" * 64),
        lambda d: d["artifacts"].__setitem__("catalogue_evidence_sha256", "0" * 64),
    ],
)
def test_rejects_run_preflight_and_portable_catalogue_mutations(
    tmp_path: Path, change: Callable[[dict[str, Any]], None]
) -> None:
    path = _receipt(tmp_path)
    _mutate(path, change)
    with pytest.raises(ValueError):
        qualification.validate_receipt(path, Path.cwd())


def test_rejects_source_cache_mutation_between_preflight_and_terminal(
    tmp_path: Path,
) -> None:
    path = _receipt(tmp_path)
    preflight = tmp_path / "preflight.json"
    payload = json.loads(preflight.read_text())
    payload["staged_cache_digest_after"] = "0" * 64
    _write_json(preflight, payload)
    _mutate(
        path,
        lambda d: d["artifacts"].__setitem__("preflight_sha256", _digest(preflight)),
    )
    with pytest.raises(ValueError, match="preflight"):
        qualification.validate_receipt(path, Path.cwd())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tooling_commit", "0" * 40),
        ("tooling_tree", "0" * 40),
        ("tooling_driver_sha256", "0" * 64),
        ("tooling_probe_sha256", "0" * 64),
        ("tooling_root", "/wrong/tooling/root"),
    ],
)
def test_rejects_tooling_identity_mutation(
    tmp_path: Path, field: str, value: str
) -> None:
    path = _receipt(tmp_path)
    _mutate(path, lambda data: data["identity"].__setitem__(field, value))
    with pytest.raises(ValueError, match="tooling"):
        qualification.validate_receipt(path, Path.cwd())


def test_rejects_outside_prefix_linkage_target(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    probe_path = tmp_path / "probe.json"
    probe = json.loads(probe_path.read_text())
    object_name = probe["linkage"]["objects"][0]
    probe["linkage"]["transcripts"][object_name] += (
        "evil.so => /opt/other/libevil.so (0x2)\n"
    )
    probe["linkage"]["targets"].append("/opt/other/libevil.so")
    _write_json(probe_path, probe)
    _mutate(
        path, lambda d: d["artifacts"].__setitem__("probe_sha256", _digest(probe_path))
    )
    with pytest.raises(ValueError, match="linkage target"):
        qualification.validate_receipt(path, Path.cwd())


def test_rejects_postflight_checkout_drift(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    postflight = tmp_path / "postflight.json"
    payload = json.loads(postflight.read_text())
    payload["source_status"] = " M recipe.eb"
    _write_json(postflight, payload)
    _mutate(
        path,
        lambda d: d["artifacts"].__setitem__("postflight_sha256", _digest(postflight)),
    )
    with pytest.raises(ValueError, match="postflight"):
        qualification.validate_receipt(path, Path.cwd())


def test_inventory_accepts_internal_relative_symlink(tmp_path: Path) -> None:
    root = tmp_path / "install"
    root.mkdir()
    target = root / "libreal.so"
    target.write_text("native")
    (root / "libalias.so").symlink_to("libreal.so")
    inventory = tmp_path / "inventory.json"
    qualification.write_inventory(root, inventory)
    entries = qualification._manifest(inventory)
    assert any(item["type"] == "symlink" for item in entries)


@pytest.mark.parametrize("kind", ["escape", "dangling", "cycle"])
def test_inventory_rejects_unsafe_symlink(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "install"
    root.mkdir()
    link = root / "bad.so"
    if kind == "escape":
        outside = tmp_path / "outside.so"
        outside.write_text("outside")
        link.symlink_to("../outside.so")
    elif kind == "dangling":
        link.symlink_to("missing.so")
    else:
        link.symlink_to("bad.so")
    with pytest.raises(RuntimeError, match="unsafe"):
        qualification.write_inventory(root, tmp_path / "inventory.json")


def test_rejects_dirty_catalogue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _receipt(tmp_path)
    evidence = tmp_path / "catalogue-evidence.json"
    payload = json.loads(evidence.read_text())
    payload["ls_tree"].append("dirty untracked claim")
    _write_json(evidence, payload)
    _mutate(
        path,
        lambda d: d["artifacts"].__setitem__(
            "catalogue_evidence_sha256", _digest(evidence)
        ),
    )
    with pytest.raises(ValueError):
        qualification.validate_receipt(path, Path.cwd())


@pytest.mark.parametrize("module_count", [0, 1, 2])
def test_executor_produces_valid_terminal_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_count: int
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prepared_sources = tmp_path / "prepared-sources"
    prepared_sources.mkdir()
    (prepared_sources / "voiage-2.2.0.tar.gz").write_text("prepared source")
    tooling_root = tmp_path / "tooling"
    for relative in (
        *qualification.TOOLING_FILES,
        "specs/native-easybuild-terminal-receipt-v1.schema.json",
    ):
        target = tooling_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path.cwd() / relative, target)
    tooling_commit = subprocess.check_output(
        ["/usr/bin/git", "rev-parse", "HEAD"], text=True
    ).strip()
    tooling_tree = subprocess.check_output(
        ["/usr/bin/git", "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    base_tree = subprocess.check_output(
        ["/usr/bin/git", "rev-parse", f"{qualification.BASE}^{{tree}}"], text=True
    ).strip()
    monkeypatch.setenv("VOIAGE_VM_LOCAL_STORAGE", "confirmed")
    monkeypatch.setattr(qualification.sys, "platform", "linux")
    monkeypatch.setattr(qualification.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(qualification.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(qualification.shutil, "which", lambda name: f"/usr/bin/{name}")
    usage = qualification.shutil.disk_usage(workspace)
    monkeypatch.setattr(
        qualification.shutil,
        "disk_usage",
        lambda path: usage._replace(free=300 * 1024**3),
    )
    real_output = subprocess.check_output

    def output(argv: list[str], **kwargs: Any) -> str:  # noqa: PLR0911
        if argv[:2] == ["eb", "--version"]:
            return "EasyBuild 5.4.0\n"
        if (
            argv[0] == qualification.sys.executable
            and "easybuild.easyblocks" in argv[-1]
        ):
            return "5.4.0\n"
        if argv[:2] == ["modulecmd", "--version"]:
            return "5.4.0\n"
        if argv[-2:] == ["status", "--porcelain"]:
            return ""
        checkout = argv[argv.index("-C") + 1] if "-C" in argv else None
        if checkout == str(tooling_root):
            if argv[-2:] == ["rev-parse", "HEAD"]:
                return tooling_commit + "\n"
            if argv[-2:] == ["rev-parse", "HEAD^{tree}"]:
                return tooling_tree + "\n"
            if argv[-2] == "rev-parse" and argv[-1].endswith("^{tree}"):
                return tooling_tree + "\n"
            if "show" in argv:
                relative = argv[-1].split(":", 1)[1]
                payload = (tooling_root / relative).read_bytes()
                return payload if not kwargs.get("text") else payload.decode()
        if checkout == str(Path.cwd()):
            if argv[-2:] == ["rev-parse", "HEAD"]:
                return qualification.BASE + "\n"
            if argv[-2:] == ["rev-parse", "HEAD^{tree}"]:
                return base_tree + "\n"
            if argv[-3:] == ["ls-tree", "-r", "HEAD"]:
                replacement = [*argv[:-1], qualification.BASE]
                return real_output(replacement, **kwargs)
        return real_output(argv, **kwargs)

    monkeypatch.setattr(qualification.subprocess, "check_output", output)

    def fake_run(argv: list[str], log: Path, env: dict[str, str]) -> int:
        log.parent.mkdir(parents=True, exist_ok=True)
        if argv[0] == "eb":
            prefix = log.parent / "install"
            for relative in [
                "bin/python",
                "lib/voiage/_core.so",
                "lib/pyarrow/lib.so",
                "lib/polars/polars.so",
            ]:
                target = prefix / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(relative)
            for index in range(module_count):
                module_file = prefix / f"modules/all/voiage/2.2.0-foss-2023a-{index}"
                module_file.parent.mkdir(parents=True, exist_ok=True)
                module_file.write_text("module")
            log.write_text(
                "== COMPLETED: Installation ended successfully (took 1 min)\n"
            )
        elif "module-probe.sh" in str(argv[3]):
            _write_json(
                log.parent / "probe.json", _probe("2023a", log.parent / "install")
            )
            log.write_text("probe passed\n")
        else:
            log.write_text("unload passed\n")
        return 0

    monkeypatch.setattr(qualification, "run", fake_run)
    monkeypatch.setattr(
        qualification.sys,
        "argv",
        [
            "native",
            "2023a",
            "--root",
            str(Path.cwd()),
            "--tooling-root",
            str(tooling_root),
            "--catalogue",
            str(Path.cwd()),
            "--workspace",
            str(workspace),
            "--source-cache",
            str(prepared_sources),
            "--run-id",
            "test-run",
            "--execute",
        ],
    )
    assert qualification.main() == (0 if module_count == 1 else 1)
    receipt = workspace / "voiage-easybuild-2023a/receipt.json"
    data = qualification.validate_receipt(receipt, Path.cwd(), tooling_root)
    assert data["commands"][0]["exit_code"] == 0
    if module_count == 1:
        assert data["outcome"] == "passed"
    else:
        assert data["outcome"] == "failed_terminal"
        assert data["failure"]["stage"] == "module-discovery"
        assert data["commands"][1]["parsed"]["modulefile_match_count"] == module_count


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux shell semantics")
def test_generated_module_scripts_with_fake_environment_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the emitted probe and unload scripts in fresh real shells."""
    source_root = Path.cwd()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prepared_sources = tmp_path / "prepared-sources-linux"
    prepared_sources.mkdir()
    (prepared_sources / "voiage-2.2.0.tar.gz").write_text("prepared")
    tooling_root = tmp_path / "tooling"
    for relative in (
        *qualification.TOOLING_FILES,
        "specs/native-easybuild-terminal-receipt-v1.schema.json",
    ):
        target = tooling_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path.cwd() / relative, target)
    tooling_commit = subprocess.check_output(
        ["/usr/bin/git", "rev-parse", "HEAD"], text=True
    ).strip()
    tooling_tree = subprocess.check_output(
        ["/usr/bin/git", "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    base_tree = subprocess.check_output(
        ["/usr/bin/git", "rev-parse", f"{qualification.BASE}^{{tree}}"], text=True
    ).strip()
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    modulecmd = fake_bin / "modulecmd"
    modulecmd.write_text(
        """#!/usr/bin/env python3
import pathlib
import shlex
import sys

if sys.argv[1:] == ["--version"]:
    print("Environment Modules 5.4.0")
    raise SystemExit
if len(sys.argv) < 3 or sys.argv[1] != "bash":
    raise SystemExit(2)
action = sys.argv[2]
if action == "purge":
    print("unset MODULEPATH PYTHONPATH VOIAGE_FAKE_MODULE_PREFIX")
elif action == "use":
    root = pathlib.Path(sys.argv[3]).resolve()
    if not root.is_dir():
        raise SystemExit(3)
    print(f"export MODULEPATH={shlex.quote(str(root))}")
elif action == "load":
    root = pathlib.Path(__import__("os").environ["MODULEPATH"])
    module_file = root / sys.argv[3]
    if not module_file.is_file():
        raise SystemExit(4)
    prefix = module_file.read_text().strip()
    print(f"export PATH={shlex.quote(prefix + '/bin')}:\\\"$PATH\\\"")
    print(f"export PYTHONPATH={shlex.quote(prefix + '/lib/python')}")
    print(f"export VOIAGE_FAKE_MODULE_PREFIX={shlex.quote(prefix)}")
elif action == "unload":
    prefix = __import__("os").environ.get("VOIAGE_FAKE_MODULE_PREFIX")
    if not prefix:
        raise SystemExit(5)
    print('export PATH="${PATH#*:}"')
    print("unset PYTHONPATH VOIAGE_FAKE_MODULE_PREFIX")
else:
    raise SystemExit(6)
"""
    )
    modulecmd.chmod(0o755)
    fake_probe = tmp_path / "probe.py"
    fake_probe.write_text(
        """import argparse
import importlib.util
import json
import os

p = argparse.ArgumentParser()
p.add_argument("--prefix")
p.add_argument("--generation")
p.add_argument("--output")
p.add_argument("--opposite-prefix")
a = p.parse_args()
assert os.environ["VOIAGE_FAKE_MODULE_PREFIX"] == a.prefix
spec = importlib.util.find_spec("voiage")
assert spec is not None and str(spec.origin).startswith(a.prefix)
with open(a.output, "w") as stream:
    json.dump({"fake_probe": "passed", "generation": a.generation}, stream)
"""
    )

    monkeypatch.setenv("VOIAGE_VM_LOCAL_STORAGE", "confirmed")
    monkeypatch.setattr(qualification.sys, "platform", "linux")
    # Hosted test environments contain the wheel; use the clean system Python
    # for preflight and the fake installed-module interpreter.
    monkeypatch.setattr(qualification.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(qualification.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(qualification.shutil, "which", lambda name: f"/usr/bin/{name}")
    usage = qualification.shutil.disk_usage(workspace)
    monkeypatch.setattr(
        qualification.shutil,
        "disk_usage",
        lambda path: usage._replace(free=300 * 1024**3),
    )
    real_output = subprocess.check_output

    def output(argv: list[str], **kwargs: Any) -> str:  # noqa: PLR0911
        if argv[:2] == ["eb", "--version"]:
            return "EasyBuild 5.4.0\n"
        if (
            argv[0] == qualification.sys.executable
            and "easybuild.easyblocks" in argv[-1]
        ):
            return "5.4.0\n"
        if argv[:2] == ["modulecmd", "--version"]:
            return "Environment Modules 5.4.0\n"
        if argv[-2:] == ["status", "--porcelain"]:
            return ""
        checkout = argv[argv.index("-C") + 1] if "-C" in argv else None
        if checkout == str(tooling_root):
            if argv[-2:] == ["rev-parse", "HEAD"]:
                return tooling_commit + "\n"
            if argv[-2:] == ["rev-parse", "HEAD^{tree}"]:
                return tooling_tree + "\n"
            if argv[-2] == "rev-parse" and argv[-1].endswith("^{tree}"):
                return tooling_tree + "\n"
            if "show" in argv:
                relative = argv[-1].split(":", 1)[1]
                payload = (tooling_root / relative).read_bytes()
                return payload if not kwargs.get("text") else payload.decode()
        if checkout == str(Path.cwd()):
            if argv[-2:] == ["rev-parse", "HEAD"]:
                return qualification.BASE + "\n"
            if argv[-2:] == ["rev-parse", "HEAD^{tree}"]:
                return base_tree + "\n"
            if argv[-3:] == ["ls-tree", "-r", "HEAD"]:
                return real_output([*argv[:-1], qualification.BASE], **kwargs)
        return real_output(argv, **kwargs)

    monkeypatch.setattr(qualification.subprocess, "check_output", output)
    original_clean_environment = qualification.clean_environment

    def clean_environment(home: Path) -> tuple[dict[str, str], list[str], str]:
        env, _, _ = original_clean_environment(home)
        # Keep fixture preflight and unloaded python3 on the same clean system
        # interpreter; hosted pytest PATH otherwise contains its installed wheel.
        env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
        allowlist = [f"{key}={env[key]}" for key in sorted(env)]
        digest = hashlib.sha256(
            json.dumps(allowlist, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return env, allowlist, digest

    monkeypatch.setattr(qualification, "clean_environment", clean_environment)

    def hybrid_run(argv: list[str], log: Path, env: dict[str, str]) -> int:
        if argv[0] == "eb":
            prefix = log.parent / "install"
            module_file = prefix / "modules/all/voiage/2.2.0-foss-2023a"
            module_file.parent.mkdir(parents=True)
            module_file.write_text(str(prefix))
            python = prefix / "bin/python3"
            python.parent.mkdir(parents=True)
            python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
            python.chmod(0o755)
            package = prefix / "lib/python/voiage"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("installed = True\n")
            (package / "_core.so").write_text("fake native object")
            for dependency in ("numpy", "pyarrow", "polars"):
                dep = prefix / f"lib/python/site-packages/{dependency}"
                dep.mkdir(parents=True)
                (dep / "__init__.py").write_text("installed = True\n")
            (prefix / "lib/voiage").mkdir(parents=True)
            (prefix / "lib/voiage/_core.so").write_text("fake native object")
            (prefix / "lib/pyarrow").mkdir(parents=True)
            (prefix / "lib/pyarrow/lib.so").write_text("fake arrow object")
            (prefix / "lib/polars").mkdir(parents=True)
            (prefix / "lib/polars/polars.so").write_text("fake polars object")
            log.write_text(
                "== COMPLETED: Installation ended successfully (took 1 min)\n"
            )
            return 0
        actual = list(argv)
        if "module-probe.sh" in actual[3]:
            actual[6] = str(fake_probe)
        with log.open("wb") as stream:
            result = subprocess.run(
                actual,
                cwd=source_root,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if "module-probe.sh" in actual[3] and result.returncode == 0:
            _write_json(
                log.parent / "probe.json", _probe("2023a", log.parent / "install")
            )
        return result.returncode

    monkeypatch.setattr(qualification, "run", hybrid_run)
    monkeypatch.setattr(
        qualification.sys,
        "argv",
        [
            "native",
            "2023a",
            "--root",
            str(source_root),
            "--tooling-root",
            str(tooling_root),
            "--catalogue",
            str(source_root),
            "--workspace",
            str(workspace),
            "--source-cache",
            str(prepared_sources),
            "--run-id",
            "linux-integration",
            "--execute",
        ],
    )

    assert qualification.main() == 0
    run_root = workspace / "voiage-easybuild-2023a"
    assert (
        qualification.validate_receipt(
            run_root / "receipt.json", source_root, tooling_root
        )["outcome"]
        == "passed"
    )
    assert (run_root / "probe.log").read_text() == ""
    assert (run_root / "unload.log").read_text() == ""
