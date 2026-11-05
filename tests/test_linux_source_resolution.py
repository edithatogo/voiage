"""Challenge resolved source boundaries before native dispatch."""

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import spack_source_witness
from scripts.easybuild_source_hooks import post_fetch_hook, resolved_file
from scripts.source_cache_manifest import inventory, validate_replay_cache


def test_resolved_source_binds_actual_bytes(tmp_path: Path) -> None:
    source = tmp_path / "archive.tar.gz"
    source.write_bytes(b"authenticated archive")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert resolved_file(source, [tmp_path], expected)["sha256"] == expected


@pytest.mark.parametrize("failure", ["missing", "digest", "shadow", "ancestor"])
def test_resolved_source_rejects_missing_changed_or_shadowed_input(
    tmp_path: Path, failure: str
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    source = cache / "archive.tar.gz"
    source.write_bytes(b"expected")
    expected = hashlib.sha256(b"expected").hexdigest()
    if failure == "missing":
        source.unlink()
    elif failure == "digest":
        source.write_bytes(b"changed")
    elif failure == "shadow":
        other = tmp_path / "shadow.tar.gz"
        other.write_bytes(b"expected")
        source = other
    else:
        link = tmp_path / "linked"
        link.symlink_to(cache, target_is_directory=True)
        source = link / "archive.tar.gz"
    with pytest.raises((ValueError, FileNotFoundError), match="source|digest|No such"):
        resolved_file(source, [cache], expected)


def test_hook_records_sources_and_extension_patches(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"verified source")
    monkeypatch.setenv("VOIAGE_SOURCE_ROOTS", str(tmp_path))
    monkeypatch.setenv("VOIAGE_SOURCE_RECEIPTS", str(tmp_path / "receipts"))
    app = SimpleNamespace(
        name="probe",
        version="1.0",
        toolchain=SimpleNamespace(name="system", version=""),
        src=[{"path": str(source)}],
        patches=[],
        exts=[{"src": str(source), "patches": [{"path": str(source)}]}],
    )
    post_fetch_hook(app)
    receipt = json.loads(next((tmp_path / "receipts").glob("*.json")).read_text())
    assert len(receipt["sources"]) == 3
    assert {entry["sha256"] for entry in receipt["sources"]} == {
        hashlib.sha256(source.read_bytes()).hexdigest()
    }


def test_hook_rejects_unresolved_auxiliary_config_guess(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("VOIAGE_SOURCE_ROOTS", str(tmp_path))
    monkeypatch.setenv("VOIAGE_SOURCE_RECEIPTS", str(tmp_path / "receipts"))
    app = SimpleNamespace(src=[], patches=[], exts=[], config_guess=None)
    with pytest.raises(ValueError, match="config.guess"):
        post_fetch_hook(app)


def test_mirror_inventory_binds_relative_aliases_and_changes(tmp_path: Path) -> None:
    source = tmp_path / "archive"
    source.write_bytes(b"authenticated")
    (tmp_path / "alias").symlink_to("archive")
    original = inventory(tmp_path)
    assert original[0]["sha256"] == original[1]["sha256"]
    source.write_bytes(b"altered")
    assert inventory(tmp_path) != original


def test_mirror_inventory_rejects_escaping_alias(tmp_path: Path) -> None:
    cache = tmp_path / "mirror"
    cache.mkdir()
    source = tmp_path / "outside"
    source.write_bytes(b"shadow")
    (cache / "alias").symlink_to("../outside")
    with pytest.raises(ValueError, match="escapes"):
        inventory(cache)


@pytest.mark.parametrize("authenticated", [True, False])
def test_spack_witness_binds_archive_to_concrete_dag(
    tmp_path: Path, monkeypatch, authenticated: bool
) -> None:
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"staged")
    manifest = [
        {
            "sha256": hashlib.sha256(
                b"staged" if authenticated else b"altered"
            ).hexdigest()
        }
    ]
    for name in ("cache", "mirror"):
        (tmp_path / f"{name}-manifest.json").write_text(json.dumps(manifest))
    spec = SimpleNamespace(
        external=False,
        package=SimpleNamespace(
            has_code=True, stage=[SimpleNamespace(archive_file=str(archive))]
        ),
        dag_hash=lambda: "concrete-dag",
    )
    environment = SimpleNamespace(concrete_roots=lambda: [spec])
    modules = {
        "spack.environment": SimpleNamespace(active_environment=lambda: environment),
        "spack.traverse": SimpleNamespace(traverse_nodes=lambda roots: roots),
    }
    monkeypatch.setattr(
        spack_source_witness.importlib, "import_module", modules.__getitem__
    )
    monkeypatch.setattr(
        spack_source_witness.sys, "argv", ["probe", str(tmp_path), str(tmp_path)]
    )
    if authenticated:
        spack_source_witness.main()
        records = json.loads((tmp_path / "staged-sources.json").read_text())
        assert records[0]["dag_hash"] == "concrete-dag"
        assert records[0]["source"]["path"] == str(archive)
    else:
        with pytest.raises(ValueError, match="frozen mirror"):
            spack_source_witness.main()
        assert not (tmp_path / "staged-sources.json").exists()


@pytest.mark.parametrize("safe", [True, False])
def test_staged_source_allows_only_private_cache_aliases(
    tmp_path: Path, safe: bool
) -> None:
    work = tmp_path / "work"
    cache = work / "cache"
    cache.mkdir(parents=True)
    target = cache / "archive" if safe else tmp_path / "outside"
    target.write_bytes(b"checked source")
    stage = work / "archive"
    stage.symlink_to(target)
    if safe:
        result = spack_source_witness.staged_source(stage, work)
        assert result["path"] == str(target)
        assert result["link"] == str(target)
    else:
        with pytest.raises(ValueError, match="frozen cache"):
            spack_source_witness.staged_source(stage, work)


@pytest.mark.parametrize(
    ("generation", "recipe", "url", "checksum"),
    [
        (
            "2023a",
            "FlexiBLAS-3.3.1-GCC-12.3.0.eb",
            "https://csc.mpi-magdeburg.mpg.de/mpcsc/software/flexiblas/",
            "bbeebf5e5a006924558fec43f49affbe1aaa4cbacfc472a9ff6066ffda142e18",
        ),
        (
            "2024a",
            "FlexiBLAS-3.4.4-GCC-13.3.0.eb",
            "https://github.com/mpimd-csc/flexiblas/releases/download/v3.4.4/",
            "05040ae092142dd0bf38d1bb9ce33f6b475d9f9bb455e33be997932ae855c22b",
        ),
        (
            "2023a",
            "hwloc-2.9.1-GCCcore-12.3.0.eb",
            "https://download.open-mpi.org/release/hwloc/v%(version_major_minor)s/",
            "a440e2299f7451dc10a57ddbfa3f116c2a6c4be1bb97c663edd3b9c7b3b3b4cf",
        ),
        (
            "2024a",
            "hwloc-2.10.0-GCCcore-13.3.0.eb",
            "https://download.open-mpi.org/release/hwloc/v%(version_major_minor)s/",
            "c7fd8a1404a9719c76aadc642864b9f77aed1dc1fc8882d6af861a9260ba240d",
        ),
        (
            "2023a",
            "OpenMPI-4.1.5-GCC-12.3.0.eb",
            "https://download.open-mpi.org/release/open-mpi/v%(version_major_minor)s/",
            "a640986bc257389dd379886fdae6264c8cfa56bc98b71ce3ae3dfbd8ce61dbe3",
        ),
        (
            "2024a",
            "OpenMPI-5.0.3-GCC-13.3.0.eb",
            "https://download.open-mpi.org/release/open-mpi/v%(version_major_minor)s/",
            "990582f206b3ab32e938aa31bbf07c639368e4405dca196fabe7f0f76eeda90b",
        ),
    ],
)
def test_recipe_resolves_authenticated_release_source(
    generation: str, recipe: str, url: str, checksum: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse(
        (root / "packaging/current/easybuild" / generation / recipe).read_text()
    )
    assignments = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    if "source_urls" not in assignments:
        component = assignments["components"].elts[0].elts[2]
        assignments.update(
            (key.value, value)
            for key, value in zip(component.keys, component.values, strict=True)
            if key.value in {"source_urls", "checksums"}
        )
    assert ast.literal_eval(assignments["source_urls"]) == [url]
    checksums = ast.literal_eval(assignments["checksums"])
    assert checksum in {
        value
        for entry in checksums
        for value in (entry.values() if isinstance(entry, dict) else [entry])
    }


@pytest.mark.parametrize(
    ("generation", "recipe"),
    [
        ("2023a", "OpenMPI-4.1.5-GCC-12.3.0.eb"),
        ("2024a", "OpenMPI-5.0.3-GCC-13.3.0.eb"),
    ],
)
def test_openmpi_overlay_retains_authenticated_patch_closure(
    generation: str, recipe: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    directory = root / "packaging/current/easybuild" / generation
    assignments = {
        node.targets[0].id: node.value
        for node in ast.parse((directory / recipe).read_text()).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    checksums = {
        name: checksum
        for entry in ast.literal_eval(assignments["checksums"])
        for name, checksum in entry.items()
    }
    for entry in ast.literal_eval(assignments["patches"]):
        name = entry[0] if isinstance(entry, tuple) else entry
        assert (
            hashlib.sha256((directory / name).read_bytes()).hexdigest()
            == checksums[name]
        )


def test_hook_accepts_exact_authenticated_lock_patch_outside_cache(
    tmp_path: Path, monkeypatch
) -> None:
    repository = Path(__file__).resolve().parents[1]
    inputs = json.loads(
        (repository / "packaging/current/source-vendor-inputs.json").read_text()
    )
    patch = inputs["entries"]["polars-runtime-32"]["lock_patch"]
    monkeypatch.setenv("VOIAGE_SOURCE_ROOTS", str(tmp_path))
    monkeypatch.setenv("VOIAGE_SOURCE_RECEIPTS", str(tmp_path / "receipts"))
    app = SimpleNamespace(
        name="polars",
        version="2.1.0",
        toolchain=SimpleNamespace(name="system", version=""),
        src=[],
        patches=[{"path": str(repository / patch["path"])}],
        exts=[],
    )
    post_fetch_hook(app)
    receipt = json.loads(next((tmp_path / "receipts").glob("*.json")).read_text())
    assert receipt["sources"][0]["sha256"] == patch["sha256"]


def test_source_driver_retains_structured_failure(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "qualify_linux_graph.sh").write_text(
        'mkdir -p "$2"\nprintf %s "$VOIAGE_SOURCE_WORK" > "$2/source-work.txt"\nexit 23\n'
    )
    runner = tmp_path / "runner"
    runner.mkdir()
    output = tmp_path / "receipts"
    result = subprocess.run(
        [
            "/bin/bash",
            str(repository / "scripts/qualify_linux_sources.sh"),
            "spack",
            str(output),
        ],
        cwd=tmp_path,
        env=dict(os.environ, RUNNER_TEMP=str(runner)),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 23
    receipt = json.loads((output / "result.json").read_text())
    assert receipt["source_replay_passed"] is False
    assert receipt["installed_acceptance"] is False
    assert receipt["exit_code"] == 23
    assert (output / "graph/source-work.txt").read_text() == str(
        runner / "voiage-sources-spack"
    )


def test_spack_witness_argument_error_has_usage(monkeypatch, capsys) -> None:
    monkeypatch.setattr(spack_source_witness.sys, "argv", ["probe"])
    with pytest.raises(SystemExit) as error:
        spack_source_witness.main()
    assert error.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_replay_cache_allows_only_frozen_mirror_additions() -> None:
    original = [{"path": "original", "sha256": "a"}]
    mirror = [{"path": "mirror", "sha256": "b"}]
    validate_replay_cache(
        original, [*original, {"path": "copy", "sha256": "b"}], mirror
    )


@pytest.mark.parametrize(
    "after",
    [
        [],
        [{"path": "original", "sha256": "changed"}],
        [{"path": "original", "sha256": "a"}, {"path": "extra", "sha256": "untrusted"}],
    ],
)
def test_replay_cache_rejects_changed_missing_and_untrusted_entries(after) -> None:
    with pytest.raises(ValueError, match="cache"):
        validate_replay_cache(
            [{"path": "original", "sha256": "a"}],
            after,
            [{"path": "mirror", "sha256": "b"}],
        )
