"""Negative controls for source and native-provenance observation claims."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.observe_arrow_capacity import (
    Observation,
    descendant_usage,
    download,
    native_objects,
    validate_linkage,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_linkage_accepts_both_built_libraries(tmp_path: Path) -> None:
    prefix = tmp_path / "install"
    text = "\n".join(
        f"{name}.so.2600 => {prefix}/lib/{name}.so.2600 (0x1234)"
        for name in ("libarrow", "libparquet", "libarrow_dataset")
    )
    validate_linkage(text, prefix)


@pytest.mark.parametrize(
    "damage", ["outside-prefix", "unresolved", "empty", "missing-parquet"]
)
def test_linkage_rejects_false_native_acceptance(tmp_path: Path, damage: str) -> None:
    prefix = tmp_path / "install"
    text = (
        f"libarrow.so.2600 => {prefix}/lib/libarrow.so.2600 (0x1234)\n"
        f"libparquet.so.2600 => {prefix}/lib/libparquet.so.2600 (0x1234)"
    )
    if damage == "outside-prefix":
        text = text.replace(
            str(prefix) + "/lib/libarrow", str(tmp_path / "other/lib/libarrow")
        )
    elif damage == "unresolved":
        text += "\nlibarrow_compute.so.2600 => not found"
    elif damage == "empty":
        text = ""
    elif damage == "missing-parquet":
        text = text.splitlines()[0]
    with pytest.raises(RuntimeError, match="Native linkage"):
        validate_linkage(text, prefix)


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "https://example.com/source.tar.gz"]
)
def test_source_download_rejects_nonprovider_sources(tmp_path: Path, url: str) -> None:
    with pytest.raises(ValueError, match="Unsupported source host"):
        download({"url": url}, tmp_path)


def test_native_object_scan_includes_parquet_extensions(tmp_path: Path) -> None:
    names = ["lib.cpython-315.so", "_parquet.cpython-315.so", "libarrow_python.so.2600"]
    for name in names:
        (tmp_path / name).write_bytes(b"\x7fELF")
    (tmp_path / "metadata.txt").touch()
    (tmp_path / "v0.7.1.some-named-index.parquet").write_bytes(b"PAR1")
    (tmp_path / "not-elf.so").write_text("linker script")
    assert {path.name for path in native_objects(tmp_path, tmp_path)} == set(names)


def test_native_object_scan_rejects_another_environment(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="escaped"):
        native_objects(tmp_path.parent, tmp_path)


def test_resource_sampling_counts_children_in_other_process_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Table:
        stdout = "100 50 100 20\n101 100 101 100\n102 101 102 10000\n200 50 200 99999\n"

    monkeypatch.setattr(
        "scripts.observe_arrow_capacity.subprocess.run",
        lambda *_args, **_kwargs: Table(),
    )
    pids, rss = descendant_usage(100)
    assert pids == {100, 101, 102}
    assert rss == 10120


def test_sampler_reports_unexpected_base_exception_and_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}")
    observation = Observation(tmp_path / "workspace", tmp_path / "evidence", inputs)

    def fail(_path: Path) -> None:
        raise SystemExit("sampling interrupted")

    monkeypatch.setattr("scripts.observe_arrow_capacity.shutil.disk_usage", fail)
    with pytest.raises(SystemExit, match="sampling interrupted"):
        observation.sample()
    assert observation.monitor_errors
