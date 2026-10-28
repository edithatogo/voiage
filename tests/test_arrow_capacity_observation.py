"""Negative controls for source and native-provenance observation claims."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.observe_arrow_capacity import download, validate_linkage

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
