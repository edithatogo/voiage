# SPDX-License-Identifier: Apache-2.0
"""Verify the installed files and the module used by a qualification probe."""

from hashlib import file_digest
from importlib import import_module
from importlib.metadata import distribution
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit
from urllib.request import url2pathname
from zipfile import ZipFile


def verify_wheel(
    name: str, module: str, version: str, expected_sha: str, wheel: Path | None = None
) -> int:
    """Return verified member count for an isolated exact-version installation."""
    dist = distribution(name)
    prefix = Path(sys.prefix).resolve()
    if sys.version_info[:3] != (3, 15, 1) or dist.version != version:
        raise ValueError(f"Unexpected qualification runtime or {name} version")
    if wheel is None:
        origin = json.loads(dist.read_text("direct_url.json") or "{}")
        url = urlsplit(origin.get("url", ""))
        if url.scheme != "file" or url.netloc or "archive_info" not in origin:
            raise ValueError(f"{name} requires a local wheel installation")
        wheel = Path(url2pathname(url.path))
    member_paths: set[Path] = set()
    with wheel.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != expected_sha:
            raise ValueError(f"{name} wheel SHA-256 mismatch")
        stream.seek(0)
        with ZipFile(stream) as archive:
            for member in archive.infolist():
                if member.is_dir() or member.filename.endswith(".dist-info/RECORD"):
                    continue
                path = Path(dist.locate_file(member.filename)).resolve()
                if not path.is_relative_to(prefix) or path in member_paths:
                    raise ValueError(
                        f"Unsafe or duplicate installed path: {member.filename}"
                    )
                with archive.open(member) as expected, path.open("rb") as actual:
                    if (
                        file_digest(expected, "sha256").digest()
                        != file_digest(actual, "sha256").digest()
                    ):
                        raise ValueError(f"Changed installed file: {member.filename}")
                member_paths.add(path)
    module_file = getattr(import_module(module), "__file__", None)
    if not module_file or Path(module_file).resolve() not in member_paths:
        raise ValueError(f"{name} import origin is outside the verified wheel")
    return len(member_paths)
