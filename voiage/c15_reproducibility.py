"""Cross-platform normalized distribution identity with semantic wheel RECORD checks."""

from __future__ import annotations

import base64
import csv
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import TYPE_CHECKING, TypedDict, cast
import zipfile

if TYPE_CHECKING:
    from collections.abc import Iterator

_TEXT_SUFFIXES = frozenset(
    {
        ".ambr",
        ".bib",
        ".cfg",
        ".cli",
        ".cs",
        ".csproj",
        ".csv",
        ".eb",
        ".go",
        ".html",
        ".ini",
        ".ipynb",
        ".jupyter",
        ".jl",
        ".js",
        ".json",
        ".md",
        ".mdx",
        ".mjs",
        ".mod",
        ".qmd",
        ".r",
        ".rd",
        ".rmd",
        ".py",
        ".rs",
        ".rst",
        ".sh",
        ".toml",
        ".tex",
        ".ts",
        ".template",
        ".txt",
        ".v",
        ".xml",
        ".yaml",
        ".yml",
    }
)
_TEXT_FILENAMES = frozenset(
    {
        ".editorconfig",
        ".gitattributes",
        ".gitignore",
        "authors",
        "changelog",
        "codeowners",
        "copying",
        "description",
        "dockerfile",
        "license",
        "metadata",
        "namespace",
        "notice",
        "pkg-info",
        "readme",
        "snakefile",
    }
)
NORMALIZATION = "sorted-paths+declared-utf8-text-lf+scm-paths+content-sha256+record-semantics+portable-v3"


class ArtifactMismatchError(ValueError):
    """Raised when independently built distributions differ."""


class ArchiveEntry(TypedDict):
    path: str
    sha256: str
    size: int


def _safe_path(name: str) -> str:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe archive path: {name}")
    return path.as_posix()


def _normalized_content(name: str, content: bytes) -> bytes:
    path = PurePosixPath(name)
    filename = path.name.casefold()
    stem = filename.split(".", maxsplit=1)[0]
    if (
        path.suffix.casefold() not in _TEXT_SUFFIXES
        and stem not in _TEXT_FILENAMES
        and filename not in _TEXT_FILENAMES
    ):
        return content
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if filename == "scm_file_list.json":
        try:
            raw_payload = cast("object", json.loads(normalized))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "scm_file_list.json must contain a string file list"
            ) from exc
        if not isinstance(raw_payload, dict):
            raise ValueError("scm_file_list.json must contain a string file list")
        payload = cast("dict[str, object]", raw_payload)
        raw_files = payload.get("files")
        if not isinstance(raw_files, list):
            raise ValueError("scm_file_list.json must contain a string file list")
        files: list[str] = []
        for item in raw_files:
            if not isinstance(item, str):
                raise TypeError("scm_file_list.json must contain a string file list")
            files.append(item)
        payload["files"] = sorted(item.replace("\\", "/") for item in files)
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return normalized.encode()


def _zip_entries(path: Path) -> Iterator[tuple[str, bytes]]:
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if not member.is_dir():
                yield member.filename, archive.read(member)


def _tar_entries(path: Path) -> Iterator[tuple[str, bytes]]:
    with tarfile.open(path, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"archive member cannot be read: {member.name}")
            yield member.name, extracted.read()


def _wheel_record_entry(
    raw: dict[str, bytes], record_name: str, normalized: list[ArchiveEntry]
) -> ArchiveEntry:
    try:
        rows = list(csv.reader(io.StringIO(raw[record_name].decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("wheel RECORD is not valid UTF-8 CSV") from exc
    recorded: set[str] = set()
    for row in rows:
        if len(row) != 3:
            raise ValueError("wheel RECORD rows must contain path, digest, and size")
        name = _safe_path(row[0])
        if name in recorded:
            raise ValueError(f"duplicate wheel RECORD path: {name}")
        recorded.add(name)
        digest, size = row[1], row[2]
        if name == record_name:
            if digest or size:
                raise ValueError("wheel RECORD must not hash itself")
            continue
        content = raw.get(name)
        if content is None:
            raise ValueError(f"wheel RECORD references missing member: {name}")
        expected = (
            base64.urlsafe_b64encode(sha256(content).digest()).rstrip(b"=").decode()
        )
        if digest != f"sha256={expected}" or size != str(len(content)):
            raise ValueError(f"wheel RECORD integrity mismatch: {name}")
    if recorded != set(raw):
        raise ValueError("wheel RECORD inventory does not match archive inventory")
    rows = [
        [entry["path"], f"sha256={entry['sha256']}", str(entry["size"])]
        for entry in normalized
    ]
    rows.append([record_name, "", ""])
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(sorted(rows))
    content = stream.getvalue().encode()
    return {
        "path": record_name,
        "sha256": sha256(content).hexdigest(),
        "size": len(content),
    }


def _raw_archive(path: Path) -> dict[str, bytes]:
    if zipfile.is_zipfile(path):
        entries = _zip_entries(path)
    elif tarfile.is_tarfile(path):
        entries = _tar_entries(path)
    else:
        raise ValueError(f"unsupported archive format: {path.name}")
    raw: dict[str, bytes] = {}
    for raw_name, content in entries:
        name = _safe_path(raw_name)
        if name in raw:
            raise ValueError(f"duplicate archive path: {name}")
        raw[name] = content
    return raw


def _entries(path: Path) -> list[ArchiveEntry]:
    raw = _raw_archive(path)
    records = [name for name in raw if name.endswith(".dist-info/RECORD")]
    if path.suffix == ".whl" and len(records) != 1:
        raise ValueError("wheel must contain exactly one .dist-info/RECORD")
    if len(records) > 1:
        raise ValueError("archive contains multiple .dist-info/RECORD files")
    normalized: list[ArchiveEntry] = []
    for name, raw_content in raw.items():
        if name in records:
            continue
        content = _normalized_content(name, raw_content)
        normalized.append(
            {"path": name, "sha256": sha256(content).hexdigest(), "size": len(content)}
        )
    normalized.sort(key=lambda item: item["path"])
    if records:
        normalized.append(_wheel_record_entry(raw, records[0], normalized))
    if not normalized:
        raise ValueError("archive contains no regular files")
    return sorted(normalized, key=lambda item: item["path"])


def _portable_entries(entries: list[ArchiveEntry]) -> list[ArchiveEntry]:
    """Return content that can be identical across native wheel platforms."""
    portable: list[ArchiveEntry] = []
    native = 0
    for entry in entries:
        path = entry["path"]
        if path.endswith((".dist-info/WHEEL", ".dist-info/RECORD")):
            continue
        if path.startswith("voiage/_core.") and path.endswith((".so", ".pyd")):
            native += 1
            continue
        portable.append(entry)
    if native > 1:
        raise ValueError("wheel inventory contains multiple native core extensions")
    return portable


def _entries_digest(entries: list[ArchiveEntry]) -> str:
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode()).hexdigest()


def normalized_archive_report(path: Path, *, runner: str) -> dict[str, object]:
    """Return metadata-independent content identity for a wheel, zip, or tar archive."""
    if not runner.strip():
        raise ValueError("runner identity must not be empty")
    entries = _entries(path)
    portable_entries = _portable_entries(entries)
    return {
        "schema_version": "1.0.0",
        "artifact_name": path.name,
        "runner": runner,
        "normalization": NORMALIZATION,
        "normalized_sha256": _entries_digest(entries),
        "portable_sha256": _entries_digest(portable_entries),
        "entries": entries,
    }


def compare_digest_reports(
    left: dict[str, object], right: dict[str, object]
) -> dict[str, object]:
    """Fail closed unless reports from distinct runners have equal normalized content."""
    for label, report in (("left", left), ("right", right)):
        if report.get("schema_version") != "1.0.0":
            raise ArtifactMismatchError(f"{label} digest report schema is unsupported")
        if report.get("normalization") != NORMALIZATION:
            raise ArtifactMismatchError(f"{label} normalization policy is unsupported")
        digest = report.get("normalized_sha256")
        portable_digest = report.get("portable_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ArtifactMismatchError(f"{label} normalized digest is invalid")
        if not isinstance(portable_digest, str) or len(portable_digest) != 64:
            raise ArtifactMismatchError(f"{label} portable digest is invalid")
        raw_entries = report.get("entries")
        if not isinstance(raw_entries, list):
            raise ArtifactMismatchError(f"{label} archive inventory is invalid")
        entries = cast("list[ArchiveEntry]", raw_entries)
        if _entries_digest(entries) != digest:
            raise ArtifactMismatchError(f"{label} normalized digest is inconsistent")
        if _entries_digest(_portable_entries(entries)) != portable_digest:
            raise ArtifactMismatchError(f"{label} portable digest is inconsistent")
    if left.get("runner") == right.get("runner"):
        raise ArtifactMismatchError("independent runner identities must differ")
    if left["portable_sha256"] != right["portable_sha256"]:
        raise ArtifactMismatchError("portable artifact digests differ")
    left_portable = _portable_entries(cast("list[ArchiveEntry]", left["entries"]))
    right_portable = _portable_entries(cast("list[ArchiveEntry]", right["entries"]))
    if left_portable != right_portable:
        raise ArtifactMismatchError("portable archive inventories differ")
    return {
        "schema_version": "1.0.0",
        "matched": True,
        "normalized_sha256": left["portable_sha256"],
        "runners": [left.get("runner"), right.get("runner")],
    }


__all__ = [
    "ArtifactMismatchError",
    "compare_digest_reports",
    "normalized_archive_report",
]
