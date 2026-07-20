from __future__ import annotations

import base64
import csv
from hashlib import sha256
import io
import json
import sys
import tarfile
from typing import TYPE_CHECKING, Self
import zipfile

import pytest

from scripts.c15_artifact_digest import main as artifact_digest_main
from voiage import c15_reproducibility
from voiage.c15_reproducibility import (
    ArtifactMismatchError,
    compare_digest_reports,
    normalized_archive_report,
)

if TYPE_CHECKING:
    from pathlib import Path


def _record(contents: dict[str, bytes], record_name: str) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name, content in contents.items():
        digest = (
            base64.urlsafe_b64encode(sha256(content).digest()).rstrip(b"=").decode()
        )
        writer.writerow([name, f"sha256={digest}", len(content)])
    writer.writerow([record_name, "", ""])
    return stream.getvalue().encode()


def _wheel(path: Path, newline: str) -> None:
    contents = {
        "voiage/data.txt": f"alpha{newline}beta{newline}".encode(),
        "voiage/data.bin": b"\x00\x01",
        "voiage-1.dist-info/licenses/LICENSE": f"terms{newline}".encode(),
    }
    record_name = "voiage-1.dist-info/RECORD"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
        archive.writestr(record_name, _record(contents, record_name))


def test_wheel_identity_normalizes_line_endings_and_validates_record(
    tmp_path: Path,
) -> None:
    linux = tmp_path / "linux.whl"
    windows = tmp_path / "windows.whl"
    _wheel(linux, "\n")
    _wheel(windows, "\r\n")
    left = normalized_archive_report(linux, runner="linux-x64")
    right = normalized_archive_report(windows, runner="windows-x64")
    assert left["normalized_sha256"] == right["normalized_sha256"]
    assert any(item["path"].endswith("RECORD") for item in left["entries"])
    assert compare_digest_reports(left, right)["matched"] is True


def test_wheel_record_tampering_and_missing_record_fail_closed(tmp_path: Path) -> None:
    tampered = tmp_path / "tampered.whl"
    original = {"voiage/data.txt": b"original"}
    record_name = "voiage-1.dist-info/RECORD"
    with zipfile.ZipFile(tampered, "w") as archive:
        archive.writestr("voiage/data.txt", b"tampered")
        archive.writestr(record_name, _record(original, record_name))
    with pytest.raises(ValueError, match="integrity mismatch"):
        normalized_archive_report(tampered, runner="runner")
    missing = tmp_path / "missing.whl"
    with zipfile.ZipFile(missing, "w") as archive:
        archive.writestr("voiage/data.txt", b"x")
    with pytest.raises(ValueError, match="exactly one"):
        normalized_archive_report(missing, runner="runner")


def test_sdist_ignores_metadata_but_not_content(tmp_path: Path) -> None:
    source = tmp_path / "payload.json"
    source.write_text('{"value":1}\n', encoding="utf-8")
    archive = tmp_path / "voiage.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(source, arcname="voiage/payload.json")
    report = normalized_archive_report(archive, runner="linux")
    assert report["entries"][0]["path"] == "voiage/payload.json"


def test_comparison_rejects_same_runner_or_drift(tmp_path: Path) -> None:
    wheel = tmp_path / "one.whl"
    _wheel(wheel, "\n")
    report = normalized_archive_report(wheel, runner="same")
    with pytest.raises(ArtifactMismatchError, match="runner"):
        compare_digest_reports(report, report)
    other = {**report, "runner": "other", "normalized_sha256": "0" * 64}
    with pytest.raises(ArtifactMismatchError, match="digests"):
        compare_digest_reports(report, other)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(schema_version="2.0.0"),
        lambda value: value.update(normalization="unknown"),
        lambda value: value.update(normalized_sha256="short"),
        lambda value: value.update(entries="invalid"),
    ],
)
def test_comparison_rejects_malformed_reports(tmp_path: Path, mutation) -> None:
    wheel = tmp_path / "one.whl"
    _wheel(wheel, "\n")
    left = normalized_archive_report(wheel, runner="left")
    right = {**left, "runner": "right"}
    mutation(right)
    with pytest.raises(ArtifactMismatchError):
        compare_digest_reports(left, right)


def test_archives_reject_unsafe_duplicate_and_empty_members(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.tar.gz"
    with tarfile.open(unsafe, "w:gz") as archive:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        normalized_archive_report(unsafe, runner="runner")

    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("value.bin", b"one")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("value.bin", b"two")
    with pytest.raises(ValueError, match="duplicate"):
        normalized_archive_report(duplicate, runner="runner")

    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w"):
        pass
    with pytest.raises(ValueError, match="no regular files"):
        normalized_archive_report(empty, runner="runner")


def test_wheel_record_rejects_malformed_inventory(tmp_path: Path) -> None:
    wheel = tmp_path / "malformed.whl"
    record_name = "voiage-1.dist-info/RECORD"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("voiage/data.bin", b"x")
        archive.writestr(record_name, "voiage/data.bin,invalid\n")
    with pytest.raises(ValueError, match="rows must contain"):
        normalized_archive_report(wheel, runner="runner")


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (b"\xff", "UTF-8"),
        (
            b"voiage-1.dist-info/RECORD,,\nvoiage-1.dist-info/RECORD,,\n",
            "duplicate",
        ),
        (b"voiage-1.dist-info/RECORD,sha256=x,1\n", "must not hash itself"),
        (b"missing.bin,sha256=x,1\nvoiage-1.dist-info/RECORD,,\n", "missing"),
        (b"voiage-1.dist-info/RECORD,,\n", "inventory"),
    ],
)
def test_wheel_record_semantics_fail_closed(
    tmp_path: Path, record: bytes, message: str
) -> None:
    wheel = tmp_path / "semantic.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("voiage/data.bin", b"x")
        archive.writestr("voiage-1.dist-info/RECORD", record)
    with pytest.raises(ValueError, match=message):
        normalized_archive_report(wheel, runner="runner")


def test_archive_and_runner_validation_cover_non_wheel_paths(tmp_path: Path) -> None:
    unsupported = tmp_path / "plain.txt"
    unsupported.write_text("plain", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        normalized_archive_report(unsupported, runner="runner")

    archive_path = tmp_path / "content.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.mkdir("folder/")
        archive.writestr("folder/value.txt", b"\xff")
    with pytest.raises(ValueError, match="empty"):
        normalized_archive_report(archive_path, runner=" ")
    assert normalized_archive_report(archive_path, runner="runner")["entries"]


def test_comparison_checks_inventory_after_digest(tmp_path: Path) -> None:
    wheel = tmp_path / "one.whl"
    _wheel(wheel, "\n")
    left = normalized_archive_report(wheel, runner="left")
    right = {**left, "runner": "right", "entries": []}
    with pytest.raises(ArtifactMismatchError, match="inconsistent"):
        compare_digest_reports(left, right)


def test_cross_platform_wheels_compare_portable_content_and_retain_native_evidence(
    tmp_path: Path,
) -> None:
    reports: list[dict[str, object]] = []
    for runner, native_name, native, tag in (
        ("linux", "voiage/_core.abi3.so", b"ELF", "cp312-abi3-linux_x86_64"),
        ("windows", "voiage/_core.pyd", b"PE", "cp312-abi3-win_amd64"),
    ):
        wheel = tmp_path / f"{runner}.whl"
        contents = {
            "voiage/module.py": b"VALUE = 1\n",
            native_name: native,
            "voiage-1.dist-info/WHEEL": f"Wheel-Version: 1.0\nTag: {tag}\n".encode(),
        }
        record_name = "voiage-1.dist-info/RECORD"
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, content in contents.items():
                archive.writestr(name, content)
            archive.writestr(record_name, _record(contents, record_name))
        reports.append(normalized_archive_report(wheel, runner=runner))

    assert reports[0]["normalized_sha256"] != reports[1]["normalized_sha256"]
    assert reports[0]["portable_sha256"] == reports[1]["portable_sha256"]
    assert compare_digest_reports(reports[0], reports[1])["matched"] is True


def test_source_archives_normalize_declared_text_and_scm_paths(tmp_path: Path) -> None:
    linux = tmp_path / "linux.zip"
    windows = tmp_path / "windows.zip"
    linux_files = {
        "project/Dockerfile": b"FROM scratch\n",
        "project/src/module.rs": b"fn main() {}\n",
        "project/data/refs.bib": b"@misc{x}\n",
        "project/notebook.jupyter": b"metadata\n",
        "project/template.ambr": b"template\n",
        "project/scm_file_list.json": json.dumps(
            {"files": ["project/Dockerfile", "project/src/module.rs"]}
        ).encode(),
    }
    windows_files = {
        name: content.replace(b"\n", b"\r\n")
        for name, content in linux_files.items()
        if name != "project/scm_file_list.json"
    }
    windows_files["project/scm_file_list.json"] = json.dumps(
        {"files": ["project\\src\\module.rs", "project\\Dockerfile"]}
    ).encode()
    for path, contents in ((linux, linux_files), (windows, windows_files)):
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in contents.items():
                archive.writestr(name, content)
    left = normalized_archive_report(linux, runner="linux")
    right = normalized_archive_report(windows, runner="windows")
    assert compare_digest_reports(left, right)["matched"] is True


@pytest.mark.parametrize(
    "payload", ["not-json", "[]", '{"files":"bad"}', '{"files":[1]}']
)
def test_invalid_scm_inventory_fails_closed(tmp_path: Path, payload: str) -> None:
    archive_path = tmp_path / "invalid.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("scm_file_list.json", payload)
    with pytest.raises((TypeError, ValueError), match="string file list"):
        normalized_archive_report(archive_path, runner="runner")


def test_tar_directory_and_unreadable_member_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "directory.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        directory = tarfile.TarInfo("folder")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        value = tarfile.TarInfo("folder/value.txt")
        value.size = 1
        archive.addfile(value, io.BytesIO(b"x"))
    assert normalized_archive_report(archive_path, runner="runner")["entries"]

    class Member:
        name = "unreadable.txt"

        def isfile(self) -> bool:
            return True

    class Archive:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def getmembers(self) -> list[Member]:
            return [Member()]

        def extractfile(self, _member: Member) -> None:
            return None

    monkeypatch.setattr(c15_reproducibility.zipfile, "is_zipfile", lambda _path: False)
    monkeypatch.setattr(c15_reproducibility.tarfile, "is_tarfile", lambda _path: True)
    monkeypatch.setattr(
        c15_reproducibility.tarfile, "open", lambda *_a, **_k: Archive()
    )
    with pytest.raises(ValueError, match="cannot be read"):
        normalized_archive_report(archive_path, runner="runner")


def test_multiple_record_files_fail_closed(tmp_path: Path) -> None:
    archive_path = tmp_path / "multiple.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("one.dist-info/RECORD", b"")
        archive.writestr("two.dist-info/RECORD", b"")
    with pytest.raises(ValueError, match="multiple"):
        normalized_archive_report(archive_path, runner="runner")


def test_compare_cli_retains_failure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "one.whl"
    _wheel(wheel, "\n")
    report = normalized_archive_report(wheel, runner="same")
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    output = tmp_path / "failure.json"
    left.write_text(json.dumps(report), encoding="utf-8")
    right.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "c15_artifact_digest.py",
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
            "--output",
            str(output),
        ],
    )
    assert artifact_digest_main() == 2
    failure = json.loads(output.read_text(encoding="utf-8"))
    assert failure["passed"] is False
    assert failure["operation"] == "compare"
