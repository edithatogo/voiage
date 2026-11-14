"""Exercise the portable manuscript package validator on small DOCX fixtures."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "scripts" / "docx_validator" / "DocxGate.csproj"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


@pytest.fixture(scope="session")
def docx_gate(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, Path]:
    """Build the gate when a .NET 10 SDK and runtime are available."""
    dotnet = shutil.which("dotnet")
    if dotnet is None:
        pytest.skip("DOCX validator qualification requires a .NET 10 SDK and runtime")
    sdks = subprocess.run(
        [dotnet, "--list-sdks"], check=True, capture_output=True, text=True
    ).stdout
    runtimes = subprocess.run(
        [dotnet, "--list-runtimes"], check=True, capture_output=True, text=True
    ).stdout
    if not any(line.startswith("10.") for line in sdks.splitlines()) or not any(
        line.startswith("Microsoft.NETCore.App 10.") for line in runtimes.splitlines()
    ):
        pytest.skip("DOCX validator qualification requires a .NET 10 SDK and runtime")

    build_dir = tmp_path_factory.mktemp("docx-validator-build")
    env = os.environ.copy()
    env.update(
        {
            "DOTNET_CLI_HOME": str(build_dir / "home"),
            "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
            "DOTNET_GENERATE_ASPNET_CERTIFICATE": "false",
            "NUGET_PACKAGES": str(build_dir / "nuget"),
        }
    )
    subprocess.run(
        [
            dotnet,
            "build",
            str(PROJECT),
            "--configuration",
            "Release",
            "--artifacts-path",
            str(build_dir / "artifacts"),
            "--nologo",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    return (
        dotnet,
        build_dir / "artifacts" / "bin" / "DocxGate" / "release" / "DocxGate.dll",
    )


def write_docx(
    path: Path, *, document: str | None = None, relationships: str | None = None
) -> None:
    """Write a minimal, standards-conforming Word package."""
    types = (
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    if relationships is None:
        relationships = (
            f'<Relationships xmlns="{REL_NS}">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/></Relationships>'
        )
    if document is None:
        document = (
            f'<w:document xmlns:w="{WORD_NS}"><w:body><w:p><w:r>'
            "<w:t>Test document</w:t></w:r></w:p></w:body></w:document>"
        )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("word/document.xml", document)


def run_gate(
    gate: tuple[str, Path], report: Path, *files: Path
) -> subprocess.CompletedProcess[str]:
    """Run the compiled gate with an explicit new receipt."""
    return subprocess.run(
        [
            gate[0],
            str(gate[1]),
            "--report",
            str(report),
            *(str(file) for file in files),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_valid_manuscript_and_supplement_are_hash_bound(
    docx_gate: tuple[str, Path], tmp_path: Path
) -> None:
    manuscript, supplement = tmp_path / "main.docx", tmp_path / "supplement.docx"
    write_docx(manuscript)
    write_docx(supplement)
    hashes = [
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (manuscript, supplement)
    ]
    report = tmp_path / "receipt.json"
    result = run_gate(docx_gate, report, manuscript, supplement)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(report.read_text())
    assert receipt["passed"] is True
    assert [item["sha256"] for item in receipt["files"]] == hashes
    assert all(item["schemaErrors"] == 0 for item in receipt["files"])
    assert len(receipt["tool"]["assembly_sha256"]) == 64
    assert len(receipt["tool"]["openxml_sha256"]) == 64
    assert len(receipt["tool"]["zip_library_sha256"]) == 64
    assert len(receipt["tool"]["packaging_sha256"]) == 64
    assert [
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (manuscript, supplement)
    ] == hashes


@pytest.mark.parametrize(
    "document",
    [
        f'<w:document xmlns:w="{WORD_NS}"><w:body>',
        f'<w:document xmlns:w="{WORD_NS}"><w:body><w:r/></w:body></w:document>',
        f'<w:document xmlns:w="{WORD_NS}"><w:body><w:p><w:r/><w:pPr/></w:p></w:body></w:document>',
        f'<w:document xmlns:w="{WORD_NS}"><w:body><w:p><w:r><w:rPr><w:b w:val="invalid"/>'
        "</w:rPr></w:r></w:p></w:body></w:document>",
        f'<w:document xmlns:w="{WORD_NS}" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'mc:Ignorable="w99"><w:body/></w:document>',
    ],
    ids=[
        "malformed-xml",
        "invalid-element",
        "element-order",
        "invalid-formatting",
        "compatibility-prefix",
    ],
)
def test_invalid_xml_and_openxml_are_rejected(
    docx_gate: tuple[str, Path], tmp_path: Path, document: str
) -> None:
    path, report = tmp_path / "invalid.docx", tmp_path / "receipt.json"
    write_docx(path, document=document)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = run_gate(docx_gate, report, path)
    assert result.returncode == 1, result.stdout + result.stderr
    receipt = json.loads(report.read_text())
    assert receipt["passed"] is False
    assert receipt["files"][0]["issues"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_missing_internal_target_and_crc_mismatch_are_rejected(
    docx_gate: tuple[str, Path], tmp_path: Path
) -> None:
    missing = tmp_path / "missing.docx"
    corrupt = tmp_path / "central-crc.docx"
    local_corrupt = tmp_path / "local-crc.docx"
    relationships = (
        f'<Relationships xmlns="{REL_NS}">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/missing.xml"/></Relationships>'
    )
    write_docx(missing, relationships=relationships)
    write_docx(corrupt)
    data = bytearray(corrupt.read_bytes())
    central_header = data.index(b"PK\x01\x02")
    crc = struct.unpack_from("<I", data, central_header + 16)[0]
    struct.pack_into("<I", data, central_header + 16, crc ^ 1)
    corrupt.write_bytes(data)
    write_docx(local_corrupt)
    data = bytearray(local_corrupt.read_bytes())
    local_header = data.index(b"PK\x03\x04")
    crc = struct.unpack_from("<I", data, local_header + 14)[0]
    struct.pack_into("<I", data, local_header + 14, crc ^ 1)
    local_corrupt.write_bytes(data)
    for index, path in enumerate((missing, corrupt, local_corrupt)):
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        report = tmp_path / f"receipt-{index}.json"
        result = run_gate(docx_gate, report, path)
        assert result.returncode == 1, result.stdout + result.stderr
        assert json.loads(report.read_text())["files"][0]["issues"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_valid_data_descriptors_and_zip64_are_accepted(
    docx_gate: tuple[str, Path], tmp_path: Path
) -> None:
    class UnseekableBuffer(io.BytesIO):
        def seek(self, offset: int, whence: int = 0) -> int:
            raise io.UnsupportedOperation("stream is not seekable")

    source = tmp_path / "source.docx"
    write_docx(source)
    stream = UnseekableBuffer()
    with zipfile.ZipFile(source) as original:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as result:
            for entry in original.infolist():
                result.writestr(entry.filename, original.read(entry))
        zip64_path = tmp_path / "zip64.docx"
        with zipfile.ZipFile(zip64_path, "w") as result:
            for entry in original.infolist():
                with result.open(entry.filename, "w", force_zip64=True) as output:
                    output.write(original.read(entry))
    descriptor_path = tmp_path / "descriptor.docx"
    descriptor_path.write_bytes(stream.getvalue())
    report = tmp_path / "valid-zip-variants.json"
    result = run_gate(docx_gate, report, descriptor_path, zip64_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(report.read_text())["passed"] is True


def test_duplicate_parts_and_external_xml_entities_are_rejected(
    docx_gate: tuple[str, Path], tmp_path: Path
) -> None:
    duplicate, external_entity = tmp_path / "duplicate.docx", tmp_path / "entity.docx"
    write_docx(duplicate)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate, "a") as archive:
            archive.writestr("word/document.xml", "<duplicate/>")
    document = (
        '<!DOCTYPE w:document [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
        f'<w:document xmlns:w="{WORD_NS}"><w:body><w:p><w:r><w:t>'
        "&xxe;</w:t></w:r></w:p></w:body></w:document>"
    )
    write_docx(external_entity, document=document)
    for index, path in enumerate((duplicate, external_entity)):
        report = tmp_path / f"invalid-package-{index}.json"
        result = run_gate(docx_gate, report, path)
        assert result.returncode == 1, result.stdout + result.stderr
        assert json.loads(report.read_text())["files"][0]["issues"]


def test_invalid_invocation_and_missing_input_fail_closed(
    docx_gate: tuple[str, Path], tmp_path: Path
) -> None:
    existing_report = tmp_path / "existing.json"
    existing_report.write_text("keep")
    assert (
        run_gate(docx_gate, existing_report, tmp_path / "missing.docx").returncode == 2
    )
    assert existing_report.read_text() == "keep"
    original = tmp_path / "original.docx"
    write_docx(original)
    original_bytes = original.read_bytes()
    assert run_gate(docx_gate, original, original).returncode == 2
    assert original.read_bytes() == original_bytes
    missing_report = tmp_path / "missing-receipt.json"
    assert (
        run_gate(docx_gate, missing_report, tmp_path / "missing.docx").returncode == 1
    )
    assert json.loads(missing_report.read_text())["passed"] is False
    assert (
        subprocess.run([docx_gate[0], str(docx_gate[1])], check=False).returncode == 2
    )
