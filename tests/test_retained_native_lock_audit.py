"""Reject drift when preparing the retained native lock for advisory checks."""

from pathlib import Path
import shutil

import pytest

from scripts import audit_retained_native_lock as audit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def packet(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    for name in audit.FROZEN:
        source = ROOT / audit.PACKET / "native" / name
        destination = root / audit.PACKET / "native" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    identity = audit.PACKET / "package-identities.json"
    shutil.copyfile(ROOT / identity, root / identity)
    return root


def resolved_replica(root: Path, destination: Path) -> Path:
    audit.prepare(root, destination)
    lock = destination / "Cargo.lock"
    text = lock.read_text()
    marker = 'name = "innovate-rs"\nversion = "0.5.0"\n'
    assert text.count(marker) == 1
    lock.write_text(text.replace(marker, marker + f'source = "{audit.GIT_SOURCE}"\n'))
    return lock


def test_exact_source_mirror_preserves_the_entire_locked_graph(
    packet: Path, tmp_path: Path
) -> None:
    replica = tmp_path / "replica"
    resolved_replica(packet, replica)
    result = audit.verify(packet, replica)
    assert result["passed"] is True
    assert result["innovate_revision"] == audit.REVISION
    assert result["frozen_lock_sha256"] == audit.FROZEN["Cargo.lock"]


@pytest.mark.parametrize(
    "damage",
    [
        "manifest",
        "native-source",
        "version",
        "checksum",
        "source",
        "missing",
        "original",
        "identity",
    ],
)
def test_replica_and_source_drift_fail_closed(
    packet: Path, tmp_path: Path, damage: str
) -> None:
    replica = tmp_path / "replica"
    lock = resolved_replica(packet, replica)
    if damage == "manifest":
        path = replica / "Cargo.toml"
        path.write_text(path.read_text() + "\n# altered\n")
    elif damage == "native-source":
        path = replica / "src/main.rs"
        path.write_text(path.read_text() + "\n// altered\n")
    elif damage == "version":
        text = lock.read_text()
        assert 'version = "1.0.18"' in text
        lock.write_text(text.replace('version = "1.0.18"', 'version = "1.0.19"', 1))
    elif damage == "checksum":
        text = lock.read_text()
        assert "8f42a60cbdf9a97f" in text
        lock.write_text(text.replace("8f42a60cbdf9a97f", "0f42a60cbdf9a97f", 1))
    elif damage == "source":
        text = lock.read_text()
        lock.write_text(text.replace(audit.GIT_SOURCE, audit.GIT_SOURCE + "0"))
    elif damage == "missing":
        lock.unlink()
    elif damage == "original":
        path = packet / audit.PACKET / "native/Cargo.lock"
        path.write_text(path.read_text() + "\n# altered\n")
    else:
        path = packet / audit.PACKET / "package-identities.json"
        path.write_text(path.read_text().replace(audit.REVISION, "0" * 40))
    with pytest.raises((OSError, ValueError)):
        audit.verify(packet, replica)


def test_prepare_refuses_to_replace_existing_work(packet: Path, tmp_path: Path) -> None:
    destination = tmp_path / "replica"
    destination.mkdir()
    marker_file = destination / "keep.txt"
    marker_file.write_text("preserve")
    with pytest.raises(FileExistsError):
        audit.prepare(packet, destination)
    assert marker_file.read_text() == "preserve"


def test_unresolved_replica_cannot_be_reported_as_equivalent(
    packet: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "replica"
    audit.prepare(packet, destination)
    with pytest.raises(ValueError, match="source identity"):
        audit.verify(packet, destination)
