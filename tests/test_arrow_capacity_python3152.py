"""Preserve prior capacity evidence while preparing the 3.15.2 experiment."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import observe_arrow_capacity as observer

ROOT = Path(__file__).resolve().parents[1]
OLD_INPUT = ROOT / "docs/reviews/arrow-capacity-inputs-20261029.json"
NEW_INPUT = ROOT / "docs/reviews/arrow-capacity-inputs-20261115-python3152.json"
OLD_TOOL = (
    ROOT / "conductor/tracks/remaining_backlog_delivery_20260831/historical/"
    "observe-arrow-capacity-python3151-20261115.py.txt"
)


def test_new_capacity_profile_preserves_the_previous_source_evidence() -> None:
    original_bytes = OLD_INPUT.read_bytes()
    assert hashlib.sha256(original_bytes).hexdigest() == (
        "7b4fccfb9ac0e96cae259051d584ba3b2da8eb8480da0128ef0a52960c77e38c"
    )
    original = json.loads(original_bytes)
    current = json.loads(NEW_INPUT.read_text())
    manifest_path = ROOT / "packaging/current/arrow-provider-sources.json"
    manifest = json.loads(manifest_path.read_text())
    expected = dict(original)
    expected.update(
        python_profile="3.15.2",
        provider_snapshot_commit="b0d6bc662bf1206f489fd4506b856783e6cb98d4",
        provider_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        previous_inputs={
            "path": "docs/reviews/arrow-capacity-inputs-20261029.json",
            "sha256": hashlib.sha256(original_bytes).hexdigest(),
        },
        execution_status="prepared-not-executed",
    )
    assert current == expected
    for name, provider in current["providers"].items():
        assert provider == manifest["providers"][name]


def test_previous_observer_bytes_and_version_gate_are_retained() -> None:
    original = OLD_TOOL.read_bytes()
    assert hashlib.sha256(original).hexdigest() == (
        "b920992eb58190965d1d1f7dbde3024e04b91eaac699e7d1771c7849c8670a4a"
    )
    expected = original.replace(b"(3, 15, 1)", b"(3, 15, 2)").replace(
        b"3.15.1", b"3.15.2"
    )
    assert (ROOT / "scripts/observe_arrow_capacity.py").read_bytes() == expected


def test_current_observer_rejects_the_previous_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(observer.sys, "version_info", (3, 15, 1))
    monkeypatch.setattr(
        observer.sys,
        "argv",
        [
            "observe_arrow_capacity.py",
            "--inputs",
            str(NEW_INPUT),
            "--workspace",
            str(tmp_path / "work"),
            "--evidence",
            str(tmp_path / "evidence"),
        ],
    )
    with pytest.raises(RuntimeError, match="Select Python 3.15.2"):
        observer.main()
    assert not (tmp_path / "work").exists()


def test_current_observer_rejects_the_old_profile_before_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(observer.sys, "version_info", (3, 15, 2))
    monkeypatch.setattr(observer.sys, "platform", "linux")
    monkeypatch.setattr(
        observer.sys,
        "argv",
        [
            "observe_arrow_capacity.py",
            "--inputs",
            str(OLD_INPUT),
            "--workspace",
            str(tmp_path / "work"),
            "--evidence",
            str(tmp_path / "evidence"),
        ],
    )
    with pytest.raises(ValueError, match="Unexpected Python input profile"):
        observer.main()
    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "evidence").exists()
