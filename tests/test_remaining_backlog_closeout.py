"""Fail-closed tests for the remaining-backlog pre-closeout checkpoint."""

import hashlib
import json
from pathlib import Path
import shutil

import pytest

import scripts.validate_remaining_backlog_closeout as closeout
from scripts.validate_remaining_backlog_closeout import (
    EXPECTED_NOTES_REFS,
    validate_closeout,
)

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "conductor/tracks/remaining_backlog_delivery_20260831/pre-closeout-20260903.json"
)
RELEVANT_ROWS = [
    {
        "mode": "100644",
        "type": "blob",
        "oid": "03390c6b1bb1180663c2da21ef52c296933ecd7e",
        "path": "docs/release/hpc-distribution-handoff.md",
    },
    {
        "mode": "100644",
        "type": "blob",
        "oid": "b9a43d3125dd066fed79d295cc5a3291f4077618",
        "path": "packaging/spack/package.py",
    },
    {
        "mode": "100644",
        "type": "blob",
        "oid": "16b0f719c7e217d0703f46e6153e769ea5d0e55c",
        "path": "scripts/hpc_package_smoke.py",
    },
]


def test_pending_closeout_is_honest_and_valid() -> None:
    result = validate_closeout(RECEIPT, ROOT)
    assert result == {
        "state": "awaiting_native_spack_receipt",
        "merge_count": 2,
        "external_issue_count": 13,
        "open_issue_count": 15,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("merged_delivery", "1087", "head_sha"), "PR #1087"),
        (("closed_repository_issues", "614"), "closed repository"),
        (("inventory_snapshot", "stash_count"), "branch, worktree, or PR"),
        (("inventory_snapshot", "notes_refs"), "branch, worktree, or PR"),
        (("inventory_snapshot", "stash_reflog"), "branch, worktree, or PR"),
        (("recovery_preservation", "commit"), "recovery ref or external bundle"),
        (
            ("recovery_preservation", "restore_verification_sha256"),
            "recovery ref or external bundle",
        ),
        (("open_issues",), "open issue inventory"),
        (("open_issue_boundaries", "human_external_issue_numbers"), "external issue"),
    ],
)
def test_closeout_rejects_stale_inventory(
    tmp_path: Path, mutation: tuple[str, ...], message: str
) -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    target = payload
    for key in mutation[:-1]:
        target = target[key]
    target[mutation[-1]] = (
        [] if mutation[-1] == "human_external_issue_numbers" else "stale"
    )
    candidate = tmp_path / "closeout.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_closeout(candidate, ROOT)


def test_closeout_cannot_advance_without_native_receipt(tmp_path: Path) -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    payload["state"] = "ready_for_final_validation"
    candidate = tmp_path / "closeout.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="binding is incomplete"):
        validate_closeout(candidate, ROOT)


@pytest.mark.parametrize("notes_ref", EXPECTED_NOTES_REFS)
def test_closeout_rejects_each_stale_notes_ref(tmp_path: Path, notes_ref: str) -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    payload["inventory_snapshot"]["notes_refs"][notes_ref]["entry_count"] += 1
    candidate = tmp_path / "closeout.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="branch, worktree, or PR inventory"):
        validate_closeout(candidate, ROOT)


def test_closeout_rejects_custom_ref_oid_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    manifest = payload["inventory_snapshot"]["custom_refs"]
    manifest["refs"][0]["oid"] = "b" * 40
    raw = json.dumps(manifest["refs"], sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(
        closeout, "EXPECTED_CUSTOM_REF_MANIFEST_SHA256", manifest["manifest_sha256"]
    )
    candidate = tmp_path / "closeout.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="custom ref name or OID"):
        validate_closeout(candidate, ROOT)


def _recovery_payload(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    recovery = dict(payload["recovery_preservation"])
    bundle = tmp_path / "recovery.bundle"
    record = tmp_path / "restore-verification.txt"
    shutil.copyfile(recovery["bundle_path"], bundle)
    shutil.copyfile(recovery["restore_verification_path"], record)
    recovery["bundle_path"] = str(bundle)
    recovery["restore_verification_path"] = str(record)
    payload["recovery_preservation"] = recovery
    return payload, recovery


def test_closeout_rejects_deleted_recovery_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, recovery = _recovery_payload(tmp_path)
    Path(recovery["restore_verification_path"]).unlink()
    monkeypatch.setattr(closeout, "EXPECTED_RECOVERY_PRESERVATION", recovery)
    candidate = tmp_path / "closeout.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact is missing"):
        validate_closeout(candidate, ROOT)


def test_closeout_rejects_corrupt_recovery_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, recovery = _recovery_payload(tmp_path)
    bundle = Path(recovery["bundle_path"])
    bundle.write_bytes(b"not a git bundle")
    recovery["bundle_sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
    monkeypatch.setattr(closeout, "EXPECTED_RECOVERY_PRESERVATION", recovery)
    candidate = tmp_path / "closeout.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="git recovery verification failed"):
        validate_closeout(candidate, ROOT)


def test_closeout_rejects_recovery_ref_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, recovery = _recovery_payload(tmp_path)
    recovery["recovery_ref"] = "refs/heads/main"
    monkeypatch.setattr(closeout, "EXPECTED_RECOVERY_PRESERVATION", recovery)
    candidate = tmp_path / "closeout.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ref commit has drifted"):
        validate_closeout(candidate, ROOT)


def _ready_payload(tmp_path: Path) -> tuple[dict[str, object], Path]:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    bindings = {}
    for name, content in {
        "guest_receipt": json.dumps({"terminal": True}),
        "input": json.dumps({"candidate": closeout.EXPECTED_NATIVE_CANDIDATE}),
        "candidate_commit_file": closeout.EXPECTED_NATIVE_CANDIDATE + "\n",
    }.items():
        target = tmp_path / f"raw-{name}.txt"
        target.write_text(content, encoding="utf-8")
        bindings[name] = {
            "path": target.name,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    commands = []
    for identity in sorted(closeout.EXPECTED_NATIVE_COMMANDS):
        argv = ["native-run", identity]
        transcript_data = {
            "identity": identity,
            "argv": argv,
            "exit_code": 0,
            "outcome": "passed",
        }
        transcript = tmp_path / f"command-{identity}.json"
        transcript.write_text(json.dumps(transcript_data), encoding="utf-8")
        commands.append(
            {
                "identity": identity,
                "argv": argv,
                "exit_code": 0,
                "transcript": {
                    "path": transcript.name,
                    "sha256": hashlib.sha256(transcript.read_bytes()).hexdigest(),
                },
            }
        )
    probes = []
    for identity in sorted(closeout.EXPECTED_NATIVE_PROBES):
        command = ["native-probe", identity]
        transcript_data = {
            "identity": identity,
            "command": command,
            "exit_code": 0,
            "result": "passed",
            "outcome": "passed",
        }
        transcript = tmp_path / f"probe-{identity}.json"
        transcript.write_text(json.dumps(transcript_data), encoding="utf-8")
        probes.append(
            {
                "identity": identity,
                "command": command,
                "exit_code": 0,
                "result": "passed",
                "transcript": {
                    "path": transcript.name,
                    "sha256": hashlib.sha256(transcript.read_bytes()).hexdigest(),
                },
            }
        )
    evidence = {
        "schema_version": "voiage.native-spack-terminal-receipt.v1",
        "terminal": True,
        "outcome": "passed",
        "identity": {
            "platform": "linux-arm64",
            "host": "native-spack-runner",
            "run_id": "native-20260903-01",
            "candidate_commit": closeout.EXPECTED_NATIVE_CANDIDATE,
            "base_main": payload["base_main"],
        },
        "relevant_path_manifest": {
            "paths": closeout.EXPECTED_NATIVE_RELEVANT_PATHS,
            "sha256": closeout.EXPECTED_NATIVE_RELEVANT_MANIFEST_SHA256,
            "entries": RELEVANT_ROWS,
        },
        "raw_bindings": bindings,
        "qualification_matrix": dict.fromkeys(
            sorted(closeout.EXPECTED_NATIVE_QUALIFICATIONS), "passed"
        ),
        "commands": commands,
        "probes": probes,
    }
    evidence_path = tmp_path / "native.json"
    raw = json.dumps(evidence).encode()
    evidence_path.write_bytes(raw)
    payload["state"] = "ready_for_final_validation"
    payload["native_spack_receipt"] = {
        "path": evidence_path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "outcome": evidence["outcome"],
        "relevant_path_manifest_sha256": closeout.EXPECTED_NATIVE_RELEVANT_MANIFEST_SHA256,
        **evidence["identity"],
    }
    return payload, evidence_path


def _isolate_native_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        closeout,
        "_validate_custom_refs",
        lambda snapshot, root: snapshot.pop("custom_refs"),
    )
    monkeypatch.setattr(closeout, "_validate_recovery", lambda recovery, root: None)
    stash_output = "\n".join(
        f"{item['oid']}\t{item['selector']}\t{item['subject']}"
        for item in closeout.EXPECTED_STASH_REFLOG
    )
    monkeypatch.setattr(
        closeout,
        "_git",
        lambda root, *args: (
            stash_output
            if args[:2] == ("stash", "list")
            else "\n".join(
                f"{x['mode']} {x['type']} {x['oid']}\t{x['path']}"
                for x in RELEVANT_ROWS
            )
            if args and args[0] == "ls-tree"
            else ""
        ),
    )


def test_closeout_accepts_schema_bound_native_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_native_validation(monkeypatch)
    payload, _ = _ready_payload(tmp_path)
    candidate = tmp_path / "closeout.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    result = validate_closeout(candidate, tmp_path)
    assert result["state"] == "ready_for_final_validation"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "unknown", "schema or identity is invalid"),
        ("outcome", "running", "outcome is not exact"),
        ("platform", "wrong-platform", "schema or identity is invalid"),
        ("host", "wrong-host", "schema or identity is invalid"),
        ("run_id", "wrong-run", "schema or identity is invalid"),
        ("candidate_commit", "b" * 40, "schema or identity is invalid"),
        ("base_main", "b" * 40, "schema or identity is invalid"),
    ],
)
def test_closeout_rejects_native_receipt_semantic_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    _isolate_native_validation(monkeypatch)
    payload, evidence_path = _ready_payload(tmp_path)
    evidence = json.loads(evidence_path.read_text())
    if field in {"schema_version", "outcome"}:
        evidence[field] = value
    else:
        evidence["identity"][field] = value
    raw = json.dumps(evidence).encode()
    evidence_path.write_bytes(raw)
    payload["native_spack_receipt"]["sha256"] = hashlib.sha256(raw).hexdigest()
    candidate = tmp_path / "closeout.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_closeout(candidate, tmp_path)


def test_closeout_rejects_arbitrary_native_receipt_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_native_validation(monkeypatch)
    payload, evidence_path = _ready_payload(tmp_path)
    raw = b"not a structured terminal receipt"
    evidence_path.write_bytes(raw)
    payload["native_spack_receipt"]["sha256"] = hashlib.sha256(raw).hexdigest()
    candidate = tmp_path / "closeout.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt transcript is not JSON"):
        validate_closeout(candidate, tmp_path)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("commands", "command identities"),
        ("qualification_matrix", "qualification matrix"),
        ("probes", "probe identities"),
    ],
)
def test_closeout_rejects_passed_native_receipt_without_proof_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    message: str,
) -> None:
    _isolate_native_validation(monkeypatch)
    payload, evidence_path = _ready_payload(tmp_path)
    evidence = json.loads(evidence_path.read_text())
    evidence[field] = []
    raw = json.dumps(evidence).encode()
    evidence_path.write_bytes(raw)
    payload["native_spack_receipt"]["sha256"] = hashlib.sha256(raw).hexdigest()
    candidate = tmp_path / "closeout.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_closeout(candidate, tmp_path)


def test_closeout_rejects_native_receipt_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_native_validation(monkeypatch)
    payload, evidence_path = _ready_payload(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_bytes(evidence_path.read_bytes())
    evidence_path.unlink()
    evidence_path.symlink_to(outside)
    candidate = tmp_path / "closeout.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="must not be a symlink"):
            validate_closeout(candidate, tmp_path)
    finally:
        outside.unlink()


def test_closeout_rejects_stale_real_ancestor_as_native_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_native_validation(monkeypatch)
    payload, evidence_path = _ready_payload(tmp_path)
    stale = "fea90d41898ac31c970b0c2b7a8a80ef3366ab96"
    evidence = json.loads(evidence_path.read_text())
    evidence["identity"]["candidate_commit"] = stale
    payload["native_spack_receipt"]["candidate_commit"] = stale
    raw = json.dumps(evidence).encode()
    evidence_path.write_bytes(raw)
    payload["native_spack_receipt"]["sha256"] = hashlib.sha256(raw).hexdigest()
    candidate = tmp_path / "closeout.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not explicitly reviewed"):
        validate_closeout(candidate, tmp_path)


def test_closeout_rejects_mismatched_command_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_native_validation(monkeypatch)
    payload, evidence_path = _ready_payload(tmp_path)
    evidence = json.loads(evidence_path.read_text())
    command = evidence["commands"][0]
    transcript = tmp_path / command["transcript"]["path"]
    parsed = json.loads(transcript.read_text())
    parsed["outcome"] = "dummy"
    transcript.write_text(json.dumps(parsed), encoding="utf-8")
    command["transcript"]["sha256"] = hashlib.sha256(
        transcript.read_bytes()
    ).hexdigest()
    raw = json.dumps(evidence).encode()
    evidence_path.write_bytes(raw)
    payload["native_spack_receipt"]["sha256"] = hashlib.sha256(raw).hexdigest()
    candidate = tmp_path / "closeout.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="transcript outcome does not match"):
        validate_closeout(candidate, tmp_path)


def test_closeout_rejects_mismatched_candidate_commit_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_native_validation(monkeypatch)
    payload, evidence_path = _ready_payload(tmp_path)
    evidence = json.loads(evidence_path.read_text())
    binding = evidence["raw_bindings"]["candidate_commit_file"]
    source_file = tmp_path / binding["path"]
    source_file.write_text("0" * 40 + "\n", encoding="utf-8")
    binding["sha256"] = hashlib.sha256(source_file.read_bytes()).hexdigest()
    raw = json.dumps(evidence).encode()
    evidence_path.write_bytes(raw)
    payload["native_spack_receipt"]["sha256"] = hashlib.sha256(raw).hexdigest()
    candidate = tmp_path / "closeout.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="commit file is inconsistent"):
        validate_closeout(candidate, tmp_path)


def test_closeout_rejects_missing_probe_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_native_validation(monkeypatch)
    payload, evidence_path = _ready_payload(tmp_path)
    evidence = json.loads(evidence_path.read_text())
    (tmp_path / evidence["probes"][0]["transcript"]["path"]).unlink()
    candidate = tmp_path / "closeout.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="transcript is missing"):
        validate_closeout(candidate, tmp_path)


def test_closeout_rejects_failed_terminal_with_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_native_validation(monkeypatch)
    payload, evidence_path = _ready_payload(tmp_path)
    evidence = json.loads(evidence_path.read_text())
    for field in ("commands", "probes", "qualification_matrix"):
        evidence.pop(field)
    evidence["outcome"] = "failed_terminal"
    failure = {
        "stage": "spack_install",
        "identity": "spack_install",
        "argv": ["spack", "install", "py-voiage"],
        "exit_code": 0,
    }
    transcript = tmp_path / "failure.json"
    transcript.write_text(
        json.dumps({**failure, "outcome": "failed_terminal"}), encoding="utf-8"
    )
    evidence["failure"] = {
        **failure,
        "transcript": {
            "path": transcript.name,
            "sha256": hashlib.sha256(transcript.read_bytes()).hexdigest(),
        },
    }
    raw = json.dumps(evidence).encode()
    evidence_path.write_bytes(raw)
    payload["native_spack_receipt"].update(
        outcome="failed_terminal", sha256=hashlib.sha256(raw).hexdigest()
    )
    candidate = tmp_path / "closeout.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="command or stage is invalid"):
        validate_closeout(candidate, tmp_path)
