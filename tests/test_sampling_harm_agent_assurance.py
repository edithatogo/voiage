"""Challenge the non-authorizing assurance record's previously unchecked bindings."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import pytest

from voiage.sampling_harm_agent_assurance import (
    ASSURANCE_PATH,
    CONTRACT_ROOT,
    SamplingHarmAgentAssuranceError,
    load_and_validate_sampling_harm_agent_assurance,
)
from voiage.scientific_review_evidence import canonical_json_sha256

ROOT = Path(__file__).parents[1]
NOW = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    for relative in (
        CONTRACT_ROOT,
        Path("specs/frontier/governance/scientific-review/v1/schemas"),
    ):
        shutil.copytree(ROOT / relative, tmp_path / relative)
    return tmp_path


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value))


def validate(root: Path) -> dict[str, object]:
    return load_and_validate_sampling_harm_agent_assurance(
        repository_root=root, now=NOW
    )


def test_existing_record_uses_mixed_digests_without_promoting_the_packet() -> None:
    result = validate(ROOT)
    assert result["pending_findings"] == 19
    assert result["role_report_count"] == 5
    assert result["historical_packet_only"] is True
    assert result["qualified_replacement_packet"] is False
    assert result["human_review"] == "not_performed"
    assert not any(
        result[key]
        for key in ("source_authority", "finding_disposition", "runtime", "real_study")
    )


@pytest.mark.parametrize("mutation", ["report_graph", "record_narrative"])
def test_rewritten_history_fails_even_with_a_consistent_digest_graph(
    repository: Path, mutation: str
) -> None:
    path = repository / ASSURANCE_PATH
    record = read(path)
    if mutation == "record_narrative":
        record["provisional_conclusion"] = "A rewritten historical conclusion."
    else:
        reference = record["panel_reports"][0]
        report_path = repository / CONTRACT_ROOT / reference["path"]
        report = read(report_path)
        report["rubric"][0]["rationale"] = "A rewritten historical rationale."
        digest = canonical_json_sha256(
            report, excluded_json_pointers={"/report_sha256"}
        )
        report["report_sha256"] = digest
        write(report_path, report)
        reference["sha256"] = digest
        synthesis_path = repository / CONTRACT_ROOT / record["synthesis"]["path"]
        synthesis = read(synthesis_path)
        synthesis["role_reports"][0]["report_sha256"] = digest
        for finding in synthesis["findings"]:
            if finding["source_role"] == reference["role"]:
                finding["source_report_sha256"] = digest
        synthesis["synthesis_sha256"] = canonical_json_sha256(
            synthesis, excluded_json_pointers={"/synthesis_sha256"}
        )
        write(synthesis_path, synthesis)
        record["synthesis"]["sha256"] = hashlib.sha256(
            synthesis_path.read_bytes()
        ).hexdigest()
        register_path = repository / CONTRACT_ROOT / record["findings"]["register_path"]
        register = read(register_path)
        register["bindings"]["synthesis_sha256"] = synthesis["synthesis_sha256"]
        write(register_path, register)
        record["findings"]["register_sha256"] = hashlib.sha256(
            register_path.read_bytes()
        ).hexdigest()
    write(path, record)
    expected = (
        "remediation-register schema"
        if mutation == "report_graph"
        else "frozen assurance digest"
    )
    with pytest.raises(SamplingHarmAgentAssuranceError, match=expected):
        validate(repository)


def test_assurance_anchor_ignores_json_serialization(repository: Path) -> None:
    path = repository / ASSURANCE_PATH
    path.write_text(json.dumps(read(path), sort_keys=True, indent=4))
    assert validate(repository)["historical_packet_only"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        "digest",
        "duplicate_role",
        "missing_role",
        "candidate",
        "tree",
        "packet",
        "path_escape",
        "absolute_path",
        "source_path",
        "synthesis_path",
        "register_path",
        "human_review",
        "authority",
        "report_authority",
        "extended_deadline",
        "missing_observation",
        "supersession",
        "panel_roles",
        "missing_manifest_field",
        "rights_claim",
        "missing_prohibition",
    ],
)
def test_manifest_mutations_fail_closed(repository: Path, mutation: str) -> None:
    path = repository / ASSURANCE_PATH
    record = read(path)
    if mutation == "digest":
        record["panel_reports"][0]["sha256"] = "0" * 64
    elif mutation == "duplicate_role":
        record["panel_reports"][1] = record["panel_reports"][0]
    elif mutation == "missing_role":
        record["panel_reports"].pop()
    elif mutation in {"candidate", "tree", "packet"}:
        key = {
            "candidate": "candidate_commit",
            "tree": "candidate_tree",
            "packet": "packet_sha256",
        }[mutation]
        record["candidate_binding"][key] = "0" * len(record["candidate_binding"][key])
    elif mutation in {"path_escape", "absolute_path"}:
        record["panel_reports"][0]["path"] = (
            "../escape.json"
            if mutation == "path_escape"
            else str(repository / "escape.json")
        )
    elif mutation in {"source_path", "synthesis_path", "register_path"}:
        key = {
            "source_path": "source_receipts",
            "synthesis_path": "synthesis",
            "register_path": "findings",
        }[mutation]
        record[key]["register_path" if key == "findings" else "path"] = "../escape.json"
    elif mutation == "human_review":
        record["protocol"]["human_review_status"] = "performed"
    elif mutation == "authority":
        record["authority"]["h8_d"] = True
    elif mutation == "report_authority":
        record["panel_reports"][0]["authorizing"] = True
    elif mutation == "extended_deadline":
        record["expiry"]["review_by"] = "2099-11-30"
    elif mutation == "missing_observation":
        del record["observed_at"]
    elif mutation == "supersession":
        record["expiry"]["supersede_on"][0] = "unrelated_event"
    elif mutation == "panel_roles":
        record["panel_roles"][0] = "human_scientist"
    elif mutation == "rights_claim":
        record["source_assessment"]["rights_status"] = "confirmed_for_subset"
    elif mutation == "missing_prohibition":
        record["prohibited_claims"][0] = "unrelated_prohibition"
    else:
        del record["synthesis"]
    write(path, record)
    with pytest.raises(SamplingHarmAgentAssuranceError):
        validate(repository)


@pytest.mark.parametrize("reference", ["report", "synthesis", "source", "register"])
def test_changed_referenced_bytes_fail(repository: Path, reference: str) -> None:
    record = read(repository / ASSURANCE_PATH)
    relative = {
        "report": record["panel_reports"][0]["path"],
        "synthesis": record["synthesis"]["path"],
        "source": record["source_receipts"]["path"],
        "register": record["findings"]["register_path"],
    }[reference]
    path = repository / CONTRACT_ROOT / relative
    value = read(path)
    value["unapproved_change"] = True
    write(path, value)
    with pytest.raises(SamplingHarmAgentAssuranceError):
        validate(repository)


def test_report_symlink_cannot_redirect_even_identical_bytes(
    repository: Path, tmp_path: Path
) -> None:
    record = read(repository / ASSURANCE_PATH)
    path = repository / CONTRACT_ROOT / record["panel_reports"][0]["path"]
    outside = tmp_path / "outside-report.json"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(SamplingHarmAgentAssuranceError, match="redirected"):
        validate(repository)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "disposed",
        "summary",
        "authority",
        "empty_authority",
        "selection",
        "candidate_binding",
        "synthesis_binding",
    ],
)
def test_rebinding_register_hash_does_not_hide_invalid_findings(
    repository: Path, mutation: str
) -> None:
    path = repository / ASSURANCE_PATH
    record = read(path)
    register_path = repository / CONTRACT_ROOT / record["findings"]["register_path"]
    register = read(register_path)
    if mutation == "missing":
        register["findings"].pop()
    elif mutation == "duplicate":
        register["findings"][1] = register["findings"][0]
    elif mutation == "disposed":
        register["findings"][0]["disposition_status"] = "resolved"
    elif mutation == "summary":
        register["summary"]["pending"] = 0
    elif mutation == "empty_authority":
        register["authority"] = {}
    elif mutation == "selection":
        register["disposition_paths"]["selection_authorized"] = True
    elif mutation == "candidate_binding":
        register["bindings"]["candidate_commit"] = "0" * 40
    elif mutation == "synthesis_binding":
        register["bindings"]["synthesis_sha256"] = "0" * 64
    else:
        register["authority"]["runtime"] = True
    write(register_path, register)
    record["findings"]["register_sha256"] = hashlib.sha256(
        register_path.read_bytes()
    ).hexdigest()
    write(path, record)
    expected = (
        "remediation-register schema" if mutation in {"summary", "authority"} else None
    )
    with pytest.raises(SamplingHarmAgentAssuranceError, match=expected):
        validate(repository)


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 9, 1),
        datetime(2026, 8, 1, tzinfo=UTC),
        datetime(2026, 12, 1, tzinfo=UTC),
    ],
)
def test_invalid_time_cannot_claim_freshness(now: datetime) -> None:
    with pytest.raises(SamplingHarmAgentAssuranceError):
        load_and_validate_sampling_harm_agent_assurance(repository_root=ROOT, now=now)


def test_known_supersession_cannot_reuse_historical_receipt() -> None:
    with pytest.raises(SamplingHarmAgentAssuranceError, match="superseded"):
        load_and_validate_sampling_harm_agent_assurance(
            repository_root=ROOT, now=NOW, superseded_by=("source_drift",)
        )


@pytest.mark.parametrize("content", ["not json", "[]"])
def test_invalid_manifest_document(repository: Path, content: str) -> None:
    (repository / ASSURANCE_PATH).write_text(content)
    with pytest.raises(SamplingHarmAgentAssuranceError):
        validate(repository)


@pytest.mark.parametrize(
    "mutation",
    [
        "report_count",
        "candidate",
        "dissent",
        "human_report",
        "null_reports",
        "renamed_report",
        "null_reviewer",
        "synthesis_only_rename",
        "synthesis_register_drift",
    ],
)
def test_rebound_synthesis_cannot_hide_semantic_changes(
    repository: Path, mutation: str
) -> None:
    path = repository / ASSURANCE_PATH
    record = read(path)
    synthesis_path = repository / CONTRACT_ROOT / record["synthesis"]["path"]
    synthesis = read(synthesis_path)
    if mutation == "report_count":
        synthesis["role_reports"].pop()
    elif mutation == "null_reports":
        synthesis["role_reports"] = None
    elif mutation == "renamed_report":
        old_path = repository / CONTRACT_ROOT / record["panel_reports"][0]["path"]
        relative = "reviews/unbound-alternative-role-report.json"
        old_path.rename(repository / CONTRACT_ROOT / relative)
        record["panel_reports"][0]["path"] = relative
        synthesis["role_reports"][0]["path"] = (CONTRACT_ROOT / relative).as_posix()
    elif mutation == "synthesis_only_rename":
        synthesis["role_reports"][0]["path"] = (
            CONTRACT_ROOT / "reviews/unbound-alternative-role-report.json"
        ).as_posix()
    elif mutation == "synthesis_register_drift":
        synthesis["actor"]["limitations"].append("Additional unreviewed limitation.")
    elif mutation == "candidate":
        synthesis["bindings"]["candidate_commit"]["value"] = "0" * 40
    elif mutation == "dissent":
        synthesis["dissent"][0]["statement"] = " "
    else:
        report_path = repository / CONTRACT_ROOT / record["panel_reports"][0]["path"]
        report = read(report_path)
        if mutation == "null_reviewer":
            report["reviewer"] = None
        else:
            report["reviewer"]["actor_type"] = "human"
        digest = canonical_json_sha256(
            report, excluded_json_pointers={"/report_sha256"}
        )
        report["report_sha256"] = digest
        write(report_path, report)
        record["panel_reports"][0]["sha256"] = digest
        synthesis["role_reports"][0]["report_sha256"] = digest
        for finding in synthesis["findings"]:
            if finding["source_role"] == record["panel_reports"][0]["role"]:
                finding["source_report_sha256"] = digest
    synthesis["synthesis_sha256"] = canonical_json_sha256(
        synthesis, excluded_json_pointers={"/synthesis_sha256"}
    )
    write(synthesis_path, synthesis)
    record["synthesis"]["sha256"] = hashlib.sha256(
        synthesis_path.read_bytes()
    ).hexdigest()
    write(path, record)
    expected = (
        "automated-challenge-synthesis schema" if mutation == "report_count" else None
    )
    with pytest.raises(SamplingHarmAgentAssuranceError, match=expected):
        validate(repository)


def test_rebound_source_cannot_grant_rights_authority(repository: Path) -> None:
    path = repository / ASSURANCE_PATH
    record = read(path)
    source_path = repository / CONTRACT_ROOT / record["source_receipts"]["path"]
    source = read(source_path)
    source["sources"][0]["source_authority"] = True
    write(source_path, source)
    record["source_receipts"]["sha256"] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    write(path, record)
    with pytest.raises(SamplingHarmAgentAssuranceError, match="source readiness"):
        validate(repository)


@pytest.mark.parametrize("valid", [False, True])
def test_cli_emits_only_verified_receipts(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    valid: bool,
) -> None:
    import sys

    from scripts import validate_sampling_harm_agent_assurance as cli

    monkeypatch.setattr(sys, "argv", ["validate", "--repository-root", str(repository)])
    # A fixed clock makes this historical test independent of the runner's date.
    monkeypatch.setattr(
        cli,
        "load_and_validate_sampling_harm_agent_assurance",
        lambda *, repository_root: validate(repository_root),
    )
    if valid:
        assert cli.main() == 0
        assert json.loads(capsys.readouterr().out)["human_review"] == "not_performed"
    else:
        (repository / ASSURANCE_PATH).unlink()
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2
        assert capsys.readouterr().out == ""
