"""Contracts for the SciCrunch/RRID registration handoff."""

import json
from pathlib import Path

from scripts.validate_scicrunch_rrid import validate_registration

PACKET = Path("docs/release/scicrunch-rrid-registration.json")


def test_scicrunch_registration_packet_is_ready_for_account_submission() -> None:
    summary = validate_registration(PACKET)

    assert summary == {
        "resource_name": "voiage",
        "release": "v2.0.0",
        "state": "ready_for_account_submission",
        "required_answer_count": 3,
        "evidence_count": 12,
    }


def test_scicrunch_packet_preserves_external_state_boundaries() -> None:
    packet = json.loads(PACKET.read_text())

    assert packet["submission"]["performed"] is False
    assert packet["submission"]["submitted_at"] is None
    assert packet["curation"]["status"] == "not_started"
    assert packet["curation"]["rrid"] is None
    assert packet["curation"]["assigned_at"] is None
    assert packet["declarations"]["resource_owner"]["answer"] is True
    assert packet["declarations"]["information_accurate"]["answer"] is None
    assert packet["declarations"]["account_terms_accepted"]["answer"] is None
    assert packet["answers"]["url"] == "https://github.com/edithatogo/voiage"
    assert packet["evidence"]["release"]["tag"] == "v2.0.0"
    assert packet["evidence"]["release"]["commit"] == (
        "e849e89152c306e79c96d0a8a9815ee5faca0529"
    )
    assert packet["evidence"]["software_heritage"]["swhid"] == (
        "swh:1:snp:31f89375852737bb9eb62ebc03fadfbc7ff70c2d"
    )
