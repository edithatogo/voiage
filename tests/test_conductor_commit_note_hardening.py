from __future__ import annotations

from pathlib import Path
import re


def _hardening_track_dir() -> Path:
    """Return the canonical active or archived hardening track directory."""
    track_id = "conductor-commit-note-checkpoint-hardening_20260625"
    active = Path("conductor/tracks") / track_id
    return active if active.is_dir() else Path("conductor/archive") / track_id


def test_conductor_commit_notes_required():
    """Conductor tracks should require commits and commit notes."""

    # This test should fail until the Conductor Commit Note And Checkpoint Hardening track
    # is implemented and generates tracks with proper commit notes

    # Read the workflow file to verify it requires commit notes
    workflow_path = Path("conductor/workflow.md")
    workflow_content = workflow_path.read_text(encoding="utf-8")

    # Check that workflow requires commits
    assert "git notes" in workflow_content, "Workflow should require git notes"
    assert "short commit SHA" in workflow_content, (
        "Workflow should require short commit SHA"
    )
    assert "commit note" in workflow_content.lower(), (
        "Workflow should require commit notes"
    )

    # Check that plan updates are required
    assert "plan-update commits" in workflow_content, (
        "Workflow should require plan-update commits"
    )

    # Check that GitHub Actions monitoring is required
    assert "GitHub Actions" in workflow_content, (
        "Workflow should require GitHub Actions monitoring"
    )


def test_phase_checkpoints_required():
    """Conductor tracks should require phase checkpoints."""

    workflow_path = Path("conductor/workflow.md")
    workflow_content = workflow_path.read_text(encoding="utf-8")

    # Check that workflow requires phase checkpoints
    assert "phase checkpoint" in workflow_content.lower(), (
        "Workflow should require phase checkpoints"
    )
    assert "protocol" in workflow_content.lower(), (
        "Workflow should require verification protocols"
    )

    # Check that verification steps are defined
    assert "Phase Completion" in workflow_content, (
        "Workflow should have phase completion protocol"
    )


def test_generated_tracks_require_commit_notes():
    """Generated tracks should follow the hardened template with commit notes."""

    # Read all conductor tracks
    tracks_dir = Path("conductor/tracks")
    tracks_registry = Path("conductor/tracks.md")
    tracks_md_content = tracks_registry.read_text(encoding="utf-8")

    # Parse track IDs from tracks.md
    # Simple parsing to find tracks
    track_patterns = re.findall(
        r"# \[.*?] Track: (.*?)\n\s*\*Link: \[\.\./tracks/([^\]]+)\]", tracks_md_content
    )

    # For each track, verify it has the proper Conductor template requirements
    for _track_name, track_id in track_patterns:
        if track_id == "conductor-commit-note-checkpoint-hardening_20260625":
            continue  # Skip the current hardening track itself

        track_dir = tracks_dir / track_id
        if track_dir.exists():
            plan_md = track_dir / "plan.md"
            if plan_md.exists():
                plan_content = plan_md.read_text(encoding="utf-8")

                # Check that the plan requires commit notes
                assert "commit notes" in plan_content.lower(), (
                    f"Track {track_id} should require commit notes"
                )
                assert "git note" in plan_content.lower(), (
                    f"Track {track_id} should require git notes"
                )
                assert "phase checkpoint" in plan_content.lower(), (
                    f"Track {track_id} should require phase checkpoints"
                )


def test_plan_contains_git_notes():
    """Track plans should contain git notes requirements."""

    # Test that the Conductor Commit Note And Checkpoint Hardening track itself
    # requires git notes in its plan
    track_dir = _hardening_track_dir()
    plan_md = track_dir / "plan.md"

    plan_content = plan_md.read_text(encoding="utf-8")

    # Check for git notes related content
    required_phrases = [
        "commit notes",
        "git notes",
        "short commit SHA",
        "phase checkpoint",
        "git note summary",
        "plan SHA updates",
    ]

    for phrase in required_phrases:
        assert phrase in plan_content, f"Plan should require {phrase}"


def test_verification_commands_updated():
    """Verification commands should include new CI/CD requirements."""

    track_dir = _hardening_track_dir()
    plan_md = track_dir / "plan.md"

    plan_content = plan_md.read_text(encoding="utf-8")

    # Check that verification commands include the track's validation test.
    assert "test_conductor_commit_note_hardening.py" in plan_content, (
        "Plan should include its Conductor validation test"
    )


def test_external_gates_documentation():
    """Tracks should document how to record blocked external evidence."""

    track_dir = _hardening_track_dir()
    spec_md = track_dir / "spec.md"

    spec_content = spec_md.read_text(encoding="utf-8")

    # Check that spec documents external gate handling
    assert "external" in spec_content, "Spec should mention external gates"
    assert "blocked" in spec_content.lower(), "Spec should document blocked states"
    assert "evidence" in spec_content.lower(), (
        "Spec should document evidence requirements"
    )
