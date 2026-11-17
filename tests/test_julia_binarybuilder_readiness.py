"""Contracts for Julia artifact packaging and General registration."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tomllib

import pytest
import yaml

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def tagbot_act() -> str:
    """Select act only when its offline dry-run controls are available."""
    act = shutil.which("act")
    if act is None:
        pytest.skip(
            "act is unavailable for offline GitHub Actions expression simulation"
        )
    try:
        result = subprocess.run(
            [act, "--help"], capture_output=True, text=True, timeout=10, check=False
        )
    except OSError, subprocess.TimeoutExpired:
        pytest.skip("act cannot provide its offline dry-run capabilities")
    required = (
        "--dryrun",
        "--workflows",
        "--eventpath",
        "--actor",
        "--container-daemon-socket",
        "--no-cache-server",
        "--action-offline-mode",
        "--pull",
        "--secret-file",
        "--env-file",
        "--var-file",
        "--platform",
        "--action-cache-path",
    )
    if result.returncode != 0 or any(flag not in result.stdout for flag in required):
        pytest.skip("installed act lacks the required offline dry-run controls")
    return act


def test_binarybuilder_recipe_pins_release_and_supported_products() -> None:
    recipe = (ROOT / "packaging/yggdrasil/V/voiage_ffi/build_tarballs.jl").read_text()

    assert 'name = "voiage_ffi"' in recipe
    assert 'version = v"2.1.0"' in recipe
    assert "964a0fc334ece9509387cd07d43776adf38be240" in recipe
    assert 'LibraryProduct("libvoiage_ffi", :libvoiage_ffi)' in recipe
    assert "compilers = [:c, :rust]" in recipe
    assert "platforms = supported_platforms()" in recipe
    assert (
        'filter!(p -> !(Sys.isfreebsd(p) && arch(p) == "aarch64"), platforms)' in recipe
    )
    assert 'filter!(p -> arch(p) != "riscv64", platforms)' in recipe
    assert 'RUSTFLAGS="-C target-feature=-crt-static"' in recipe
    assert "install_license ../LICENSE" in recipe


def test_julia_package_has_general_quality_metadata() -> None:
    project_path = ROOT / "bindings/julia/Project.toml"
    project = tomllib.loads(project_path.read_text())

    assert project["name"] == "Voiage"
    assert project["authors"] == ["Dylan Mordaunt <voiage@users.noreply.github.com>"]
    assert "Statistics" not in project.get("deps", {})
    assert "JSON" not in project.get("deps", {})
    assert {"Aqua", "JSON", "Test"} <= set(project["extras"])
    assert {"Aqua", "JSON", "Test"} <= set(project["targets"]["test"])
    assert project["compat"]["Aqua"] == "0.8"
    assert project["compat"]["Libdl"] == "1.10"
    assert project["compat"]["Test"] == "1.10"
    assert (ROOT / "bindings/julia/LICENSE").read_bytes() == (
        ROOT / "LICENSE"
    ).read_bytes()
    assert not (ROOT / "bindings/julia/Manifest.toml").exists()


def test_tagbot_is_configured_for_the_julia_subpackage() -> None:
    workflow_path = ROOT / ".github/workflows/julia-tagbot.yml"
    workflow = yaml.safe_load(workflow_path.read_text())
    rendered = workflow_path.read_text()

    assert workflow[True] == {
        "issue_comment": {"types": ["created"]},
        "workflow_dispatch": None,
    }
    assert "issue_comment: # zizmor: ignore[dangerous-triggers]" in rendered
    assert workflow["permissions"] == {"contents": "read"}
    assert list(workflow["jobs"]) == ["tagbot"]
    job = workflow["jobs"]["tagbot"]
    assert " ".join(job["if"].split()) == (
        "github.actor == 'JuliaTagBot' && "
        "( github.event_name == 'workflow_dispatch' || "
        "( github.event_name == 'issue_comment' && "
        "github.event.action == 'created' && "
        "github.event.sender.id == 50554310 && "
        "github.event.comment.user.id == 50554310 && "
        "github.event.issue.user.id == 50554310 && "
        "github.event.issue.state == 'closed' && "
        "!github.event.issue.pull_request ) )"
    )
    assert job["permissions"] == {"contents": "write"}
    assert len(job["steps"]) == 1
    step = job["steps"][0]
    assert step["uses"] == (
        "JuliaRegistries/TagBot@6b7c22e7bc2b8f4d1c56b7199a63421cf2667ed1"
    )
    assert step["with"] == {
        "token": "${{ secrets.GITHUB_TOKEN }}",
        "ssh": "${{ secrets.DOCUMENTER_KEY }}",
        "subdir": "bindings/julia",
        "tag_prefix": "julia",
    }


@pytest.mark.parametrize(
    ("event_name", "actor", "overrides", "allowed"),
    [
        ("issue_comment", "JuliaTagBot", {}, True),
        ("issue_comment", "attacker", {}, False),
        ("issue_comment", "JuliaTagBot", {"sender": 42}, False),
        ("issue_comment", "JuliaTagBot", {"commenter": 42}, False),
        ("issue_comment", "JuliaTagBot", {"owner": 42}, False),
        ("issue_comment", "JuliaTagBot", {"state": "open"}, False),
        ("issue_comment", "JuliaTagBot", {"is_pr": True}, False),
        ("issue_comment", "JuliaTagBot", {"sender": None}, False),
        ("issue_comment", "JuliaTagBot", {"action": "edited"}, False),
        ("workflow_dispatch", "JuliaTagBot", {}, True),
        ("workflow_dispatch", "maintainer", {}, False),
    ],
)
def test_tagbot_guard_with_synthetic_events(
    tmp_path: Path,
    tagbot_act: str,
    event_name: str,
    actor: str,
    overrides: dict[str, object],
    allowed: bool,
) -> None:
    """Evaluate the real job condition with no secrets or TagBot execution."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/julia-tagbot.yml").read_text())
    fixture = {
        "name": "Synthetic TagBot guard",
        "on": {"issue_comment": {"types": ["created"]}, "workflow_dispatch": {}},
        "permissions": {},
        "jobs": {
            "guard": {
                "if": workflow["jobs"]["tagbot"]["if"],
                "runs-on": "ubuntu-latest",
                "steps": [{"run": "printf synthetic"}],
            }
        },
    }
    issue: dict[str, object] = {
        "user": {"id": overrides.get("owner", 50554310)},
        "state": overrides.get("state", "closed"),
    }
    if overrides.get("is_pr"):
        issue["pull_request"] = {"url": "https://example.invalid/pull"}
    event: dict[str, object] = {"repository": {"full_name": "edithatogo/voiage"}}
    if event_name == "issue_comment":
        event.update(
            {
                "action": overrides.get("action", "created"),
                "sender": {"id": overrides.get("sender", 50554310)},
                "comment": {
                    "user": {"id": overrides.get("commenter", 50554310)},
                    "body": "synthetic",
                },
                "issue": issue,
            }
        )
    else:
        event["inputs"] = {}
    fixture_path = tmp_path / "workflow.yml"
    event_path = tmp_path / "event.json"
    fixture_path.write_text(yaml.safe_dump(fixture, sort_keys=False))
    event_path.write_text(json.dumps(event))
    result = subprocess.run(
        [
            tagbot_act,
            event_name,
            "--dryrun",
            "--workflows",
            str(fixture_path),
            "--eventpath",
            str(event_path),
            "--actor",
            actor,
            "--container-daemon-socket",
            "-",
            "--no-cache-server",
            "--action-offline-mode",
            "--pull=false",
            "--secret-file",
            os.devnull,
            "--env-file",
            os.devnull,
            "--var-file",
            os.devnull,
            "--platform",
            "ubuntu-latest=-self-hosted",
            "--action-cache-path",
            str(tmp_path / "action-cache"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert ("*DRYRUN*" in result.stdout + result.stderr) is allowed


def test_julia_ci_covers_supported_runtimes_and_platforms() -> None:
    workflow_path = ROOT / ".github/workflows/bindings-ci.yml"
    rendered = workflow_path.read_text()

    for runtime in ("1.10", "1.11", "1.12"):
        assert f'- "{runtime}"' in rendered
    for runner in ("ubuntu-latest", "macos-latest", "windows-latest"):
        assert f"runner: {runner}" in rendered
    assert "matrix.os.library" in rendered


def test_registration_documentation_preserves_the_two_external_gates() -> None:
    readme = (ROOT / "bindings/julia/README.md").read_text()

    assert "JuliaPackaging/Yggdrasil/pull/14292" in readme
    assert "@JuliaRegistrator register subdir=bindings/julia" in readme
    assert "JLL registration" in readme
    assert "General registry merge" in readme


def test_current_release_candidate_preserves_historical_build_evidence() -> None:
    receipt = json.loads(
        (
            ROOT / "specs/submission-readiness/yggdrasil-v2-2-candidate-20260831.json"
        ).read_text()
    )
    project = tomllib.loads((ROOT / "bindings/julia/Project.toml").read_text())
    workspace = tomllib.loads((ROOT / "rust/Cargo.toml").read_text())
    assert receipt["version"] == "2.2.0"
    assert project["version"] == workspace["workspace"]["package"]["version"]
    assert receipt["source_commit"] == "7af563c8cb373057d30662650b3f332f39e05b83"
    historical = receipt["historical_recipe"]
    candidate = (ROOT / receipt["recipe"]["path"]).read_bytes()
    original = (ROOT / historical["path"]).read_bytes()
    assert hashlib.sha256(candidate).hexdigest() == receipt["recipe"]["sha256"]
    assert hashlib.sha256(original).hexdigest() == historical["sha256"]
    assert candidate.decode() == original.decode().replace(
        'version = v"2.1.0"', 'version = v"2.2.0"'
    ).replace(historical["source_commit"], receipt["source_commit"])
    assert receipt["state"] == "prepared_not_built_or_submitted"
    assert all(
        value is None or value == []
        for value in receipt["new_candidate_evidence"].values()
    )
    assert historical["upstream_head"] == "2528e2efb90e4197924d45c98873ca5cdb1a9d42"
    assert historical["included_platforms_passed"] == 15
