"""Exercise the Git credential setup used by automated proposal workflows."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest
import yaml

ROOT = Path(__file__).parents[1]


def test_tagbot_credential_arguments_are_not_sent_to_process_monitor() -> None:
    # The pinned action constructs a token-bearing Git clone URL:
    # https://github.com/JuliaRegistries/TagBot/blob/6b7c22e7bc2b8f4d1c56b7199a63421cf2667ed1/tagbot/action/git.py#L90-L92
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "julia-tagbot.yml").read_text(
            encoding="utf-8"
        )
    )
    steps = workflow["jobs"]["tagbot"]["steps"]
    assert any(
        str(step.get("uses", "")).startswith("JuliaRegistries/TagBot@")
        for step in steps
    )
    assert all(
        "step-security/harden-runner@" not in str(step.get("uses", ""))
        for step in steps
    )


@pytest.mark.skipif(
    os.name == "nt", reason="The fixture credential helper is a POSIX shell script"
)
@pytest.mark.parametrize(
    ("workflow_name", "job_name", "step_name", "remote"),
    [
        (
            "canonical-specialized-voi-sync.yml",
            "propose-mirror-update",
            "Propose only a bounded projection-mirror update",
            "https://github.com/edithatogo/voiage.git",
        ),
        (
            "conda-update.yml",
            "update-feedstock",
            "Create conda-forge submission PR",
            "https://github.com/edithatogo/staged-recipes.git",
        ),
    ],
)
def test_proposal_workflow_uses_credential_protocol_without_token_arguments(
    tmp_path: Path, workflow_name: str, job_name: str, step_name: str, remote: str
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / workflow_name).read_text(encoding="utf-8")
    )
    step = next(
        item
        for item in workflow["jobs"][job_name]["steps"]
        if item.get("name") == step_name
    )
    commands = [
        shlex.split(line.strip())
        for line in step["run"].splitlines()
        if line.strip().startswith(("git config --local", "git remote set-url origin"))
    ]
    assert len(commands) in (2, 3)

    repo = tmp_path / "repo"
    repo.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    helper = bin_dir / "gh"
    helper.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$CALL_TRACE"\n'
        'while IFS= read -r line && [ -n "$line" ]; do :; done\n'
        'printf "username=x-access-token\\npassword=%s\\n\\n" "$GH_TOKEN"\n',
        encoding="utf-8",
    )
    helper.chmod(0o755)
    marker = "fixture-token-not-a-credential"
    git_path = shutil.which("git")
    assert git_path is not None
    env = os.environ.copy()
    env.update(
        HOME=str(tmp_path),
        PATH=f"{bin_dir}{os.pathsep}{env['PATH']}",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=str(tmp_path / "global-gitconfig"),
        GIT_TERMINAL_PROMPT="0",
        GH_TOKEN=marker,
        CALL_TRACE=str(tmp_path / "helper-arguments"),
    )
    env.pop("GIT_CONFIG_COUNT", None)
    env.pop("GIT_CONFIG_PARAMETERS", None)

    def git(
        *args: str, input_text: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [git_path, *args],
            cwd=repo,
            env=env,
            input=input_text,
            text=True,
            capture_output=True,
            check=True,
        )

    git("init", "--quiet")
    git("remote", "add", "origin", remote)
    git("config", "--global", "credential.helper", "!exit 99")
    for command in commands:
        git(*command[1:])

    result = git("credential", "fill", input_text="protocol=https\nhost=github.com\n\n")
    assert f"password={marker}" in result.stdout
    assert (tmp_path / "helper-arguments").read_text(encoding="utf-8").splitlines() == [
        "auth",
        "git-credential",
        "get",
    ]
    assert marker not in (repo / ".git" / "config").read_text(encoding="utf-8")
    assert git("remote", "get-url", "origin").stdout.strip() == remote
