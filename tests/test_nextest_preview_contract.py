"""Keep the optional nextest replay limited to its producing job."""

from pathlib import Path
import tomllib

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_preview_uses_group_and_same_run_archive() -> None:
    config = tomllib.loads((ROOT / "rust/.config/nextest.toml").read_text())
    assert config["test-groups"]["checkpoint-io"]["max-threads"] == 1
    assert {
        "filter": "binary(evpi_resume)",
        "test-group": "checkpoint-io",
    } in config["profile"]["default"]["overrides"]
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/dependency-preview.yml").read_text()
    )
    job = workflow["jobs"]["rust-acceleration-observation"]
    assert job["continue-on-error"] is True
    steps = {step["name"]: step for step in job["steps"] if "name" in step}
    archive = steps["Archive checkpoint tests from this checkout"]
    replay = steps["Replay the same-run checkpoint archive"]
    assert archive["working-directory"] == replay["working-directory"] == "rust"
    assert (
        '--archive-file "$RUNNER_TEMP/voiage-checkpoint-tests.tar.zst"'
        in archive["run"]
    )
    assert (
        '--archive-file "$RUNNER_TEMP/voiage-checkpoint-tests.tar.zst"' in replay["run"]
    )
    assert "--retries 0" in replay["run"]
