"""Required automation must test real installed external producers."""

import configparser
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_producer_lane_uses_the_base_wheel_with_explicit_producer_tooling() -> None:
    config = configparser.ConfigParser(interpolation=None)
    config.read(ROOT / "tox.ini")
    assert "producer-compatibility" in config["tox"]["env_list"].split()
    lane = config["testenv:producer-compatibility"]
    assert lane["extras"].strip() == ""
    assert lane["dependency_groups"].strip() == "producer-tests"
    assert "python -I -m pytest" in lane["commands"]
    assert "--import-mode=importlib" in lane["commands"]
    assert "--confcutdir=" in lane["commands"]
    assert "VOIAGE_PRODUCER_PREFIX = {env_dir}" in lane["set_env"]


def test_protected_and_platform_jobs_execute_producer_acceptance() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    for name in ("test-unit", "pandas-free-runtime"):
        job = jobs[name]
        assert not job.get("continue-on-error", False)
        commands = "\n".join(step.get("run", "") for step in job["steps"])
        assert "tox run -e pandas-free,producer-compatibility" in commands
