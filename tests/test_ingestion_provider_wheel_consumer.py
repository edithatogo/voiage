"""Black-box provider-SDK consumer evidence using a real installed wheel."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONSUMER_PACKAGE = ROOT / "tests/fixtures/provider_sdk_wheel_consumer"


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    """Run one build/install boundary with its diagnostic retained by pytest."""
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"command failed: {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.integration
def test_external_provider_wheel_discovers_only_after_explicit_opt_in(
    tmp_path: Path,
) -> None:
    """A separately installed package uses only frozen public SDK surfaces."""
    uv = shutil.which("uv")
    assert uv is not None, "the wheel-consumer contract requires uv"
    # Keep the build cache inside pytest's owned temporary directory.  A
    # developer may legitimately point UV_CACHE_DIR at a machine-local cache,
    # but a test must never inherit a host-specific path that is not writable
    # on a clean hosted runner.
    cache_dir = tmp_path / "uv-cache"
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(cache_dir)
    environment["UV_LINK_MODE"] = "copy"

    voiage_dist = tmp_path / "voiage-dist"
    consumer_dist = tmp_path / "consumer-dist"
    _run(
        [uv, "build", "--wheel", "--out-dir", str(voiage_dist)],
        cwd=ROOT,
        env=environment,
    )
    _run(
        [uv, "build", "--wheel", "--out-dir", str(consumer_dist)],
        cwd=CONSUMER_PACKAGE,
        env=environment,
    )

    environment_dir = tmp_path / "consumer-venv"
    _run(
        [uv, "venv", "--python", sys.executable, str(environment_dir)],
        cwd=tmp_path,
        env=environment,
    )
    interpreter = environment_dir / "bin/python"
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(interpreter),
            str(next(voiage_dist.glob("*.whl"))),
            str(next(consumer_dist.glob("*.whl"))),
        ],
        cwd=tmp_path,
        env=environment,
    )

    (tmp_path / "data.csv").write_text("value\n1\n", encoding="utf-8")
    descriptor = tmp_path / "example.json"
    descriptor.write_text(
        json.dumps({"voiage_wheel_example": "1", "resource": "data.csv"}),
        encoding="utf-8",
    )
    probe = "\n".join(
        (
            "import json, sys",
            "from pathlib import Path",
            "import voiage.ingestion",
            "assert 'voiage_sdk_wheel_consumer' not in sys.modules",
            "from voiage.ingestion import (ProviderRegistry, SourceAccessPolicy, discover_entry_point_providers)",
            "providers = discover_entry_point_providers(allowlist={'example-wheel'})",
            "assert len(providers) == 1",
            "bundle = ProviderRegistry(providers).ingest(Path('example.json'), policy=SourceAccessPolicy(Path('.')))",
            "print(json.dumps({'provider_id': bundle.manifest.provenance.provider_id, 'rows': bundle.table('data').to_pylist()}))",
        )
    )
    result = subprocess.run(
        [str(interpreter), "-I", "-c", probe],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "provider_id": "example-wheel-provider",
        "rows": [{"value": 1}],
    }
