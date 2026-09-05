"""Build the pinned HPC source and probe an isolated installed CPU package.

This is source-wheel evidence on the current host, not a Spack/EasyBuild build.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import tempfile
from urllib.request import urlopen

SOURCE_URL = "https://files.pythonhosted.org/packages/ac/85/697bbefd2f9e78b28fba8a8b981217e8dafe7f0b28ec6ee233960cca9896/voiage-2.2.0.tar.gz"
SOURCE_SHA256 = "e4edfd41011891a94cbc2b144ff1d20340fcc32481e7a2b24157494b7490a16b"
RUNTIME_PINS = {
    "click": "8.5.0",
    "numpy": "2.2.6",
    "scipy": "1.16.3",
    "pandas": "2.3.3",
    "xarray": "2024.11.0",
    "scikit-learn": "1.7.2",
    "pyarrow": "25.0.0",
    "polars": "1.42.1",
    "pydantic": "2.13.4",
    "jsonschema": "4.26.0",
    "typing-extensions": "4.16.0",
    "typer": "0.27.2",
}
PROBE = """
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import voiage
from voiage import _core
from voiage.methods.basic import evpi
from voiage.schema import ValueArray
assert importlib.metadata.version('voiage') == '2.2.0'
assert Path(voiage.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
info = _core.runtime_info()
assert info['engine'] == 'rust' and info['core_version'] == '2.2.0'
assert info['source_dirty'] is False
value = evpi(ValueArray.from_numpy(np.array([[0., 2.], [2., 0.]]), ["A", "B"]))
assert np.isclose(value, 1.0), value
table = pa.table({'decision': ['A', 'B'], 'value': [0., 2.]})
sink = pa.BufferOutputStream()
with ipc.new_stream(sink, table.schema) as writer:
    writer.write_table(table)
assert ipc.open_stream(sink.getvalue()).read_all().equals(table)
print(json.dumps({'version': '2.2.0', 'engine': info['engine'],
                  'source_revision': info['source_revision'],
                  'source_tree_git_oid': info['source_tree_git_oid'],
                  'evpi': float(value), 'arrow_round_trip': True}))
"""


def extract_source(archive: Path, target: Path) -> Path:
    """Reject changed archives and non-regular members before extraction."""
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("HPC source archive SHA-256 mismatch")
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            path = Path(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.parts[0] != "voiage-2.2.0"
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("Unsafe HPC source archive member")
        stream.extractall(target, filter="data")
    return target / "voiage-2.2.0"


def run_smoke(output: Path, source_archive: Path | None = None) -> None:
    """Write success only after source build, isolated install and probes pass."""
    receipt: dict[str, object] = {
        "schema": "voiage.hpc-source-smoke.v1",
        "recorded_at": datetime.now(UTC).isoformat(),
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "host": platform.platform(),
        "evidence_scope": "local_source_wheel_with_recipe_runtime_pins_not_hpc_toolchain_build",
        "runtime_pins": RUNTIME_PINS,
        "spack_build_executed": False,
        "easybuild_build_executed": False,
        "status": "failed",
        "steps": [],
    }
    steps: list[dict[str, object]] = []
    receipt["steps"] = steps
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    # Do not allow inherited source identity to override the release sdist.
    for key in (
        "VOIAGE_SOURCE_REVISION",
        "VOIAGE_SOURCE_TREE_GIT_OID",
        "VOIAGE_SOURCE_CLEAN",
    ):
        env.pop(key, None)

    def execute(argv: list[str], cwd: Path) -> str:
        completed = subprocess.run(  # noqa: S603 - fixed tool arguments, no shell
            argv,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=1200,
            check=False,
        )
        steps.append({"argv": argv, "exit_code": completed.returncode})
        if completed.returncode:
            raise RuntimeError(completed.stderr[-8000:] or completed.stdout[-8000:])
        return completed.stdout

    try:
        with tempfile.TemporaryDirectory(prefix="voiage-hpc-smoke-") as directory:
            work = Path(directory)
            archive = work / "voiage-2.2.0.tar.gz"
            if source_archive is None:
                with urlopen(SOURCE_URL, timeout=120) as response:
                    archive.write_bytes(response.read())
            else:
                archive.write_bytes(source_archive.read_bytes())
            source = extract_source(archive, work / "source")
            execute(
                [
                    "uv",
                    "build",
                    "--wheel",
                    "--out-dir",
                    str(work / "wheels"),
                    str(source),
                ],
                work,
            )
            wheels = list((work / "wheels").glob("voiage-2.2.0-*.whl"))
            if len(wheels) != 1:
                raise RuntimeError("Expected exactly one built voiage wheel")  # noqa: TRY301
            receipt["wheel_sha256"] = hashlib.sha256(wheels[0].read_bytes()).hexdigest()
            execute(["uv", "venv", "--python", "3.12", str(work / "consumer")], work)
            python = work / "consumer/bin/python"
            constraints = work / "runtime-constraints.txt"
            constraints.write_text(
                "".join(
                    f"{name}=={version}\n" for name, version in RUNTIME_PINS.items()
                )
            )
            execute(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--constraint",
                    str(constraints),
                    str(wheels[0]),
                ],
                work,
            )
            execute(["uv", "pip", "check", "--python", str(python)], work)
            versions_probe = (
                "import importlib.metadata,json; print(json.dumps({name:importlib.metadata.version(name) for name in "
                + repr(list(RUNTIME_PINS))
                + "}))"
            )
            installed = json.loads(
                execute([str(python), "-I", "-c", versions_probe], work)
            )
            if installed != RUNTIME_PINS:
                raise RuntimeError("Installed runtime versions differ from recipe pins")  # noqa: TRY301
            receipt["installed_runtime_versions"] = installed
            receipt["probe"] = json.loads(
                execute([str(python), "-I", "-c", PROBE], work)
            )
            execute([str(work / "consumer/bin/voiage"), "--help"], work)
            receipt["status"] = "passed"
    except Exception as error:
        receipt["error"] = str(error)
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Run the explicitly requested source-build smoke check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    args = parser.parse_args()
    run_smoke(args.output, args.source_archive)


if __name__ == "__main__":
    main()
