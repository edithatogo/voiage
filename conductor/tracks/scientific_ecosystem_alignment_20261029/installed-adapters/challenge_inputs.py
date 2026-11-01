# SPDX-License-Identifier: Apache-2.0
"""Challenge an altered Innovate manifest and a self-consistent substituted VOP handoff."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import voiage
import runpy
import shutil
import sys
import tempfile

packet, fixture_source, export_source, handoff_script, wheel = map(Path, sys.argv[1:6])
sha = sys.argv[6]


def check(name: str, script: Path, args: list[str | Path]) -> list[str]:
    sys.argv = [str(script), *map(str, args)]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(script), run_name="__main__")
    except ValueError as e:
        return [name, "rejected", str(e)]
    return [name, "accepted"]


with tempfile.TemporaryDirectory() as t:
    t = Path(t)
    fixture = Path(shutil.copytree(fixture_source, t / "fixture"))
    m = fixture / "manifest.json"
    m.write_bytes(m.read_bytes() + b" ")
    result = [
        check(
            "changed_manifest",
            packet / "probe.py",
            ["innovate", fixture, "--voiage-sha", sha],
        )
    ]
    bundle = Path(shutil.copytree(export_source, t / "bundle"))
    csv = bundle / "hpv_vaccination_net_benefit.csv"
    rows = csv.read_bytes().splitlines(keepends=True)
    csv.write_bytes(rows[0] + b"0.0,0.0\r\n" * 500)
    rec = bundle / "export.json"
    val = json.loads(rec.read_text())
    val["net_benefit_csv_sha256"] = hashlib.sha256(csv.read_bytes()).hexdigest()
    rec.write_text(json.dumps(val))
    result.append(
        check(
            "substitute_csv_rehashed_receipt",
            packet / "candidate_vop.py",
            [bundle, handoff_script, wheel, sha],
        )
    )
    with patch.object(voiage, "__file__", str(t / "uninstalled" / "__init__.py")):
        result.append(
            check(
                "outside_import",
                packet / "probe.py",
                ["innovate", fixture_source, "--voiage-sha", sha],
            )
        )
    expected = json.loads((packet / "negative-receipt.json").read_text())
    if result != expected:
        raise AssertionError(result)
    print(json.dumps(result, indent=2))
