"""CLI coverage for dataset descriptor inspection and normalization."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from voiage.cli import app


def test_ingest_inspect_and_normalize(tmp_path) -> None:
    (tmp_path / "samples.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    descriptor = tmp_path / "datapackage.json"
    descriptor.write_text(
        json.dumps(
            {
                "name": "cli-fixture",
                "resources": [{"name": "samples", "path": "samples.csv", "schema": {"fields": [{"name": "a"}, {"name": "b"}]}}],
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    inspected = runner.invoke(app, ["ingest", "inspect", str(descriptor)])
    output = tmp_path / "normalized.arrow"
    normalized = runner.invoke(app, ["ingest", "normalize", str(descriptor), "--output", str(output)])
    calculated = runner.invoke(
        app,
        [
            "ingest",
            "calculate-from-dataset",
            str(descriptor),
            "--table",
            "samples",
            "--field",
            "a",
            "--field",
            "b",
        ],
    )

    assert inspected.exit_code == 0
    assert json.loads(inspected.output)["provider"] == "frictionless"
    assert normalized.exit_code == 0
    assert output.is_file()
    assert calculated.exit_code == 0
    assert "input_digest" in json.loads(calculated.output)
