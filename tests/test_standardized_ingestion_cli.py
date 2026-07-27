"""CLI coverage for dataset descriptor inspection and normalization."""

from __future__ import annotations

from hashlib import sha256
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
                "licenses": [{"name": "CC-BY-4.0"}],
                "contributors": [{"title": "Fixture maintainer", "role": "author"}],
                "resources": [
                    {
                        "name": "samples",
                        "path": "samples.csv",
                        "schema": {"fields": [{"name": "a"}, {"name": "b"}]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    validated = runner.invoke(app, ["ingest", "validate", str(descriptor)])
    default_inspected = runner.invoke(app, ["ingest", "inspect", str(descriptor)])
    inspected = runner.invoke(
        app,
        [
            "ingest",
            "inspect",
            str(descriptor),
            "--table",
            "samples",
            "--field",
            "a",
            "--field",
            "b",
        ],
    )
    output = tmp_path / "normalized.arrow"
    normalized = runner.invoke(
        app, ["ingest", "normalize", str(descriptor), "--output", str(output)]
    )
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

    assert validated.exit_code == 0
    assert json.loads(validated.output)["valid"] is True
    assert default_inspected.exit_code == 0
    assert json.loads(default_inspected.output)["binding_resolution"] is None
    assert inspected.exit_code == 0
    inspection = json.loads(inspected.output)
    assert inspection["provider"] == "frictionless"
    assert inspection["capabilities"] == {
        "format_versions": ["1"],
        "media_types": ["text/csv"],
        "provider_id": "frictionless",
        "supported_transforms": [],
        "supports_filtering": False,
        "supports_projection": False,
        "supports_random_access": False,
        "supports_streaming": False,
    }
    assert inspection["provenance"]["license"] == "CC-BY-4.0"
    assert inspection["governance"] == {
        "frictionlessdata.org:contributors": [
            {"role": "author", "title": "Fixture maintainer"}
        ],
        "frictionlessdata.org:licenses": [{"name": "CC-BY-4.0"}],
    }
    resolution = inspection["binding_resolution"]
    assert resolution["binding"]["role"] == "net_benefit"
    assert resolution["binding"]["table_id"] == "samples"
    assert resolution["binding"]["field_ids"] == ["a", "b"]
    assert len(resolution["binding_profile_digest"]) == 64
    assert len(resolution["input_digest"]) == 64
    assert resolution["data_quality"] == {
        **resolution["data_quality"],
        "null_counts": {"a": 0, "b": 0},
        "row_count": 1,
        "selected_field_ids": ["a", "b"],
        "table_id": "samples",
        "unique_value_counts": {"a": 1, "b": 1},
    }
    assert inspection["resources"] == [
        {
            "byte_size": (tmp_path / "samples.csv").stat().st_size,
            "media_type": "text/csv",
            "resource_id": "samples",
            "sha256": sha256((tmp_path / "samples.csv").read_bytes()).hexdigest(),
            "uri": (tmp_path / "samples.csv").resolve().as_uri(),
        }
    ]
    assert normalized.exit_code == 0
    assert output.is_file()
    assert calculated.exit_code == 0
    assert "input_digest" in json.loads(calculated.output)


def test_inspect_requires_complete_explicit_binding_options(tmp_path) -> None:
    (tmp_path / "samples.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    descriptor = tmp_path / "datapackage.json"
    descriptor.write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "name": "samples",
                        "path": "samples.csv",
                        "schema": {"fields": [{"name": "a"}, {"name": "b"}]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    table_only = runner.invoke(
        app, ["ingest", "inspect", str(descriptor), "--table", "samples"]
    )
    fields_only = runner.invoke(
        app, ["ingest", "inspect", str(descriptor), "--field", "a"]
    )

    assert table_only.exit_code == 2
    assert fields_only.exit_code == 2
    assert "--table and at least one --field" in table_only.output
    assert "--table and at least one --field" in fields_only.output


def test_ingest_cli_returns_safe_error_for_unrecognized_descriptor(tmp_path) -> None:
    descriptor = tmp_path / "unknown.json"
    descriptor.write_text("{}", encoding="utf-8")

    result = CliRunner().invoke(app, ["ingest", "inspect", str(descriptor)])
    validated = CliRunner().invoke(app, ["ingest", "validate", str(descriptor)])

    assert result.exit_code == 2
    assert "exactly one" in result.output
    assert validated.exit_code == 2
    assert "exactly one" in validated.output


def test_normalize_and_calculate_return_safe_errors(tmp_path) -> None:
    descriptor = tmp_path / "unknown.json"
    descriptor.write_text("{}", encoding="utf-8")
    runner = CliRunner()
    assert (
        runner.invoke(
            app,
            ["ingest", "normalize", str(descriptor), "-o", str(tmp_path / "x.arrow")],
        ).exit_code
        == 2
    )
    assert (
        runner.invoke(
            app,
            [
                "ingest",
                "calculate-from-dataset",
                str(descriptor),
                "--table",
                "x",
                "--field",
                "a",
            ],
        ).exit_code
        == 2
    )


def test_ingest_cli_applies_explicit_resource_size_policy(tmp_path) -> None:
    """CLI policy flags constrain provider materialization without network opt-in."""
    (tmp_path / "samples.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    descriptor = tmp_path / "datapackage.json"
    descriptor.write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "name": "samples",
                        "path": "samples.csv",
                        "schema": {"fields": [{"name": "a"}, {"name": "b"}]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "ingest",
            "validate",
            str(descriptor),
            "--max-resource-bytes",
            "1",
        ],
    )

    assert result.exit_code == 2
    assert "exceeds configured size limit" in result.output
