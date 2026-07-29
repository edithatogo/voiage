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


def test_ingest_cli_redacts_credentials_from_rejected_resource_uris(tmp_path) -> None:
    """User-facing diagnostics must not disclose credentials in a descriptor."""
    descriptor = tmp_path / "croissant.json"
    descriptor.write_text(
        json.dumps(
            {
                "@context": "https://mlcommons.org/croissant/1.1",
                "distribution": [
                    {"contentUrl": "ssh://user:super-secret@example.invalid/data.csv"}
                ],
                "recordSet": [
                    {"name": "samples", "field": [{"name": "a"}, {"name": "b"}]}
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["ingest", "validate", str(descriptor)])

    assert result.exit_code == 2
    assert "network resource access is disabled" in result.output
    assert "super-secret" not in result.output
    assert "example.invalid" not in result.output


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


def test_ingest_cli_replays_a_declared_resource_from_offline_cache(tmp_path) -> None:
    """The CLI forwards its cache and offline policy flags to the provider."""
    source = tmp_path / "samples.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    descriptor = tmp_path / "croissant.json"
    descriptor.write_text(
        json.dumps(
            {
                "@context": "https://mlcommons.org/croissant/1.1",
                "distribution": [
                    {
                        "contentUrl": "samples.csv",
                        "sha256": sha256(source.read_bytes()).hexdigest(),
                    }
                ],
                "recordSet": [
                    {"name": "samples", "field": [{"name": "a"}, {"name": "b"}]}
                ],
            }
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    runner = CliRunner()

    cached = runner.invoke(
        app, ["ingest", "validate", str(descriptor), "--cache-dir", str(cache)]
    )
    source.unlink()
    replayed = runner.invoke(
        app,
        [
            "ingest",
            "validate",
            str(descriptor),
            "--cache-dir",
            str(cache),
            "--offline",
        ],
    )

    assert cached.exit_code == 0
    assert replayed.exit_code == 0
    assert json.loads(replayed.output)["valid"] is True


def test_croissant_inspection_exposes_governance_and_receipt_identity(tmp_path) -> None:
    """Croissant inspection retains governance without inferring VOI semantics."""
    source = tmp_path / "samples.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    descriptor = tmp_path / "croissant.json"
    descriptor.write_text(
        json.dumps(
            {
                "@context": "https://mlcommons.org/croissant/1.1",
                "name": "governed-ml-fixture",
                "license": "CC-BY-4.0",
                "citation": "Example et al. (2026)",
                "usageInfo": "Synthetic test data only.",
                "rai": {"dataBiases": "None asserted for this synthetic fixture."},
                "distribution": [
                    {
                        "contentUrl": "samples.csv",
                        "encodingFormat": "text/csv",
                        "sha256": sha256(source.read_bytes()).hexdigest(),
                    }
                ],
                "recordSet": [
                    {"name": "samples", "field": [{"name": "a"}, {"name": "b"}]}
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["ingest", "inspect", str(descriptor)])

    assert result.exit_code == 0
    inspection = json.loads(result.output)
    assert inspection["provider"] == "croissant"
    assert inspection["provenance"]["license"] == "CC-BY-4.0"
    assert inspection["provenance"]["citation"] == "Example et al. (2026)"
    assert inspection["governance"] == {
        "mlcommons.org:croissant-governance": {
            "citation": "Example et al. (2026)",
            "license": "CC-BY-4.0",
            "rai": {"dataBiases": "None asserted for this synthetic fixture."},
            "usageInfo": "Synthetic test data only.",
        }
    }
    assert inspection["resources"] == [
        {
            "byte_size": source.stat().st_size,
            "media_type": "text/csv",
            "resource_id": "samples",
            "sha256": sha256(source.read_bytes()).hexdigest(),
            "uri": source.resolve().as_uri(),
        }
    ]


def test_frictionless_inspection_exposes_governance_and_receipt_identity(
    tmp_path,
) -> None:
    """Frictionless inspection preserves explicit package governance metadata."""
    source = tmp_path / "samples.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    descriptor = tmp_path / "datapackage.json"
    descriptor.write_text(
        json.dumps(
            {
                "name": "governed-operations-fixture",
                "title": "Governed operations fixture",
                "description": "Synthetic, rights-cleared test data.",
                "profile": "tabular-data-package",
                "version": "1.0.0",
                "citation": "Example et al. (2026)",
                "licenses": [
                    {"name": "CC-BY-4.0", "path": "https://example.invalid/license"}
                ],
                "sources": [{"title": "Synthetic source", "path": "source.md"}],
                "contributors": [{"title": "Maintainer", "role": "author"}],
                "resources": [
                    {
                        "name": "samples",
                        "path": "samples.csv",
                        "hash": sha256(source.read_bytes()).hexdigest(),
                        "bytes": source.stat().st_size,
                        "schema": {"fields": [{"name": "a"}, {"name": "b"}]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["ingest", "inspect", str(descriptor)])

    assert result.exit_code == 0
    inspection = json.loads(result.output)
    assert inspection["provider"] == "frictionless"
    assert inspection["provenance"]["license"] == "CC-BY-4.0"
    assert inspection["provenance"]["citation"] == "Example et al. (2026)"
    assert inspection["governance"] == {
        "frictionlessdata.org:contributors": [
            {"role": "author", "title": "Maintainer"}
        ],
        "frictionlessdata.org:description": "Synthetic, rights-cleared test data.",
        "frictionlessdata.org:licenses": [
            {"name": "CC-BY-4.0", "path": "https://example.invalid/license"}
        ],
        "frictionlessdata.org:profile": "tabular-data-package",
        "frictionlessdata.org:sources": [
            {"path": "source.md", "title": "Synthetic source"}
        ],
        "frictionlessdata.org:title": "Governed operations fixture",
        "frictionlessdata.org:version": "1.0.0",
    }
    assert inspection["resources"] == [
        {
            "byte_size": source.stat().st_size,
            "media_type": "text/csv",
            "resource_id": "samples",
            "sha256": sha256(source.read_bytes()).hexdigest(),
            "uri": source.resolve().as_uri(),
        }
    ]
