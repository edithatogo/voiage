"""CLI coverage for dataset descriptor inspection and normalization."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from voiage.cli import app
from voiage.contracts import NormalizedInputBundle
from voiage.ingestion import cli as ingestion_cli
from voiage.ingestion.registry import default_registry as real_default_registry


def test_descriptor_metadata_rejects_a_non_object_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = tmp_path / "array.json"
    descriptor.write_text("[]", encoding="utf-8")

    class _InspectedRegistry:
        def inspect(self, _descriptor: Path) -> dict[str, object]:
            return {
                "capabilities": {},
                "descriptor": str(descriptor),
                "provider_id": "test-provider",
            }

    monkeypatch.setattr(
        ingestion_cli,
        "default_registry",
        _InspectedRegistry,
    )

    with pytest.raises(ingestion_cli.IngestionError, match="root must be an object"):
        ingestion_cli._inspection_summary(descriptor)


def test_ingest_cli_publishes_stable_domain_exit_codes(tmp_path) -> None:
    """CLI syntax, source, binding, and output failures remain distinguishable."""
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

    usage = runner.invoke(app, ["ingest", "validate", str(descriptor), "--unknown"])
    source = runner.invoke(
        app, ["ingest", "validate", str(descriptor), "--provider", "croissant"]
    )
    binding = runner.invoke(
        app,
        [
            "ingest",
            "calculate-from-dataset",
            str(descriptor),
            "--table",
            "samples",
            "--field",
            "missing",
        ],
    )
    output = runner.invoke(
        app,
        ["ingest", "normalize", str(descriptor), "--output", str(tmp_path)],
    )

    assert usage.exit_code == 2
    assert source.exit_code == 3
    assert binding.exit_code == 4
    assert output.exit_code == 5


def test_ingest_validate_rejects_a_resource_above_the_explicit_row_limit(
    tmp_path,
) -> None:
    """The CLI maps row-policy rejection to the stable ingestion exit code."""
    (tmp_path / "samples.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
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
            "--max-resource-rows",
            "1",
        ],
    )

    assert result.exit_code == 3
    assert "row limit" in result.output


def test_ingest_cli_enforces_explicit_provider_and_binding_profile(
    tmp_path, monkeypatch
) -> None:
    """Explicit provider and profile inputs are asserted, never inferred."""
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
    profile = tmp_path / "binding-profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "bindings": [
                    {
                        "role": "net_benefit",
                        "table_id": "samples",
                        "field_ids": ("a", "b"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    monkeypatch.setattr(ingestion_cli, "evpi", lambda _: 0.0)

    validated = runner.invoke(
        app, ["ingest", "validate", str(descriptor), "--provider", "frictionless"]
    )
    calculated = runner.invoke(
        app,
        [
            "ingest",
            "calculate-from-dataset",
            str(descriptor),
            "--binding-profile",
            str(profile),
        ],
    )
    conflicting = runner.invoke(
        app,
        [
            "ingest",
            "calculate-from-dataset",
            str(descriptor),
            "--binding-profile",
            str(profile),
            "--table",
            "samples",
            "--field",
            "a",
        ],
    )

    assert validated.exit_code == 0, validated.output
    assert calculated.exit_code == 0, calculated.output
    assert json.loads(calculated.output)["binding_profile_digest"]
    assert conflicting.exit_code == 4
    assert "cannot be combined" in conflicting.output


@pytest.mark.parametrize(
    ("descriptor_name", "provider"),
    [
        (
            "canonical-decision.croissant.json",
            "croissant",
        ),
        (
            "canonical-decision.datapackage.json",
            "frictionless",
        ),
    ],
)
def test_reference_descriptors_have_safe_cli_walkthroughs(
    descriptor_name: str, provider: str
) -> None:
    """ML and engineering fixtures expose the documented CLI evidence path."""
    fixture_root = Path(__file__).parent / "fixtures" / "standardized_ingestion"
    descriptor = fixture_root / descriptor_name
    runner = CliRunner()

    validated = runner.invoke(app, ["ingest", "validate", str(descriptor)])
    inspected = runner.invoke(app, ["ingest", "inspect", str(descriptor)])

    assert validated.exit_code == 0
    assert json.loads(validated.output)["valid"] is True
    assert inspected.exit_code == 0
    result = json.loads(inspected.output)
    assert result["provider"] == provider
    assert result["binding_resolution"] is None
    assert result["capabilities"]["provider_id"] == provider


@pytest.mark.parametrize(
    ("descriptor_name", "provider"),
    [
        ("canonical-decision.croissant.json", "croissant"),
        ("canonical-decision.datapackage.json", "frictionless"),
        ("cost-outcome-decision.croissant.json", "croissant"),
        ("cost-outcome-decision.datapackage.json", "frictionless"),
    ],
)
def test_cross_domain_reference_descriptors_validate_and_inspect(
    descriptor_name: str, provider: str
) -> None:
    """P9 fixture descriptors retain one local, non-materializing CLI route."""
    fixture_root = Path(__file__).parent / "fixtures" / "standardized_ingestion"
    descriptor = fixture_root / descriptor_name
    runner = CliRunner()

    validated = runner.invoke(app, ["ingest", "validate", str(descriptor)])
    inspected = runner.invoke(app, ["ingest", "inspect", str(descriptor)])

    assert validated.exit_code == 0
    validation = json.loads(validated.output)
    assert validation["valid"] is True
    assert validation["resources"][0]["sha256"]
    assert inspected.exit_code == 0
    inspection = json.loads(inspected.output)
    assert inspection["provider"] == provider
    assert inspection["binding_resolution"] is None


def test_ingest_commands_publish_stable_help_surfaces() -> None:
    """All documented ingestion commands remain discoverable through the CLI."""
    runner = CliRunner()

    for command, description in (
        ("validate", "Validate a supported descriptor"),
        ("inspect", "Inspect descriptor identity and provider capabilities"),
        ("normalize", "Normalize a descriptor into a deterministic Arrow IPC"),
        (
            "calculate-from-dataset",
            "Calculate EVPI from explicitly selected normalized net-benefit fields",
        ),
    ):
        result = runner.invoke(app, ["ingest", command, "--help"])

        assert result.exit_code == 0
        assert description in result.output


def test_ingest_inspect_and_normalize(tmp_path, monkeypatch) -> None:
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
    monkeypatch.setattr(ingestion_cli, "evpi", lambda _: 0.0)

    validated = runner.invoke(app, ["ingest", "validate", str(descriptor)])
    default_inspected = runner.invoke(app, ["ingest", "inspect", str(descriptor)])
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
    inspection = json.loads(default_inspected.output)
    assert inspection["provider"] == "frictionless"
    assert inspection["capabilities"] == {
        "format_versions": ["1"],
        "media_types": [
            "text/csv",
            "text/tab-separated-values",
            "application/json",
            "application/vnd.apache.parquet",
            "application/vnd.apache.arrow.file",
        ],
        "provider_id": "frictionless",
        "supported_transforms": [],
        "supports_filtering": False,
        "supports_projection": False,
        "supports_random_access": False,
        "source_selection_keys": [],
        "supports_streaming": False,
    }
    assert inspection["binding_resolution"] is None
    assert inspection["descriptor"] == str(descriptor)
    assert set(inspection) == {
        "binding_resolution",
        "capabilities",
        "descriptor",
        "provider",
        "schema",
    }
    assert normalized.exit_code == 0
    assert output.is_file()
    assert calculated.exit_code == 0
    assert "input_digest" in json.loads(calculated.output)


def test_inspect_rejects_binding_options_without_loading_data(tmp_path) -> None:
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
    assert "No such option" in table_only.output
    assert "No such option" in fields_only.output


def test_inspect_does_not_materialize_declared_resource(tmp_path) -> None:
    """Inspection can identify a descriptor whose local resource is absent."""
    descriptor = tmp_path / "datapackage.json"
    descriptor.write_text(
        json.dumps(
            {
                "name": "metadata-only",
                "resources": [
                    {
                        "name": "missing",
                        "path": "missing.csv",
                        "schema": {"fields": [{"name": "a"}]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    inspected = runner.invoke(app, ["ingest", "inspect", str(descriptor)])
    validated = runner.invoke(app, ["ingest", "validate", str(descriptor)])
    normalized = runner.invoke(
        app,
        ["ingest", "normalize", str(descriptor), "--output", str(tmp_path / "x.arrow")],
    )

    assert inspected.exit_code == 0
    assert json.loads(inspected.output)["provider"] == "frictionless"
    assert json.loads(inspected.output)["binding_resolution"] is None
    assert validated.exit_code == 3
    assert normalized.exit_code == 3
    assert "declared resource does not exist" in validated.output
    assert "declared resource does not exist" in normalized.output


def test_cli_croissant_source_selection_materializes_the_requested_pair(
    tmp_path, monkeypatch
) -> None:
    """Materializing commands expose source-pair choice without projection flags."""
    fixture_root = Path(__file__).parent / "fixtures" / "croissant_1_1"
    descriptor = fixture_root / "valid" / "multi-pair-croissant.json"
    output = tmp_path / "alternate.arrow"
    runner = CliRunner()
    monkeypatch.setattr(ingestion_cli, "evpi", lambda _: 0.0)
    options = [
        "--provider",
        "croissant",
        "--record-set",
        "alternate_samples",
        "--distribution",
        "#alternate",
        "--source-root",
        str(fixture_root),
    ]

    validated = runner.invoke(app, ["ingest", "validate", str(descriptor), *options])
    normalized = runner.invoke(
        app,
        ["ingest", "normalize", str(descriptor), "--output", str(output), *options],
    )
    calculated = runner.invoke(
        app,
        [
            "ingest",
            "calculate-from-dataset",
            str(descriptor),
            "--table",
            "alternate_samples",
            "--field",
            "strategy_c",
            "--field",
            "strategy_d",
            *options,
        ],
    )
    missing_distribution = runner.invoke(
        app,
        [
            "ingest",
            "validate",
            str(descriptor),
            "--record-set",
            "alternate_samples",
        ],
    )

    validation = json.loads(validated.output)
    normalization = json.loads(normalized.output)
    assert validated.exit_code == normalized.exit_code == calculated.exit_code == 0
    assert validation["tables"] == {"alternate_samples": 2}
    assert (
        normalization["content_digest"]
        == NormalizedInputBundle.read_ipc(output).content_digest
    )
    assert json.loads(calculated.output)["evpi"] == 0.0
    assert missing_distribution.exit_code == 3
    assert (
        "requires both --record-set and --distribution" in missing_distribution.output
    )


def test_normalize_materializes_once_and_reports_the_written_bundle_digest(
    tmp_path, monkeypatch
) -> None:
    """Normalization does not re-ingest sources merely to render its summary."""
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
    output = tmp_path / "normalized.arrow"
    observed = 0
    registry = real_default_registry()
    original_ingest = registry.ingest

    def spy_ingest(*args, **kwargs):
        nonlocal observed
        observed += 1
        return original_ingest(*args, **kwargs)

    monkeypatch.setattr(registry, "ingest", spy_ingest)
    monkeypatch.setattr(ingestion_cli, "default_registry", lambda: registry)

    result = CliRunner().invoke(
        app, ["ingest", "normalize", str(descriptor), "--output", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert observed == 1
    assert (
        json.loads(result.output)["content_digest"]
        == NormalizedInputBundle.read_ipc(output).content_digest
    )


def test_inspect_reports_declared_schema_boundaries_without_resource_io(
    tmp_path,
) -> None:
    """Inspection projects unsupported metadata but never opens missing data."""
    descriptor = tmp_path / "datapackage.json"
    descriptor.write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "name": "missing",
                        "path": "absent.csv",
                        "format": "xlsx",
                        "dialect": {"delimiter": ";"},
                        "transform": [],
                        "schema": {
                            "fields": [{"name": "id"}],
                            "primaryKey": "id",
                            "missingValues": ["NA"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["ingest", "inspect", str(descriptor)])
    assert result.exit_code == 0
    assert json.loads(result.output)["schema"] == {
        "tables": [
            {
                "table_id": "missing",
                "field_ids": ["id"],
                "primary_key": ["id"],
                "foreign_keys": [],
            }
        ],
        "unsupported_features": [
            {"code": "resource-dialect", "path": "resources[0].dialect"},
            {"code": "resource-transform", "path": "resources[0].transform"},
            {"code": "resource-format", "path": "resources[0].format"},
            {
                "code": "schema-missing-values",
                "path": "resources[0].schema.missingValues",
            },
        ],
    }


def test_descriptor_schema_summary_covers_non_materializing_fallbacks() -> None:
    """Every defensive descriptor-only branch remains safe and deterministic."""
    assert ingestion_cli._descriptor_schema_summary({}, provider_id="unknown") == {
        "tables": [],
        "unsupported_features": [],
    }
    assert ingestion_cli._frictionless_descriptor_schema_summary(
        {"resources": "not-a-list"}
    ) == {"tables": [], "unsupported_features": []}
    assert ingestion_cli._frictionless_descriptor_schema_summary(
        {"resources": ["not-an-object"]}
    ) == {
        "tables": [],
        "unsupported_features": [
            {"code": "resource-not-object", "path": "resources[0]"}
        ],
    }
    assert ingestion_cli._croissant_descriptor_schema_summary(
        {"recordSet": "not-a-list"}
    ) == {"tables": [], "unsupported_features": []}
    assert ingestion_cli._croissant_descriptor_schema_summary(
        {"recordSet": ["not-an-object"]}
    ) == {
        "tables": [],
        "unsupported_features": [
            {"code": "record-set-not-object", "path": "recordSet[0]"}
        ],
    }


def test_descriptor_schema_summary_reports_croissant_field_boundaries() -> None:
    """Field diagnostics are descriptor-only and do not select a source pair."""
    summary = ingestion_cli._croissant_descriptor_schema_summary(
        {
            "recordSet": [
                {
                    "name": "samples",
                    "field": [
                        "not-an-object",
                        {
                            "name": "net_benefit",
                            "references": "outside",
                            "subField": [],
                            "source": "derived",
                        },
                    ],
                }
            ]
        }
    )
    assert summary["tables"] == [
        {
            "table_id": "samples",
            "field_ids": ["net_benefit"],
            "primary_key": [],
            "foreign_keys": [],
        }
    ]
    assert summary["unsupported_features"] == [
        {"code": "field-references", "path": "recordSet[0].field[1].references"},
        {"code": "field-subField", "path": "recordSet[0].field[1].subField"},
        {"code": "field-source", "path": "recordSet[0].field[1].source"},
    ]


def test_descriptor_schema_summary_continues_after_a_non_list_field_declaration() -> (
    None
):
    """One malformed record set cannot suppress later descriptor diagnostics."""
    summary = ingestion_cli._croissant_descriptor_schema_summary(
        {
            "recordSet": [
                {"name": "first", "field": "not-a-list"},
                "not-an-object",
            ]
        }
    )

    assert summary == {
        "tables": [
            {
                "table_id": "first",
                "field_ids": [],
                "primary_key": [],
                "foreign_keys": [],
            }
        ],
        "unsupported_features": [
            {"code": "record-set-not-object", "path": "recordSet[1]"}
        ],
    }


def test_ingest_cli_returns_safe_error_for_unrecognized_descriptor(tmp_path) -> None:
    descriptor = tmp_path / "unknown.json"
    descriptor.write_text("{}", encoding="utf-8")

    result = CliRunner().invoke(app, ["ingest", "inspect", str(descriptor)])
    validated = CliRunner().invoke(app, ["ingest", "validate", str(descriptor)])

    assert result.exit_code == 3
    assert "exactly one" in result.output
    assert validated.exit_code == 3
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

    assert result.exit_code == 3
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
        == 3
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
        == 3
    )


def test_materializing_ingest_commands_expose_explicit_source_policy_controls(
    tmp_path, monkeypatch
) -> None:
    """CLI callers can make materialization policy explicit and fail closed."""
    source_root = tmp_path / "declared-source-root"
    source_root.mkdir()
    (source_root / "samples.csv").write_text("a,b\n1,2\n", encoding="utf-8")
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
    monkeypatch.setattr(ingestion_cli, "evpi", lambda _: 0.0)
    source_policy = ["--source-root", str(source_root)]
    resolved = [
        runner.invoke(app, ["ingest", "validate", str(descriptor), *source_policy]),
        runner.invoke(
            app,
            [
                "ingest",
                "normalize",
                str(descriptor),
                "--output",
                str(tmp_path / "normalized.arrow"),
                *source_policy,
            ],
        ),
        runner.invoke(
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
                *source_policy,
            ],
        ),
    ]
    constrained = runner.invoke(
        app,
        [
            "ingest",
            "validate",
            str(descriptor),
            "--source-root",
            str(source_root),
            "--max-resource-bytes",
            "1",
        ],
    )
    invalid_limit = runner.invoke(
        app,
        [
            "ingest",
            "validate",
            str(descriptor),
            "--source-root",
            str(source_root),
            "--max-resource-bytes",
            "0",
        ],
    )

    assert all(result.exit_code == 0 for result in resolved)
    assert json.loads(resolved[0].output)["valid"] is True
    assert constrained.exit_code == 3
    assert "exceeds configured size limit" in constrained.output
    assert invalid_limit.exit_code == 2
    assert "Invalid value" in invalid_limit.output


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

    assert result.exit_code == 3
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


def test_croissant_validation_exposes_governance_and_receipt_identity(tmp_path) -> None:
    """Materializing validation retains Croissant governance and receipts."""
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

    result = CliRunner().invoke(app, ["ingest", "validate", str(descriptor)])

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


def test_frictionless_validation_exposes_governance_and_receipt_identity(
    tmp_path,
) -> None:
    """Materializing validation preserves package governance metadata."""
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

    result = CliRunner().invoke(app, ["ingest", "validate", str(descriptor)])

    assert result.exit_code == 0
    inspection = json.loads(result.output)
    assert inspection["provider"] == "frictionless"
    assert inspection["provenance"]["license"] == "CC-BY-4.0"
    assert inspection["provenance"]["citation"] == "Example et al. (2026)"
    assert inspection["governance"] == {
        "frictionlessdata.org:citation": "Example et al. (2026)",
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
