"""Built-in descriptor adapters stay isolated from the conductor contract."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from voiage import ingestion
from voiage.contracts import (
    DatasetManifest,
    FieldManifest,
    NormalizedInputBundle,
    SourceProvenance,
    TableManifest,
    VOIBinding,
    prepare_analysis_inputs,
)
from voiage.ingestion import (
    IngestionError,
    IngestionProvider,
    ProviderCapabilities,
    SourceAccessPolicy,
    default_registry,
    discover_entry_point_providers,
    from_dataframe,
)
from voiage.ingestion._tabular import digest_file, read_csv
from voiage.ingestion.croissant import CroissantProvider
from voiage.ingestion.frictionless import (
    FrictionlessProvider,
    _citation_label,
    _governance_extensions,
    _license_label,
)
from voiage.ingestion.registry import ProviderRegistry


def _write_csv(tmp_path) -> None:
    (tmp_path / "samples.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("name", "descriptor"),
    [
        (
            "croissant.json",
            {
                "@context": "http://mlcommons.org/croissant/1.1",
                "name": "ml-fixture",
                "distribution": [{"contentUrl": "samples.csv"}],
                "recordSet": [
                    {"name": "samples", "field": [{"name": "a"}, {"name": "b"}]}
                ],
            },
        ),
        (
            "datapackage.json",
            {
                "name": "operations-fixture",
                "resources": [
                    {
                        "name": "samples",
                        "path": "samples.csv",
                        "schema": {"fields": [{"name": "a"}, {"name": "b"}]},
                    }
                ],
            },
        ),
    ],
)
def test_built_in_providers_normalize_supported_csv_profile(
    tmp_path, name, descriptor
) -> None:
    _write_csv(tmp_path)
    descriptor_path = tmp_path / name
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")

    bundle = default_registry().ingest(descriptor_path)

    assert bundle.manifest.dataset_id in {"ml-fixture", "operations-fixture"}
    assert bundle.table("samples").column_names == ["a", "b"]
    assert bundle.manifest.resources[0].resource_id == "samples"
    assert bundle.manifest.resources[0].sha256 == digest_file(tmp_path / "samples.csv")
    assert (
        bundle.manifest.resources[0].byte_size
        == (tmp_path / "samples.csv").stat().st_size
    )


def test_registry_inspect_reports_capabilities_without_materializing(tmp_path) -> None:
    descriptor_path = tmp_path / "croissant.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "@context": "http://mlcommons.org/croissant/1.1",
                "name": "inspect-fixture",
                "distribution": [{"contentUrl": "missing.csv"}],
                "recordSet": [{"name": "samples", "field": [{"name": "a"}]}],
            }
        ),
        encoding="utf-8",
    )

    inspection = default_registry().inspect(descriptor_path)

    assert inspection["provider_id"] == "croissant"
    assert inspection["descriptor"] == str(descriptor_path)
    assert inspection["capabilities"] == {
        "format_versions": ("1.1",),
        "media_types": ("text/csv",),
        "supported_transforms": (),
        "supports_projection": False,
        "supports_filtering": False,
        "supports_streaming": False,
        "supports_random_access": False,
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "descriptor root must be a JSON object"),
        (
            {"unrecognized": True},
            "descriptor must match exactly one registered provider",
        ),
    ],
)
def test_registry_inspect_rejects_invalid_or_unrecognized_descriptors(
    tmp_path, payload, message
) -> None:
    """Metadata-only inspection keeps the same descriptor boundary as ingest."""
    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(IngestionError, match=message):
        default_registry().inspect(descriptor_path)


def test_registry_inspect_rejects_malformed_json_descriptor(tmp_path) -> None:
    """Inspection exposes the same stable malformed-descriptor error as ingest."""
    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text("{not valid JSON", encoding="utf-8")

    with pytest.raises(IngestionError, match="descriptor is not valid UTF-8 JSON"):
        default_registry().inspect(descriptor_path)


@pytest.mark.parametrize("provider", ["croissant", "frictionless"])
def test_built_in_provider_replays_a_verified_cached_resource_offline(
    tmp_path, provider
) -> None:
    """Built-ins consume the verified cache rather than bypassing it on replay."""
    _write_csv(tmp_path)
    source = tmp_path / "samples.csv"
    digest = digest_file(source)
    if provider == "croissant":
        descriptor = {
            "@context": "https://mlcommons.org/croissant/1.1",
            "distribution": [{"contentUrl": "samples.csv", "sha256": digest}],
            "recordSet": [{"name": "samples", "field": [{"name": "a"}, {"name": "b"}]}],
        }
        descriptor_path = tmp_path / "croissant.json"
    else:
        descriptor = {
            "resources": [
                {
                    "name": "samples",
                    "path": "samples.csv",
                    "hash": digest,
                    "bytes": source.stat().st_size,
                    "schema": {"fields": [{"name": "a"}, {"name": "b"}]},
                }
            ]
        }
        descriptor_path = tmp_path / "datapackage.json"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    cache = tmp_path / "cache"

    default_registry().ingest(
        descriptor_path,
        policy=SourceAccessPolicy(tmp_path, cache_dir=cache, cache_namespace="test"),
    )
    source.unlink()

    bundle = default_registry().ingest(
        descriptor_path,
        policy=SourceAccessPolicy(
            tmp_path, cache_dir=cache, cache_namespace="test", offline=True
        ),
    )

    assert bundle.table("samples").num_rows == 2
    assert bundle.manifest.resources[0].uri == source.as_uri()


def test_frictionless_provider_validates_declared_types_constraints_and_primary_key(
    tmp_path,
) -> None:
    (tmp_path / "samples.csv").write_text("id,value\n1,1.5\n2,2.5\n", encoding="utf-8")
    descriptor_path = tmp_path / "datapackage.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "name": "strict-operations-fixture",
                "resources": [
                    {
                        "name": "samples",
                        "path": "samples.csv",
                        "dialect": {"delimiter": ","},
                        "schema": {
                            "primaryKey": "id",
                            "fields": [
                                {
                                    "name": "id",
                                    "type": "integer",
                                    "constraints": {"required": True, "unique": True},
                                },
                                {"name": "value", "type": "number"},
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = FrictionlessProvider().ingest(
        descriptor_path, policy=SourceAccessPolicy(tmp_path)
    )

    assert bundle.manifest.tables[0].primary_key == ("id",)
    assert bundle.table("samples").column_names == ["id", "value"]


def test_frictionless_provider_preserves_package_governance_metadata(tmp_path) -> None:
    (tmp_path / "samples.csv").write_text("id,value\n1,1.5\n", encoding="utf-8")
    descriptor_path = tmp_path / "datapackage.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "name": "governed-operations-fixture",
                "title": "Governed operations fixture",
                "description": "Synthetic, rights-cleared test data.",
                "version": "1.0.0",
                "profile": "tabular-data-package",
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
                        "schema": {
                            "fields": [
                                {"name": "id", "type": "integer"},
                                {"name": "value", "type": "number"},
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = FrictionlessProvider().ingest(
        descriptor_path, policy=SourceAccessPolicy(tmp_path)
    )

    assert bundle.manifest.provenance.license == "CC-BY-4.0"
    assert bundle.manifest.provenance.citation == "Example et al. (2026)"
    assert bundle.manifest.extensions == {
        "frictionlessdata.org:contributors": (
            {"role": "author", "title": "Maintainer"},
        ),
        "frictionlessdata.org:description": "Synthetic, rights-cleared test data.",
        "frictionlessdata.org:licenses": (
            {"name": "CC-BY-4.0", "path": "https://example.invalid/license"},
        ),
        "frictionlessdata.org:profile": "tabular-data-package",
        "frictionlessdata.org:sources": (
            {"path": "source.md", "title": "Synthetic source"},
        ),
        "frictionlessdata.org:title": "Governed operations fixture",
        "frictionlessdata.org:version": "1.0.0",
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("MIT", "MIT"),
        (None, None),
        (["Apache-2.0"], "Apache-2.0"),
        ([{"title": "BSD-3-Clause"}], "BSD-3-Clause"),
        ([{"path": "LICENSE"}], "LICENSE"),
        ([{}, "GPL-3.0-only"], "GPL-3.0-only"),
        ([0], None),
        ([], None),
    ],
)
def test_frictionless_governance_helpers_preserve_only_explicit_labels(
    value, expected
) -> None:
    assert _license_label(value) == expected
    assert _citation_label(value) == (value if isinstance(value, str) else None)
    assert _governance_extensions({"title": "fixture", "resources": []}) == {
        "frictionlessdata.org:title": "fixture"
    }


@pytest.mark.parametrize(
    ("csv", "schema", "message"),
    [
        (
            "id,value\n,1\n",
            {
                "primaryKey": "id",
                "fields": [{"name": "id"}, {"name": "value"}],
            },
            "primaryKey contains null",
        ),
        (
            "id\n1\n1\n",
            {"primaryKey": "id", "fields": [{"name": "id", "type": "integer"}]},
            "primaryKey contains duplicate",
        ),
        (
            "id\nnot-a-number\n",
            {"fields": [{"name": "id", "type": "integer"}]},
            "field type does not match",
        ),
        (
            "id\n1\n",
            {"fields": [{"name": "id", "type": "date"}]},
            "unsupported Data Package field type",
        ),
    ],
)
def test_frictionless_provider_rejects_unsupported_or_invalid_schema_claims(
    tmp_path, csv, schema, message
) -> None:
    (tmp_path / "samples.csv").write_text(csv, encoding="utf-8")
    descriptor_path = tmp_path / "datapackage.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "resources": [
                    {"name": "samples", "path": "samples.csv", "schema": schema}
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(IngestionError, match=message):
        FrictionlessProvider().ingest(
            descriptor_path, policy=SourceAccessPolicy(tmp_path)
        )


@pytest.mark.parametrize(
    ("csv", "resource", "message"),
    [
        (
            "id\n1\n",
            {
                "name": "samples",
                "path": "samples.csv",
                "dialect": {"delimiter": ";"},
                "schema": {"fields": [{"name": "id"}]},
            },
            "only CSV comma dialect",
        ),
        (
            "id\n1\n",
            {
                "name": "samples",
                "path": "samples.csv",
                "schema": {"primaryKey": 1, "fields": [{"name": "id"}]},
            },
            "primaryKey must be a string or field list",
        ),
        (
            "id\n1\n",
            {
                "name": "samples",
                "path": "samples.csv",
                "schema": {"primaryKey": ["missing"], "fields": [{"name": "id"}]},
            },
            "primaryKey references an unknown",
        ),
        (
            "id\n1\n",
            {
                "name": "samples",
                "path": "samples.csv",
                "schema": {"fields": [{}]},
            },
            "fields require string names",
        ),
        (
            "id\n1\n",
            {
                "name": "samples",
                "path": "samples.csv",
                "schema": {"fields": [{"name": "missing"}]},
            },
            "exactly declare",
        ),
        (
            "id\n1\n",
            {
                "name": "samples",
                "path": "samples.csv",
                "schema": {"fields": [{"name": "id", "constraints": []}]},
            },
            "constraints must be an object",
        ),
        (
            "id\n1\n1\n",
            {
                "name": "samples",
                "path": "samples.csv",
                "schema": {"fields": [{"name": "id", "constraints": {"unique": True}}]},
            },
            "unique field contains duplicate",
        ),
    ],
)
def test_frictionless_provider_covers_remaining_strict_profile_rejections(
    tmp_path, csv, resource, message
) -> None:
    (tmp_path / "samples.csv").write_text(csv, encoding="utf-8")
    descriptor_path = tmp_path / "datapackage.json"
    descriptor_path.write_text(json.dumps({"resources": [resource]}), encoding="utf-8")

    with pytest.raises(IngestionError, match=message):
        FrictionlessProvider().ingest(
            descriptor_path, policy=SourceAccessPolicy(tmp_path)
        )


def test_frictionless_provider_rejects_required_null_field(tmp_path) -> None:
    (tmp_path / "samples.csv").write_text("id,value\n,1\n", encoding="utf-8")
    descriptor_path = tmp_path / "datapackage.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "name": "samples",
                        "path": "samples.csv",
                        "schema": {
                            "fields": [
                                {"name": "id", "constraints": {"required": True}},
                                {"name": "value"},
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(IngestionError, match="required field contains null"):
        FrictionlessProvider().ingest(
            descriptor_path, policy=SourceAccessPolicy(tmp_path)
        )


@pytest.mark.parametrize(
    ("provider", "version"),
    [(CroissantProvider(), "1.1"), (FrictionlessProvider(), "1")],
)
def test_provider_capabilities_declare_conservative_supported_profiles(
    provider, version
) -> None:
    capabilities = provider.capabilities

    assert capabilities.provider_id == provider.provider_id
    assert version in capabilities.format_versions
    assert capabilities.media_types == ("text/csv",)
    assert capabilities.supports_projection is False
    assert capabilities.supports_filtering is False
    assert capabilities.supports_streaming is False
    assert capabilities.supports_random_access is False


def test_registry_capability_lookup_rejects_unknown_provider() -> None:
    """Inspection must not report a capability profile for an unregistered ID."""
    registry = default_registry()

    assert registry.capabilities_for("croissant").provider_id == "croissant"
    with pytest.raises(IngestionError, match="unregistered provider"):
        registry.capabilities_for("unknown")


def test_source_policy_blocks_path_traversal_and_network(tmp_path) -> None:
    policy = SourceAccessPolicy(tmp_path)

    with pytest.raises(IngestionError, match="escapes"):
        policy.resolve("../outside.csv")
    with pytest.raises(IngestionError, match="network"):
        policy.resolve("https://example.invalid/input.csv")
    with pytest.raises(IngestionError, match="not implemented"):
        SourceAccessPolicy(tmp_path, allow_network=True).resolve(
            "https://example.invalid/input.csv"
        )
    with pytest.raises(IngestionError, match="does not exist"):
        policy.resolve("missing.csv")


def test_source_policy_enforces_an_explicit_resource_size_limit(tmp_path) -> None:
    """Oversized local inputs are rejected before a parser can load them."""
    resource = tmp_path / "oversized.csv"
    resource.write_bytes(b"abcde")

    with pytest.raises(IngestionError, match="configured size limit"):
        SourceAccessPolicy(tmp_path, max_resource_bytes=4).resolve(resource.name)

    with pytest.raises(ValueError, match="must be positive"):
        SourceAccessPolicy(tmp_path, max_resource_bytes=0)


@pytest.mark.parametrize(
    "reference",
    [
        "http://127.0.0.1/private.csv",
        "https://example.invalid/redirect.csv",
        "ftp://example.invalid/resource.csv",
        "s3://bucket/private.csv",
        "file:///etc/passwd",
    ],
)
def test_source_policy_rejects_every_url_scheme_before_any_network_access(
    tmp_path, reference
) -> None:
    with pytest.raises(IngestionError, match="network resource access is disabled"):
        SourceAccessPolicy(tmp_path).resolve(reference)


def test_dataframe_interchange_adapter_does_not_require_a_specific_frame_library() -> (
    None
):
    bundle = from_dataframe(
        pl.DataFrame({"a": [1.0], "b": [2.0]}), dataset_id="business"
    )

    assert bundle.manifest.provenance.provider_id == "dataframe-interchange"
    assert bundle.table("data").column_names == ["a", "b"]


def test_dataframe_interchange_preserves_supported_nullable_and_temporal_values() -> (
    None
):
    frame = pl.DataFrame(
        {
            "active": [True, None],
            "label": ["alpha", None],
            "observed_at": [
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 1, 2, tzinfo=UTC),
            ],
        }
    )

    bundle = from_dataframe(frame, dataset_id="engineering")

    assert bundle.table("data").to_pylist() == [
        {
            "active": True,
            "label": "alpha",
            "observed_at": datetime(2026, 1, 1, tzinfo=UTC),
        },
        {
            "active": None,
            "label": None,
            "observed_at": datetime(2026, 1, 2, tzinfo=UTC),
        },
    ]


def test_dataframe_interchange_preserves_pandas_category_null_and_timezone_values() -> (
    None
):
    index = pd.Index(["first", "second"], name="scenario")
    frame = pd.DataFrame(
        {
            "tier": pd.Series(["standard", None], dtype="category", index=index),
            "cost": pd.Series([10, None], dtype="Int64", index=index),
            "observed_at": pd.Series(
                pd.to_datetime(["2026-01-01T00:00:00Z", None]), index=index
            ),
        },
        index=index,
    )

    bundle = from_dataframe(frame, dataset_id="business")

    assert bundle.table("data").to_pylist() == [
        {
            "tier": "standard",
            "cost": 10,
            "observed_at": datetime(2026, 1, 1, tzinfo=UTC),
        },
        {"tier": None, "cost": None, "observed_at": None},
    ]
    assert bundle.table("data").column_names == ["tier", "cost", "observed_at"]


def test_dataframe_interchange_reports_a_disallowed_copy() -> None:
    frame = pl.DataFrame({"label": ["one", "two"]})

    with pytest.raises(ValueError, match="requested copy policy"):
        from_dataframe(frame, dataset_id="no-copy", allow_copy=False)


def test_dataframe_interchange_rejects_unsupported_nested_values_with_stable_error() -> (
    None
):
    frame = pl.DataFrame({"scenario": [["one", "two"]]})

    with pytest.raises(ValueError, match="dataframe interchange protocol"):
        from_dataframe(frame, dataset_id="nested")


@pytest.mark.parametrize(
    ("provider", "name", "descriptor", "message"),
    [
        (
            CroissantProvider(),
            "croissant.json",
            {"@context": "mlcommons.org/croissant/1.1"},
            "recordSet",
        ),
        (
            FrictionlessProvider(),
            "datapackage.json",
            {"resources": []},
            "requires resources",
        ),
        (
            FrictionlessProvider(),
            "datapackage.json",
            {"resources": [{"name": "x", "path": "x.csv", "schema": {}}]},
            "requires fields",
        ),
    ],
)
def test_providers_reject_ambiguous_or_incomplete_descriptors(
    tmp_path, provider, name, descriptor, message
) -> None:
    path = tmp_path / name
    path.write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(IngestionError, match=message):
        provider.ingest(path, policy=SourceAccessPolicy(tmp_path))


def test_registry_rejects_invalid_and_ambiguous_descriptors(tmp_path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[1]", encoding="utf-8")
    with pytest.raises(IngestionError, match="root"):
        ProviderRegistry().ingest(invalid)
    invalid.write_text("not-json", encoding="utf-8")
    with pytest.raises(IngestionError, match="valid UTF-8 JSON"):
        ProviderRegistry().ingest(invalid)
    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(IngestionError, match="exactly one"):
        ProviderRegistry().ingest(invalid)


def test_registry_supports_a_fake_provider_with_injected_source_policy(
    tmp_path,
) -> None:
    source_path = tmp_path / "example.json"
    source_path.write_text('{"provider": "fake"}', encoding="utf-8")
    supplied_policy = SourceAccessPolicy(tmp_path)
    observed: list[SourceAccessPolicy] = []

    class FakeProvider:
        provider_id = "fake"
        capabilities = ProviderCapabilities(
            provider_id="fake",
            format_versions=("1",),
            media_types=("application/json",),
        )

        def can_handle(self, descriptor: dict[str, object]) -> bool:
            return descriptor.get("provider") == "fake"

        def ingest(
            self, descriptor_path, *, policy: SourceAccessPolicy
        ) -> NormalizedInputBundle:
            assert descriptor_path == source_path
            observed.append(policy)
            return from_dataframe(pl.DataFrame({"value": [1]}), dataset_id="fake")

    bundle = ProviderRegistry((FakeProvider(),)).ingest(
        source_path, policy=supplied_policy
    )

    assert bundle.manifest.dataset_id == "fake"
    assert observed == [supplied_policy]


def test_public_provider_protocol_supports_consumer_runtime_validation() -> None:
    class ConsumerProvider:
        provider_id = "consumer"
        capabilities = ProviderCapabilities(
            provider_id="consumer",
            format_versions=("1",),
            media_types=("application/json",),
        )

        def can_handle(self, descriptor: dict[str, object]) -> bool:
            return bool(descriptor)

        def ingest(
            self, descriptor_path, *, policy: SourceAccessPolicy
        ) -> NormalizedInputBundle:
            raise AssertionError("not called")

    assert isinstance(ConsumerProvider(), IngestionProvider)
    assert not isinstance(object(), IngestionProvider)


def test_registry_rejects_a_provider_with_mismatched_capability_identity() -> None:
    """A published provider cannot claim a capability manifest for another ID."""

    class MismatchedProvider:
        provider_id = "consumer"
        capabilities = ProviderCapabilities(
            provider_id="different",
            format_versions=("1",),
            media_types=("application/json",),
        )

        def can_handle(self, descriptor: dict[str, object]) -> bool:
            return bool(descriptor)

        def ingest(
            self, descriptor_path, *, policy: SourceAccessPolicy
        ) -> NormalizedInputBundle:
            raise AssertionError("not called")

    with pytest.raises(IngestionError, match="provider contract"):
        ProviderRegistry((MismatchedProvider(),))


def test_base_import_does_not_load_builtin_provider_modules() -> None:
    script = "; ".join(
        (
            "import sys",
            "import voiage",
            "assert 'voiage.ingestion.croissant' not in sys.modules",
            "assert 'voiage.ingestion.frictionless' not in sys.modules",
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_ingestion_package_import_does_not_load_builtin_provider_modules() -> None:
    """Public ingestion helpers stay usable without loading source adapters."""
    script = "; ".join(
        (
            "import sys",
            "import voiage.ingestion",
            "assert 'voiage.ingestion.croissant' not in sys.modules",
            "assert 'voiage.ingestion.frictionless' not in sys.modules",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_public_provider_exports_load_the_requested_adapter_on_demand() -> None:
    """Lazy exports retain the public provider-class import contract."""
    script = "; ".join(
        (
            "import sys",
            "from voiage.ingestion import CroissantProvider",
            "assert CroissantProvider.provider_id == 'croissant'",
            "assert 'voiage.ingestion.croissant' in sys.modules",
            "assert 'voiage.ingestion.frictionless' not in sys.modules",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_public_provider_exports_cover_lazy_lookup_branches() -> None:
    """Public names resolve through the lazy module lookup under coverage."""
    assert ingestion.CroissantProvider is CroissantProvider
    assert ingestion.FrictionlessProvider is FrictionlessProvider

    with pytest.raises(AttributeError, match="has no attribute 'UnknownProvider'"):
        _ = ingestion.UnknownProvider


def test_entry_point_discovery_is_opt_in_and_allow_listed() -> None:
    loaded: list[str] = []

    class EntryPoint:
        def __init__(self, name: str, value: object) -> None:
            self.name = name
            self._value = value

        def load(self) -> object:
            loaded.append(self.name)
            return self._value

    allowed = EntryPoint("example-provider", CroissantProvider())
    ignored = EntryPoint("untrusted-provider", FrictionlessProvider())
    resolver_calls: list[str] = []

    def resolver(*, group: str):
        resolver_calls.append(group)
        return (ignored, allowed)

    assert discover_entry_point_providers(allowlist=(), resolver=resolver) == ()
    assert resolver_calls == []
    assert loaded == []

    providers = discover_entry_point_providers(
        allowlist=("example-provider",), resolver=resolver
    )

    assert resolver_calls == ["voiage.ingestion.providers"]
    assert loaded == ["example-provider"]
    assert providers == (allowed._value,)


def test_entry_point_discovery_rejects_missing_invalid_and_failing_providers() -> None:
    class EntryPoint:
        name = "example-provider"

        def __init__(self, value: object, *, fail: bool = False) -> None:
            self.value = value
            self.fail = fail

        def load(self) -> object:
            if self.fail:
                raise RuntimeError("private source details")
            return self.value

    def resolver(*, group: str):
        assert group == "voiage.ingestion.providers"
        return (EntryPoint(object()),)

    with pytest.raises(IngestionError, match="does not satisfy"):
        discover_entry_point_providers(
            allowlist=("example-provider",), resolver=resolver
        )
    with pytest.raises(IngestionError, match="unavailable"):
        discover_entry_point_providers(allowlist=("missing",), resolver=resolver)

    def failing_resolver(*, group: str):
        return (EntryPoint(CroissantProvider(), fail=True),)

    with pytest.raises(IngestionError, match="could not be loaded") as error:
        discover_entry_point_providers(
            allowlist=("example-provider",), resolver=failing_resolver
        )
    assert "private source details" not in str(error.value)


def test_entry_point_discovery_converts_a_missing_optional_extra_to_stable_error() -> (
    None
):
    class EntryPoint:
        name = "requires-extra"

        def load(self) -> object:
            raise ModuleNotFoundError("No module named 'optional_provider_extra'")

    def resolver(*, group: str):
        assert group == "voiage.ingestion.providers"
        return (EntryPoint(),)

    with pytest.raises(IngestionError, match="could not be loaded") as error:
        discover_entry_point_providers(allowlist=("requires-extra",), resolver=resolver)

    assert "optional_provider_extra" not in str(error.value)


def test_tabular_and_preparation_rejection_paths(tmp_path) -> None:
    source = tmp_path / "samples.txt"
    source.write_text("a\n1\n", encoding="utf-8")
    descriptor = tmp_path / "datapackage.json"
    descriptor.write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "name": "t",
                        "path": "samples.txt",
                        "schema": {"fields": [{"name": "a"}]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(IngestionError, match="CSV"):
        default_registry().ingest(descriptor)
    csv_source = tmp_path / "unreadable.csv"
    csv_source.write_text("a\n1\n", encoding="utf-8")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "voiage.ingestion._tabular.csv.read_csv",
            lambda _: (_ for _ in ()).throw(pa.ArrowInvalid("bad CSV")),
        )
        with pytest.raises(IngestionError, match="cannot be parsed"):
            read_csv("unreadable.csv", SourceAccessPolicy(tmp_path))
    assert (
        digest_file(csv_source)
        == "309b0e45a73d3fc5325e2b6ed0a01ef8b9cde6b05a5633c1f893f970d52bfddc"
    )
    manifest = DatasetManifest(
        dataset_id="x",
        tables=(
            TableManifest(
                table_id="t", fields=(FieldManifest(field_id="a", dtype="float64"),)
            ),
        ),
        provenance=SourceProvenance(
            provider_id="direct", source_uri="file:///x", descriptor_digest="a" * 64
        ),
    )
    empty = NormalizedInputBundle(manifest=manifest, tables={"t": pa.table({"a": []})})
    with pytest.raises(ValueError, match="exactly one"):
        prepare_analysis_inputs(empty)
    bound = manifest.model_copy(
        update={
            "bindings": (
                VOIBinding(role="net_benefit", table_id="t", field_ids=("a",)),
            )
        }
    )
    with pytest.raises(ValueError, match="at least one row"):
        prepare_analysis_inputs(
            NormalizedInputBundle(manifest=bound, tables={"t": pa.table({"a": []})})
        )
    with pytest.raises(ValueError, match="contains nulls"):
        prepare_analysis_inputs(
            NormalizedInputBundle(manifest=bound, tables={"t": pa.table({"a": [None]})})
        )


def test_preparation_rejects_non_numeric_arrow_column() -> None:
    class NonNumericColumn:
        null_count = 0

        def combine_chunks(self):
            return self

        def to_numpy(self, *, zero_copy_only: bool):
            raise pa.ArrowInvalid("cannot convert")

    class SelectedTable:
        num_rows = 1

        def __getitem__(self, field: str) -> NonNumericColumn:
            assert field == "a"
            return NonNumericColumn()

    class Table:
        def select(self, fields: tuple[str, ...]) -> SelectedTable:
            assert fields == ("a",)
            return SelectedTable()

    binding = VOIBinding(role="net_benefit", table_id="t", field_ids=("a",))
    bundle = SimpleNamespace(
        manifest=SimpleNamespace(bindings=(binding,)), table=lambda _: Table()
    )
    with pytest.raises(ValueError, match="not numeric"):
        prepare_analysis_inputs(bundle)


@pytest.mark.parametrize(
    ("provider", "name", "descriptor", "message"),
    [
        (
            CroissantProvider(),
            "croissant.json",
            {
                "@context": "mlcommons.org/croissant/1.1",
                "recordSet": [{"name": "samples", "field": []}],
            },
            "distribution",
        ),
        (
            CroissantProvider(),
            "croissant.json",
            {
                "@context": "mlcommons.org/croissant/1.1",
                "recordSet": [{"field": []}],
                "distribution": [{"contentUrl": "samples.csv"}],
            },
            "requires name",
        ),
        (
            CroissantProvider(),
            "croissant.json",
            {
                "@context": "mlcommons.org/croissant/1.1",
                "recordSet": [{"name": "samples", "field": [{"name": "missing"}]}],
                "distribution": [{"contentUrl": "samples.csv"}],
            },
            "exactly declare",
        ),
        (
            FrictionlessProvider(),
            "datapackage.json",
            {"resources": [{"path": "samples.csv", "schema": {"fields": []}}]},
            "requires name",
        ),
        (
            FrictionlessProvider(),
            "datapackage.json",
            {
                "resources": [
                    {
                        "name": "samples",
                        "path": "samples.csv",
                        "schema": {"fields": [{"name": "missing"}]},
                    }
                ]
            },
            "exactly declare",
        ),
    ],
)
def test_providers_reject_incomplete_or_mismatched_declarations(
    tmp_path, provider, name, descriptor, message
) -> None:
    _write_csv(tmp_path)
    path = tmp_path / name
    path.write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(IngestionError, match=message):
        provider.ingest(path, policy=SourceAccessPolicy(tmp_path))


def test_croissant_provider_rejects_unsupported_version_explicitly(tmp_path) -> None:
    _write_csv(tmp_path)
    descriptor_path = tmp_path / "croissant.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "@context": "https://mlcommons.org/croissant/1.0",
                "name": "unsupported-version",
                "distribution": [{"contentUrl": "samples.csv"}],
                "recordSet": [
                    {"name": "samples", "field": [{"name": "a"}, {"name": "b"}]}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(IngestionError, match="version 1.1"):
        CroissantProvider().ingest(descriptor_path, policy=SourceAccessPolicy(tmp_path))


@pytest.mark.parametrize(
    ("distribution", "message"),
    [
        ({"contentUrl": "samples.zip"}, "archives"),
        (
            {"contentUrl": "samples.csv", "transform": {"script": "x"}},
            "transformations",
        ),
    ],
)
def test_croissant_provider_rejects_unsupported_archives_and_transformations(
    tmp_path, distribution, message
) -> None:
    _write_csv(tmp_path)
    descriptor_path = tmp_path / "croissant.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "@context": "https://mlcommons.org/croissant/1.1",
                "name": "unsupported-profile-feature",
                "distribution": [distribution],
                "recordSet": [
                    {"name": "samples", "field": [{"name": "a"}, {"name": "b"}]}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(IngestionError, match=message):
        CroissantProvider().ingest(descriptor_path, policy=SourceAccessPolicy(tmp_path))


@pytest.mark.parametrize(
    (
        "descriptor_update",
        "distribution_update",
        "record_set_update",
        "field_update",
        "message",
    ),
    [
        (
            {"conformsTo": "http://mlcommons.org/croissant/1.0"},
            {},
            {},
            {},
            "conformsTo",
        ),
        ({}, {"encodingFormat": "application/json"}, {}, {}, "CSV media type"),
        ({}, {"contentChecksum": "00"}, {}, {}, "integrity declarations"),
        ({}, {}, {"key": ["a"]}, {}, "keys"),
        ({}, {}, {"split": "train"}, {}, "splits"),
        ({}, {}, {}, {"references": "other-table"}, "references"),
        ({}, {}, {}, {"subField": [{"name": "nested"}]}, "nested fields"),
        ({}, {}, {}, {"source": {"fileObject": "samples.csv"}}, "field sources"),
    ],
)
def test_croissant_provider_rejects_unsupported_profile_semantics_explicitly(
    tmp_path,
    descriptor_update,
    distribution_update,
    record_set_update,
    field_update,
    message,
) -> None:
    """Unsupported descriptor semantics must never be silently ignored."""
    _write_csv(tmp_path)
    descriptor_path = tmp_path / "croissant.json"
    field = {"name": "a", **field_update}
    record_set = {
        "name": "samples",
        "field": [field, {"name": "b"}],
        **record_set_update,
    }
    descriptor = {
        "@context": "https://mlcommons.org/croissant/1.1",
        "name": "unsupported-profile-semantics",
        "distribution": [{"contentUrl": "samples.csv", **distribution_update}],
        "recordSet": [record_set],
        **descriptor_update,
    }
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")

    with pytest.raises(IngestionError, match=message):
        CroissantProvider().ingest(descriptor_path, policy=SourceAccessPolicy(tmp_path))


def test_croissant_provider_preserves_governance_metadata_without_inference(
    tmp_path,
) -> None:
    """Governance descriptors remain inspectable but never change calculation input."""
    _write_csv(tmp_path)
    descriptor_path = tmp_path / "croissant.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "@context": "https://mlcommons.org/croissant/1.1",
                "@id": "https://example.invalid/datasets/posterior-samples",
                "name": "governed-ml-fixture",
                "citation": "Example et al. (2026)",
                "license": "CC-BY-4.0",
                "creator": [{"name": "ML team"}],
                "datePublished": "2026-01-01",
                "description": "Synthetic posterior samples.",
                "isAccessibleForFree": True,
                "keywords": ["VOI", "synthetic"],
                "odrl": {"permission": "use"},
                "provenance": {"wasGeneratedBy": "simulation"},
                "rai": {"risk": "low"},
                "sameAs": ["https://example.invalid/catalog/record"],
                "usageInfo": "Use only for deterministic tests.",
                "distribution": [{"contentUrl": "samples.csv"}],
                "recordSet": [
                    {"name": "samples", "field": [{"name": "a"}, {"name": "b"}]}
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = CroissantProvider().ingest(
        descriptor_path, policy=SourceAccessPolicy(tmp_path)
    )

    assert bundle.manifest.provenance.license == "CC-BY-4.0"
    assert bundle.manifest.provenance.citation == "Example et al. (2026)"
    assert bundle.manifest.extensions == {
        "mlcommons.org:croissant-governance": {
            "@id": "https://example.invalid/datasets/posterior-samples",
            "citation": "Example et al. (2026)",
            "creator": ({"name": "ML team"},),
            "datePublished": "2026-01-01",
            "description": "Synthetic posterior samples.",
            "isAccessibleForFree": True,
            "keywords": ("VOI", "synthetic"),
            "license": "CC-BY-4.0",
            "odrl": {"permission": "use"},
            "provenance": {"wasGeneratedBy": "simulation"},
            "rai": {"risk": "low"},
            "sameAs": ("https://example.invalid/catalog/record",),
            "usageInfo": "Use only for deterministic tests.",
        }
    }


@pytest.mark.parametrize(
    ("checksum", "message"),
    [
        (None, None),
        ("0" * 64, "SHA-256"),
    ],
)
def test_croissant_provider_validates_declared_local_sha256(
    tmp_path, checksum, message
) -> None:
    """A declared local FileObject checksum must be verified before parsing."""
    _write_csv(tmp_path)
    descriptor_path = tmp_path / "croissant.json"
    distribution = {"contentUrl": "samples.csv"}
    if checksum is None:
        distribution["sha256"] = digest_file(tmp_path / "samples.csv")
    else:
        distribution["sha256"] = checksum
    descriptor_path.write_text(
        json.dumps(
            {
                "@context": "https://mlcommons.org/croissant/1.1",
                "name": "checksum-fixture",
                "distribution": [distribution],
                "recordSet": [
                    {"name": "samples", "field": [{"name": "a"}, {"name": "b"}]}
                ],
            }
        ),
        encoding="utf-8",
    )

    if message is not None:
        with pytest.raises(IngestionError, match=message):
            CroissantProvider().ingest(
                descriptor_path, policy=SourceAccessPolicy(tmp_path)
            )
    else:
        assert (
            CroissantProvider()
            .ingest(descriptor_path, policy=SourceAccessPolicy(tmp_path))
            .table("samples")
            .num_rows
            == 2
        )


def test_croissant_provider_preserves_non_checksum_ingestion_error(tmp_path) -> None:
    """Only checksum failures are translated into Croissant integrity diagnostics."""
    descriptor_path = tmp_path / "croissant.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "@context": "https://mlcommons.org/croissant/1.1",
                "distribution": [{"contentUrl": "missing.csv", "sha256": "0" * 64}],
                "recordSet": [
                    {"name": "samples", "field": [{"name": "a"}, {"name": "b"}]}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(IngestionError, match="declared resource does not exist"):
        CroissantProvider().ingest(descriptor_path, policy=SourceAccessPolicy(tmp_path))


@pytest.mark.parametrize(
    ("resource_update", "message"),
    [
        ({"checksum": "unsupported"}, "integrity declarations"),
        ({"hash": 42}, "hash must be a SHA-256"),
        ({"bytes": True}, "bytes must be a non-negative"),
    ],
)
def test_frictionless_provider_rejects_unsupported_integrity_declarations(
    tmp_path, resource_update, message
) -> None:
    """Only SHA-256 hash and integer byte-size declarations are accepted."""
    _write_csv(tmp_path)
    descriptor_path = tmp_path / "datapackage.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "name": "samples",
                        "path": "samples.csv",
                        "schema": {"fields": [{"name": "a"}, {"name": "b"}]},
                        **resource_update,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(IngestionError, match=message):
        FrictionlessProvider().ingest(
            descriptor_path, policy=SourceAccessPolicy(tmp_path)
        )


def test_dataframe_adapter_rejects_non_dataframe() -> None:
    with pytest.raises(ValueError, match="dataframe interchange"):
        from_dataframe(object(), dataset_id="bad")
