"""Built-in descriptor adapters stay isolated from the conductor contract."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import polars as pl
import pyarrow as pa
import pytest

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
    ProviderCapabilities,
    SourceAccessPolicy,
    default_registry,
    discover_entry_point_providers,
    from_dataframe,
)
from voiage.ingestion._tabular import digest_file, read_csv
from voiage.ingestion.croissant import CroissantProvider
from voiage.ingestion.frictionless import FrictionlessProvider
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


def test_dataframe_interchange_adapter_does_not_require_a_specific_frame_library() -> (
    None
):
    bundle = from_dataframe(
        pl.DataFrame({"a": [1.0], "b": [2.0]}), dataset_id="business"
    )

    assert bundle.manifest.provenance.provider_id == "dataframe-interchange"
    assert bundle.table("data").column_names == ["a", "b"]


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
            "exactly one resource",
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


def test_dataframe_adapter_rejects_non_dataframe() -> None:
    with pytest.raises(ValueError, match="dataframe interchange"):
        from_dataframe(object(), dataset_id="bad")
