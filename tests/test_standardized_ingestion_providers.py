"""Built-in descriptor adapters stay isolated from the conductor contract."""

from __future__ import annotations

import json

import polars as pl
import pytest

from voiage.ingestion import (
    IngestionError,
    SourceAccessPolicy,
    default_registry,
    from_dataframe,
)


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
                "recordSet": [{"name": "samples", "field": [{"name": "a"}, {"name": "b"}]}],
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
def test_built_in_providers_normalize_supported_csv_profile(tmp_path, name, descriptor) -> None:
    _write_csv(tmp_path)
    descriptor_path = tmp_path / name
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")

    bundle = default_registry().ingest(descriptor_path)

    assert bundle.manifest.dataset_id in {"ml-fixture", "operations-fixture"}
    assert bundle.table("samples").column_names == ["a", "b"]


def test_source_policy_blocks_path_traversal_and_network(tmp_path) -> None:
    policy = SourceAccessPolicy(tmp_path)

    with pytest.raises(IngestionError, match="escapes"):
        policy.resolve("../outside.csv")
    with pytest.raises(IngestionError, match="network"):
        policy.resolve("https://example.invalid/input.csv")


def test_dataframe_interchange_adapter_does_not_require_a_specific_frame_library() -> None:
    bundle = from_dataframe(pl.DataFrame({"a": [1.0], "b": [2.0]}), dataset_id="business")

    assert bundle.manifest.provenance.provider_id == "dataframe-interchange"
    assert bundle.table("data").column_names == ["a", "b"]
