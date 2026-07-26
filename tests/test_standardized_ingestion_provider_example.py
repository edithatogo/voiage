"""Consumer-style evidence for the documented third-party provider surface."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from voiage.ingestion import ProviderRegistry, SourceAccessPolicy


def test_example_provider_uses_only_the_public_provider_contract(tmp_path) -> None:
    (tmp_path / "data.csv").write_text("value\n1\n", encoding="utf-8")
    descriptor = tmp_path / "example.json"
    descriptor.write_text(
        '{"voiage_example": "1", "resource": "data.csv"}', encoding="utf-8"
    )
    source = (
        Path(__file__).parents[1]
        / "examples"
        / "standardized_ingestion"
        / "example_provider.py"
    )
    spec = importlib.util.spec_from_file_location("example_provider", source)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    bundle = ProviderRegistry((module.provider,)).ingest(
        descriptor, policy=SourceAccessPolicy(tmp_path)
    )

    assert bundle.manifest.provenance.provider_id == "example-csv"
    assert bundle.table("data").to_pylist() == [{"value": 1}]
