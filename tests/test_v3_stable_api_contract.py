"""Executable governance checks for the normative v3 public API contract."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from scripts import validate_core_api_contract as validator

ROOT = Path(__file__).resolve().parents[1]
DECISION_RECORD_PATH = ROOT / "specs/core-api/decision-record.md"
FOUNDATION_PATH = ROOT / "specs/core-api/foundation.md"
CONTRACT_INDEX_PATH = ROOT / "specs/core-api/contract-index.md"
EXTENSION_EVOLUTION_PATH = ROOT / "specs/core-api/extension-evolution.md"
V1_CONTRACT_PATH = ROOT / "specs/v1/stable-api.json"
V2_CONTRACT_PATH = ROOT / "specs/v2/stable-api.json"
V3_CONTRACT_PATH = ROOT / "specs/v3/stable-api.json"
V1_POLICY_PATH = ROOT / "specs/v1/compatibility-policy.json"
V2_POLICY_PATH = ROOT / "specs/v2/compatibility-policy.json"
V3_POLICY_PATH = ROOT / "specs/v3/compatibility-policy.json"
VALUE_ARRAY_EXAMPLE_PATH = ROOT / "specs/core-api/examples/v1/value-array.example.json"
PARAMETER_SET_EXAMPLE_PATH = (
    ROOT / "specs/core-api/examples/v1/parameter-set.example.json"
)
NORMALIZED_BUNDLE_SCHEMA_PATH = (
    ROOT / "specs/core-api/schemas/v2/normalized-input-bundle.schema.json"
)
MIGRATION_GUIDE_PATH = (
    ROOT / "docs/astro-site/src/content/docs/user-guide/migration-guide.mdx"
)

FROZEN_V1_CONTRACT_SHA256 = (
    "ff77ce326e29d17354dbe274b30d42e0ed89ae80efb7700bba94eef8ae68249d"
)
FROZEN_V1_POLICY_SHA256 = (
    "3027c936cfced144113ada048b42c1ac9e461d9634849b7a45add618946bce8a"
)
FROZEN_V2_CONTRACT_SHA256 = (
    "5cfe375b551a10d1764fd9a18f4ddf359409a58d9b5879456a91db82b9f71fb3"
)
FROZEN_V2_POLICY_SHA256 = (
    "d7a66a41b652f5681ed5f4a5f43193e069a45d1c872189588b6e4ac656596985"
)
FROZEN_VALUE_ARRAY_EXAMPLE_SHA256 = (
    "f354303ba6566d0578fb8a429d70edb23dbdd7fa1daed9a5500d77e7cd51aa62"
)
FROZEN_PARAMETER_SET_EXAMPLE_SHA256 = (
    "6e2a2de7df1acac10c43a6e421b2a4e69690530170d0c0594d16dfec25038e4e"
)
REMOVED_V3_SYMBOLS = (
    "ValueArray.dataset",
    "ParameterSet.dataset",
    "from_dataset",
    "to_dataset",
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_v3_authority_revision_records_polars_tabular_model_and_xarray_removal() -> (
    None
):
    """AC1: Normative core-api documents record Polars as the v3 tabular model."""
    decision = DECISION_RECORD_PATH.read_text(encoding="utf-8")
    foundation = FOUNDATION_PATH.read_text(encoding="utf-8")

    assert "Xarray-labeled arrays are the Python in-memory reference model." not in (
        decision
    )
    assert (
        "Polars is adapter-only tabular tooling, not the canonical computational model."
        not in decision
    )
    assert "Polars DataFrames are the Python in-memory reference model" in decision
    assert "3.0.0" in decision
    assert "2026-10-26" in decision
    assert "NumPy is the reference execution baseline." in decision
    assert "JAX is optional acceleration, not public contract surface." in decision
    assert "Arrow/Parquet is the cross-language interchange boundary." in decision

    assert (
        "Polars remains adapter-only tabular tooling and is not the canonical compute model."
        not in foundation
    )
    assert (
        "Xarray-labeled arrays, NumPy, optional JAX, Arrow/Parquet, and adapter-only Polars are fixed infrastructure choices."
        not in foundation
    )
    assert "Polars" in foundation
    assert "3.0.0" in foundation


def test_v3_contract_is_a_major_version_boundary_and_preserves_v1_v2_digests() -> None:
    """AC2: v3 supersedes v2 with removed xarray symbols while v1/v2 remain frozen."""
    assert _sha256(V1_CONTRACT_PATH) == FROZEN_V1_CONTRACT_SHA256
    assert _sha256(V1_POLICY_PATH) == FROZEN_V1_POLICY_SHA256
    assert _sha256(V2_CONTRACT_PATH) == FROZEN_V2_CONTRACT_SHA256
    assert _sha256(V2_POLICY_PATH) == FROZEN_V2_POLICY_SHA256

    v2 = _json(V2_CONTRACT_PATH)
    v3 = _json(V3_CONTRACT_PATH)

    assert v3["contract_version"] == "3.0.0"
    assert v3["status"] == "normative"
    assert v3["supersedes"] == "../v2/stable-api.json"
    assert v3["release_gate"] == "runtime-and-binding-conformance-required"
    assert v3["numerics"] == v2["numerics"]
    assert v3["errors"] == v2["errors"]
    assert v3["methods"] == v2["methods"]
    assert v3["surfaces"] == v2["surfaces"]

    symbols = v3["symbols"]
    assert isinstance(symbols, dict)
    assert symbols["stable"] == v2["symbols"]["stable"]
    assert symbols["provisional"] == v2["symbols"]["provisional"]
    assert symbols["experimental"] == v2["symbols"]["experimental"]
    assert list(REMOVED_V3_SYMBOLS) == symbols["removed"]
    for removed_symbol in REMOVED_V3_SYMBOLS:
        assert removed_symbol not in symbols["deprecated"]

    transition = v3["v2_to_v3"]
    assert isinstance(transition, dict)
    assert transition["classification"] == "major-breaking-change"
    assert transition["v2_contract"] == "../v2/stable-api.json"
    assert transition["removed_symbols"] == list(REMOVED_V3_SYMBOLS)
    assert transition["replacement_surface"] == {
        "frame": "polars.DataFrame",
        "from_polars": "classmethod",
        "to_polars": "method",
        "values": "numpy.ndarray",
    }
    assert (
        transition["migration_guide"]
        == "../../docs/astro-site/src/content/docs/user-guide/migration-guide.mdx"
    )


def test_v3_policy_records_major_breaking_tabular_change() -> None:
    """AC2 & AC5: v3 compatibility policy preserves semver rules and records non-goals."""
    v2_policy = _json(V2_POLICY_PATH)
    v3_policy = _json(V3_POLICY_PATH)

    assert v3_policy["policy_version"] == "3.0.0"
    assert v3_policy["status"] == "normative"
    assert v3_policy["applies_from"] == "3.0.0"
    assert v3_policy["supersedes"] == "../v2/compatibility-policy.json"
    assert v3_policy["semantic_versioning"] == v2_policy["semantic_versioning"]
    assert v3_policy["surfaces"] == v2_policy["surfaces"]
    assert v3_policy["deprecation"] == v2_policy["deprecation"]
    assert v3_policy["excluded_stabilities"] == v2_policy["excluded_stabilities"]
    assert v3_policy["promotion_requirements"] == v2_policy["promotion_requirements"]
    assert v3_policy["compatibility_evidence"] == v2_policy["compatibility_evidence"]

    v2_to_v3 = v3_policy["v2_to_v3"]
    assert isinstance(v2_to_v3, dict)
    assert v2_to_v3["classification"] == "major-breaking-change"
    assert v2_to_v3["removed_symbols"] == list(REMOVED_V3_SYMBOLS)
    assert v2_to_v3["non_goals"] == [
        "no-new-persistence-format",
        "no-wire-contract-change",
        "no-binding-surface-change",
        "no-algorithm-rng-or-numerical-change",
    ]


def test_v3_schema_vocabulary_reuses_core_api_identifiers_and_preserves_examples() -> (
    None
):
    """AC3: Polars container contract reuses existing schema vocabulary and fixtures."""
    assert _sha256(VALUE_ARRAY_EXAMPLE_PATH) == FROZEN_VALUE_ARRAY_EXAMPLE_SHA256
    assert _sha256(PARAMETER_SET_EXAMPLE_PATH) == FROZEN_PARAMETER_SET_EXAMPLE_SHA256

    validator._validate_example(
        VALUE_ARRAY_EXAMPLE_PATH,
        ROOT / "specs/core-api/schemas/v1/value-array.schema.json",
    )
    validator._validate_example(
        PARAMETER_SET_EXAMPLE_PATH,
        ROOT / "specs/core-api/schemas/v1/parameter-set.schema.json",
    )

    bundle_schema = _json(NORMALIZED_BUNDLE_SCHEMA_PATH)
    binding_props = bundle_schema["$defs"]["binding"]["properties"]
    assert "net_benefit" in binding_props["role"]["enum"]
    assert "parameter" in binding_props["role"]["enum"]
    assert "wide" in binding_props["layout"]["enum"]
    assert "table_id" in binding_props
    assert "field_ids" in binding_props

    v3 = _json(V3_CONTRACT_PATH)
    containers = v3["tabular_containers"]
    assert isinstance(containers, dict)
    assert containers["vocabulary_source"] == (
        "../core-api/schemas/v2/normalized-input-bundle.schema.json"
    )
    assert containers["ValueArray"] == {
        "table_id": "value_array",
        "role": "net_benefit",
        "layout": "wide",
        "dtype": "float64",
        "field_id_source": "strategy_names",
        "multi_perspective_dtype": "array[float64, n_perspectives]",
        "multi_perspective_metadata": "perspective_names",
        "polars_surface": ["frame", "from_polars", "to_polars"],
        "values_return_type": "numpy.ndarray",
    }
    assert containers["ParameterSet"] == {
        "table_id": "parameter_set",
        "role": "parameter",
        "layout": "wide",
        "dtype": "float64",
        "field_id_source": "parameter_names",
        "polars_surface": ["frame", "from_polars", "to_polars"],
    }


def test_v3_migration_guide_and_core_api_docs_cover_boundary_and_non_goals() -> None:
    """AC4 & AC5: Migration guide, contract index, and extension rules document v3."""
    migration = MIGRATION_GUIDE_PATH.read_text(encoding="utf-8")
    contract_index = CONTRACT_INDEX_PATH.read_text(encoding="utf-8")
    extension_evolution = EXTENSION_EVOLUTION_PATH.read_text(encoding="utf-8")

    for required in (
        "## Migrating from 2.x to 3.0",
        "`ValueArray.dataset`",
        "`ParameterSet.dataset`",
        "`from_dataset`",
        "`to_dataset`",
        "`from_polars`",
        "`to_polars`",
        "`ValueArray.frame`",
        "`ParameterSet.frame`",
        "`ValueArray.values`",
        "`numpy_values`",
        "`ParameterSet.parameters`",
        "`ValueArray.from_numpy`",
        "`ValueArray.from_numpy_perspectives`",
        "`__dataframe__`",
        "`__arrow_c_stream__`",
        "no longer installed transitively",
    ):
        assert required in migration

    assert "../v3/stable-api.json" in contract_index
    assert "../v3/compatibility-policy.json" in contract_index
    assert "../v3/compatibility-policy.json" in extension_evolution
