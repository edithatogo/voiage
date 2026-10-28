"""Caller-owned producer compatibility; these libraries are test tooling only."""

from __future__ import annotations

import builtins
from contextlib import contextmanager
from datetime import UTC, datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from voiage.contracts import VOIBinding, prepare_analysis_inputs
from voiage.ingestion import from_dataframe

if TYPE_CHECKING:
    from collections.abc import Iterator

FIXTURE = Path(__file__).with_name("producer.json")


@contextmanager
def _watch_runtime_imports() -> Iterator[None]:
    """Reject legacy import requests originating in Voiage, not producer code."""
    original = builtins.__import__
    original_module = importlib.import_module

    def check(name: str, caller: dict[str, Any]) -> None:
        if name.split(".", 1)[0] in {"pandas", "xarray"}:
            module = caller.get("__name__", "")
            if module == "voiage" or module.startswith("voiage."):
                raise AssertionError(f"Voiage requested legacy import: {name}")

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        caller_globals = kwargs.get("globals", args[0] if args else {}) or {}
        check(name, caller_globals)
        return original(name, *args, **kwargs)

    def guarded_module(name: str, package: str | None = None) -> Any:
        check(name, sys._getframe(1).f_globals)
        return original_module(name, package)

    builtins.__import__ = guarded
    importlib.import_module = guarded_module
    try:
        yield
    finally:
        builtins.__import__ = original
        importlib.import_module = original_module


def _frame(kind: str, columns: dict[str, list[float]]) -> Any:
    if kind == "pandas":
        return pd.DataFrame(
            columns, index=pd.Index([101, 102, 103], name="source_index")
        )
    return pl.DataFrame(columns)


def _bundle(frame: object, fixture: dict[str, Any], *, allow_copy: bool = True) -> Any:
    with _watch_runtime_imports():
        return from_dataframe(
            frame,
            dataset_id=fixture["dataset_id"],
            table_id=fixture["table_id"],
            bindings=(
                VOIBinding(
                    role="net_benefit",
                    table_id=fixture["table_id"],
                    field_ids=tuple(fixture["columns"]),
                    strategy_names=("A", "B"),
                ),
            ),
            allow_copy=allow_copy,
        )


@pytest.fixture
def frozen() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


@pytest.mark.parametrize("kind", ["pandas", "polars"])
def test_real_producer_preserves_rows_binding_and_index_boundary(
    kind: str, frozen: dict[str, Any]
) -> None:
    frame = _frame(kind, frozen["columns"])
    bundle = _bundle(frame, frozen)
    assert bundle.table("samples").to_pylist() == frozen["rows"]
    assert bundle.table("samples").column_names == list(frozen["columns"])
    assert prepare_analysis_inputs(bundle).net_benefits.numpy_values.tolist() == [
        [10.0, 20.0],
        [30.0, 10.0],
        [20.0, 25.0],
    ]
    expected = frozen["expected"][kind]
    assert bundle.schema_fingerprint == expected["schema_fingerprint"]
    assert bundle.manifest.provenance.descriptor_digest == expected["descriptor_digest"]
    assert (
        hashlib.sha256(bundle.canonical_json.encode()).hexdigest()
        == expected["canonical_manifest_sha256"]
    )
    details = bundle.manifest.model_dump(mode="json")["extensions"][
        "voiage.dev:dataframe-interchange"
    ]
    assert details["copy_policy"] == "allow_copy"
    assert details["copy_outcome"] == "not_observable"
    assert details["index_policy"] == "excluded_by_dataframe_interchange_protocol"
    assert details["conversion_protocol"] == (
        "dataframe_interchange_fallback" if kind == "pandas" else "arrow_py_capsule"
    )


def test_polars_capsule_shares_real_numeric_buffers(frozen: dict[str, Any]) -> None:
    frame = _frame("polars", frozen["columns"])
    reference = frame.to_arrow()
    actual = _bundle(frame, frozen).table("samples")
    for name in reference.column_names:
        source = reference.column(name).chunk(0)
        target = actual.column(name).chunk(0)
        assert target.equals(source)
        assert target.buffers()[1].address == source.buffers()[1].address
        assert target.buffers()[1].size == source.buffers()[1].size
    del frame, reference
    assert actual.to_pylist() == frozen["rows"]


def test_arrow_table_supports_strict_no_copy(frozen: dict[str, Any]) -> None:
    frame = pa.table(frozen["columns"])
    bundle = _bundle(frame, frozen, allow_copy=False)
    for name in frame.column_names:
        assert (
            bundle.table("samples").column(name).chunk(0).buffers()[1].address
            == frame.column(name).chunk(0).buffers()[1].address
        )
    assert bundle.manifest.diagnostics[0].code == "dataframe_interchange.copy.zero_copy"


@pytest.mark.parametrize("kind", ["pandas", "polars"])
def test_strict_copy_policy_keeps_existing_rejection(
    kind: str, frozen: dict[str, Any]
) -> None:
    with pytest.raises(ValueError) as error:
        _bundle(_frame(kind, frozen["columns"]), frozen, allow_copy=False)
    assert str(error.value) == frozen["rejection"]
    assert error.value.__cause__ is not None


@pytest.mark.parametrize(
    "bad", [None, object(), {"strategy_a": [1.0]}, pl.DataFrame({"nested": [[1, 2]]})]
)
def test_nonconforming_producer_retains_diagnostic(
    bad: object, frozen: dict[str, Any]
) -> None:
    with pytest.raises(ValueError) as error:
        _bundle(bad, frozen)
    assert str(error.value) == frozen["rejection"]


@pytest.mark.parametrize(
    "statement", ["import pandas", "import xarray", "importlib.import_module('pandas')"]
)
def test_import_watcher_rejects_runtime_regression(statement: str) -> None:
    with (
        _watch_runtime_imports(),
        pytest.raises(AssertionError, match="Voiage requested legacy import"),
    ):
        exec(  # noqa: S102 - fixed regression snippets, no external input
            statement, {"__name__": "voiage.regression", "importlib": importlib}
        )


def test_import_watcher_allows_external_producer_ownership() -> None:
    with _watch_runtime_imports():
        module = builtins.__import__("pandas", {"__name__": "caller_owned_producer"})
    assert module is pd


def test_fresh_consumer_import_and_conversion_keeps_legacy_imports_external() -> None:
    # The caller loads its producer first. Voiage must not request either legacy import.
    code = """
import builtins, importlib, json, pathlib, sys
import pandas as pd
import polars as pl
original_module = importlib.import_module
def guarded_module(name, package=None):
    owner = sys._getframe(1).f_globals.get('__name__', '')
    if name.split('.',1)[0] in {'pandas','xarray'} and (owner == 'voiage' or owner.startswith('voiage.')):
        raise AssertionError('Voiage requested legacy import: ' + name)
    return original_module(name, package)
importlib.import_module = guarded_module
original = builtins.__import__
def guarded(name, globals=None, *args, **kwargs):
    owner = (globals or {}).get('__name__', '')
    if name.split('.',1)[0] in {'pandas','xarray'} and (owner == 'voiage' or owner.startswith('voiage.')):
        raise AssertionError('Voiage requested legacy import: ' + name)
    return original(name, globals, *args, **kwargs)
builtins.__import__ = guarded
import voiage
from voiage.ingestion import from_dataframe
expected = pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(voiage.__file__).resolve().is_relative_to(expected)
for frame in [pd.DataFrame({'value':[1,2]}), pl.DataFrame({'value':[1,2]})]:
    assert from_dataframe(frame, dataset_id='external').table('data').to_pylist() == [{'value':1},{'value':2}]
assert not any(n.split('.',1)[0] == 'xarray' for n in sys.modules)
print(json.dumps({'python':sys.version.split()[0], 'package':voiage.__file__, 'legacy_runtime_imports':[]}))
"""
    import voiage

    prefix = os.environ.get(
        "VOIAGE_PRODUCER_PREFIX", str(Path(voiage.__file__).parents[1])
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, prefix],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["python"] == "3.15.1"


@pytest.fixture(autouse=True)
def qualified_interpreter_and_install_location() -> None:
    """The dedicated lane must consume its own wheel in this process too."""
    import voiage

    assert sys.version_info[:3] == (3, 15, 1)
    prefix = os.environ.get("VOIAGE_PRODUCER_PREFIX")
    if prefix is not None:
        assert Path(voiage.__file__).resolve().is_relative_to(Path(prefix).resolve())


@pytest.mark.parametrize("kind", ["pandas", "polars"])
def test_nullable_category_and_timezone_survive_external_handoff(kind: str) -> None:
    """Keep each real producer's supported null/category/time semantics."""
    instant = datetime(2026, 1, 1, tzinfo=UTC)
    if kind == "pandas":
        frame = pd.DataFrame(
            {
                "tier": pd.Series(["standard", None], dtype="category"),
                "cost": pd.Series([10, None], dtype="Int64"),
                "observed_at": pd.Series(pd.to_datetime([instant, None])),
            }
        )
        frame.index = pd.Index([101, 102], name="source_index")
    else:
        frame = pl.DataFrame(
            {
                "tier": pl.Series(["standard", None], dtype=pl.Categorical),
                "cost": pl.Series([10, None], dtype=pl.Int64),
                "observed_at": pl.Series(
                    [instant, None], dtype=pl.Datetime(time_zone="UTC")
                ),
            }
        )
    with _watch_runtime_imports():
        bundle = from_dataframe(frame, dataset_id="nullable-producer")
    assert bundle.table("data").to_pylist() == [
        {"tier": "standard", "cost": 10, "observed_at": instant},
        {"tier": None, "cost": None, "observed_at": None},
    ]
    details = bundle.manifest.model_dump(mode="json")["extensions"][
        "voiage.dev:dataframe-interchange"
    ]
    fields = {item["field_id"]: item for item in details["field_decisions"]}
    assert fields["tier"]["categorical"] is True
    assert fields["observed_at"]["timezone"] == "UTC"
    assert "source_index" not in bundle.table("data").column_names


@pytest.mark.parametrize("kind", ["pandas", "polars"])
def test_external_empty_typed_frame_retains_schema(kind: str) -> None:
    """Zero rows must not silently discard caller-declared columns."""
    frame = (
        pd.DataFrame({"value": pd.Series([], dtype="float64")})
        if kind == "pandas"
        else pl.DataFrame({"value": pl.Series([], dtype=pl.Float64)})
    )
    with _watch_runtime_imports():
        bundle = from_dataframe(frame, dataset_id="empty-producer")
    assert bundle.table("data").num_rows == 0
    assert bundle.table("data").schema == pa.schema([("value", pa.float64())])


@pytest.mark.parametrize("kind", ["pandas", "polars"])
def test_sliced_external_producer_preserves_selected_rows(
    kind: str, frozen: dict[str, Any]
) -> None:
    """Sliced caller buffers may require copying, but cannot change rows."""
    columns = {
        name: [item for value in values for item in (value, 99.0)]
        for name, values in frozen["columns"].items()
    }
    if kind == "pandas":
        frame = pd.DataFrame(columns)
        frame.index = pd.Index(range(6), name="source_index")
        sliced = frame.iloc[::2]
    else:
        frame = pl.DataFrame(columns)
        sliced = frame[::2]
    assert _bundle(sliced, frozen).table("samples").to_pylist() == frozen["rows"]
