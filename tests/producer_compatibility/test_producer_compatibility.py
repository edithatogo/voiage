"""Caller-owned producer compatibility; these libraries are test tooling only."""

from __future__ import annotations

import builtins
from contextlib import contextmanager
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


def test_fresh_consumer_import_and_conversion_keeps_legacy_imports_external(
    frozen: dict[str, Any],
) -> None:
    # The caller loads its producer first. Voiage must not request either legacy import.
    code = """
import builtins, importlib, json, pathlib, sys
import pandas as pd
import polars as pl
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
