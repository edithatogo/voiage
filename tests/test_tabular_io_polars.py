"""Behavioral parity for the bounded T1.2 tabular migration."""

from __future__ import annotations

from contextlib import nullcontext
import io
import json
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
lazy from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from voiage import _tabular_io
from voiage.ecosystem_integration import DataFormatConnector, _load_tabular_artifact
from voiage.widgets.voi_widgets import VOIAnalysisWidget


@pytest.mark.parametrize(
    "text",
    [
        "treatment,cost,utility\nDrug A,10,0.8\nDrug B,,NA\n",
        "name,value\nalpha,1\nbeta,text\n",
        "name,flag\na,True\nb,False\n",
        "a,b\n",
        "a,a,\n1,2,3\n",
        'name,value\n"quoted, name",1e-09\nUnicode é,2\n',
    ],
)
def test_csv_payload_matches_previous_reader(tmp_path: Path, text: str) -> None:
    path = tmp_path / "data.csv"
    path.write_text(text)
    legacy = pd.read_csv(path)
    result = DataFormatConnector().import_health_data(str(path), "generic")
    assert result["columns"] == list(legacy.columns)
    assert result["shape"] == legacy.shape
    assert result["missing_values"] == legacy.isnull().sum().to_dict()
    for actual, expected in zip(result["data"], legacy.to_dict("records"), strict=True):
        for name, value in expected.items():
            if pd.isna(value):
                assert pd.isna(actual[name])
            else:
                assert actual[name] == value
                assert type(actual[name]) is type(value)


def test_parquet_index_and_payload(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    legacy = pd.DataFrame(
        {"cost": [1.0, float("nan")], "name": ["a", "b"]},
        index=pd.Index([7, 8], name="record"),
    )
    legacy.to_parquet(path)
    frame = _load_tabular_artifact(path)
    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == ["cost", "name"]
    result = DataFormatConnector().import_health_data(str(path))
    assert result["shape"] == (2, 2)
    assert result["missing_values"] == {"cost": 1, "name": 0}


def test_csv_serializer_quoting_and_empty(tmp_path: Path) -> None:
    for index, data in enumerate(
        [{}, {"name": ["", 'a,"b\n'], "value": [float("nan"), 1e-9]}]
    ):
        path = tmp_path / f"{index}.csv"
        _tabular_io.write_csv(pl.DataFrame(data), path)
        assert path.read_text() == pd.DataFrame(data).to_csv(index=False)


@pytest.mark.parametrize("kind", ["net_benefits", "parameters"])
def test_widget_csv_without_pandas_reader(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("pandas reader was called")

    monkeypatch.setattr(pd, "read_csv", forbidden)
    widget = object.__new__(VOIAnalysisWidget)
    widget.status_label = SimpleNamespace(value="")
    widget.output_area = nullcontext()
    widget.net_benefits = None
    widget.parameters = None
    handler = getattr(widget, f"_on_{kind}_upload")
    handler({"new": {"upload": {"name": "data.csv", "content": b"a,b\n1,2\n3,4\n"}}})
    assert widget.status_label.value.endswith("loaded from data.csv")
    loaded = getattr(widget, kind)
    assert loaded is not None
    if kind == "net_benefits":
        np.testing.assert_array_equal(loaded.values, [[1.0, 2.0], [3.0, 4.0]])
    handler({"new": {"upload": {"name": "broken.csv", "content": b"a,b\nwrong,2\n"}}})
    assert widget.status_label.value == f"Error loading {kind.replace('_', ' ')}"
    assert getattr(widget, kind) is loaded


def test_csv_connector_does_not_use_pandas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("legacy reader")

    monkeypatch.setattr(pd, "read_csv", forbidden)
    path = tmp_path / "data.csv"
    path.write_text("cost\n1\n2\n")
    assert DataFormatConnector().import_health_data(str(path))["data_type"] == "costs"


def test_empty_csv_fails_soft(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.touch()
    with pytest.warns(UserWarning, match="Error importing health data: No columns"):
        assert DataFormatConnector().import_health_data(str(path)) == {}


def test_parquet_without_pandas_metadata(tmp_path: Path) -> None:
    path = tmp_path / "native.parquet"
    pl.DataFrame({"sample_index": [0, 1], "cost": [1.0, 2.0]}).write_parquet(path)
    assert _load_tabular_artifact(path).columns == ["sample_index", "cost"]


def test_csv_header_collision(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.csv"
    path.write_text("a,a,a.1,\n1,2,3,4\n")
    assert _tabular_io.read_csv(path).columns == list(pd.read_csv(path).columns)


def test_migrated_paths_with_pandas_unavailable() -> None:
    """Qualify only T1.2 paths; this is not the later package dependency gate."""
    script = r"""
import importlib.abc
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from contextlib import nullcontext
import json

class NoPandas(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pandas' or fullname.startswith('pandas.'):
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, NoPandas())
from voiage.ecosystem_integration import DataFormatConnector, _load_tabular_artifact
from voiage.widgets.voi_widgets import VOIAnalysisWidget
import polars as pl
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    csv_path=root/'input.csv'
    csv_path.write_text('cost,utility\n1,0.8\n2,0.9\n')
    assert DataFormatConnector().import_health_data(str(csv_path))['shape']==(2,2)
    parquet_path=root/'input.parquet'
    pl.DataFrame({'a':[1.,2.]}).write_parquet(parquet_path)
    assert _load_tabular_artifact(parquet_path).shape==(2,1)
    for kind in ['net_benefits','parameters']:
        widget=object.__new__(VOIAnalysisWidget)
        widget.status_label=SimpleNamespace(value='')
        widget.output_area=nullcontext()
        getattr(widget,f'_on_{kind}_upload')({'new':{'input':{'name':'input.csv','content':csv_path.read_bytes()}}})
        assert getattr(widget,kind) is not None
assert not any(name=='pandas' or name.startswith('pandas.') for name in sys.modules)
print('CSV, Parquet and widget paths passed with pandas imports blocked')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "text",
    [
        "a,b\n1,2\n\n3,4\n",
        "a,b\n1,2\n   \n3,4\n",
        'a,b\n"quoted\nline",2\n\n,\n',
        ",Unnamed: 0\n1,2\n",
    ],
)
def test_csv_review_regressions(tmp_path: Path, text: str) -> None:
    path = tmp_path / "review.csv"
    path.write_text(text)
    expected = pd.read_csv(path)
    result = DataFormatConnector().import_health_data(str(path), "generic")
    assert result["columns"] == list(expected.columns)
    assert result["shape"] == expected.shape
    assert result["missing_values"] == expected.isnull().sum().to_dict()
    for actual, previous in zip(
        result["data"], expected.to_dict("records"), strict=True
    ):
        for name, value in previous.items():
            assert pd.isna(actual[name]) if pd.isna(value) else actual[name] == value


def test_export_infers_late_fractional_values(tmp_path: Path) -> None:
    from voiage.health_economics import HealthEconomicsAnalysis, HealthState

    analysis = HealthEconomicsAnalysis()
    for index in range(101):
        analysis.add_health_state(
            HealthState(
                state_id=str(index),
                description="state",
                utility=0.8,
                cost=1 if index < 100 else 1.5,
                duration=1.0,
            )
        )
    path = tmp_path / "export.json"
    DataFormatConnector().export_health_data(analysis, str(path), format_type="json")
    assert json.loads(path.read_text())["health_states"][-1]["cost"] == 1.5


def test_csv_preserves_bom_and_quoted_carriage_returns(tmp_path: Path) -> None:
    text = '\ufeffname,value\r\n"line\rreturn",1\r\n'
    path = tmp_path / "bom.csv"
    path.write_bytes(text.encode("utf-8"))
    expected = pd.read_csv(path).to_dict("records")
    assert _tabular_io.records(_tabular_io.read_csv(path)) == expected
    assert _tabular_io.records(_tabular_io.read_csv(io.StringIO(text))) == expected


def test_malformed_csv_fails_soft(tmp_path: Path) -> None:
    path = tmp_path / "malformed.csv"
    path.write_text('name,cost\n"unterminated,1\n')
    with pytest.warns(UserWarning, match="Error importing health data"):
        assert DataFormatConnector().import_health_data(str(path)) == {}
