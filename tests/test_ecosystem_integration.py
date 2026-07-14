"""Deterministic tests for ecosystem connector import/export helpers."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from voiage.ecosystem_integration import (
    DataFormatConnector,
    EcosystemIntegration,
    RPackageConnector,
    TreeAgeConnector,
    WorkflowConnector,
    quick_export_notebook,
    quick_import_health_data,
    quick_r_export,
)
from voiage.health_economics import HealthEconomicsAnalysis, HealthState, Treatment


def _build_analysis() -> HealthEconomicsAnalysis:
    analysis = HealthEconomicsAnalysis(willingness_to_pay=50_000.0, currency="AUD")
    analysis.add_treatment(
        Treatment(
            name="Standard Care",
            description="Baseline",
            effectiveness=0.75,
            cost_per_cycle=100.0,
            cycles_required=10,
        )
    )
    analysis.add_treatment(
        Treatment(
            name="Drug A",
            description="Intervention",
            effectiveness=0.9,
            cost_per_cycle=150.0,
            cycles_required=10,
        )
    )
    analysis.add_health_state(
        HealthState(
            state_id="stable",
            description="Stable disease",
            utility=0.85,
            cost=50.0,
            duration=1.0,
        )
    )
    return analysis


def test_treeage_import_and_convert_to_voi(tmp_path: Path) -> None:
    """TreeAge XML should be parsed into a VOI-compatible structure."""
    connector = TreeAgeConnector()
    xml_path = tmp_path / "treeage.xml"
    xml_path.write_text(
        """
        <treeage>
          <model type="decision_tree" />
          <node id="decision_1" type="decision" name="Choose treatment" />
          <node id="terminal_1" type="terminal" name="Outcome" />
          <probability from="decision_1" to="terminal_1" value="0.7" />
        </treeage>
        """.strip(),
        encoding="utf-8",
    )

    model = connector.import_treeage_model(str(xml_path))
    voi_input = connector.convert_to_voi_analysis(model)

    assert model["model_type"] == "decision_tree"
    assert model["nodes"][0]["id"] == "decision_1"
    assert model["probabilities"][0]["value"] == pytest.approx(0.7)
    assert voi_input["decision_options"][0]["name"] == "Choose treatment"
    assert voi_input["uncertainty_parameters"][0]["value"] == pytest.approx(0.7)


def test_treeage_import_returns_empty_dict_for_invalid_xml(tmp_path: Path) -> None:
    """Malformed TreeAge files should fail soft with an empty payload."""
    connector = TreeAgeConnector()
    xml_path = tmp_path / "broken.xml"
    xml_path.write_text("<treeage><model>", encoding="utf-8")

    with pytest.warns(UserWarning, match="Error importing TreeAge model"):
        model = connector.import_treeage_model(str(xml_path))

    assert model == {}


def test_treeage_export_writes_treatment_nodes(tmp_path: Path) -> None:
    """TreeAge export should emit decision and terminal nodes for treatments."""
    connector = TreeAgeConnector()
    output_path = tmp_path / "treeage_out.xml"

    connector.export_to_treeage(_build_analysis(), str(output_path))

    xml_text = output_path.read_text(encoding="utf-8")
    assert "Drug A" in xml_text
    assert "Standard Care" in xml_text
    assert "<node" in xml_text


def test_treeage_export_exception(tmp_path: Path) -> None:
    """TreeAge export should warn on failure."""
    connector = TreeAgeConnector()
    output_path = tmp_path / "treeage_out.xml"

    with patch("pathlib.Path.write_text", side_effect=Exception("Disk full")):
        with pytest.warns(UserWarning, match="Error exporting to TreeAge: Disk full"):
            connector.export_to_treeage(_build_analysis(), str(output_path))


def test_rpackage_import_helpers_round_trip_json(tmp_path: Path) -> None:
    """BCEA and HE-Sim JSON inputs should be normalized correctly."""
    connector = RPackageConnector()
    bcea_path = tmp_path / "bcea.json"
    hesim_path = tmp_path / "hesim.json"

    bcea_payload = {
        "treatments": [
            {"name": "Standard Care", "effectiveness": 1.0, "cost": 100.0},
            {"name": "Drug A", "effectiveness": 1.2, "cost": 150.0},
        ],
        "cost_effectiveness": {
            "Drug A": {
                "mean_cost": 150.0,
                "mean_effectiveness": 1.2,
                "icer": 25000.0,
            }
        },
        "icer_distributions": {"Drug A": [25000.0, 26000.0]},
    }
    hesim_payload = {
        "markov_models": [{"name": "base"}],
        "state_transitions": [{"from": "stable", "to": "dead"}],
        "cost_effectiveness_results": {"Drug A": {"nmb": 1000.0}},
    }

    bcea_path.write_text(json.dumps(bcea_payload), encoding="utf-8")
    hesim_path.write_text(json.dumps(hesim_payload), encoding="utf-8")

    bcea_results = connector.import_bcea_results(str(bcea_path))
    hesim_results = connector.import_hesim_results(str(hesim_path))

    assert bcea_results["analysis_type"] == "bcea"
    assert bcea_results["treatments"][1]["name"] == "Drug A"
    assert bcea_results["cost_effectiveness_results"]["Drug A"]["icer"] == 25000.0
    assert hesim_results["analysis_type"] == "hesim"
    assert hesim_results["markov_models"][0]["name"] == "base"
    assert hesim_results["cost_effectiveness"]["Drug A"]["nmb"] == 1000.0


def test_rpackage_export_for_bcea_writes_simulation_payload(tmp_path: Path) -> None:
    """BCEA export should serialize treatments and simulation metadata."""
    connector = RPackageConnector()
    output_path = tmp_path / "bcea_export.json"

    connector.export_for_bcea(_build_analysis(), str(output_path), num_simulations=5)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert [item["name"] for item in payload["treatments"]] == [
        "Standard Care",
        "Drug A",
    ]
    assert len(payload["simulation_data"]["costs"]) == 5
    assert len(payload["simulation_data"]["effectiveness"]) == 5
    assert payload["parameters"]["num_simulations"] == 5


def test_rpackage_import_helpers_fail_soft_on_invalid_json(tmp_path: Path) -> None:
    """Broken BCEA and HE-Sim payloads should return empty dictionaries."""
    connector = RPackageConnector()
    broken_bcea = tmp_path / "broken_bcea.json"
    broken_hesim = tmp_path / "broken_hesim.json"
    broken_bcea.write_text("{not-json", encoding="utf-8")
    broken_hesim.write_text("{still-not-json", encoding="utf-8")

    assert connector.import_bcea_results(str(broken_bcea)) == {}
    assert connector.import_hesim_results(str(broken_hesim)) == {}


def test_data_format_connector_imports_csv_and_json(tmp_path: Path) -> None:
    """CSV and JSON inputs should be normalized into VOI data structures."""
    connector = DataFormatConnector()
    csv_path = tmp_path / "transitions.csv"
    json_path = tmp_path / "payload.json"

    pd.DataFrame(
        {
            "from_state": ["stable"],
            "to_state": ["dead"],
            "probability": [0.2],
        }
    ).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps({"costs": [{"state": "stable", "cost": 10.0}]}),
        encoding="utf-8",
    )

    csv_results = connector.import_health_data(str(csv_path), data_type="transitions")
    json_results = connector.import_health_data(str(json_path))

    assert csv_results["data_type"] == "transitions"
    assert csv_results["data"][0]["probability"] == 0.2
    assert json_results["data"]["costs"][0]["state"] == "stable"


@pytest.mark.parametrize(
    ("filename", "columns", "expected_type"),
    [
        ("treatments.csv", {"treatment": ["Drug A"], "cost": [10.0]}, "treatments"),
        (
            "states.csv",
            {"state": ["stable"], "description": ["Stable"]},
            "health_states",
        ),
        ("costs.csv", {"cost": [25.0], "category": ["drug"]}, "costs"),
        ("utilities.csv", {"utility": [0.8], "state_name": ["stable"]}, "utilities"),
        ("generic.csv", {"name": ["row"], "value": [1]}, "generic"),
    ],
)
def test_data_format_connector_auto_detects_csv_payload_type(
    tmp_path: Path,
    filename: str,
    columns: dict[str, list[object]],
    expected_type: str,
) -> None:
    connector = DataFormatConnector()
    csv_path = tmp_path / filename
    pd.DataFrame(columns).to_csv(csv_path, index=False)

    results = connector.import_health_data(str(csv_path), data_type="auto")

    assert results["data_type"] == expected_type
    assert results["data"]


def test_data_format_connector_imports_xlsx(tmp_path: Path) -> None:
    """Excel imports should follow the same normalization path as CSV files."""
    pytest.importorskip("openpyxl")

    connector = DataFormatConnector()
    xlsx_path = tmp_path / "costs.xlsx"
    pd.DataFrame({"state": ["stable"], "cost": [10.0]}).to_excel(xlsx_path, index=False)

    results = connector.import_health_data(str(xlsx_path), data_type="costs")

    assert results["data_type"] == "costs"
    assert results["data"][0]["cost"] == 10.0


def test_data_format_connector_rejects_unknown_file_suffix(tmp_path: Path) -> None:
    """Unsupported health data formats should raise a value error."""
    connector = DataFormatConnector()
    bad_path = tmp_path / "data.unsupported"
    bad_path.write_text("irrelevant", encoding="utf-8")

    with pytest.raises(ValueError):
        connector.import_health_data(str(bad_path))


def test_data_format_connector_exports_json_bundle(tmp_path: Path) -> None:
    """Export should persist a JSON bundle with the main analysis tables."""
    connector = DataFormatConnector()
    output_path = tmp_path / "health_export.json"

    connector.export_health_data(
        _build_analysis(), str(output_path), format_type="json"
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "treatments" in payload
    assert "health_states" in payload
    assert "cost_effectiveness" in payload
    assert payload["treatments"][0]["name"] == "Standard Care"


def test_data_format_connector_exports_csv_tables(tmp_path: Path) -> None:
    """CSV export should write the expected table files."""
    connector = DataFormatConnector()
    output_path = tmp_path / "health_export"

    connector.export_health_data(_build_analysis(), str(output_path), format_type="csv")

    treatments_path = tmp_path / "health_export_treatments.csv"
    states_path = tmp_path / "health_export_health_states.csv"
    ce_path = tmp_path / "health_export_cost_effectiveness.csv"

    assert treatments_path.exists()
    assert states_path.exists()
    assert ce_path.exists()
    assert "Standard Care" in treatments_path.read_text(encoding="utf-8")


def test_data_format_connector_exports_xlsx_workbook(tmp_path: Path) -> None:
    """Excel export should create a workbook with the analysis sheets."""
    pytest.importorskip("openpyxl")

    connector = DataFormatConnector()
    output_path = tmp_path / "health_export.xlsx"

    connector.export_health_data(
        _build_analysis(), str(output_path), format_type="xlsx"
    )

    workbook = pd.read_excel(output_path, sheet_name=None)

    assert {"treatments", "health_states", "cost_effectiveness"} <= set(workbook)
    assert workbook["treatments"].iloc[0]["name"] == "Standard Care"


def test_data_format_connector_import_fails_soft_when_reader_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reader exceptions should trigger the warning path and return an empty payload."""
    connector = DataFormatConnector()
    csv_path = tmp_path / "broken.csv"
    csv_path.write_text("treatment,cost\nDrug A,10\n", encoding="utf-8")

    def _raise_read_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pd, "read_csv", _raise_read_error)

    assert connector.import_health_data(str(csv_path), data_type="auto") == {}


def test_data_format_connector_export_fails_soft_when_writer_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writer exceptions should exercise the export warning path without raising."""
    connector = DataFormatConnector()

    def _raise_to_csv(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pd.DataFrame, "to_csv", _raise_to_csv)

    connector.export_health_data(
        _build_analysis(), str(tmp_path / "health_export"), format_type="csv"
    )


def test_workflow_connector_creates_jupyter_and_r_artifacts(tmp_path: Path) -> None:
    """Workflow exports should write deterministic notebook and R script artifacts."""
    connector = WorkflowConnector()
    analysis = _build_analysis()
    notebook_path = tmp_path / "analysis.ipynb"
    r_path = tmp_path / "analysis.R"

    connector.create_jupyter_analysis(analysis, str(notebook_path))
    connector.create_r_workflow(analysis, str(r_path))

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    r_script = r_path.read_text(encoding="utf-8")

    assert notebook["nbformat"] == 4
    assert notebook["cells"][0]["source"][0].startswith("# Health Economics")
    assert any("Drug A" in "".join(cell["source"]) for cell in notebook["cells"])
    assert "willingness_to_pay <- 50000.0" in r_script
    assert 'currency <- "AUD"' in r_script
    assert 'name = "Drug A"' in r_script


def test_workflow_connector_fails_soft_when_file_write_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workflow writer errors should warn instead of raising."""
    connector = WorkflowConnector()

    def _raise_open_error(*args, **kwargs):
        raise OSError("cannot write")

    monkeypatch.setattr("builtins.open", _raise_open_error)

    with pytest.warns(UserWarning, match="Error creating Jupyter notebook"):
        connector.create_jupyter_analysis(
            _build_analysis(), str(tmp_path / "bad.ipynb")
        )

    with pytest.warns(UserWarning, match="Error creating R workflow"):
        connector.create_r_workflow(_build_analysis(), str(tmp_path / "bad.R"))


def test_ecosystem_integration_routes_imports_exports_and_reports(
    tmp_path: Path,
) -> None:
    """The integration manager should route common import and export operations."""
    integration = EcosystemIntegration()
    analysis = _build_analysis()
    csv_path = tmp_path / "treatments.csv"
    bcea_path = tmp_path / "bcea.json"
    notebook_path = tmp_path / "routed.ipynb"
    r_path = tmp_path / "routed.R"
    export_path = tmp_path / "health_export.json"

    pd.DataFrame({"treatment": ["Drug A"], "cost": [10.0]}).to_csv(
        csv_path, index=False
    )
    bcea_path.write_text(
        json.dumps({"treatments": [{"name": "Drug A"}]}),
        encoding="utf-8",
    )

    health_data = integration.import_from_external(
        "data_formats", str(csv_path), data_type="auto"
    )
    bcea_data = integration.import_from_external(
        "r_packages", str(bcea_path), format="bcea"
    )
    unknown_r_format = integration.import_from_external(
        "r_packages", str(bcea_path), format="unknown"
    )

    integration.export_to_external(
        "data_formats", analysis, str(export_path), format_type="json"
    )
    integration.export_to_external(
        "workflows", analysis, str(notebook_path), format="jupyter"
    )
    integration.export_to_external("workflows", analysis, str(r_path), format="r")

    formats = integration.list_supported_formats()
    report = integration.create_integration_report()

    assert health_data["data_type"] == "treatments"
    assert bcea_data["analysis_type"] == "bcea"
    assert unknown_r_format == {}
    assert export_path.exists()
    assert notebook_path.exists()
    assert r_path.exists()
    assert "treeage" in formats
    assert report["available_connectors"] == [
        "treeage",
        "r_packages",
        "data_formats",
        "workflows",
    ]

    with pytest.raises(ValueError, match="Unknown integration type"):
        integration.import_from_external("missing", str(csv_path))

    with pytest.raises(ValueError, match="Unknown integration type"):
        integration.export_to_external("missing", analysis, str(export_path))


def test_quick_ecosystem_helpers_route_to_default_connectors(tmp_path: Path) -> None:
    """Quick helper functions should delegate to the integration manager."""
    csv_path = tmp_path / "costs.csv"
    notebook_path = tmp_path / "quick.ipynb"
    r_path = tmp_path / "quick.R"
    analysis = _build_analysis()

    pd.DataFrame({"cost": [25.0], "category": ["drug"]}).to_csv(csv_path, index=False)

    imported = quick_import_health_data(str(csv_path))
    quick_export_notebook(analysis, str(notebook_path))
    quick_r_export(analysis, str(r_path))

    assert imported["data_type"] == "costs"
    assert notebook_path.exists()
    assert r_path.exists()
