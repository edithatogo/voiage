"""Executable generated-code and current declaration cleanup contracts."""

import ast
import json
from pathlib import Path

import polars as pl

from voiage.ecosystem_integration import WorkflowConnector
from voiage.governance import DEPENDENCY_POLICY
from voiage.health_economics import HealthEconomicsAnalysis, Treatment


def test_generated_notebook_code_compiles_and_uses_native_results(
    tmp_path: Path,
) -> None:
    analysis = HealthEconomicsAnalysis(willingness_to_pay=50000.0, currency="AUD")
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
            description="Alternative",
            effectiveness=0.9,
            cost_per_cycle=150.0,
            cycles_required=10,
        )
    )
    path = tmp_path / "analysis.ipynb"
    WorkflowConnector().create_jupyter_analysis(analysis, str(path))
    notebook = json.loads(path.read_text())
    assert notebook["metadata"]["language_info"]["version"] == "3.15.1"
    scope: dict[str, object] = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        ast.parse(source)
        if "# Create visualizations" not in source:
            exec(compile(source, str(path), "exec"), scope)  # noqa: S102 - owned generated test code
    frame = scope["results_df"]
    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == ["Treatment", "Cost", "QALY", "NMB", "ICER"]
    assert frame["Treatment"].to_list() == ["Standard Care", "Drug A"]
    assert frame["Cost"].to_list() == [
        analysis._calculate_treatment_totals(
            treatment, analysis._create_default_health_states(treatment)
        )[0]
        for treatment in analysis.treatments.values()
    ]


def test_runtime_governance_names_match_actual_core_declarations() -> None:
    import tomllib

    from packaging.requirements import Requirement

    root = Path(__file__).parents[1]
    declared = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    assert set(DEPENDENCY_POLICY["base"]) == {Requirement(s).name for s in declared}
