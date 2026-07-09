"""
Ecosystem Integration Module for voiage.

This module provides integration capabilities with popular software and workflows:
- TreeAge Pro compatibility
- R package integration (hesim, BCEA, etc.)
- Health economic modeling software interoperability
- Research workflow integration (Jupyter, RStudio, etc.)
- Data format compatibility (CSV, Excel, SAS, Stata)

Author: voiage Development Team
Version: 2.0.0
"""

import concurrent.futures
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any
import warnings

from defusedxml import ElementTree  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from voiage.exceptions import raise_type_error, raise_value_error
from voiage.health_economics import HealthEconomicsAnalysis
from voiage.schema import ParameterSet, ValueArray


@dataclass
class EcosystemConnector:
    """Base class for ecosystem integrations."""

    name: str
    version: str
    supported_formats: list[str]
    integration_type: str  # "import", "export", "bidirectional"


@dataclass(frozen=True)
class HeomlRunBundle:
    """Loaded HEOML run bundle and its core VOI contract objects."""

    manifest: dict[str, Any]
    value_array: ValueArray | None
    parameter_set: ParameterSet | None
    provenance: dict[str, Any]


class TreeAgeConnector(EcosystemConnector):
    """TreeAge Pro integration connector."""

    def __init__(self) -> None:
        super().__init__(
            name="TreeAge Pro",
            version="2023+",
            supported_formats=[".treeage", ".csv", ".xlsx"],
            integration_type="bidirectional",
        )

    def import_treeage_model(self, file_path: str) -> dict[str, Any]:
        """
        Import TreeAge Pro model file.

        Args:
            file_path: Path to TreeAge model file

        Returns
        -------
            Parsed model structure compatible with voiage
        """
        try:
            tree = ElementTree.parse(file_path)
            root = tree.getroot()

            model_structure: dict[str, Any] = {
                "model_type": "decision_tree",
                "nodes": [],
                "branches": [],
                "probabilities": [],
                "costs": [],
                "outcomes": [],
            }
            nodes: list[dict[str, Any]] = model_structure["nodes"]
            probabilities: list[dict[str, Any]] = model_structure["probabilities"]
            costs: list[dict[str, Any]] = model_structure["costs"]

            # Extract decision tree structure
            for node in root.findall(".//node"):
                node_data = {
                    "id": node.get("id"),
                    "type": node.get("type"),  # decision, chance, terminal
                    "name": node.get("name"),
                    "description": node.get("description", ""),
                }
                nodes.append(node_data)

            # Extract probabilities
            for prob in root.findall(".//probability"):
                prob_data = {
                    "from_node": prob.get("from"),
                    "to_node": prob.get("to"),
                    "value": float(prob.get("value", 0.0)),
                }
                probabilities.append(prob_data)

            # Extract costs and outcomes
            for cost in root.findall(".//cost"):
                cost_data = {
                    "node_id": cost.get("node_id"),
                    "value": float(cost.get("value", 0.0)),
                    "type": cost.get("type", "cost"),
                }
                costs.append(cost_data)

        except Exception as e:
            warnings.warn(f"Error importing TreeAge model: {e}", stacklevel=2)
            return {}
        else:
            return model_structure

    def export_to_treeage(
        self, health_analysis: HealthEconomicsAnalysis, output_path: str
    ) -> None:
        """
        Export voiage model to TreeAge Pro format.

        Args:
            health_analysis: Health economics analysis to export
            output_path: Output file path
        """
        try:
            lines = [
                '<?xml version="1.0" encoding="utf-8"?>',
                '<TreeAgeModel version="1.0">',
            ]

            for treatment in health_analysis.treatments.values():
                lines.append(
                    "  <node"
                    f' id="{escape(treatment.name, quote=True)}"'
                    ' type="decision"'
                    f' name="{escape(treatment.description, quote=True)}"'
                    f' description="{escape(f"Treatment: {treatment.name}", quote=True)}">'
                )

                health_states = health_analysis._create_default_health_states(treatment)
                for i, state in enumerate(health_states):
                    qaly = health_analysis.calculate_qaly(state)
                    cost = health_analysis.calculate_cost(state)
                    lines.append(
                        "    <branch"
                        f' id="{escape(f"{treatment.name}_outcome_{i}", quote=True)}"'
                        ' probability="1.0"'
                        f' cost="{cost}"'
                        f' qaly="{qaly}" />'
                    )

                lines.append("  </node>")

            lines.append("</TreeAgeModel>")
            Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

        except Exception as e:
            warnings.warn(f"Error exporting to TreeAge: {e}", stacklevel=2)

    def convert_to_voi_analysis(
        self, model_structure: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert TreeAge model to voiage analysis format."""
        voi_format: dict[str, Any] = {
            "decision_options": [],
            "uncertainty_parameters": [],
            "outcome_function": None,
        }
        decision_options: list[dict[str, Any]] = voi_format["decision_options"]
        uncertainty_parameters: list[dict[str, Any]] = voi_format[
            "uncertainty_parameters"
        ]

        # Extract decision options from nodes
        decision_options.extend(
            {
                "name": node.get("name"),
                "description": node.get("description"),
                "cost_outcomes": [],
            }
            for node in model_structure.get("nodes", [])
            if node.get("type") == "decision"
        )

        # Extract probabilities and costs
        uncertainty_parameters.extend(
            {
                "name": f"prob_{prob.get('from_node')}_{prob.get('to_node')}",
                "value": prob.get("value"),
                "type": "probability",
            }
            for prob in model_structure.get("probabilities", [])
        )

        return voi_format


class RPackageConnector(EcosystemConnector):
    """R package integration connector for health economics."""

    def __init__(self) -> None:
        super().__init__(
            name="R Health Economics Packages",
            version="1.0",
            supported_formats=[".rds", ".csv", ".json"],
            integration_type="bidirectional",
        )

    def import_bcea_results(self, file_path: str) -> dict[str, Any]:
        """
        Import Bayesian Cost-Effectiveness Analysis results.

        Args:
            file_path: Path to BCEA results file

        Returns
        -------
            BCEA results in voiage compatible format
        """
        try:
            # Read BCEA results (assuming JSON export from R)
            with open(file_path) as f:
                bcea_data = json.load(f)

            voi_results: dict[str, Any] = {
                "analysis_type": "bcea",
                "treatments": [],
                "cost_effectiveness_results": {},
                "icer_distributions": {},
                "ceac_data": {},
            }
            treatments: list[dict[str, Any]] = voi_results["treatments"]
            cost_effectiveness_results: dict[str, dict[str, Any]] = voi_results[
                "cost_effectiveness_results"
            ]

            # Extract treatment information
            if "treatments" in bcea_data:
                treatments.extend(
                    {
                        "name": treatment.get("name"),
                        "effectiveness": treatment.get("effectiveness"),
                        "cost": treatment.get("cost"),
                    }
                    for treatment in bcea_data["treatments"]
                )

            # Extract cost-effectiveness results
            if "cost_effectiveness" in bcea_data:
                for treatment, results in bcea_data["cost_effectiveness"].items():
                    cost_effectiveness_results[treatment] = {
                        "mean_cost": results.get("mean_cost"),
                        "mean_effectiveness": results.get("mean_effectiveness"),
                        "icer": results.get("icer"),
                    }

            # Extract ICER distributions
            if "icer_distributions" in bcea_data:
                voi_results["icer_distributions"] = bcea_data["icer_distributions"]

        except Exception as e:
            warnings.warn(f"Error importing BCEA results: {e}", stacklevel=2)
            return {}
        else:
            return voi_results

    def export_for_bcea(
        self,
        health_analysis: HealthEconomicsAnalysis,
        output_path: str,
        num_simulations: int = 10000,
    ) -> None:
        """
        Export analysis results for BCEA package.

        Args:
            health_analysis: Health economics analysis to export
            output_path: Output file path
            num_simulations: Number of Monte Carlo simulations
        """
        try:
            # Create data structure for BCEA
            bcea_data: dict[str, Any] = {
                "treatments": [],
                "simulation_data": {"costs": [], "effectiveness": [], "icer": []},
                "parameters": {
                    "willingness_to_pay": health_analysis.willingness_to_pay,
                    "currency": health_analysis.currency,
                    "num_simulations": num_simulations,
                },
            }
            treatments_data: list[dict[str, Any]] = bcea_data["treatments"]
            simulation_costs: list[list[float]] = bcea_data["simulation_data"]["costs"]
            simulation_effectiveness: list[list[float]] = bcea_data["simulation_data"][
                "effectiveness"
            ]
            simulation_icer: list[list[float]] = bcea_data["simulation_data"]["icer"]

            # Add treatment information
            treatments_data.extend(
                {
                    "name": treatment.name,
                    "description": treatment.description,
                    "effectiveness": treatment.effectiveness,
                    "cost_per_cycle": treatment.cost_per_cycle,
                }
                for treatment in health_analysis.treatments.values()
            )

            # Generate simulation data
            treatments = list(health_analysis.treatments.values())
            base_costs = np.array(
                [
                    treatment.cost_per_cycle * treatment.cycles_required
                    for treatment in treatments
                ]
            )
            costs_arr = base_costs * (
                1 + np.random.normal(0, 0.1, (num_simulations, len(treatments)))
            )
            simulation_costs.extend(costs_arr.tolist())

            base_effs = np.array([treatment.effectiveness for treatment in treatments])
            effs_arr = base_effs * (
                1 + np.random.normal(0, 0.05, (num_simulations, len(treatments)))
            )
            simulation_effectiveness.extend(effs_arr.tolist())

            # Calculate ICER for each simulation
            if len(treatments_data) >= 2:
                cost_diffs = costs_arr[:, 1:] - costs_arr[:, 0:1]
                eff_diffs = effs_arr[:, 1:] - effs_arr[:, 0:1]

                # Calculate ICER: cost_diff / eff_diff if eff_diff > 0 else np.inf
                with np.errstate(divide="ignore", invalid="ignore"):
                    icers = np.where(eff_diffs > 0, cost_diffs / eff_diffs, np.inf)

                simulation_icer.extend(icers.T.tolist())

            # Save to JSON
            with open(output_path, "w") as f:
                json.dump(bcea_data, f, indent=2, default=str)

        except Exception as e:
            warnings.warn(f"Error exporting for BCEA: {e}", stacklevel=2)

    def import_hesim_results(self, file_path: str) -> dict[str, Any]:
        """
        Import HE-Sim simulation results.

        Args:
            file_path: Path to HE-Sim results file

        Returns
        -------
            HE-Sim results in voiage compatible format
        """
        try:
            # Read HE-Sim results (JSON format assumed)
            with open(file_path) as f:
                hesim_data = json.load(f)

            voi_format: dict[str, Any] = {
                "analysis_type": "hesim",
                "markov_models": [],
                "state_transitions": [],
                "cost_effectiveness": {},
            }
            markov_models: list[Any] = voi_format["markov_models"]
            state_transitions: list[Any] = voi_format["state_transitions"]
            cost_effectiveness: dict[str, Any] = voi_format["cost_effectiveness"]

            if "markov_models" in hesim_data:
                markov_models.extend(hesim_data["markov_models"])

            if "state_transitions" in hesim_data:
                state_transitions.extend(hesim_data["state_transitions"])

            if "cost_effectiveness_results" in hesim_data:
                cost_effectiveness.update(hesim_data["cost_effectiveness_results"])

        except Exception as e:
            warnings.warn(f"Error importing HE-Sim results: {e}", stacklevel=2)
            return {}
        else:
            return voi_format


class DataFormatConnector(EcosystemConnector):
    """Data format compatibility connector."""

    def __init__(self) -> None:
        super().__init__(
            name="Data Format Connector",
            version="1.0",
            supported_formats=[".csv", ".xlsx", ".xls", ".json", ".parquet"],
            integration_type="bidirectional",
        )

    def import_health_data(
        self, file_path: str, data_type: str = "auto"
    ) -> dict[str, Any]:
        """
        Import health economics data from various formats.

        Args:
            file_path: Path to data file
            data_type: Type of data ("costs", "utilities", "transitions", "auto")

        Returns
        -------
            Imported data in voiage format
        """
        path_obj = Path(file_path)
        suffix = path_obj.suffix.lower()
        if suffix not in {".csv", ".xlsx", ".xls", ".parquet", ".json"}:
            raise_value_error(f"Unsupported file format: {path_obj.suffix}")

        try:
            if suffix == ".csv":
                df = pd.read_csv(path_obj)
            elif suffix in {".xlsx", ".xls"}:
                df = pd.read_excel(path_obj)
            elif suffix == ".parquet":
                df = pd.read_parquet(path_obj)
            elif suffix == ".json":
                with open(path_obj) as f:
                    data = json.load(f)
                return self._convert_json_to_dataframe(data)

            return self._process_dataframe(df, data_type)

        except Exception as e:
            warnings.warn(f"Error importing health data: {e}", stacklevel=2)
            return {}

    def export_health_data(
        self,
        health_analysis: HealthEconomicsAnalysis,
        output_path: str,
        format_type: str = "csv",
    ) -> None:
        """
        Export health economics data.

        Args:
            health_analysis: Health economics analysis to export
            output_path: Output file path
            format_type: Export format ("csv", "xlsx", "json")
        """
        try:
            # Create export data structure
            export_data: dict[str, pd.DataFrame] = {}

            # Export treatments
            treatments_data = [
                {
                    "name": treatment.name,
                    "description": treatment.description,
                    "effectiveness": treatment.effectiveness,
                    "cost_per_cycle": treatment.cost_per_cycle,
                    "cycles_required": treatment.cycles_required,
                    "side_effect_utility": treatment.side_effect_utility,
                    "side_effect_cost": treatment.side_effect_cost,
                }
                for treatment in health_analysis.treatments.values()
            ]
            export_data["treatments"] = pd.DataFrame(treatments_data)

            # Export health states
            states_data = [
                {
                    "state_id": state.state_id,
                    "description": state.description,
                    "utility": state.utility,
                    "cost": state.cost,
                    "duration": state.duration,
                }
                for state in health_analysis.health_states.values()
            ]
            export_data["health_states"] = pd.DataFrame(states_data)

            # Export cost-effectiveness results
            ce_results = [
                {
                    "treatment": treatment_name,
                    "total_cost": cost,
                    "total_qaly": qaly,
                    "net_monetary_benefit": nmb,
                }
                for treatment_name, treatment in health_analysis.treatments.items()
                for health_states in [
                    health_analysis._create_default_health_states(treatment)
                ]
                for cost, qaly in [
                    health_analysis._calculate_treatment_totals(
                        treatment, health_states
                    )
                ]
                for nmb in [
                    health_analysis.calculate_net_monetary_benefit(
                        treatment, health_states
                    )
                ]
            ]
            export_data["cost_effectiveness"] = pd.DataFrame(ce_results)

            # Save to file
            output_path_obj = Path(output_path)

            if format_type.lower() == "csv":
                for table_name, df in export_data.items():
                    table_path = (
                        output_path_obj.parent
                        / f"{output_path_obj.stem}_{table_name}.csv"
                    )
                    df.to_csv(table_path, index=False)

            elif format_type.lower() == "xlsx":
                with pd.ExcelWriter(output_path_obj) as writer:
                    for table_name, df in export_data.items():
                        df.to_excel(writer, sheet_name=table_name, index=False)

            elif format_type.lower() == "json":
                # Convert DataFrames to JSON-serializable format
                json_data: dict[str, list[dict[str, Any]]] = {}
                for table_name, df in export_data.items():
                    json_data[table_name] = df.to_dict("records")

                with open(output_path_obj, "w") as f:
                    json.dump(json_data, f, indent=2, default=str)

        except Exception as e:
            warnings.warn(f"Error exporting health data: {e}", stacklevel=2)

    def _process_dataframe(self, df: pd.DataFrame, data_type: str) -> dict[str, Any]:
        """Process imported dataframe."""
        if data_type == "auto":
            # Auto-detect data type based on columns
            columns = [col.lower() for col in df.columns]

            if any(col in ["treatment", "intervention", "drug"] for col in columns):
                data_type = "treatments"
            elif any(col in ["state", "health_state", "condition"] for col in columns):
                data_type = "health_states"
            elif any(col in ["cost", "price", "expense"] for col in columns):
                data_type = "costs"
            elif any(col in ["utility", "qaly", "effectiveness"] for col in columns):
                data_type = "utilities"
            else:
                data_type = "generic"

        return {
            "data_type": data_type,
            "data": df.to_dict("records"),
            "columns": list(df.columns),
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
        }

    def _convert_json_to_dataframe(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert JSON data to processed format."""
        return {
            "data_type": "json",
            "data": data,
            "keys": list(data.keys()) if isinstance(data, dict) else [],
            "structure": "nested" if isinstance(data, dict) else "flat",
        }


def _load_tabular_artifact(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return raise_value_error(
        f"Unsupported HEOML tabular artifact format: {path.suffix}"
    )


def load_heoml_run_bundle(manifest_path: str) -> HeomlRunBundle:
    """Load a HEOML run-bundle manifest and its core VOI artifacts."""
    path_obj = Path(manifest_path)
    try:
        manifest = json.loads(path_obj.read_text(encoding="utf-8"))
    except Exception as exc:
        raise_value_error(f"Unable to read HEOML manifest: {exc}")

    if not isinstance(manifest, dict):
        raise_type_error("HEOML manifest must be a JSON object.")

    heoml_version = manifest.get("heoml_version", manifest.get("heoml_profile_version"))
    if heoml_version != "0.1":
        raise_value_error("HEOML manifest must declare heoml_version '0.1'.")
    profile = manifest.get("profile", "heoml.run_bundle")
    if profile != "heoml.run_bundle":
        raise_value_error("HEOML manifest must declare profile 'heoml.run_bundle'.")

    artifacts = manifest.get("artifacts")
    artifact_by_category: dict[str, dict[str, Any]] = {}
    if isinstance(artifacts, dict):
        for category, path_value in artifacts.items():
            if not isinstance(category, str) or not isinstance(path_value, str):
                raise_type_error("HEOML artifact entries must map strings to strings.")
            artifact_by_category[category] = {"category": category, "path": path_value}
    elif isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise_type_error("HEOML artifact entries must be objects.")
            category = artifact.get("category")
            path_value = artifact.get("path")
            if not isinstance(category, str) or not isinstance(path_value, str):
                raise_type_error(
                    "HEOML artifact entries must include category and path."
                )
            artifact_by_category.setdefault(category, artifact)
    else:
        raise_type_error("HEOML manifest artifacts must be an array or object.")

    bundle_root = path_obj.parent

    voi_artifact = artifact_by_category.get("voi") or artifact_by_category.get(
        "net_benefits"
    )
    parameter_artifact = artifact_by_category.get(
        "parameter"
    ) or artifact_by_category.get("parameter_samples")

    def load_voi() -> ValueArray | None:
        if voi_artifact is None:
            return None
        voi_path = bundle_root / voi_artifact["path"]
        voi_frame = _load_tabular_artifact(voi_path)
        if "sample_index" in voi_frame.columns:
            voi_frame = voi_frame.drop(columns=["sample_index"])
        return ValueArray.from_numpy(
            voi_frame.to_numpy(),
            strategy_names=[str(column) for column in voi_frame.columns],
        )

    def load_parameter() -> ParameterSet | None:
        if parameter_artifact is None:
            return None
        parameter_path = bundle_root / parameter_artifact["path"]
        parameter_frame = _load_tabular_artifact(parameter_path)
        if "sample_index" in parameter_frame.columns:
            parameter_frame = parameter_frame.drop(columns=["sample_index"])
        return ParameterSet.from_numpy_or_dict(
            {
                str(column): parameter_frame[column].to_numpy()
                for column in parameter_frame.columns
            }
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_voi = executor.submit(load_voi)
        future_param = executor.submit(load_parameter)

        value_array = future_voi.result()
        parameter_set = future_param.result()

    run_section = manifest.get("run")
    manifest_id = None
    if isinstance(run_section, dict):
        run_id = run_section.get("id")
        if isinstance(run_id, str):
            manifest_id = run_id

    provenance = {
        "heoml_version": heoml_version,
        "profile": profile,
        "manifest_id": manifest.get("heoml_manifest_id", manifest_id),
        "producer": manifest.get("producer"),
    }

    normalized_manifest = dict(manifest)
    normalized_manifest["heoml_version"] = heoml_version
    normalized_manifest["profile"] = profile
    normalized_manifest["heoml_manifest_id"] = provenance["manifest_id"]

    return HeomlRunBundle(
        manifest=normalized_manifest,
        value_array=value_array,
        parameter_set=parameter_set,
        provenance=provenance,
    )


class WorkflowConnector(EcosystemConnector):
    """Research workflow integration connector."""

    def __init__(self) -> None:
        super().__init__(
            name="Research Workflow Connector",
            version="1.0",
            supported_formats=[".ipynb", ".r", ".py", ".md"],
            integration_type="bidirectional",
        )

    def create_jupyter_analysis(
        self, health_analysis: HealthEconomicsAnalysis, output_path: str
    ) -> None:
        """
        Create Jupyter notebook for health economics analysis.

        Args:
            health_analysis: Health economics analysis
            output_path: Output notebook path
        """
        try:
            notebook_cells: list[dict[str, Any]] = [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# Health Economics Analysis with voiage\\n\\n",
                        "This notebook demonstrates health economics analysis using the voiage library.\\n\\n",
                        f"**Willingness to Pay:** {health_analysis.willingness_to_pay} {health_analysis.currency}\\n",
                        f"**Currency:** {health_analysis.currency}\\n",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import sys\\n",
                        'sys.path.append("/path/to/voiage")\\n',
                        "from voiage.health_economics import HealthEconomicsAnalysis, HealthState, Treatment\\n",
                        "import numpy as np\\n",
                        "import pandas as pd\\n\\n",
                        "# Initialize analysis\\n",
                        "health_analysis = HealthEconomicsAnalysis(\\n",
                        f"    willingness_to_pay={health_analysis.willingness_to_pay},\\n",
                        f'    currency="{health_analysis.currency}"\\n',
                        ")\\n",
                    ],
                },
            ]
            notebook_content: dict[str, Any] = {
                "cells": notebook_cells,
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3",
                    },
                    "language_info": {"name": "python", "version": "3.8+"},
                },
                "nbformat": 4,
                "nbformat_minor": 4,
            }

            # Add treatment cells
            for treatment in health_analysis.treatments.values():
                cell: dict[str, Any] = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        f"# Add {treatment.name} treatment\\n",
                        f"treatment_{treatment.name.lower().replace(' ', '_')} = Treatment(\\n",
                        f'    name="{treatment.name}",\\n',
                        f'    description="{treatment.description}",\\n',
                        f"    effectiveness={treatment.effectiveness},\\n",
                        f"    cost_per_cycle={treatment.cost_per_cycle},\\n",
                        f"    cycles_required={treatment.cycles_required}\\n",
                        ")\\n",
                        f"health_analysis.add_treatment(treatment_{treatment.name.lower().replace(' ', '_')})\\n",
                    ],
                }
                notebook_content["cells"].append(cell)

            # Add analysis cells
            analysis_cells: list[dict[str, Any]] = [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Perform cost-effectiveness analysis\\n",
                        "treatments = list(health_analysis.treatments.values())\\n",
                        "results = []\\n",
                        "\\n",
                        "for treatment in treatments:\\n",
                        "    health_states = health_analysis._create_default_health_states(treatment)\\n",
                        "    cost, qaly = health_analysis._calculate_treatment_totals(treatment, health_states)\\n",
                        "    nmb = health_analysis.calculate_net_monetary_benefit(treatment, health_states)\\n",
                        "    icer = health_analysis.calculate_icer(treatment)\\n",
                        "    \\n",
                        "    results.append({\\n",
                        "        'Treatment': treatment.name,\\n",
                        "        'Cost': cost,\\n",
                        "        'QALY': qaly,\\n",
                        "        'NMB': nmb,\\n",
                        "        'ICER': icer\\n",
                        "    })\\n",
                        "\\n",
                        "results_df = pd.DataFrame(results)\\n",
                        "print(results_df)\\n",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Create visualizations\\n",
                        "import matplotlib.pyplot as plt\\n",
                        "import seaborn as sns\\n",
                        "\\n",
                        "# Cost-effectiveness plane\\n",
                        "plt.figure(figsize=(10, 6))\\n",
                        'plt.scatter(results_df["QALY"], results_df["Cost"])\\n',
                        "for row in results_df.itertuples():\\n",
                        "    plt.annotate(row.Treatment, (row.QALY, row.Cost))\\n",
                        'plt.xlabel("Effectiveness (QALYs)")\\n',
                        'plt.ylabel("Cost")\\n',
                        'plt.title("Cost-Effectiveness Plane")\\n',
                        "plt.grid(True)\\n",
                        "plt.show()\\n",
                    ],
                },
            ]

            notebook_content["cells"].extend(analysis_cells)

            # Save notebook
            import json

            with open(output_path, "w") as f:
                json.dump(notebook_content, f, indent=1)

        except Exception as e:
            warnings.warn(f"Error creating Jupyter notebook: {e}", stacklevel=2)

    def create_r_workflow(
        self, health_analysis: HealthEconomicsAnalysis, output_path: str
    ) -> None:
        """
        Create R script for health economics analysis.

        Args:
            health_analysis: Health economics analysis
            output_path: Output R script path
        """
        try:
            r_content = [
                "# Health Economics Analysis with voiage Integration\\n",
                "# This R script demonstrates integration with voiage analysis\\n",
                "\\n",
                "# Load required libraries\\n",
                "library(ggplot2)\\n",
                "library(dplyr)\\n",
                "library(jsonlite)\\n",
                "\\n",
                "# Analysis parameters\\n",
                f"willingness_to_pay <- {health_analysis.willingness_to_pay}\\n",
                f'currency <- "{health_analysis.currency}"\\n',
                "\\n",
                "# Data preparation\\n",
                "treatments_data <- data.frame(\\n",
            ]

            # Add treatment data
            first_treatment = True
            for treatment in health_analysis.treatments.values():
                if not first_treatment:
                    r_content.append("                           ,\\n")
                r_content.append(
                    f'    name = "{treatment.name}",\\n'
                    f"    effectiveness = {treatment.effectiveness},\\n"
                    f"    cost_per_cycle = {treatment.cost_per_cycle},\\n"
                    f"    cycles_required = {treatment.cycles_required}\\n"
                )
                first_treatment = False

            r_content.extend(
                [
                    ")\\n",
                    "\\n",
                    "# Cost-effectiveness analysis\\n",
                    "results <- treatments_data %>%\\n",
                    "    mutate(\\n",
                    "        total_cost = cost_per_cycle * cycles_required,\\n",
                    "        qaly_estimate = effectiveness * 5,  # Simplified QALY estimate\\n",
                    "        nmb = qaly_estimate * willingness_to_pay - total_cost\\n",
                    "    )\\n",
                    "\\n",
                    "# Display results\\n",
                    "print(results)\\n",
                    "\\n",
                    "# Create cost-effectiveness plane\\n",
                    "ggplot(results, aes(x = qaly_estimate, y = total_cost, label = name)) +\\n",
                    "    geom_point(size = 3) +\\n",
                    "    geom_text(hjust = 0, vjust = 0) +\\n",
                    '    labs(x = "Effectiveness (QALYs)",\\n',
                    '         y = "Cost",\\n',
                    '         title = "Cost-Effectiveness Plane") +\\n',
                    "    theme_minimal()\\n",
                    "\\n",
                    "# Export results\\n",
                    'write.csv(results, "cost_effectiveness_results.csv", row.names = FALSE)\\n',
                    "\\n",
                    'cat("Analysis complete. Results saved to cost_effectiveness_results.csv\\n")\\n',
                ]
            )

            # Write R script
            with open(output_path, "w") as f:
                f.writelines(r_content)

        except Exception as e:
            warnings.warn(f"Error creating R workflow: {e}", stacklevel=2)


class EcosystemIntegration:
    """Main ecosystem integration manager."""

    def __init__(self) -> None:
        self.connectors = {
            "treeage": TreeAgeConnector(),
            "r_packages": RPackageConnector(),
            "data_formats": DataFormatConnector(),
            "workflows": WorkflowConnector(),
        }

    def get_connector(self, integration_type: str) -> EcosystemConnector | None:
        """Get connector for specific integration type."""
        return self.connectors.get(integration_type.lower())

    def import_from_external(
        self, integration_type: str, file_path: str, **kwargs: object
    ) -> dict[str, object]:
        """Import data from external software."""
        connector = self.get_connector(integration_type)
        if connector is None:
            raise_value_error(f"Unknown integration type: {integration_type}")

        if integration_type.lower() == "r_packages":
            if isinstance(connector, RPackageConnector):
                format_name = kwargs.get("format")
                if format_name == "bcea":
                    return connector.import_bcea_results(file_path)
                if format_name == "hesim":
                    return connector.import_hesim_results(file_path)
        elif integration_type.lower() == "data_formats" and isinstance(
            connector, DataFormatConnector
        ):
            return connector.import_health_data(
                file_path, kwargs.get("data_type", "auto")
            )

        return {}

    def export_to_external(
        self,
        integration_type: str,
        analysis_object: object,
        output_path: str,
        **kwargs: object,
    ) -> None:
        """Export analysis to external software format."""
        connector = self.get_connector(integration_type)
        if connector is None:
            raise_value_error(f"Unknown integration type: {integration_type}")

        if integration_type.lower() == "treeage" and isinstance(
            connector, TreeAgeConnector
        ):
            if hasattr(analysis_object, "treatments"):
                connector.export_to_treeage(analysis_object, output_path)
        elif integration_type.lower() == "r_packages" and isinstance(
            connector, RPackageConnector
        ):
            if kwargs.get("format") == "bcea":
                connector.export_for_bcea(
                    analysis_object,
                    output_path,
                    kwargs.get("num_simulations", 10000),
                )
        elif integration_type.lower() == "data_formats" and isinstance(
            connector, DataFormatConnector
        ):
            connector.export_health_data(
                analysis_object,
                output_path,
                kwargs.get("format_type", "csv"),
            )
        elif integration_type.lower() == "workflows" and isinstance(
            connector, WorkflowConnector
        ):
            if kwargs.get("format") == "jupyter":
                connector.create_jupyter_analysis(analysis_object, output_path)
            elif kwargs.get("format") == "r":
                connector.create_r_workflow(analysis_object, output_path)

    def list_supported_formats(self) -> dict[str, list[str]]:
        """List all supported file formats across connectors."""
        formats: dict[str, list[str]] = {}
        for name, connector in self.connectors.items():
            formats[name] = connector.supported_formats
        return formats

    def create_integration_report(self) -> dict[str, Any]:
        """Create comprehensive integration capabilities report."""
        return {
            "available_connectors": list(self.connectors.keys()),
            "supported_formats": self.list_supported_formats(),
            "integration_capabilities": {
                "treeage": {
                    "import": True,
                    "export": True,
                    "bidirectional": True,
                    "description": "TreeAge Pro decision tree and Markov model compatibility",
                },
                "r_packages": {
                    "import": True,
                    "export": True,
                    "bidirectional": True,
                    "description": "BCEA, HE-Sim, and other R health economics packages",
                },
                "data_formats": {
                    "import": True,
                    "export": True,
                    "bidirectional": True,
                    "description": "CSV, Excel, Parquet, JSON data format compatibility",
                },
                "workflows": {
                    "import": False,
                    "export": True,
                    "bidirectional": False,
                    "description": "Jupyter notebooks, R scripts, and research workflows",
                },
            },
            "version": "2.0.0",
            "last_updated": "2025-11-09",
        }


# Utility functions for common integration tasks


def quick_import_health_data(
    file_path: str, analysis_type: str = "auto"
) -> dict[str, Any]:
    """Quick import function for health economics data."""
    integration = EcosystemIntegration()
    return integration.import_from_external(
        "data_formats", file_path, data_type=analysis_type
    )


def quick_export_notebook(
    health_analysis: HealthEconomicsAnalysis, output_path: str
) -> None:
    """Quick export to Jupyter notebook."""
    integration = EcosystemIntegration()
    integration.export_to_external(
        "workflows", health_analysis, output_path, format="jupyter"
    )


def quick_r_export(health_analysis: HealthEconomicsAnalysis, output_path: str) -> None:
    """Quick export to R script."""
    integration = EcosystemIntegration()
    integration.export_to_external(
        "workflows", health_analysis, output_path, format="r"
    )


def convert_treeage_to_voi(file_path: str) -> dict[str, Any]:
    """Convert TreeAge model to voiage format."""
    integration = EcosystemIntegration()
    treeage_connector = integration.get_connector("treeage")
    if not isinstance(treeage_connector, TreeAgeConnector):
        raise_type_error("TreeAge connector is unavailable.")

    assert isinstance(treeage_connector, TreeAgeConnector)
    model_structure = treeage_connector.import_treeage_model(file_path)
    return treeage_connector.convert_to_voi_analysis(model_structure)
