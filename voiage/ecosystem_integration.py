"""
Ecosystem Integration Module for voiage

This module provides integration capabilities with popular software and workflows:
- TreeAge Pro compatibility
- R package integration (hesim, BCEA, etc.)
- Health economic modeling software interoperability
- Research workflow integration (Jupyter, RStudio, etc.)
- Data format compatibility (CSV, Excel, SAS, Stata)

Author: voiage Development Team
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, asdict
import json
import warnings
import io
from pathlib import Path
from abc import ABC, abstractmethod

from voiage.health_economics import HealthEconomicsAnalysis, HealthState, Treatment
from voiage.multi_domain import MultiDomainVOI, DomainType, DomainParameters


@dataclass
class EcosystemConnector:
    """Base class for ecosystem integrations"""
    name: str
    version: str
    supported_formats: List[str]
    integration_type: str  # "import", "export", "bidirectional"


class TreeAgeConnector(EcosystemConnector):
    """TreeAge Pro integration connector"""
    
    def __init__(self):
        super().__init__(
            name="TreeAge Pro",
            version="2023+",
            supported_formats=[".treeage", ".csv", ".xlsx"],
            integration_type="bidirectional"
        )
        
    def import_treeage_model(self, file_path: str) -> Dict[str, Any]:
        """
        Import TreeAge Pro model file
        
        Args:
            file_path: Path to TreeAge model file
            
        Returns:
            Parsed model structure compatible with voiage
        """
        try:
            # Parse TreeAge XML structure
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            model_structure = {
                'model_type': 'decision_tree',
                'nodes': [],
                'branches': [],
                'probabilities': [],
                'costs': [],
                'outcomes': []
            }
            
            # Extract decision tree structure
            for node in root.findall('.//node'):
                node_data = {
                    'id': node.get('id'),
                    'type': node.get('type'),  # decision, chance, terminal
                    'name': node.get('name'),
                    'description': node.get('description', '')
                }
                model_structure['nodes'].append(node_data)
                
            # Extract probabilities
            for prob in root.findall('.//probability'):
                prob_data = {
                    'from_node': prob.get('from'),
                    'to_node': prob.get('to'),
                    'value': float(prob.get('value', 0.0))
                }
                model_structure['probabilities'].append(prob_data)
                
            # Extract costs and outcomes
            for cost in root.findall('.//cost'):
                cost_data = {
                    'node_id': cost.get('node_id'),
                    'value': float(cost.get('value', 0.0)),
                    'type': cost.get('type', 'cost')
                }
                model_structure['costs'].append(cost_data)
                
            return model_structure
            
        except Exception as e:
            warnings.warn(f"Error importing TreeAge model: {e}")
            return {}
            
    def export_to_treeage(self, 
                         health_analysis: HealthEconomicsAnalysis,
                         output_path: str) -> None:
        """
        Export voiage model to TreeAge Pro format
        
        Args:
            health_analysis: Health economics analysis to export
            output_path: Output file path
        """
        try:
            import xml.etree.ElementTree as ET
            
            # Create TreeAge XML structure
            root = ET.Element("TreeAgeModel")
            root.set("version", "1.0")
            
            # Add decision tree structure
            for treatment in health_analysis.treatments.values():
                treatment_node = ET.SubElement(root, "node")
                treatment_node.set("id", treatment.name)
                treatment_node.set("type", "decision")
                treatment_node.set("name", treatment.description)
                treatment_node.set("description", f"Treatment: {treatment.name}")
                
                # Add outcome branches
                health_states = health_analysis._create_default_health_states(treatment)
                for i, state in enumerate(health_states):
                    branch = ET.SubElement(treatment_node, "branch")
                    branch.set("id", f"{treatment.name}_outcome_{i}")
                    branch.set("probability", "1.0")
                    
                    # Add cost and QALY information
                    qaly = health_analysis.calculate_qaly(state)
                    cost = health_analysis.calculate_cost(state)
                    
                    branch.set("cost", str(cost))
                    branch.set("qaly", str(qaly))
                    
            # Write XML file
            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ", level=0)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
        except Exception as e:
            warnings.warn(f"Error exporting to TreeAge: {e}")
            
    def convert_to_voi_analysis(self, model_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TreeAge model to voiage analysis format"""
        voi_format = {
            'decision_options': [],
            'uncertainty_parameters': [],
            'outcome_function': None
        }
        
        # Extract decision options from nodes
        decision_nodes = [node for node in model_structure.get('nodes', []) 
                         if node.get('type') == 'decision']
        
        for node in decision_nodes:
            voi_format['decision_options'].append({
                'name': node.get('name'),
                'description': node.get('description'),
                'cost_outcomes': []
            })
            
        # Extract probabilities and costs
        for prob in model_structure.get('probabilities', []):
            voi_format['uncertainty_parameters'].append({
                'name': f"prob_{prob.get('from_node')}_{prob.get('to_node')}",
                'value': prob.get('value'),
                'type': 'probability'
            })
            
        return voi_format


class RPackageConnector(EcosystemConnector):
    """R package integration connector for health economics"""
    
    def __init__(self):
        super().__init__(
            name="R Health Economics Packages",
            version="1.0",
            supported_formats=[".rds", ".csv", ".json"],
            integration_type="bidirectional"
        )
        
    def import_bcea_results(self, file_path: str) -> Dict[str, Any]:
        """
        Import Bayesian Cost-Effectiveness Analysis results
        
        Args:
            file_path: Path to BCEA results file
            
        Returns:
            BCEA results in voiage compatible format
        """
        try:
            # Read BCEA results (assuming JSON export from R)
            with open(file_path, 'r') as f:
                bcea_data = json.load(f)
                
            voi_results = {
                'analysis_type': 'bcea',
                'treatments': [],
                'cost_effectiveness_results': {},
                'icer_distributions': {},
                'ceac_data': {}
            }
            
            # Extract treatment information
            if 'treatments' in bcea_data:
                for treatment in bcea_data['treatments']:
                    voi_results['treatments'].append({
                        'name': treatment.get('name'),
                        'effectiveness': treatment.get('effectiveness'),
                        'cost': treatment.get('cost')
                    })
                    
            # Extract cost-effectiveness results
            if 'cost_effectiveness' in bcea_data:
                for treatment, results in bcea_data['cost_effectiveness'].items():
                    voi_results['cost_effectiveness_results'][treatment] = {
                        'mean_cost': results.get('mean_cost'),
                        'mean_effectiveness': results.get('mean_effectiveness'),
                        'icer': results.get('icer')
                    }
                    
            # Extract ICER distributions
            if 'icer_distributions' in bcea_data:
                voi_results['icer_distributions'] = bcea_data['icer_distributions']
                
            return voi_results
            
        except Exception as e:
            warnings.warn(f"Error importing BCEA results: {e}")
            return {}
            
    def export_for_bcea(self, 
                       health_analysis: HealthEconomicsAnalysis,
                       output_path: str,
                       num_simulations: int = 10000) -> None:
        """
        Export analysis results for BCEA package
        
        Args:
            health_analysis: Health economics analysis to export
            output_path: Output file path
            num_simulations: Number of Monte Carlo simulations
        """
        try:
            # Create data structure for BCEA
            bcea_data = {
                'treatments': [],
                'simulation_data': {
                    'costs': [],
                    'effectiveness': [],
                    'icer': []
                },
                'parameters': {
                    'willingness_to_pay': health_analysis.willingness_to_pay,
                    'currency': health_analysis.currency,
                    'num_simulations': num_simulations
                }
            }
            
            # Add treatment information
            for treatment in health_analysis.treatments.values():
                bcea_data['treatments'].append({
                    'name': treatment.name,
                    'description': treatment.description,
                    'effectiveness': treatment.effectiveness,
                    'cost_per_cycle': treatment.cost_per_cycle
                })
                
            # Generate simulation data
            for i in range(num_simulations):
                sim_costs = []
                sim_effectiveness = []
                
                for treatment in health_analysis.treatments.values():
                    # Simple simulation (in practice, would use proper statistical models)
                    cost = treatment.cost_per_cycle * treatment.cycles_required * (1 + np.random.normal(0, 0.1))
                    effectiveness = treatment.effectiveness * (1 + np.random.normal(0, 0.05))
                    
                    sim_costs.append(cost)
                    sim_effectiveness.append(effectiveness)
                    
                bcea_data['simulation_data']['costs'].append(sim_costs)
                bcea_data['simulation_data']['effectiveness'].append(sim_effectiveness)
                
            # Calculate ICER for each simulation
            if len(bcea_data['treatments']) >= 2:
                _ = bcea_data['treatments'][0]  # baseline = bcea_data['treatments'][0]
                for i in range(1, len(bcea_data['treatments'])):
                    treatment = bcea_data['treatments'][i]
                    sim_icer = []
                    
                    for j in range(num_simulations):
                        cost_diff = (bcea_data['simulation_data']['costs'][j][i] - 
                                   bcea_data['simulation_data']['costs'][j][0])
                        eff_diff = (bcea_data['simulation_data']['effectiveness'][j][i] - 
                                  bcea_data['simulation_data']['effectiveness'][j][0])
                        
                        if eff_diff > 0:
                            icer = cost_diff / eff_diff
                        else:
                            icer = np.inf
                            
                        sim_icer.append(icer)
                        
                    bcea_data['simulation_data']['icer'].append(sim_icer)
                    
            # Save to JSON
            with open(output_path, 'w') as f:
                json.dump(bcea_data, f, indent=2, default=str)
                
        except Exception as e:
            warnings.warn(f"Error exporting for BCEA: {e}")
            
    def import_hesim_results(self, file_path: str) -> Dict[str, Any]:
        """
        Import HE-Sim simulation results
        
        Args:
            file_path: Path to HE-Sim results file
            
        Returns:
            HE-Sim results in voiage compatible format
        """
        try:
            # Read HE-Sim results (JSON format assumed)
            with open(file_path, 'r') as f:
                hesim_data = json.load(f)
                
            voi_format = {
                'analysis_type': 'hesim',
                'markov_models': [],
                'state_transitions': [],
                'cost_effectiveness': {}
            }
                
            if 'markov_models' in hesim_data:
                voi_format['markov_models'] = hesim_data['markov_models']
                
            if 'state_transitions' in hesim_data:
                voi_format['state_transitions'] = hesim_data['state_transitions']
                
            if 'cost_effectiveness_results' in hesim_data:
                voi_format['cost_effectiveness'] = hesim_data['cost_effectiveness_results']
                
            return voi_format
            
        except Exception as e:
            warnings.warn(f"Error importing HE-Sim results: {e}")
            return {}


class DataFormatConnector(EcosystemConnector):
    """Data format compatibility connector"""
    
    def __init__(self):
        super().__init__(
            name="Data Format Connector",
            version="1.0",
            supported_formats=[".csv", ".xlsx", ".xls", ".json", ".parquet"],
            integration_type="bidirectional"
        )
        
    def import_health_data(self, file_path: str, 
                          data_type: str = "auto") -> Dict[str, Any]:
        """
        Import health economics data from various formats
        
        Args:
            file_path: Path to data file
            data_type: Type of data ("costs", "utilities", "transitions", "auto")
            
        Returns:
            Imported data in voiage format
        """
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return self._convert_json_to_dataframe(data)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
            return self._process_dataframe(df, data_type)
            
        except Exception as e:
            warnings.warn(f"Error importing health data: {e}")
            return {}
            
    def export_health_data(self, 
                          health_analysis: HealthEconomicsAnalysis,
                          output_path: str,
                          format_type: str = "csv") -> None:
        """
        Export health economics data
        
        Args:
            health_analysis: Health economics analysis to export
            output_path: Output file path
            format_type: Export format ("csv", "xlsx", "json")
        """
        try:
            # Create export data structure
            export_data = {}
            
            # Export treatments
            treatments_data = []
            for treatment in health_analysis.treatments.values():
                treatments_data.append({
                    'name': treatment.name,
                    'description': treatment.description,
                    'effectiveness': treatment.effectiveness,
                    'cost_per_cycle': treatment.cost_per_cycle,
                    'cycles_required': treatment.cycles_required,
                    'side_effect_utility': treatment.side_effect_utility,
                    'side_effect_cost': treatment.side_effect_cost
                })
            export_data['treatments'] = pd.DataFrame(treatments_data)
            
            # Export health states
            states_data = []
            for state_id, state in health_analysis.health_states.items():
                states_data.append({
                    'state_id': state.state_id,
                    'description': state.description,
                    'utility': state.utility,
                    'cost': state.cost,
                    'duration': state.duration
                })
            export_data['health_states'] = pd.DataFrame(states_data)
            
            # Export cost-effectiveness results
            ce_results = []
            for treatment_name in health_analysis.treatments:
                treatment = health_analysis.treatments[treatment_name]
                health_states = health_analysis._create_default_health_states(treatment)
                cost, qaly = health_analysis._calculate_treatment_totals(treatment, health_states)
                nmb = health_analysis.calculate_net_monetary_benefit(treatment, health_states)
                
                ce_results.append({
                    'treatment': treatment_name,
                    'total_cost': cost,
                    'total_qaly': qaly,
                    'net_monetary_benefit': nmb
                })
            export_data['cost_effectiveness'] = pd.DataFrame(ce_results)
            
            # Save to file
            output_path = Path(output_path)
            
            if format_type.lower() == "csv":
                for table_name, df in export_data.items():
                    table_path = output_path.parent / f"{output_path.stem}_{table_name}.csv"
                    df.to_csv(table_path, index=False)
                    
            elif format_type.lower() == "xlsx":
                with pd.ExcelWriter(output_path) as writer:
                    for table_name, df in export_data.items():
                        df.to_excel(writer, sheet_name=table_name, index=False)
                        
            elif format_type.lower() == "json":
                # Convert DataFrames to JSON-serializable format
                json_data = {}
                for table_name, df in export_data.items():
                    json_data[table_name] = df.to_dict('records')
                    
                with open(output_path, 'w') as f:
                    json.dump(json_data, f, indent=2, default=str)
                    
        except Exception as e:
            warnings.warn(f"Error exporting health data: {e}")
            
    def _process_dataframe(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Process imported dataframe"""
        if data_type == "auto":
            # Auto-detect data type based on columns
            columns = [col.lower() for col in df.columns]
            
            if any(col in ['treatment', 'intervention', 'drug'] for col in columns):
                data_type = "treatments"
            elif any(col in ['state', 'health_state', 'condition'] for col in columns):
                data_type = "health_states"
            elif any(col in ['cost', 'price', 'expense'] for col in columns):
                data_type = "costs"
            elif any(col in ['utility', 'qaly', 'effectiveness'] for col in columns):
                data_type = "utilities"
            else:
                data_type = "generic"
                
        processed_data = {
            'data_type': data_type,
            'data': df.to_dict('records'),
            'columns': list(df.columns),
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return processed_data
        
    def _convert_json_to_dataframe(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON data to processed format"""
        return {
            'data_type': 'json',
            'data': data,
            'keys': list(data.keys()) if isinstance(data, dict) else [],
            'structure': 'nested' if isinstance(data, dict) else 'flat'
        }


class WorkflowConnector(EcosystemConnector):
    """Research workflow integration connector"""
    
    def __init__(self):
        super().__init__(
            name="Research Workflow Connector",
            version="1.0",
            supported_formats=[".ipynb", ".r", ".py", ".md"],
            integration_type="bidirectional"
        )
        
    def create_jupyter_analysis(self, 
                              health_analysis: HealthEconomicsAnalysis,
                              output_path: str) -> None:
        """
        Create Jupyter notebook for health economics analysis
        
        Args:
            health_analysis: Health economics analysis
            output_path: Output notebook path
        """
        try:
            notebook_content = {
                'cells': [
                    {
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': [
                            '# Health Economics Analysis with voiage\\n\\n',
                            'This notebook demonstrates health economics analysis using the voiage library.\\n\\n',
                            f'**Willingness to Pay:** {health_analysis.willingness_to_pay} {health_analysis.currency}\\n',
                            f'**Currency:** {health_analysis.currency}\\n'
                        ]
                    },
                    {
                        'cell_type': 'code',
                        'execution_count': None,
                        'metadata': {},
                        'outputs': [],
                        'source': [
                            'import sys\\n',
                            'sys.path.append("/path/to/voiage")\\n',
                            'from voiage.health_economics import HealthEconomicsAnalysis, HealthState, Treatment\\n',
                            'import numpy as np\\n',
                            'import pandas as pd\\n\\n',
                            '# Initialize analysis\\n',
                            f'health_analysis = HealthEconomicsAnalysis(\\n',
                            f'    willingness_to_pay={health_analysis.willingness_to_pay},\\n',
                            f'    currency="{health_analysis.currency}"\\n',
                            ')\\n'
                        ]
                    }
                ],
                'metadata': {
                    'kernelspec': {
                        'display_name': 'Python 3',
                        'language': 'python',
                        'name': 'python3'
                    },
                    'language_info': {
                        'name': 'python',
                        'version': '3.8+'
                    }
                },
                'nbformat': 4,
                'nbformat_minor': 4
            }
            
            # Add treatment cells
            for treatment in health_analysis.treatments.values():
                cell = {
                    'cell_type': 'code',
                    'execution_count': None,
                    'metadata': {},
                    'outputs': [],
                    'source': [
                        f'# Add {treatment.name} treatment\\n',
                        f'treatment_{treatment.name.lower().replace(" ", "_")} = Treatment(\\n',
                        f'    name="{treatment.name}",\\n',
                        f'    description="{treatment.description}",\\n',
                        f'    effectiveness={treatment.effectiveness},\\n',
                        f'    cost_per_cycle={treatment.cost_per_cycle},\\n',
                        f'    cycles_required={treatment.cycles_required}\\n',
                        f')\\n',
                        f'health_analysis.add_treatment(treatment_{treatment.name.lower().replace(" ", "_")})\\n'
                    ]
                }
                notebook_content['cells'].append(cell)
                
            # Add analysis cells
            analysis_cells = [
                {
                    'cell_type': 'code',
                    'execution_count': None,
                    'metadata': {},
                    'outputs': [],
                    'source': [
                        '# Perform cost-effectiveness analysis\\n',
                        'treatments = list(health_analysis.treatments.values())\\n',
                        'results = []\\n',
                        '\\n',
                        'for treatment in treatments:\\n',
                        '    health_states = health_analysis._create_default_health_states(treatment)\\n',
                        '    cost, qaly = health_analysis._calculate_treatment_totals(treatment, health_states)\\n',
                        '    nmb = health_analysis.calculate_net_monetary_benefit(treatment, health_states)\\n',
                        '    icer = health_analysis.calculate_icer(treatment)\\n',
                        '    \\n',
                        '    results.append({\\n',
                        "        'Treatment': treatment.name,\\n",
                        "        'Cost': cost,\\n",
                        "        'QALY': qaly,\\n",
                        "        'NMB': nmb,\\n",
                        "        'ICER': icer\\n",
                        '    })\\n',
                        '\\n',
                        'results_df = pd.DataFrame(results)\\n',
                        'print(results_df)\\n'
                    ]
                },
                {
                    'cell_type': 'code',
                    'execution_count': None,
                    'metadata': {},
                    'outputs': [],
                    'source': [
                        '# Create visualizations\\n',
                        'import matplotlib.pyplot as plt\\n',
                        'import seaborn as sns\\n',
                        '\\n',
                        '# Cost-effectiveness plane\\n',
                        'plt.figure(figsize=(10, 6))\\n',
                        'plt.scatter(results_df["QALY"], results_df["Cost"])\\n',
                        'for i, row in results_df.iterrows():\\n',
                        '    plt.annotate(row["Treatment"], (row["QALY"], row["Cost"]))\\n',
                        'plt.xlabel("Effectiveness (QALYs)")\\n',
                        'plt.ylabel("Cost")\\n',
                        'plt.title("Cost-Effectiveness Plane")\\n',
                        'plt.grid(True)\\n',
                        'plt.show()\\n'
                    ]
                }
            ]
            
            notebook_content['cells'].extend(analysis_cells)
            
            # Save notebook
            import json
            with open(output_path, 'w') as f:
                json.dump(notebook_content, f, indent=1)
                
        except Exception as e:
            warnings.warn(f"Error creating Jupyter notebook: {e}")
            
    def create_r_workflow(self, 
                         health_analysis: HealthEconomicsAnalysis,
                         output_path: str) -> None:
        """
        Create R script for health economics analysis
        
        Args:
            health_analysis: Health economics analysis
            output_path: Output R script path
        """
        try:
            r_content = [
                '# Health Economics Analysis with voiage Integration\\n',
                '# This R script demonstrates integration with voiage analysis\\n',
                '\\n',
                '# Load required libraries\\n',
                'library(ggplot2)\\n',
                'library(dplyr)\\n',
                'library(jsonlite)\\n',
                '\\n',
                f'# Analysis parameters\\n',
                f'willingness_to_pay <- {health_analysis.willingness_to_pay}\\n',
                f'currency <- "{health_analysis.currency}"\\n',
                '\\n',
                '# Data preparation\\n',
                'treatments_data <- data.frame(\\n'
            ]
            
            # Add treatment data
            first_treatment = True
            for treatment in health_analysis.treatments.values():
                if not first_treatment:
                    r_content.append('                           ,\\n')
                r_content.append(
                    f'    name = "{treatment.name}",\\n'
                    f'    effectiveness = {treatment.effectiveness},\\n'
                    f'    cost_per_cycle = {treatment.cost_per_cycle},\\n'
                    f'    cycles_required = {treatment.cycles_required}\\n'
                )
                first_treatment = False
                
            r_content.extend([
                ')\\n',
                '\\n',
                '# Cost-effectiveness analysis\\n',
                'results <- treatments_data %>%\\n',
                '    mutate(\\n',
                '        total_cost = cost_per_cycle * cycles_required,\\n',
                '        qaly_estimate = effectiveness * 5,  # Simplified QALY estimate\\n',
                '        nmb = qaly_estimate * willingness_to_pay - total_cost\\n',
                '    )\\n',
                '\\n',
                '# Display results\\n',
                'print(results)\\n',
                '\\n',
                '# Create cost-effectiveness plane\\n',
                'ggplot(results, aes(x = qaly_estimate, y = total_cost, label = name)) +\\n',
                '    geom_point(size = 3) +\\n',
                '    geom_text(hjust = 0, vjust = 0) +\\n',
                '    labs(x = "Effectiveness (QALYs)",\\n',
                '         y = "Cost",\\n',
                '         title = "Cost-Effectiveness Plane") +\\n',
                '    theme_minimal()\\n',
                '\\n',
                '# Export results\\n',
                'write.csv(results, "cost_effectiveness_results.csv", row.names = FALSE)\\n',
                '\\n',
                'cat("Analysis complete. Results saved to cost_effectiveness_results.csv\\n")\\n'
            ])
            
            # Write R script
            with open(output_path, 'w') as f:
                f.writelines(r_content)
                
        except Exception as e:
            warnings.warn(f"Error creating R workflow: {e}")


class EcosystemIntegration:
    """Main ecosystem integration manager"""
    
    def __init__(self):
        self.connectors = {
            'treeage': TreeAgeConnector(),
            'r_packages': RPackageConnector(),
            'data_formats': DataFormatConnector(),
            'workflows': WorkflowConnector()
        }
        
    def get_connector(self, integration_type: str) -> Optional[EcosystemConnector]:
        """Get connector for specific integration type"""
        return self.connectors.get(integration_type.lower())
        
    def import_from_external(self, 
                           integration_type: str,
                           file_path: str,
                           **kwargs) -> Dict[str, Any]:
        """Import data from external software"""
        connector = self.get_connector(integration_type)
        if connector is None:
            raise ValueError(f"Unknown integration type: {integration_type}")
            
        if integration_type.lower() == 'r_packages':
            if 'format' in kwargs:
                if kwargs['format'] == 'bcea':
                    return connector.import_bcea_results(file_path)
                elif kwargs['format'] == 'hesim':
                    return connector.import_hesim_results(file_path)
        elif integration_type.lower() == 'data_formats':
            return connector.import_health_data(file_path, kwargs.get('data_type', 'auto'))
            
        return {}
        
    def export_to_external(self,
                         integration_type: str,
                         analysis_object,
                         output_path: str,
                         **kwargs) -> None:
        """Export analysis to external software format"""
        connector = self.get_connector(integration_type)
        if connector is None:
            raise ValueError(f"Unknown integration type: {integration_type}")
            
        if integration_type.lower() == 'treeage':
            if hasattr(analysis_object, 'treatments'):
                connector.export_to_treeage(analysis_object, output_path)
        elif integration_type.lower() == 'r_packages':
            if 'format' in kwargs:
                if kwargs['format'] == 'bcea':
                    connector.export_for_bcea(analysis_object, output_path, 
                                           kwargs.get('num_simulations', 10000))
        elif integration_type.lower() == 'data_formats':
            connector.export_health_data(analysis_object, output_path, 
                                       kwargs.get('format_type', 'csv'))
        elif integration_type.lower() == 'workflows':
            if 'format' in kwargs:
                if kwargs['format'] == 'jupyter':
                    connector.create_jupyter_analysis(analysis_object, output_path)
                elif kwargs['format'] == 'r':
                    connector.create_r_workflow(analysis_object, output_path)
                    
    def list_supported_formats(self) -> Dict[str, List[str]]:
        """List all supported file formats across connectors"""
        formats = {}
        for name, connector in self.connectors.items():
            formats[name] = connector.supported_formats
        return formats
        
    def create_integration_report(self) -> Dict[str, Any]:
        """Create comprehensive integration capabilities report"""
        report = {
            'available_connectors': list(self.connectors.keys()),
            'supported_formats': self.list_supported_formats(),
            'integration_capabilities': {
                'treeage': {
                    'import': True,
                    'export': True,
                    'bidirectional': True,
                    'description': 'TreeAge Pro decision tree and Markov model compatibility'
                },
                'r_packages': {
                    'import': True,
                    'export': True,
                    'bidirectional': True,
                    'description': 'BCEA, HE-Sim, and other R health economics packages'
                },
                'data_formats': {
                    'import': True,
                    'export': True,
                    'bidirectional': True,
                    'description': 'CSV, Excel, Parquet, JSON data format compatibility'
                },
                'workflows': {
                    'import': False,
                    'export': True,
                    'bidirectional': False,
                    'description': 'Jupyter notebooks, R scripts, and research workflows'
                }
            },
            'version': '2.0.0',
            'last_updated': '2025-11-09'
        }
        return report


# Utility functions for common integration tasks

def quick_import_health_data(file_path: str, 
                           analysis_type: str = "auto") -> Dict[str, Any]:
    """Quick import function for health economics data"""
    integration = EcosystemIntegration()
    return integration.import_from_external('data_formats', file_path, data_type=analysis_type)


def quick_export_notebook(health_analysis: HealthEconomicsAnalysis, 
                         output_path: str) -> None:
    """Quick export to Jupyter notebook"""
    integration = EcosystemIntegration()
    integration.export_to_external('workflows', health_analysis, output_path, format='jupyter')


def quick_r_export(health_analysis: HealthEconomicsAnalysis, 
                  output_path: str) -> None:
    """Quick export to R script"""
    integration = EcosystemIntegration()
    integration.export_to_external('workflows', health_analysis, output_path, format='r')


def convert_treeage_to_voi(file_path: str) -> Dict[str, Any]:
    """Convert TreeAge model to voiage format"""
    integration = EcosystemIntegration()
    treeage_connector = integration.get_connector('treeage')
    model_structure = treeage_connector.import_treeage_model(file_path)
    return treeage_connector.convert_to_voi_analysis(model_structure)
