"""
Health Economics Module for voiage

This module provides specialized health economics analysis capabilities including:
- Quality-Adjusted Life Year (QALY) analysis
- Cost-effectiveness analysis with ICER calculations
- Net monetary benefit frameworks
- Health outcome modeling
- Budget impact analysis
- Health technology assessment integration

Author: voiage Development Team
Version: 2.0.0
"""

import jax.numpy as jnp
import jax.random as random
from jax import jit, vmap
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from dataclasses import dataclass
from voiage.analysis import DecisionAnalysis


@dataclass
class HealthState:
    """Represents a health state with utility value and cost"""
    state_id: str
    description: str
    utility: float  # Quality of life weight (0-1)
    cost: float  # Annual cost in currency units
    duration: float  # Expected duration in years
    transition_probabilities: Optional[Dict[str, float]] = None  # Probabilities to other states


@dataclass
class Treatment:
    """Represents a medical treatment with effectiveness and costs"""
    name: str
    description: str
    effectiveness: float  # Treatment effect (0-1, 1 = perfect cure)
    cost_per_cycle: float  # Cost per treatment cycle
    cycles_required: int  # Number of treatment cycles
    side_effect_utility: float = 0.0  # Utility reduction due to side effects
    side_effect_cost: float = 0.0  # Additional cost for managing side effects


class HealthEconomicsAnalysis:
    """
    Comprehensive health economics analysis class
    
    Integrates with voiage's VOI analysis to provide specialized health economic
    evaluations including cost-effectiveness, QALY analysis, and decision making
    under uncertainty.
    """
    
    def __init__(self, willingness_to_pay: float = 50000.0, currency: str = "USD"):
        """
        Initialize health economics analysis
        
        Args:
            willingness_to_pay: Willingness to pay per QALY in currency units
            currency: Currency code for cost calculations
        """
        self.willingness_to_pay = willingness_to_pay
        self.currency = currency
        self.health_states: Dict[str, HealthState] = {}
        self.treatments: Dict[str, Treatment] = {}
        self.transition_matrix: Optional[jnp.ndarray] = None
        
    def add_health_state(self, health_state: HealthState) -> None:
        """Add a health state to the analysis"""
        self.health_states[health_state.state_id] = health_state
        
    def add_treatment(self, treatment: Treatment) -> None:
        """Add a treatment option to the analysis"""
        self.treatments[treatment.name] = treatment
        
    def calculate_qaly(self, 
                      health_state: HealthState, 
                      discount_rate: float = 0.03,
                      time_horizon: float = 10.0) -> float:
        """
        Calculate Quality-Adjusted Life Years (QALY) for a health state
        
        Args:
            health_state: The health state to analyze
            discount_rate: Annual discount rate for future health benefits
            time_horizon: Time horizon for analysis in years
            
        Returns:
            QALY value
        """
        # Use health_state duration if no time_horizon specified
        if time_horizon is None:
            time_horizon = health_state.duration
            
        time_points = jnp.arange(0, time_horizon, 0.1)
        utility_stream = health_state.utility * jnp.ones_like(time_points)
        
        # Apply discounting
        discounted_utility = utility_stream * jnp.exp(-discount_rate * time_points)
        
        # Simple analytical calculation for QALY
        if discount_rate == 0:
            qaly = health_state.utility * time_horizon
        else:
            # Present value of QALY stream with discounting
            qaly = health_state.utility * (1 - jnp.exp(-discount_rate * time_horizon)) / discount_rate
        
        return jnp.minimum(qaly, time_horizon)  # Cap at time horizon
        
    def calculate_cost(self, 
                      health_state: HealthState,
                      discount_rate: float = 0.03,
                      time_horizon: float = 10.0) -> float:
        """
        Calculate discounted costs for a health state
        
        Args:
            health_state: The health state to analyze
            discount_rate: Annual discount rate for costs
            time_horizon: Time horizon for analysis in years
            
        Returns:
            Total discounted cost
        """
        time_points = jnp.arange(0, time_horizon, 0.1)
        cost_stream = health_state.cost * jnp.ones_like(time_points)
        
        # Apply discounting
        discounted_cost = cost_stream * jnp.exp(-discount_rate * time_points)
        
        # Simple analytical calculation for costs
        if discount_rate == 0:
            total_cost = health_state.cost * time_horizon
        else:
            # Present value of cost stream with discounting
            total_cost = health_state.cost * (1 - jnp.exp(-discount_rate * time_horizon)) / discount_rate
        
        return jnp.minimum(total_cost, time_horizon * health_state.cost)
        
    def calculate_icer(self, 
                      treatment1: Treatment, 
                      treatment2: Optional[Treatment] = None,
                      health_states1: List[HealthState] = None,
                      health_states2: List[HealthState] = None) -> float:
        """
        Calculate Incremental Cost-Effectiveness Ratio (ICER)
        
        Args:
            treatment1: First treatment to compare
            treatment2: Second treatment (comparator), defaults to standard care
            health_states1: Health states for treatment1
            health_states2: Health states for treatment2
            
        Returns:
            ICER value (cost per QALY gained)
        """
        if treatment2 is None:
            treatment2 = Treatment("Standard Care", "Standard treatment", 0.0, 0.0, 0)
            
        if health_states1 is None:
            # Create default health states based on treatment effectiveness
            health_states1 = self._create_default_health_states(treatment1)
        if health_states2 is None:
            health_states2 = self._create_default_health_states(treatment2)
            
        # Calculate total costs and QALYs
        cost1, qaly1 = self._calculate_treatment_totals(treatment1, health_states1)
        cost2, qaly2 = self._calculate_treatment_totals(treatment2, health_states2)
        
        # Calculate ICER
        incremental_cost = cost1 - cost2
        incremental_qaly = qaly1 - qaly2
        
        if incremental_qaly <= 0:
            return float('inf')  # Treatment is dominated (worse outcomes, higher cost)
            
        icer = incremental_cost / incremental_qaly
        return float(icer)
        
    def calculate_net_monetary_benefit(self, 
                                     treatment: Treatment,
                                     health_states: List[HealthState] = None) -> float:
        """
        Calculate Net Monetary Benefit (NMB) for a treatment
        
        Args:
            treatment: Treatment to analyze
            health_states: Health states for the treatment
            
        Returns:
            Net Monetary Benefit value
        """
        if health_states is None:
            health_states = self._create_default_health_states(treatment)
            
        cost, qaly = self._calculate_treatment_totals(treatment, health_states)
        
        # NMB = (QALY * WTP) - Cost
        nmb = (qaly * self.willingness_to_pay) - cost
        return nmb
        
    def create_cost_effectiveness_acceptability_curve(self, 
                                                     treatment: Treatment,
                                                     health_states: List[HealthState] = None,
                                                     wtp_range: Tuple[float, float] = (0, 200000),
                                                     num_points: int = 100) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Create Cost-Effectiveness Acceptability Curve (CEAC)
        
        Args:
            treatment: Treatment to analyze
            health_states: Health states for the treatment
            wtp_range: Range of willingness-to-pay values to evaluate
            num_points: Number of evaluation points
            
        Returns:
            Tuple of (wtp_values, ceac_probabilities)
        """
        wtp_values = jnp.linspace(wtp_range[0], wtp_range[1], num_points)
        
        def nmb_for_wtp(wtp):
            original_wtp = self.willingness_to_pay
            self.willingness_to_pay = wtp
            nmb = self.calculate_net_monetary_benefit(treatment, health_states)
            self.willingness_to_pay = original_wtp
            return nmb
            
        # Calculate NMB for each WTP threshold
        nmb_values = vmap(nmb_for_wtp)(wtp_values)
        
        # Probability of cost-effectiveness (NMB > 0)
        ceac_probabilities = jnp.mean(nmb_values[:, None] > 0)
        
        return wtp_values, ceac_probabilities
        
    def budget_impact_analysis(self,
                             treatment: Treatment,
                             population_size: int = 100000,
                             adoption_rate: float = 0.5,
                             time_horizon: int = 5,
                             annual_budget: float = 10000000) -> Dict[str, float]:
        """
        Perform budget impact analysis
        
        Args:
            treatment: Treatment to analyze
            population_size: Total target population
            adoption_rate: Expected adoption rate of new treatment
            time_horizon: Analysis time horizon in years
            annual_budget: Available annual budget
            
        Returns:
            Dictionary with budget impact metrics
        """
        health_states = self._create_default_health_states(treatment)
        treatment_cost, _ = self._calculate_treatment_totals(treatment, health_states)
        
        # Calculate budget impact
        patients_treated = population_size * adoption_rate
        total_budget_impact = patients_treated * treatment_cost * time_horizon
        annual_impact = total_budget_impact / time_horizon
        
        # Budget impact as percentage of available budget
        impact_percentage = (annual_impact / annual_budget) * 100
        
        return {
            'annual_budget_impact': float(annual_impact),
            'total_budget_impact': float(total_budget_impact),
            'budget_impact_percentage': float(impact_percentage),
            'patients_treated': int(patients_treated),
            'sustainability_score': max(0, 100 - impact_percentage)  # 0-100 scale
        }
        
    def probabilistic_sensitivity_analysis(self,
                                         treatment: Treatment,
                                         num_simulations: int = 1000,
                                         utility_uncertainty: float = 0.1,
                                         cost_uncertainty: float = 0.2) -> Dict[str, Any]:
        """
        Perform probabilistic sensitivity analysis (PSA)
        
        Args:
            treatment: Treatment to analyze
            num_simulations: Number of Monte Carlo simulations
            utility_uncertainty: Relative uncertainty in utility values (±%)
            cost_uncertainty: Relative uncertainty in cost values (±%)
            
        Returns:
            Dictionary with PSA results including distributions and summary statistics
        """
        # Generate random samples
        key = random.PRNGKey(42)
        
        utility_multipliers = random.normal(key, (num_simulations,)) * utility_uncertainty + 1.0
        cost_multipliers = random.normal(key, (num_simulations,)) * cost_uncertainty + 1.0
        
        # Calculate outcomes for each simulation
        def simulate_once(i):
            health_state = HealthState(
                state_id="temp",
                description="Temporary state",
                utility=jnp.maximum(0, treatment.effectiveness * utility_multipliers[i]),
                cost=treatment.cost_per_cycle * treatment.cycles_required * cost_multipliers[i],
                duration=5.0
            )
            
            qaly = self.calculate_qaly(health_state)
            cost = self.calculate_cost(health_state)
            icer = self.calculate_icer(treatment)
            nmb = self.calculate_net_monetary_benefit(treatment, [health_state])
            
            return jnp.array([qaly, cost, icer, nmb])
            
        results = vmap(simulate_once)(jnp.arange(num_simulations))
        
        qaly_samples, cost_samples, icer_samples, nmb_samples = results.T
        
        return {
            'qaly_distribution': {
                'mean': float(jnp.mean(qaly_samples)),
                'std': float(jnp.std(qaly_samples)),
                'q025': float(jnp.percentile(qaly_samples, 2.5)),
                'q975': float(jnp.percentile(qaly_samples, 97.5))
            },
            'cost_distribution': {
                'mean': float(jnp.mean(cost_samples)),
                'std': float(jnp.std(cost_samples)),
                'q025': float(jnp.percentile(cost_samples, 2.5)),
                'q975': float(jnp.percentile(cost_samples, 97.5))
            },
            'icer_distribution': {
                'mean': float(jnp.mean(icer_samples[icer_samples != float('inf')])),
                'median': float(jnp.median(icer_samples[icer_samples != float('inf')])),
                'q025': float(jnp.percentile(icer_samples[icer_samples != float('inf')], 2.5)),
                'q975': float(jnp.percentile(icer_samples[icer_samples != float('inf')], 97.5))
            },
            'net_monetary_benefit': {
                'mean': float(jnp.mean(nmb_samples)),
                'probability_positive': float(jnp.mean(nmb_samples > 0)),
                'q025': float(jnp.percentile(nmb_samples, 2.5)),
                'q975': float(jnp.percentile(nmb_samples, 97.5))
            },
            'simulation_parameters': {
                'num_simulations': num_simulations,
                'utility_uncertainty': utility_uncertainty,
                'cost_uncertainty': cost_uncertainty
            }
        }
        
    def _create_default_health_states(self, treatment: Treatment) -> List[HealthState]:
        """Create default health states based on treatment effectiveness"""
        if treatment.effectiveness > 0.8:
            # High effectiveness - mostly healthy state
            return [
                HealthState("Healthy", "Post-treatment health", 0.9, 1000, 10.0),
                HealthState("Mild", "Mild symptoms", 0.7, 5000, 2.0),
                HealthState("Dead", "Death", 0.0, 10000, 0.1)
            ]
        elif treatment.effectiveness > 0.5:
            # Medium effectiveness
            return [
                HealthState("Improved", "Moderate improvement", 0.8, 2000, 8.0),
                HealthState("Mild", "Mild symptoms", 0.6, 4000, 3.0),
                HealthState("Dead", "Death", 0.0, 10000, 0.5)
            ]
        else:
            # Low effectiveness
            return [
                HealthState("Slight", "Slight improvement", 0.7, 3000, 5.0),
                HealthState("Unchanged", "No improvement", 0.5, 6000, 3.0),
                HealthState("Dead", "Death", 0.0, 8000, 1.0)
            ]
            
    def _calculate_treatment_totals(self, treatment: Treatment, health_states: List[HealthState]) -> Tuple[float, float]:
        """Calculate total cost and QALY for a treatment across health states"""
        total_cost = treatment.cost_per_cycle * treatment.cycles_required + treatment.side_effect_cost
        total_qaly = 0.0
        
        for state in health_states:
            qaly = self.calculate_qaly(state)
            total_qaly += qaly * state.utility
            
        return total_cost, total_qaly
        
    def create_voi_analysis_for_health_decisions(self,
                                               treatments: List[Treatment],
                                               decision_outcome_function,
                                               additional_parameters: Dict[str, Any] = None) -> DecisionAnalysis:
        """
        Create VOI analysis specifically for health economic decisions
        
        Args:
            treatments: List of treatment options to compare
            decision_outcome_function: Function that returns outcomes for each treatment
            additional_parameters: Additional parameters for the analysis
            
        Returns:
            DecisionAnalysis object configured for health economics
        """
        if additional_parameters is None:
            additional_parameters = {}
            
        # Create parameters for VOI analysis
        parameters = {
            'treatments': treatments,
            'willingness_to_pay': self.willingness_to_pay,
            'currency': self.currency,
            'analysis_type': 'health_economics',
            **additional_parameters
        }
        
        # Initialize DecisionAnalysis with health economics parameters
        analysis = DecisionAnalysis(backend='jax', **parameters)
        
        # Add custom outcome function
        analysis.decision_function = lambda params, **kwargs: self._health_decision_outcomes(
            treatments, decision_outcome_function, **kwargs
        )
        
        return analysis
        
    def _health_decision_outcomes(self, treatments, decision_function, **kwargs):
        """Custom decision function for health economics outcomes"""
        outcomes = []
        
        for treatment in treatments:
            if hasattr(decision_function, '__call__'):
                outcome = decision_function(treatment, **kwargs)
            else:
                # Default outcome calculation
                health_states = self._create_default_health_states(treatment)
                cost, qaly = self._calculate_treatment_totals(treatment, health_states)
                outcome = {
                    'qaly': qaly,
                    'cost': cost,
                    'nmb': (qaly * self.willingness_to_pay) - cost,
                    'icer': self.calculate_icer(treatment)
                }
                
            outcomes.append(outcome)
            
        return outcomes


# Utility functions for common health economics calculations

def calculate_icer_simple(cost_intervention: float, 
                         effect_intervention: float,
                         cost_comparator: float = 0.0,
                         effect_comparator: float = 0.0) -> float:
    """
    Simple ICER calculation for two interventions
    
    Args:
        cost_intervention: Cost of intervention
        effect_intervention: Effect of intervention (QALYs)
        cost_comparator: Cost of comparator
        effect_comparator: Effect of comparator (QALYs)
        
    Returns:
        ICER value
    """
    incremental_cost = cost_intervention - cost_comparator
    incremental_effect = effect_intervention - effect_comparator
    
    if incremental_effect <= 0:
        return float('inf')
        
    return incremental_cost / incremental_effect


def calculate_net_monetary_benefit_simple(effect: float, 
                                        cost: float, 
                                        wtp: float) -> float:
    """
    Simple Net Monetary Benefit calculation
    
    Args:
        effect: Health effect (QALYs)
        cost: Cost
        wtp: Willingness to pay per QALY
        
    Returns:
        Net monetary benefit
    """
    return (effect * wtp) - cost


def qaly_calculator(life_years: float, 
                   utility_weight: float, 
                   discount_rate: float = 0.03) -> float:
    """
    Simple QALY calculator
    
    Args:
        life_years: Life years in health state
        utility_weight: Quality of life weight (0-1)
        discount_rate: Discount rate for future benefits
        
    Returns:
        QALY value
    """
    if life_years <= 0 or utility_weight <= 0:
        return 0.0
        
    # Simplified QALY calculation
    undiscounted_qaly = life_years * utility_weight
    
    # Apply simple discounting for multi-year horizons
    if life_years > 1:
        discounted_qaly = undiscounted_qaly * (1 - (1 + discount_rate) ** (-life_years)) / (discount_rate * life_years)
    else:
        discounted_qaly = undiscounted_qaly
        
    return min(discounted_qaly, life_years)
