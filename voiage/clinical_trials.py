"""
Clinical Trial Design Optimization Module

This module provides advanced clinical trial design optimization using Value of Information analysis:
- Sample size optimization based on VOI
- Adaptive trial design methodologies
- Interim analysis scheduling
- Multi-arm trial optimization
- Bayesian trial design integration
- Health economic endpoints in trial design

Author: voiage Development Team
Version: 2.0.0
"""

import jax.numpy as jnp
import jax.random as random
from jax import jit, vmap, grad
import jax.scipy.stats as jstats
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum
from scipy import stats
import warnings

from voiage.health_economics import HealthEconomicsAnalysis, Treatment, HealthState
from voiage.analysis import DecisionAnalysis


class TrialType(Enum):
    """Types of clinical trials"""
    SUPERIORITY = "superiority"
    NON_INFERIORITY = "non_inferiority"
    SUPERIORITY_WITH_HEALTH_ECONOMICS = "superiority_he"
    ADAPTIVE = "adaptive"
    PLATFORM = "platform"
    N_OF_1 = "n_of_1"


class EndpointType(Enum):
    """Types of clinical trial endpoints"""
    BINARY = "binary"
    CONTINUOUS = "continuous"
    TIME_TO_EVENT = "time_to_event"
    COMPOSITE = "composite"
    QALY = "qaly"
    COST = "cost"
    COST_EFFECTIVENESS = "cost_effectiveness"


class AdaptationRule(Enum):
    """Types of adaptation rules for adaptive trials"""
    SAMPLE_SIZE_REESTIMATION = "sample_size_reest"
    DROPPING_ARMS = "dropping_arms"
    DOSE_FINDING = "dose_finding"
    EARLY_SUCCESS = "early_success"
    EARLY_FUTILITY = "early_futility"


@dataclass
class TrialDesign:
    """Clinical trial design parameters"""
    trial_type: TrialType
    primary_endpoint: EndpointType
    sample_size: int
    number_of_arms: int = 1
    allocation_ratio: List[float] = field(default_factory=lambda: [1.0])
    interim_analyses: int = 0
    adaptation_schedule: List[int] = field(default_factory=list)
    alpha: float = 0.05
    beta: float = 0.2
    effect_size: float = 0.5
    variance: float = 1.0
    baseline_rate: float = 0.5
    
    # Health economics specific parameters
    willingness_to_pay: float = 50000.0
    health_economic_endpoint: bool = False
    budget_constraint: Optional[float] = None
    time_horizon: float = 5.0
    
    # Adaptive parameters
    adaptation_rules: List[AdaptationRule] = field(default_factory=list)
    adaptation_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Bayesian parameters
    prior_distribution: Dict[str, Any] = field(default_factory=dict)
    bayesian_analysis: bool = False
    posterior_threshold: float = 0.95


@dataclass
class TrialOutcome:
    """Clinical trial outcome data"""
    treatment_effect: float
    p_value: float
    confidence_interval: Tuple[float, float]
    power_achieved: float
    sample_size_used: int
    
    # Health economics outcomes
    cost_effectiveness_ratio: Optional[float] = None
    incremental_qaly: Optional[float] = None
    net_monetary_benefit: Optional[float] = None
    probability_cost_effective: Optional[float] = None
    
    # Adaptive trial outcomes
    adaptation_triggered: bool = False
    adaptation_type: Optional[AdaptationRule] = None
    final_sample_size: int = 0


class VOIBasedSampleSizeOptimizer:
    """
    Sample size optimization based on Value of Information analysis
    
    This class determines optimal sample sizes for clinical trials by considering
    the value of information gained from additional participants versus the cost
    of including them in the trial.
    """
    
    def __init__(self, trial_design: TrialDesign):
        """
        Initialize sample size optimizer
        
        Args:
            trial_design: Clinical trial design parameters
        """
        self.trial_design = trial_design
        self.health_analysis = HealthEconomicsAnalysis(
            willingness_to_pay=trial_design.willingness_to_pay
        )
        
    def calculate_voi_per_participant(self, 
                                    treatment: Treatment,
                                    sample_size: int) -> float:
        """
        Calculate the value of information gained per additional participant
        
        Args:
            treatment: Treatment being tested
            sample_size: Current sample size
            
        Returns:
            VOI per additional participant
        """
        # Simulate trial outcomes at different sample sizes
        power_increase = self._calculate_power_increase(sample_size, sample_size + 1)
        
        # Convert power increase to health economic value
        health_states = self.health_analysis._create_default_health_states(treatment)
        _, qaly = self.health_analysis._calculate_treatment_totals(treatment, health_states)
        
        # Value of additional precision in health outcome estimate
        precision_value = power_increase * qaly * self.trial_design.willingness_to_pay
        
        # Value of reduced decision uncertainty
        decision_uncertainty_reduction = self._calculate_uncertainty_reduction(sample_size)
        decision_value = decision_uncertainty_reduction * self._estimate_decision_value()
        
        total_voi = precision_value + decision_value
        return total_voi
        
    def optimize_sample_size(self,
                           treatment: Treatment,
                           min_sample_size: int = 30,
                           max_sample_size: int = 1000,
                           cost_per_participant: float = 1000.0) -> Dict[str, Any]:
        """
        Optimize sample size based on VOI analysis
        
        Args:
            treatment: Treatment being tested
            min_sample_size: Minimum acceptable sample size
            max_sample_size: Maximum feasible sample size
            cost_per_participant: Cost to include one participant
            
        Returns:
            Optimization results including optimal sample size and VOI analysis
        """
        sample_sizes = jnp.arange(min_sample_size, max_sample_size + 1, 10)
        
        # Calculate VOI and cost for each sample size
        voi_per_participant = vmap(lambda n: self.calculate_voi_per_participant(treatment, n))(sample_sizes)
        total_voi = vmap(lambda n: self._calculate_total_voi(treatment, n))(sample_sizes)
        total_costs = sample_sizes * cost_per_participant
        
        # Net benefit calculation
        net_benefits = total_voi - total_costs
        
        # Find optimal sample size
        optimal_idx = jnp.argmax(net_benefits)
        optimal_sample_size = sample_sizes[optimal_idx]
        max_net_benefit = net_benefits[optimal_idx]
        
        # Calculate efficiency metrics
        voi_efficiency = total_voi / (total_costs + 1e-6)  # Avoid division by zero
        
        return {
            'optimal_sample_size': int(optimal_sample_size),
            'max_net_benefit': float(max_net_benefit),
            'total_voi_at_optimal': float(total_voi[optimal_idx]),
            'total_cost_at_optimal': float(total_costs[optimal_idx]),
            'voi_efficiency': float(voi_efficiency[optimal_idx]),
            'sample_size_range': (int(min_sample_size), int(max_sample_size)),
            'power_at_optimal': float(self._calculate_power_at_sample_size(optimal_sample_size)),
            'optimization_curve': {
                'sample_sizes': sample_sizes.tolist(),
                'voi_per_participant': voi_per_participant.tolist(),
                'total_voi': total_voi.tolist(),
                'total_costs': total_costs.tolist(),
                'net_benefits': net_benefits.tolist()
            }
        }
        
    def _calculate_power_increase(self, current_size: int, new_size: int) -> float:
        """Calculate increase in statistical power from additional participants"""
        # Simplified power calculation for demonstration
        current_power = self._calculate_power_at_sample_size(current_size)
        new_power = self._calculate_power_at_sample_size(new_size)
        return jnp.maximum(0, new_power - current_power)
        
    def _calculate_power_at_sample_size(self, sample_size: int) -> float:
        """Calculate statistical power for given sample size"""
        if self.trial_design.number_of_arms == 1:
            n_per_group = sample_size
        else:
            n_per_group = sample_size // self.trial_design.number_of_arms
            
        return jnp.where(n_per_group <= 0, 0.0, self._calculate_power_valid_sample(n_per_group))

    def _calculate_power_valid_sample(self, n_per_group: int) -> float:
        """Calculate statistical power for valid sample size"""
        # Approximate power calculation for two-sample t-test
            
        # Approximate power calculation for two-sample t-test
        delta = self.trial_design.effect_size
        sigma = self.trial_design.variance
        
        # Non-centrality parameter
        ncp = delta * jnp.sqrt(n_per_group / (2 * sigma))
        
        # Power approximation (JAX-compatible)
        critical_value = jstats.norm.ppf(1 - self.trial_design.alpha / 2)
        power = 1 - jstats.norm.cdf(critical_value - ncp)
        
        return power
        
    def _calculate_uncertainty_reduction(self, sample_size: int) -> float:
        """Calculate reduction in parameter uncertainty with sample size"""
        # Standard error reduction with sample size
        return 1.0 / jnp.sqrt(jnp.maximum(1, sample_size))
        
    def _estimate_decision_value(self) -> float:
        """Estimate the value of making a correct decision"""
        # This would typically be based on health economic model
        return self.trial_design.willingness_to_pay * 0.1  # 10% of WTP as decision value
        
    def _calculate_total_voi(self, treatment: Treatment, sample_size: int) -> float:
        """Calculate total VOI for given sample size"""
        voi_per_participant = self.calculate_voi_per_participant(treatment, sample_size)
        
        # Diminishing returns model
        diminishing_factor = 1.0 / (1.0 + 0.001 * sample_size)
        total_voi = voi_per_participant * sample_size * diminishing_factor
        
        return total_voi


class AdaptiveTrialOptimizer:
    """
    Adaptive clinical trial design optimization using VOI
    
    This class optimizes adaptive trial designs by determining optimal
    adaptation rules, timing, and thresholds based on VOI analysis.
    """
    
    def __init__(self, trial_design: TrialDesign):
        """
        Initialize adaptive trial optimizer
        
        Args:
            trial_design: Adaptive trial design parameters
        """
        self.trial_design = trial_design
        self.health_analysis = HealthEconomicsAnalysis(
            willingness_to_pay=trial_design.willingness_to_pay
        )
        
    def optimize_adaptation_schedule(self,
                                   treatment: Treatment,
                                   max_interim_analyses: int = 5) -> Dict[str, Any]:
        """
        Optimize the schedule of interim analyses
        
        Args:
            treatment: Treatment being tested
            max_interim_analyses: Maximum number of interim analyses
            
        Returns:
            Optimal adaptation schedule and timing
        """
        final_sample_size = self.trial_design.sample_size
        
        # Test different adaptation schedules
        schedules = []
        for k in range(1, max_interim_analyses + 1):
            # Equally spaced adaptation times
            adaptation_times = [(i + 1) * final_sample_size // (k + 1) for i in range(k)]
            
            # Calculate expected VOI for this schedule
            expected_voi = self._calculate_adaptation_schedule_voi(
                treatment, adaptation_times, final_sample_size
            )
            
            # Calculate expected cost (delays, complexity)
            adaptation_cost = k * 1000  # Cost per adaptation
            
            schedules.append({
                'num_adaptations': k,
                'adaptation_times': adaptation_times,
                'expected_voi': expected_voi,
                'adaptation_cost': adaptation_cost,
                'net_benefit': expected_voi - adaptation_cost
            })
            
        # Find optimal schedule
        optimal_schedule = max(schedules, key=lambda s: s['net_benefit'])
        
        return {
            'optimal_schedule': optimal_schedule,
            'all_schedules': schedules,
            'recommendation': f"Perform {optimal_schedule['num_adaptations']} interim analyses at "
                            f"sample sizes {optimal_schedule['adaptation_times']}"
        }
        
    def optimize_adaptation_thresholds(self,
                                     treatment: Treatment,
                                     adaptation_rule: AdaptationRule) -> Dict[str, Any]:
        """
        Optimize adaptation thresholds for specific adaptation rule
        
        Args:
            treatment: Treatment being tested
            adaptation_rule: Type of adaptation rule to optimize
            
        Returns:
            Optimal thresholds and performance metrics
        """
        if adaptation_rule == AdaptationRule.EARLY_SUCCESS:
            return self._optimize_early_success_threshold(treatment)
        elif adaptation_rule == AdaptationRule.EARLY_FUTILITY:
            return self._optimize_early_futility_threshold(treatment)
        elif adaptation_rule == AdaptationRule.SAMPLE_SIZE_REESTIMATION:
            return self._optimize_sample_size_reest_threshold(treatment)
        else:
            return self._optimize_general_threshold(treatment, adaptation_rule)
            
    def _optimize_early_success_threshold(self, treatment: Treatment) -> Dict[str, Any]:
        """Optimize early success stopping threshold"""
        thresholds = jnp.arange(0.8, 0.99, 0.02)
        performance_metrics = []
        
        for threshold in thresholds:
            # Simulate early stopping scenarios
            early_stop_rate = self._simulate_early_stop_rate(threshold, treatment)
            power_loss = self._simulate_power_loss(threshold, treatment)
            time_savings = early_stop_rate * 0.3  # 30% time reduction estimate
            
            # Calculate net benefit
            net_benefit = time_savings - power_loss
            
            performance_metrics.append({
                'threshold': float(threshold),
                'early_stop_rate': float(early_stop_rate),
                'power_loss': float(power_loss),
                'net_benefit': float(net_benefit)
            })
            
        # Find optimal threshold
        optimal = max(performance_metrics, key=lambda x: x['net_benefit'])
        
        return {
            'optimal_threshold': optimal['threshold'],
            'performance_metrics': performance_metrics,
            'expected_benefit': optimal['net_benefit']
        }
        
    def _optimize_early_futility_threshold(self, treatment: Treatment) -> Dict[str, Any]:
        """Optimize early futility stopping threshold"""
        thresholds = jnp.arange(0.1, 0.5, 0.05)
        performance_metrics = []
        
        for threshold in thresholds:
            # Simulate early futility scenarios
            futility_rate = self._simulate_futility_rate(threshold, treatment)
            power_retained = 1.0 - self._simulate_power_loss(1.0 - threshold, treatment)
            cost_savings = futility_rate * 0.2  # 20% cost reduction estimate
            
            # Calculate net benefit
            net_benefit = cost_savings + power_retained
            
            performance_metrics.append({
                'threshold': float(threshold),
                'futility_rate': float(futility_rate),
                'power_retained': float(power_retained),
                'net_benefit': float(net_benefit)
            })
            
        optimal = max(performance_metrics, key=lambda x: x['net_benefit'])
        
        return {
            'optimal_threshold': optimal['threshold'],
            'performance_metrics': performance_metrics,
            'expected_benefit': optimal['net_benefit']
        }
        
    def _optimize_sample_size_reest_threshold(self, treatment: Treatment) -> Dict[str, Any]:
        """Optimize sample size re-estimation thresholds"""
        # This is more complex and would typically involve simulating
        # the entire adaptive trial process
        return {
            'optimal_threshold': 0.5,
            'performance_metrics': [],
            'expected_benefit': 0.1,
            'note': 'Sample size re-estimation optimization requires full simulation'
        }
        
    def _optimize_general_threshold(self, treatment: Treatment, rule: AdaptationRule) -> Dict[str, Any]:
        """Optimize thresholds for general adaptation rules"""
        return {
            'optimal_threshold': 0.5,
            'performance_metrics': [],
            'expected_benefit': 0.0,
            'note': f'Optimization for {rule.value} not yet implemented'
        }
        
    def _calculate_adaptation_schedule_voi(self,
                                         treatment: Treatment,
                                         adaptation_times: List[int],
                                         final_sample_size: int) -> float:
        """Calculate expected VOI for adaptation schedule"""
        total_voi = 0.0
        
        for time in adaptation_times:
            # VOI from early decision making
            early_decision_voi = self._calculate_early_decision_voi(treatment, time, final_sample_size)
            total_voi += early_decision_voi
            
        return total_voi
        
    def _calculate_early_decision_voi(self,
                                    treatment: Treatment,
                                    interim_sample_size: int,
                                    final_sample_size: int) -> float:
        """Calculate VOI from early decision making"""
        # Simplified calculation - in practice this would be more sophisticated
        interim_power = self._calculate_power_at_sample_size(interim_sample_size)
        final_power = self._calculate_power_at_sample_size(final_sample_size)
        
        # Value of early information
        early_info_value = (final_power - interim_power) * 0.1 * self.trial_design.willingness_to_pay
        
        return early_info_value
        
    def _simulate_early_stop_rate(self, threshold: float, treatment: Treatment) -> float:
        """Simulate rate of early stopping for success"""
        # Simplified simulation
        if hasattr(treatment, 'effectiveness'):
            expected_effect = treatment.effectiveness
        else:
            expected_effect = 0.5
            
        # Probability of crossing success threshold early
        early_stop_prob = jnp.maximum(0, (expected_effect - (1 - threshold)) * 2)
        return early_stop_prob
        
    def _simulate_futility_rate(self, threshold: float, treatment: Treatment) -> float:
        """Simulate rate of early stopping for futility"""
        if hasattr(treatment, 'effectiveness'):
            expected_effect = treatment.effectiveness
        else:
            expected_effect = 0.5
            
        # Probability of falling below futility threshold
        futility_prob = jnp.maximum(0, (threshold - expected_effect) * 2)
        return futility_prob
        
    def _simulate_power_loss(self, threshold: float, treatment: Treatment) -> float:
        """Simulate power loss from early stopping"""
        # Simplified model - power loss increases with stricter thresholds
        power_loss = jnp.minimum(0.2, threshold * 0.1)
        return power_loss
        
    def _calculate_power_at_sample_size(self, sample_size: int) -> float:
        """Calculate power for given sample size (simplified)"""
        if sample_size <= 0:
            return 0.0
            
        effect_size = self.trial_design.effect_size
        ncp = effect_size * jnp.sqrt(sample_size / 2)
        critical_value = stats.norm.ppf(1 - self.trial_design.alpha / 2)
        power = 1 - stats.norm.cdf(critical_value - ncp)
        
        return power


class ClinicalTrialDesignOptimizer:
    """
    Comprehensive clinical trial design optimization using VOI
    
    This class provides a unified interface for optimizing all aspects
    of clinical trial design using Value of Information analysis.
    """
    
    def __init__(self, trial_design: TrialDesign):
        """
        Initialize trial design optimizer
        
        Args:
            trial_design: Clinical trial design parameters
        """
        self.trial_design = trial_design
        self.sample_size_optimizer = VOIBasedSampleSizeOptimizer(trial_design)
        self.adaptive_optimizer = AdaptiveTrialOptimizer(trial_design)
        
    def optimize_complete_design(self,
                               treatment: Treatment,
                               budget_constraint: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize complete trial design using VOI analysis
        
        Args:
            treatment: Treatment being tested
            budget_constraint: Maximum budget for the trial
            
        Returns:
            Complete optimized trial design
        """
        optimization_results = {}
        
        # 1. Optimize sample size
        sample_size_results = self.sample_size_optimizer.optimize_sample_size(
            treatment, 
            cost_per_participant=1000.0  # Default cost
        )
        optimization_results['sample_size'] = sample_size_results
        
        # 2. Optimize adaptation schedule if applicable
        if hasattr(self.trial_design, "adaptation_rules") and self.trial_design.adaptation_rules:
            adaptation_results = self.adaptive_optimizer.optimize_adaptation_schedule(treatment)
            optimization_results['adaptation'] = adaptation_results
            
            # 3. Optimize adaptation thresholds
            if hasattr(self.trial_design, "adaptation_rules") and self.trial_design.adaptation_rules:
                threshold_results = {}
                for rule in self.trial_design.adaptation_rules:
                    threshold_results[rule.value] = self.adaptive_optimizer.optimize_adaptation_thresholds(
                        treatment, rule
                    )
                optimization_results['thresholds'] = threshold_results
                
        # 4. Calculate overall trial efficiency
        efficiency_metrics = self._calculate_trial_efficiency(
            optimization_results, budget_constraint
        )
        optimization_results['efficiency'] = efficiency_metrics
        
        # 5. Generate final recommendations
        recommendations = self._generate_design_recommendations(optimization_results)
        optimization_results['recommendations'] = recommendations
        
        return optimization_results
        
    def simulate_trial_outcomes(self,
                              treatment: Treatment,
                              optimized_design: Dict[str, Any]) -> TrialOutcome:
        """
        Simulate trial outcomes using optimized design
        
        Args:
            treatment: Treatment being tested
            optimized_design: Optimized trial design
            
        Returns:
            Simulated trial outcome
        """
        # Extract optimized parameters
        optimal_sample_size = optimized_design['sample_size']['optimal_sample_size']
        
        # Simulate primary endpoint outcome
        treatment_effect = self._simulate_treatment_effect(treatment)
        p_value = self._calculate_p_value(treatment_effect, optimal_sample_size)
        confidence_interval = self._calculate_confidence_interval(treatment_effect, optimal_sample_size)
        power_achieved = self._calculate_power_at_sample_size(optimal_sample_size)
        
        # Simulate health economic outcomes if applicable
        cost_effectiveness_ratio = None
        incremental_qaly = None
        net_monetary_benefit = None
        probability_cost_effective = None
        
        if self.trial_design.health_economic_endpoint:
            cost_effectiveness_ratio, incremental_qaly, net_monetary_benefit, probability_cost_effective = \
                self._simulate_health_economic_outcomes(treatment, optimal_sample_size)
                
        # Simulate adaptive trial outcomes if applicable
        adaptation_triggered = False
        adaptation_type = None
        
        if self.trial_design.adaptation_schedule and 'adaptation' in optimized_design:
            adaptation_triggered = random.uniform(random.PRNGKey(42)) < 0.3  # 30% chance
            if adaptation_triggered:
                adaptation_types = [rule.value for rule in self.trial_design.adaptation_rules]
                adaptation_type = adaptation_types[0] if adaptation_types else "sample_size_reest"
                
        return TrialOutcome(
            treatment_effect=treatment_effect,
            p_value=p_value,
            confidence_interval=confidence_interval,
            power_achieved=power_achieved,
            sample_size_used=optimal_sample_size,
            cost_effectiveness_ratio=cost_effectiveness_ratio,
            incremental_qaly=incremental_qaly,
            net_monetary_benefit=net_monetary_benefit,
            probability_cost_effective=probability_cost_effective,
            adaptation_triggered=adaptation_triggered,
            adaptation_type=adaptation_type,
            final_sample_size=optimal_sample_size
        )
        
    def _calculate_trial_efficiency(self,
                                  optimization_results: Dict[str, Any],
                                  budget_constraint: Optional[float] = None) -> Dict[str, Any]:
        """Calculate overall trial efficiency metrics"""
        sample_size_opt = optimization_results['sample_size']
        
        total_cost = sample_size_opt['total_cost_at_optimal']
        total_voi = sample_size_opt['total_voi_at_optimal']
        
        # Basic efficiency metrics
        voi_per_dollar = total_voi / (total_cost + 1e-6)
        cost_per_voi = total_cost / (total_voi + 1e-6)
        
        # Budget constraint check
        within_budget = budget_constraint is None or total_cost <= budget_constraint
        
        efficiency_metrics = {
            'voi_per_dollar': float(voi_per_dollar),
            'cost_per_voi': float(cost_per_voi),
            'total_cost': float(total_cost),
            'total_voi': float(total_voi),
            'within_budget': within_budget,
            'budget_utilization': float(total_cost / budget_constraint) if budget_constraint else 1.0
        }
        
        # Add adaptation efficiency if applicable
        if 'adaptation' in optimization_results:
            adaptation_benefit = optimization_results['adaptation']['optimal_schedule']['expected_voi']
            efficiency_metrics['adaptation_benefit'] = float(adaptation_benefit)
            efficiency_metrics['total_efficiency'] = float((total_voi + adaptation_benefit) / total_cost)
        else:
            efficiency_metrics['total_efficiency'] = float(voi_per_dollar)
            
        return efficiency_metrics
        
    def _generate_design_recommendations(self, optimization_results: Dict[str, Any]) -> List[str]:
        """Generate design recommendations based on optimization"""
        recommendations = []
        
        sample_size_rec = optimization_results['sample_size']
        recommendations.append(
            f"Optimal sample size: {sample_size_rec['optimal_sample_size']} participants"
        )
        
        efficiency = optimization_results['efficiency']
        recommendations.append(
            f"Trial efficiency: {efficiency['voi_per_dollar']:.3f} VOI per dollar invested"
        )
        
        if efficiency['within_budget']:
            recommendations.append("Design is within budget constraints")
        else:
            recommendations.append("Warning: Design exceeds budget constraints")
            
        # Adaptation recommendations
        if 'adaptation' in optimization_results:
            adaptation_rec = optimization_results['adaptation']['optimal_schedule']
            recommendations.append(
                f"Implement {adaptation_rec['num_adaptations']} interim analyses at sample sizes {adaptation_rec['adaptation_times']}"
            )
            
        return recommendations
        
    def _simulate_treatment_effect(self, treatment: Treatment) -> float:
        """Simulate treatment effect from trial data"""
        # Simplified simulation
        true_effect = treatment.effectiveness if hasattr(treatment, 'effectiveness') else 0.5
        # Generate normal random noise with mean=0, std=0.1
        noise = 0.1 * random.normal(random.PRNGKey(42))
        observed_effect = true_effect + noise
        return observed_effect
        
    def _calculate_p_value(self, treatment_effect: float, sample_size: int) -> float:
        """Calculate p-value for treatment effect"""
        # Simplified p-value calculation
        t_stat = treatment_effect * jnp.sqrt(sample_size / 2)
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        return p_value
        
    def _calculate_confidence_interval(self, treatment_effect: float, sample_size: int) -> Tuple[float, float]:
        """Calculate confidence interval for treatment effect"""
        se = jnp.sqrt(2 / sample_size)  # Standard error
        t_critical = stats.norm.ppf(0.975)
        
        lower = treatment_effect - t_critical * se
        upper = treatment_effect + t_critical * se
        
        return (float(lower), float(upper))
        
    def _calculate_power_at_sample_size(self, sample_size: int) -> float:
        """Calculate power for sample size"""
        effect_size = self.trial_design.effect_size
        ncp = effect_size * jnp.sqrt(sample_size / 2)
        critical_value = stats.norm.ppf(1 - self.trial_design.alpha / 2)
        power = 1 - stats.norm.cdf(critical_value - ncp)
        return power
        
    def _simulate_health_economic_outcomes(self,
                                         treatment: Treatment,
                                         sample_size: int) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Simulate health economic outcomes"""
        if not self.trial_design.health_economic_endpoint:
            return None, None, None, None
            
        # Simulate cost and QALY differences
        # Generate cost difference with mean=1000, std=500
        cost_diff = 1000 + 500 * random.normal(random.PRNGKey(42))
        # Generate QALY difference with mean=0.1, std=0.05
        qaly_diff = 0.1 + 0.05 * random.normal(random.PRNGKey(42))
        
        # Calculate cost-effectiveness ratio
        if qaly_diff > 0:
            cer = cost_diff / qaly_diff
        else:
            cer = float('inf')
            
        # Calculate net monetary benefit
        nmb = qaly_diff * self.trial_design.willingness_to_pay - cost_diff
        
        # Calculate probability cost-effective
        prob_ce = stats.norm.cdf(nmb / (jnp.sqrt(sample_size) * 100))  # Simplified
        
        return cer, float(qaly_diff), float(nmb), float(prob_ce)


# Factory functions for common trial designs

def create_superiority_trial(effect_size: float = 0.5,
                           alpha: float = 0.05,
                           beta: float = 0.2,
                           willingness_to_pay: float = 50000.0) -> TrialDesign:
    """Create standard superiority trial design"""
    return TrialDesign(
        trial_type=TrialType.SUPERIORITY,
        primary_endpoint=EndpointType.CONTINUOUS,
        sample_size=100,  # Will be optimized
        effect_size=effect_size,
        alpha=alpha,
        beta=beta,
        willingness_to_pay=willingness_to_pay,
        health_economic_endpoint=True
    )


def create_adaptive_trial(effect_size: float = 0.5,
                        adaptations: int = 2,
                        alpha: float = 0.05,
                        willingness_to_pay: float = 50000.0) -> TrialDesign:
    """Create adaptive trial design"""
    return TrialDesign(
        trial_type=TrialType.ADAPTIVE,
        primary_endpoint=EndpointType.CONTINUOUS,
        sample_size=200,  # Will be optimized
        effect_size=effect_size,
        alpha=alpha,
        interim_analyses=adaptations,
        adaptation_schedule=[100, 150],
        adaptation_rules=[AdaptationRule.EARLY_SUCCESS, AdaptationRule.EARLY_FUTILITY],
        willingness_to_pay=willingness_to_pay,
        health_economic_endpoint=True
    )


def create_health_economics_trial(effect_size: float = 0.5,
                                willingness_to_pay: float = 50000.0,
                                budget_constraint: float = 1000000.0) -> TrialDesign:
    """Create trial with health economics endpoints"""
    return TrialDesign(
        trial_type=TrialType.SUPERIORITY_WITH_HEALTH_ECONOMICS,
        primary_endpoint=EndpointType.COST_EFFECTIVENESS,
        sample_size=300,  # Larger for health economics endpoints
        effect_size=effect_size,
        alpha=0.05,
        beta=0.2,
        willingness_to_pay=willingness_to_pay,
        health_economic_endpoint=True,
        budget_constraint=budget_constraint,
        time_horizon=5.0
    )


# Utility functions

def quick_trial_optimization(treatment: Treatment,
                           trial_type: str = "superiority",
                           budget_constraint: Optional[float] = None) -> Dict[str, Any]:
    """Quick trial optimization for common scenarios"""
    if trial_type.lower() == "superiority":
        design = create_superiority_trial()
    elif trial_type.lower() == "adaptive":
        design = create_adaptive_trial()
    elif trial_type.lower() == "health_economics":
        design = create_health_economics_trial()
    else:
        design = create_superiority_trial()
        
    optimizer = ClinicalTrialDesignOptimizer(design)
    return optimizer.optimize_complete_design(treatment, budget_constraint)


def calculate_trial_voi(treatment: Treatment,
                       sample_size: int,
                       willingness_to_pay: float = 50000.0) -> float:
    """Calculate VOI for a given trial configuration"""
    design = create_superiority_trial(willingness_to_pay=willingness_to_pay)
    optimizer = ClinicalTrialDesignOptimizer(design)
    
    # Calculate VOI per participant and multiply by sample size
    voi_per_participant = optimizer.sample_size_optimizer.calculate_voi_per_participant(
        treatment, sample_size
    )
    
    return voi_per_participant * sample_size
