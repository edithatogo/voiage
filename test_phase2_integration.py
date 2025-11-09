"""
Phase 2.0.0 - Advanced Domain Applications Integration Test

This test validates all Phase 2 functionality including:
- Health Economics specialization
- Multi-domain VOI framework
- Ecosystem integration
- Clinical trial design optimization
- HTA integration

Author: voiage Development Team
Version: 2.0.0
"""

import sys
import os
sys.path.append('/Users/doughnut/GitHub/voiage')

import pytest
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Any
import json
import tempfile
import warnings

# Import all Phase 2 modules
from voiage.health_economics import (
    HealthEconomicsAnalysis, HealthState, Treatment,
    calculate_icer_simple, calculate_net_monetary_benefit_simple, qaly_calculator
)

from voiage.multi_domain import (
    MultiDomainVOI, DomainType, DomainParameters,
    ManufacturingParameters, FinanceParameters, EnvironmentalParameters, EngineeringParameters,
    create_manufacturing_voi, create_finance_voi, create_environmental_voi, create_engineering_voi
)

from voiage.ecosystem_integration import (
    EcosystemIntegration, TreeAgeConnector, RPackageConnector, DataFormatConnector, WorkflowConnector,
    quick_import_health_data, quick_export_notebook, quick_r_export, convert_treeage_to_voi
)

from voiage.clinical_trials import (
    TrialDesign, TrialType, EndpointType, AdaptationRule,
    VOIBasedSampleSizeOptimizer, AdaptiveTrialOptimizer, ClinicalTrialDesignOptimizer,
    create_superiority_trial, create_adaptive_trial, create_health_economics_trial,
    quick_trial_optimization, calculate_trial_voi
)

from voiage.hta_integration import (
    HTAFramework, DecisionType, HTAFrameworkCriteria, HTASubmission, HTAEvaluation,
    HTAIntegrationFramework, NICEFrameworkAdapter, CADTHFrameworkAdapter, ICERFrameworkAdapter,
    create_hta_submission, quick_hta_evaluation, compare_hta_decisions, generate_hta_report
)


def is_numeric(value):
    """Check if value is numeric (float or JAX array)"""
    return isinstance(value, (float, jnp.ndarray, np.ndarray))


class TestHealthEconomics:
    """Test health economics module"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.health_analysis = HealthEconomicsAnalysis(
            willingness_to_pay=50000.0,
            currency="USD"
        )
        
        # Create test treatments
        self.treatment_a = Treatment(
            name="Treatment A",
            description="New innovative treatment",
            effectiveness=0.8,
            cost_per_cycle=1000.0,
            cycles_required=6
        )
        
        self.treatment_b = Treatment(
            name="Treatment B", 
            description="Standard treatment",
            effectiveness=0.6,
            cost_per_cycle=500.0,
            cycles_required=8
        )
        
        self.health_analysis.add_treatment(self.treatment_a)
        self.health_analysis.add_treatment(self.treatment_b)
        
    def test_qaly_calculation(self):
        """Test QALY calculation"""
        # Create test health state
        health_state = HealthState(
            state_id="healthy",
            description="Healthy state",
            utility=0.9,
            cost=1000.0,
            duration=5.0
        )
        
        qaly = self.health_analysis.calculate_qaly(health_state)
        assert qaly > 0
        assert qaly <= 10.0  # Reasonable bound for 10-year analysis horizon
        
    def test_cost_calculation(self):
        """Test cost calculation"""
        health_state = HealthState(
            state_id="treatment",
            description="Treatment state",
            utility=0.7,
            cost=2000.0,
            duration=3.0
        )
        
        cost = self.health_analysis.calculate_cost(health_state)
        assert cost > 0
        assert cost <= 20000.0  # Reasonable bound for 10-year analysis horizon
        
    def test_icer_calculation(self):
        """Test ICER calculation"""
        icer = self.health_analysis.calculate_icer(self.treatment_a, self.treatment_b)
        assert isinstance(icer, float)
        assert icer >= 0
        
    def test_net_monetary_benefit(self):
        """Test Net Monetary Benefit calculation"""
        nmb = self.health_analysis.calculate_net_monetary_benefit(self.treatment_a)
        assert is_numeric(nmb)
        
    def test_budget_impact_analysis(self):
        """Test budget impact analysis"""
        bia = self.health_analysis.budget_impact_analysis(
            self.treatment_a,
            population_size=10000,
            adoption_rate=0.3
        )
        
        assert 'annual_budget_impact' in bia
        assert 'total_budget_impact' in bia
        assert 'sustainability_score' in bia
        assert bia['sustainability_score'] >= 0
        
    def test_probabilistic_sensitivity_analysis(self):
        """Test PSA"""
        psa = self.health_analysis.probabilistic_sensitivity_analysis(
            self.treatment_a,
            num_simulations=100
        )
        
        assert 'qaly_distribution' in psa
        assert 'cost_distribution' in psa
        assert 'net_monetary_benefit' in psa
        assert psa['qaly_distribution']['mean'] > 0
        
    def test_utility_functions(self):
        """Test utility functions"""
        # Test simple ICER calculation
        icer = calculate_icer_simple(5000, 2.0, 3000, 1.0)
        assert isinstance(icer, float)
        assert icer >= 0
        
        # Test NMB calculation
        nmb = calculate_net_monetary_benefit_simple(2.0, 5000, 50000)
        assert isinstance(nmb, float)
        
        # Test QALY calculator
        qaly = qaly_calculator(10.0, 0.8, 0.03)
        assert qaly > 0
        assert qaly <= 10.0


class TestMultiDomainVOI:
    """Test multi-domain VOI framework"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create domain parameters
        self.manufacturing_params = ManufacturingParameters(
            name="Manufacturing Analysis",
            description="Production optimization",
            production_capacity=1000.0,
            quality_threshold=0.95
        )
        
        self.finance_params = FinanceParameters(
            name="Investment Analysis", 
            description="Portfolio optimization",
            initial_investment=1000000.0,
            expected_return=0.08
        )
        
    def test_manufacturing_voi(self):
        """Test manufacturing domain VOI"""
        manufacturing_voi = create_manufacturing_voi(self.manufacturing_params)
        
        # Test default outcome function
        decision_vars = jnp.array([500.0, 0.9, 100.0])  # production, quality, inventory
        uncertainties = jnp.array([0.1, 0.05, 0.02])   # demand, supply, quality
        
        outcomes = manufacturing_voi._manufacturing_outcome(
            decision_vars, uncertainties, self.manufacturing_params
        )
        
        assert len(outcomes) == 5
        assert outcomes.shape == (5,)
        
    def test_finance_voi(self):
        """Test finance domain VOI"""
        finance_voi = create_finance_voi(self.finance_params)
        
        decision_vars = jnp.array([0.7, 0.5, 5.0])  # allocation, risk, horizon
        uncertainties = jnp.array([0.02, 0.1, 0.2])  # return, vol, correlation
        
        outcomes = finance_voi._finance_outcome(
            decision_vars, uncertainties, self.finance_params
        )
        
        assert len(outcomes) == 5
        assert outcomes.shape == (5,)
        
    def test_domain_specific_evpi(self):
        """Test domain-specific EVPI metrics"""
        manufacturing_voi = create_manufacturing_voi(self.manufacturing_params)
        
        # This would normally require a full decision analysis
        # For testing, we'll simulate the basic structure
        mock_metrics = {
            'evpi': 1000.0,
            'production_uncertainty_value': 600.0,
            'quality_uncertainty_value': 300.0,
            'demand_uncertainty_value': 100.0
        }
        
        evpi_metrics = manufacturing_voi.domain_specific_evpi(
            None, "profit"  # Mock decision analysis
        )
        
        # Should return domain-specific breakdown
        assert 'production_uncertainty_value' in evpi_metrics
        assert 'quality_uncertainty_value' in evpi_metrics
        assert 'demand_uncertainty_value' in evpi_metrics
        
    def test_environmental_voi(self):
        """Test environmental domain VOI"""
        env_params = EnvironmentalParameters(
            name="Environmental Policy",
            description="Pollution control optimization",
            baseline_pollution_level=100.0,
            pollution_reduction_target=0.2
        )
        
        env_voi = create_environmental_voi(env_params)
        
        decision_vars = jnp.array([50.0, 0.8, 0.7])  # control, monitoring, compliance
        uncertainties = jnp.array([0.1, 0.05, 0.02])  # baseline, ecosystem, climate
        
        outcomes = env_voi._environmental_outcome(
            decision_vars, uncertainties, env_params
        )
        
        assert len(outcomes) == 5
        assert outcomes.shape == (5,)
        
    def test_engineering_voi(self):
        """Test engineering domain VOI"""
        eng_params = EngineeringParameters(
            name="Engineering Design",
            description="System reliability optimization",
            system_reliability_target=0.99,
            safety_factor=2.0
        )
        
        eng_voi = create_engineering_voi(eng_params)
        
        decision_vars = jnp.array([3.0, 1.5, 0.8])  # complexity, safety, maintenance
        uncertainties = jnp.array([0.05, 0.1, 0.02])  # material, load, failure
        
        outcomes = eng_voi._engineering_outcome(
            decision_vars, uncertainties, eng_params
        )
        
        assert len(outcomes) == 5
        assert outcomes.shape == (5,)


class TestEcosystemIntegration:
    """Test ecosystem integration module"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.integration = EcosystemIntegration()
        
        # Create test health economics analysis
        self.health_analysis = HealthEconomicsAnalysis(willingness_to_pay=50000.0)
        treatment = Treatment(
            name="Test Treatment",
            description="Test treatment for export",
            effectiveness=0.8,
            cost_per_cycle=1000.0,
            cycles_required=6
        )
        self.health_analysis.add_treatment(treatment)
        
    def test_connector_availability(self):
        """Test that all connectors are available"""
        connectors = ['treeage', 'r_packages', 'data_formats', 'workflows']
        
        for connector_name in connectors:
            connector = self.integration.get_connector(connector_name)
            assert connector is not None
            assert connector.name is not None
            
    def test_list_supported_formats(self):
        """Test supported formats listing"""
        formats = self.integration.list_supported_formats()
        
        assert isinstance(formats, dict)
        assert 'treeage' in formats
        assert 'r_packages' in formats
        assert 'data_formats' in formats
        assert 'workflows' in formats
        
    def test_data_format_operations(self):
        """Test data format import/export"""
        connector = self.integration.get_connector('data_formats')
        assert connector is not None
        
        # Test export to CSV
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name
            
        try:
            self.integration.export_to_external(
                'data_formats',
                self.health_analysis,
                temp_path,
                format_type='csv'
            )
            
            # Check if file was created
            assert os.path.exists(temp_path)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_workflow_export(self):
        """Test workflow export (Jupyter/R)"""
        # Test Jupyter export
        with tempfile.NamedTemporaryFile(suffix='.ipynb', delete=False) as f:
            temp_path = f.name
            
        try:
            self.integration.export_to_external(
                'workflows',
                self.health_analysis,
                temp_path,
                format='jupyter'
            )
            
            # Check if file was created
            assert os.path.exists(temp_path)
            
            # Verify it's valid JSON (Jupyter notebook format)
            with open(temp_path, 'r') as f:
                notebook_data = json.load(f)
                assert 'cells' in notebook_data
                assert 'metadata' in notebook_data
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
        # Test R export
        with tempfile.NamedTemporaryFile(suffix='.R', delete=False) as f:
            temp_path = f.name
            
        try:
            self.integration.export_to_external(
                'workflows',
                self.health_analysis,
                temp_path,
                format='r'
            )
            
            # Check if file was created
            assert os.path.exists(temp_path)
            
            # Verify it contains R code
            with open(temp_path, 'r') as f:
                r_code = f.read()
                assert 'library(' in r_code
                assert 'willingness_to_pay' in r_code
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_integration_report(self):
        """Test integration capabilities report"""
        report = self.integration.create_integration_report()
        
        assert 'available_connectors' in report
        assert 'supported_formats' in report
        assert 'integration_capabilities' in report
        
        # Verify key information
        assert len(report['available_connectors']) > 0
        assert len(report['supported_formats']) > 0
        assert report['version'] is not None


class TestClinicalTrialDesign:
    """Test clinical trial design optimization"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create test trial design
        self.trial_design = create_superiority_trial(
            effect_size=0.5,
            alpha=0.05,
            beta=0.2,
            willingness_to_pay=50000.0
        )
        
        # Create test treatment
        self.treatment = Treatment(
            name="Clinical Trial Treatment",
            description="Treatment for clinical trial",
            effectiveness=0.7,
            cost_per_cycle=2000.0,
            cycles_required=4
        )
        
    def test_trial_design_creation(self):
        """Test trial design creation"""
        assert self.trial_design.trial_type == TrialType.SUPERIORITY
        assert self.trial_design.primary_endpoint == EndpointType.CONTINUOUS
        assert self.trial_design.willingness_to_pay == 50000.0
        assert self.trial_design.health_economic_endpoint == True
        
    def test_sample_size_optimizer(self):
        """Test sample size optimization"""
        optimizer = VOIBasedSampleSizeOptimizer(self.trial_design)
        
        # Test VOI per participant calculation
        voi_per_participant = optimizer.calculate_voi_per_participant(self.treatment, 100)
        assert is_numeric(voi_per_participant)
        assert voi_per_participant >= 0
        
    def test_sample_size_optimization(self):
        """Test complete sample size optimization"""
        optimizer = VOIBasedSampleSizeOptimizer(self.trial_design)
        
        results = optimizer.optimize_sample_size(
            self.treatment,
            min_sample_size=50,
            max_sample_size=500,
            cost_per_participant=1000.0
        )
        
        assert 'optimal_sample_size' in results
        assert 'max_net_benefit' in results
        assert 'optimization_curve' in results
        
        optimal_size = results['optimal_sample_size']
        assert 50 <= optimal_size <= 500
        
    def test_adaptive_trial_optimizer(self):
        """Test adaptive trial optimization"""
        adaptive_trial = create_adaptive_trial()
        optimizer = AdaptiveTrialOptimizer(adaptive_trial)
        
        # Test adaptation schedule optimization
        schedule_results = optimizer.optimize_adaptation_schedule(self.treatment)
        
        assert 'optimal_schedule' in schedule_results
        assert 'all_schedules' in schedule_results
        assert 'recommendation' in schedule_results
        
    def test_complete_trial_optimization(self):
        """Test complete trial design optimization"""
        optimizer = ClinicalTrialDesignOptimizer(self.trial_design)
        
        results = optimizer.optimize_complete_design(self.treatment)
        
        assert 'sample_size' in results
        assert 'efficiency' in results
        assert 'recommendations' in results
        
        # Verify optimization results
        sample_size_results = results['sample_size']
        assert 'optimal_sample_size' in sample_size_results
        assert sample_size_results['optimal_sample_size'] > 0
        
    def test_trial_outcome_simulation(self):
        """Test trial outcome simulation"""
        optimizer = ClinicalTrialDesignOptimizer(self.trial_design)
        
        # First optimize the design
        optimized_design = optimizer.optimize_complete_design(self.treatment)
        
        # Then simulate outcomes
        outcome = optimizer.simulate_trial_outcomes(self.treatment, optimized_design)
        
        assert is_numeric(outcome.treatment_effect)
        assert 0 <= outcome.p_value <= 1
        assert isinstance(outcome.confidence_interval, tuple)
        assert len(outcome.confidence_interval) == 2
        assert outcome.sample_size_used > 0
        
    def test_quick_trial_optimization(self):
        """Test quick trial optimization utility"""
        results = quick_trial_optimization(
            self.treatment,
            trial_type="superiority"
        )
        
        assert isinstance(results, dict)
        assert 'sample_size' in results
        assert 'efficiency' in results
        
    def test_trial_voi_calculation(self):
        """Test trial VOI calculation"""
        voi = calculate_trial_voi(
            self.treatment,
            sample_size=100,
            willingness_to_pay=50000.0
        )
        
        assert is_numeric(voi)
        assert voi >= 0


class TestHTAIntegration:
    """Test HTA integration module"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create test HTA submission
        self.submission = create_hta_submission(
            technology_name="Innovative Drug X",
            manufacturer="PharmaCorp",
            indication="Advanced cancer",
            economic_results={
                'icer': 75000.0,
                'qaly_gain': 1.2,
                'net_monetary_benefit': 15000.0
            },
            clinical_results={
                'evidence_level': 'RCT',
                'sample_size': 500,
                'primary_endpoint_met': True
            }
        )
        
        # Add framework-specific data
        self.submission.framework_specific_data = {
            'end_of_life': True,
            'rare_disease': False,
            'comparative_effectiveness': True
        }
        
    def test_hta_submission_creation(self):
        """Test HTA submission creation"""
        assert self.submission.technology_name == "Innovative Drug X"
        assert self.submission.manufacturer == "PharmaCorp"
        assert self.submission.indication == "Advanced cancer"
        assert 'icer' in self.submission.cost_effectiveness_analysis
        
    def test_nice_evaluation(self):
        """Test NICE framework evaluation"""
        nice_adapter = NICEFrameworkAdapter()
        evaluation = nice_adapter.evaluate_submission(self.submission)
        
        assert evaluation.framework == HTAFramework.NICE
        assert isinstance(evaluation.decision, DecisionType)
        assert evaluation.icer is not None
        assert evaluation.qaly_gain is not None
        
        # Test end of life consideration
        assert "end of life" in evaluation.recommendation.lower() or evaluation.recommendation != "Not recommended"
        
    def test_cadth_evaluation(self):
        """Test CADTH framework evaluation"""
        cadth_adapter = CADTHFrameworkAdapter()
        evaluation = cadth_adapter.evaluate_submission(self.submission)
        
        assert evaluation.framework == HTAFramework.CADTH
        assert evaluation.clinical_effectiveness_score is not None
        assert evaluation.cost_effectiveness_score is not None
        
    def test_icer_evaluation(self):
        """Test ICER framework evaluation"""
        icer_adapter = ICERFrameworkAdapter()
        evaluation = icer_adapter.evaluate_submission(self.submission)
        
        assert evaluation.framework == HTAFramework.ICER
        assert "value" in evaluation.recommendation.lower()
        
    def test_hta_framework_integration(self):
        """Test main HTA integration framework"""
        hta_framework = HTAIntegrationFramework()
        
        # Test single framework evaluation
        evaluation = hta_framework.evaluate_for_framework(
            self.submission,
            HTAFramework.NICE
        )
        
        assert isinstance(evaluation, HTAEvaluation)
        assert evaluation.framework == HTAFramework.NICE
        
    def test_multiple_framework_evaluation(self):
        """Test multiple framework evaluation"""
        hta_framework = HTAIntegrationFramework()
        
        frameworks = [HTAFramework.NICE, HTAFramework.CADTH, HTAFramework.ICER]
        evaluations = hta_framework.evaluate_multiple_frameworks(self.submission, frameworks)
        
        assert len(evaluations) == len(frameworks)
        assert all(isinstance(eval_result, HTAEvaluation) for eval_result in evaluations.values())
        
    def test_framework_comparison(self):
        """Test framework decision comparison"""
        hta_framework = HTAIntegrationFramework()
        
        # Create evaluations for comparison
        frameworks = [HTAFramework.NICE, HTAFramework.CADTH]
        evaluations = hta_framework.evaluate_multiple_frameworks(self.submission, frameworks)
        
        # Perform comparison
        comparison = hta_framework.compare_framework_decisions(evaluations)
        
        assert 'frameworks_evaluated' in comparison
        assert 'decisions' in comparison
        assert 'decision_agreement' in comparison
        assert 'icer_range' in comparison
        
    def test_hta_strategy_creation(self):
        """Test HTA strategy creation"""
        hta_framework = HTAIntegrationFramework()
        
        frameworks = [HTAFramework.NICE, HTAFramework.CADTH]
        strategy = hta_framework.create_hta_strategy(self.submission, frameworks)
        
        assert 'target_frameworks' in strategy
        assert 'evaluations' in strategy
        assert 'comparison' in strategy
        assert 'strategy_recommendations' in strategy
        
    def test_utility_functions(self):
        """Test HTA utility functions"""
        # Test quick evaluation
        evaluation = quick_hta_evaluation(self.submission, HTAFramework.NICE)
        assert isinstance(evaluation, HTAEvaluation)
        
        # Test comparison
        comparison = compare_hta_decisions(
            self.submission,
            [HTAFramework.NICE, HTAFramework.ICER]
        )
        
        assert isinstance(comparison, dict)
        assert 'comparison' in comparison
        
        # Test report generation
        report = generate_hta_report(self.submission, HTAFramework.NICE)
        assert isinstance(report, str)
        assert "HTA Evaluation Report" in report
        assert "DECISION:" in report


class TestPhase2Integration:
    """Test comprehensive Phase 2 integration"""
    
    def setup_method(self):
        """Setup test fixtures"""
        pass

    def test_all_modules_import(self):
        """Test that all Phase 2 modules can be imported"""
        # This test ensures all modules load without errors
        
        # Health Economics
        from voiage.health_economics import HealthEconomicsAnalysis
        assert HealthEconomicsAnalysis is not None
        
        # Multi-Domain
        from voiage.multi_domain import MultiDomainVOI
        assert MultiDomainVOI is not None
        
        # Ecosystem Integration
        from voiage.ecosystem_integration import EcosystemIntegration
        assert EcosystemIntegration is not None
        
        # Clinical Trials
        from voiage.clinical_trials import ClinicalTrialDesignOptimizer
        assert ClinicalTrialDesignOptimizer is not None
        
        # HTA Integration
        from voiage.hta_integration import HTAIntegrationFramework
        assert HTAIntegrationFramework is not None
        
    def test_cross_module_functionality(self):
        """Test functionality that spans multiple modules"""
        # Create health economics analysis
        health_analysis = HealthEconomicsAnalysis(willingness_to_pay=50000.0)
        treatment = Treatment("Test Treatment", "Test", 0.8, 1000.0, 6)
        health_analysis.add_treatment(treatment)
        
        # Create trial design and optimize
        trial_design = create_health_economics_trial()
        optimizer = ClinicalTrialDesignOptimizer(trial_design)
        trial_results = optimizer.optimize_complete_design(treatment)
        
        # Create HTA submission from results
        submission = create_hta_submission(
            technology_name="Test Technology",
            manufacturer="Test Corp",
            indication="Test Indication",
            economic_results={
                'icer': trial_results.get('sample_size', {}).get('optimal_sample_size', 100) * 100,
                'qaly_gain': 1.0,
                'net_monetary_benefit': 25000.0
            },
            clinical_results={'evidence_level': 'RCT', 'sample_size': 200}
        )
        
        # Evaluate for HTA frameworks
        hta_framework = HTAIntegrationFramework()
        evaluation = hta_framework.evaluate_for_framework(submission, HTAFramework.NICE)
        
        # Verify integration works
        assert isinstance(evaluation, HTAEvaluation)
        assert evaluation.icer is not None
        
    def test_performance_validation(self):
        """Test that Phase 2 features perform adequately"""
        import time
        
        # Test health economics performance
        start_time = time.time()
        health_analysis = HealthEconomicsAnalysis(willingness_to_pay=50000.0)
        treatment = Treatment("Performance Test", "Test", 0.8, 1000.0, 6)
        health_analysis.add_treatment(treatment)
        
        # Run PSA
        psa_results = health_analysis.probabilistic_sensitivity_analysis(treatment, num_simulations=50)
        health_time = time.time() - start_time
        
        assert psa_results is not None
        assert health_time < 5.0  # Should complete within 5 seconds
        
        # Test multi-domain performance
        start_time = time.time()
        manufacturing_params = ManufacturingParameters(name="Perf Test", description="Test")
        manufacturing_voi = create_manufacturing_voi(manufacturing_params)
        
        decision_vars = jnp.array([100.0, 0.9, 50.0])
        uncertainties = jnp.array([0.1, 0.05, 0.02])
        outcomes = manufacturing_voi._manufacturing_outcome(
            decision_vars, uncertainties, manufacturing_params
        )
        domain_time = time.time() - start_time
        
        assert len(outcomes) == 5
        assert domain_time < 2.0  # Should complete within 2 seconds
        
    def test_error_handling(self):
        """Test error handling across modules"""
        # Test invalid health economics inputs
        health_analysis = HealthEconomicsAnalysis()
        
        # Test invalid treatment
        invalid_treatment = Treatment("Invalid", "Test", -0.5, -100, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            health_analysis.add_treatment(invalid_treatment)
            
        # Test invalid domain parameters
        try:
            invalid_params = DomainParameters(
                domain_type=DomainType.MANUFACTURING,
                name="Invalid",
                description="Test"
            )
            # This should work with defaults
            assert invalid_params is not None
        except Exception as e:
            pytest.fail(f"Domain parameters creation failed: {e}")
            
    def test_comprehensive_workflow(self):
        """Test complete workflow from analysis to HTA submission"""
        # Step 1: Health Economics Analysis
        health_analysis = HealthEconomicsAnalysis(willingness_to_pay=75000.0)
        new_treatment = Treatment(
            name="Novel Therapy",
            description="Breakthrough treatment",
            effectiveness=0.85,
            cost_per_cycle=3000.0,
            cycles_required=4
        )
        health_analysis.add_treatment(new_treatment)
        
        # Step 2: Clinical Trial Design
        trial_design = create_health_economics_trial(
            willingness_to_pay=75000.0,
            budget_constraint=2000000.0
        )
        trial_optimizer = ClinicalTrialDesignOptimizer(trial_design)
        trial_results = trial_optimizer.optimize_complete_design(new_treatment)
        
        # Step 3: HTA Submission
        economic_results = {
            'icer': 45000.0,
            'qaly_gain': 2.1,
            'net_monetary_benefit': 37500.0
        }
        
        submission = create_hta_submission(
            technology_name="Novel Therapy",
            manufacturer="BioPharma Inc",
            indication="Rare genetic disorder",
            economic_results=economic_results,
            clinical_results={
                'evidence_level': 'RCT',
                'sample_size': trial_results['sample_size']['optimal_sample_size'],
                'primary_endpoint_met': True
            }
        )
        
        submission.framework_specific_data = {
            'rare_disease': True,
            'innovation_breakthrough': True,
            'comparative_effectiveness': True
        }
        
        # Step 4: Multi-framework HTA Evaluation
        hta_framework = HTAIntegrationFramework()
        strategy = hta_framework.create_hta_strategy(
            submission,
            [HTAFramework.NICE, HTAFramework.CADTH, HTAFramework.ICER]
        )
        
        # Verify complete workflow
        assert 'target_frameworks' in strategy
        assert len(strategy['evaluations']) == 3
        assert 'strategy_recommendations' in strategy
        assert isinstance(strategy['comparison'], dict)
        
        # Step 5: Generate final report
        report = generate_hta_report(submission, HTAFramework.NICE)
        assert "HTA Evaluation Report" in report
        assert "Novel Therapy" in report


def run_phase2_tests():
    """Run all Phase 2 tests"""
    print("Running Phase 2.0.0 - Advanced Domain Applications Tests")
    print("=" * 60)
    
    test_classes = [
        TestHealthEconomics,
        TestMultiDomainVOI,
        TestEcosystemIntegration,
        TestClinicalTrialDesign,
        TestHTAIntegration,
        TestPhase2Integration
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 40)
        
        test_instance = test_class()
        test_instance.setup_method()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_') and callable(getattr(test_instance, method))]
        
        for test_method_name in test_methods:
            try:
                test_method = getattr(test_instance, test_method_name)
                test_method()
                print(f"  ✓ {test_method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {test_method_name}: {str(e)}")
                failed_tests.append(f"{test_class.__name__}.{test_method_name}: {str(e)}")
            finally:
                total_tests += 1
    
    print(f"\n{'-' * 60}")
    print(f"Test Results:")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for failure in failed_tests:
            print(f"  - {failure}")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_phase2_tests()
    if success:
        print(f"\n🎉 All Phase 2.0.0 tests passed successfully!")
        print("Advanced Domain Applications implementation is complete and ready for production use.")
    else:
        print(f"\n❌ Some tests failed. Please review the errors above.")
        sys.exit(1)