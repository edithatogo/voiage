"""Run Q09 reproducible comparative evaluation across Voiage Q03-R03 estimators."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from voiage.experimental.study_design import (
    discounted_population_enbs,
    enumerate_study_portfolio,
    equity_weighted_study_enbs,
    sequential_adaptive_stopping_regret,
)
from voiage.methods.sample_information import (
    beta_binomial_two_arm_evsi,
    evsi_importance_sampling,
    evsi_qmc_normal_normal,
    exponential_time_to_event_evsi,
    gamma_poisson_evsi,
    normal_normal_two_arm_evsi,
)


def run_evaluation() -> dict[str, object]:
    """Execute deterministic comparative evaluation cases and return JSON-serializable report."""
    cases: list[dict[str, object]] = []

    # 1. Normal-Normal analytic vs Halton QMC (V07 accuracy-vs-cost)
    t0 = time.perf_counter()
    nn_exact = normal_normal_two_arm_evsi(
        prior_mean=0.1,
        prior_standard_deviation=0.5,
        outcome_standard_deviation=1.0,
        total_sample_size=100,
        net_benefit_slope=1000.0,
        net_benefit_intercept=-100.0,
    )
    t_exact_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    nn_qmc = evsi_qmc_normal_normal(
        prior_mean=0.1,
        prior_standard_deviation=0.5,
        outcome_standard_deviation=1.0,
        total_sample_size=100,
        net_benefit_slope=1000.0,
        net_benefit_intercept=-100.0,
        num_points=1024,
    )
    t_qmc_ms = (time.perf_counter() - t0) * 1000.0
    qmc_evsi = float(nn_qmc["evsi"])
    cases.append(
        {
            "case_id": "q05_normal_normal_exact_vs_qmc",
            "estimand": "two_arm_normal_evsi",
            "reference_value": nn_exact,
            "qmc_value": qmc_evsi,
            "abs_error": abs(qmc_evsi - nn_exact),
            "exact_ms": round(t_exact_ms, 4),
            "qmc_ms": round(t_qmc_ms, 4),
            "passed": abs(qmc_evsi - nn_exact) < 2.0,
        }
    )

    # 2. Non-normal conjugate models (V05)
    bb_25 = beta_binomial_two_arm_evsi(
        alpha_0=2.0,
        beta_0=8.0,
        alpha_1=2.5,
        beta_1=7.5,
        per_arm_sample_size=25,
        net_benefit_slope=1000.0,
        net_benefit_intercept=-50.0,
    )
    gp_20 = gamma_poisson_evsi(
        prior_shape=4.0,
        prior_rate=10.0,
        exposure_per_unit=1.0,
        total_sample_size=20,
        net_benefit_slope=500.0,
        net_benefit_intercept=-200.0,
    )
    tte_20 = exponential_time_to_event_evsi(
        prior_shape=3.0,
        prior_rate=6.0,
        expected_events=20.0,
        net_benefit_slope=400.0,
        net_benefit_intercept=-200.0,
    )
    cases.append(
        {
            "case_id": "q05_non_normal_study_models",
            "beta_binomial_evsi": bb_25,
            "gamma_poisson_evsi": gp_20,
            "exponential_tte_evsi": tte_20,
            "passed": bb_25 > 0.0 and gp_20 > 0.0 and tte_20 > 0.0,
        }
    )

    # 3. Importance sampling EVSI + diagnostics (V04)
    nb = np.array(
        [[0.0, float(k)] for k in [1, 2, 3, -1, -2, 1.5, 0.5, -0.5, 2.5, -1.5]],
        dtype=np.float64,
    )
    lw = np.array(
        [[0.1, 0.2, 0.3, -0.1, -0.2, 0.15, 0.05, -0.05, 0.25, -0.15]],
        dtype=np.float64,
    )
    is_res = evsi_importance_sampling(
        nb, lw, min_ess_threshold=5.0, max_khat_threshold=0.7
    )
    cases.append(
        {
            "case_id": "q05_importance_sampling_diagnostics",
            "evsi": float(is_res["evsi"]),
            "min_ess": float(is_res["min_ess"]),
            "max_khat": float(is_res["max_khat"]),
            "passed": float(is_res["min_ess"]) >= 5.0
            and float(is_res["max_khat"]) <= 0.7,
        }
    )

    # 4. Q06 portfolio optimization & equity/sequential extensions
    pop_enbs = discounted_population_enbs(
        per_patient_evsi=2.0,
        research_cost=5000.0,
        discount_rate=0.035,
        recruitment_delay_years=1.0,
        time_horizon_years=10.0,
        incident_population_per_year=1000.0,
    )
    port = enumerate_study_portfolio(
        study_ids=["s1", "s2", "s3"],
        costs=[100.0, 100.0, 80.0],
        standalone_evsis=[220.0, 210.0, 150.0],
        parameter_groups=["efficacy", "efficacy", "utility"],
        interactions=[(0, 1, -180.0)],
        budget=250.0,
        require_explicit_overlap_guard=True,
    )
    eq_enbs = equity_weighted_study_enbs(
        subgroup_evsi=[100.0, 200.0],
        equity_weights=[1.0, 3.0],
        research_cost=50.0,
    )
    seq_reg = sequential_adaptive_stopping_regret(
        stage_evsi_remaining=[100.0, 180.0, 190.0],
        stage_cumulative_costs=[20.0, 60.0, 120.0],
    )
    cases.append(
        {
            "case_id": "q06_portfolio_equity_and_sequential",
            "discounted_population_enbs": pop_enbs,
            "portfolio_selected": port["selected_indices"],
            "portfolio_enbs": float(port["enbs"]),
            "equity_weighted_enbs": eq_enbs,
            "sequential_optimal_stage": seq_reg["optimal_stage"],
            "passed": pop_enbs > 0.0
            and port["selected_indices"] == [0, 2]
            and abs(eq_enbs - 125.0) < 1e-12
            and seq_reg["optimal_stage"] == 1,
        }
    )

    return {
        "schema_version": "voiage.q09-comparative-evaluation.v1",
        "all_passed": all(bool(c["passed"]) for c in cases),
        "cases": cases,
        "external_gates_retained": [
            "pyOpenSci / JOSS / rOpenSci venue submissions (#1037)",
            "External independent reproduction & MDM manuscript evaluation",
            "Live installed innovate Q08.2 workflow (#1423)",
        ],
    }


def main() -> int:
    """Write the Q09 comparative evaluation JSON packet."""
    out_path = Path(
        "conductor/tracks/scientific_ecosystem_alignment_20261029/frontier_20261108/q09-comparative-evaluation-20261127.json"
    )
    report = run_evaluation()
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_passed": report["all_passed"], "output": str(out_path)}))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
