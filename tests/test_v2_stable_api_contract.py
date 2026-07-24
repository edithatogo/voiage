"""Executable governance checks for the normative v2 public API contract."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from voiage.methods.sample_information import evsi

ROOT = Path(__file__).resolve().parents[1]
V1_CONTRACT_PATH = ROOT / "specs/v1/stable-api.json"
V2_CONTRACT_PATH = ROOT / "specs/v2/stable-api.json"
V1_POLICY_PATH = ROOT / "specs/v1/compatibility-policy.json"
V2_POLICY_PATH = ROOT / "specs/v2/compatibility-policy.json"
API_REFERENCE_PATH = ROOT / "docs/astro-site/src/content/docs/api-reference/index.mdx"
MIGRATION_GUIDE_PATH = (
    ROOT / "docs/astro-site/src/content/docs/user-guide/migration-guide.mdx"
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v2_contract_is_a_major_version_boundary_from_frozen_v1() -> None:
    """v2 must supersede v1 without changing the historical contract."""
    v1 = _json(V1_CONTRACT_PATH)
    v2 = _json(V2_CONTRACT_PATH)

    assert v1["contract_version"] == "1.0.0"
    assert v2["contract_version"] == "2.0.0"
    assert v2["status"] == "normative"
    assert v2["supersedes"] == "../v1/stable-api.json"
    assert v2["release_gate"] == "runtime-and-binding-conformance-required"
    assert v2["v1_to_v2"]["classification"] == "major-breaking-change"
    assert v2["v1_to_v2"]["v1_contract"] == "../v1/stable-api.json"


def test_v2_carries_forward_unchanged_v1_surfaces_and_non_evsi_methods() -> None:
    """The EVSI correction must not silently redefine unrelated v1 surfaces."""
    v1 = _json(V1_CONTRACT_PATH)
    v2 = _json(V2_CONTRACT_PATH)

    for method_name, method_contract in v1["methods"].items():
        if method_name != "evsi":
            assert v2["methods"][method_name] == method_contract
    assert v2["surfaces"] == v1["surfaces"]
    assert set(v2["symbols"]["stable"]) == {
        *v1["symbols"]["stable"],
        "normal_normal_two_arm_evsi",
    }
    assert set(v2["symbols"]["provisional"]) == {
        *v1["symbols"]["provisional"],
        "ingestion",
    }
    for maturity in ("experimental", "deprecated", "removed"):
        assert v2["symbols"][maturity] == v1["symbols"][maturity]


def test_v2_two_loop_contract_defines_one_coherent_bayesian_model() -> None:
    """Every built-in EVSI integration stage must use the same fitted prior."""
    method = _json(V2_CONTRACT_PATH)["methods"]["evsi"]
    built_in = method["built_in_two_loop"]

    assert method["status"] == "stable"
    assert method["scientific_maturity"] == "method-specific"
    assert method["implementation"] == "rust+python-orchestration"
    assert method["output"] == "finite-float"
    assert method["negative_monte_carlo_estimates"] == "return-untruncated"
    assert method["loop_counts"] == "positive-non-boolean-integers"
    assert (
        built_in["prior"] == "one-joint-multivariate-normal-fitted-from-finite-psa-rows"
    )
    assert built_in["maturity"] == "fixture-backed"
    assert (
        built_in["coherence"]
        == "current-decision-prior-predictive-data-and-posterior-all-use-the-fitted-joint-normal-prior"
    )
    assert (
        built_in["outer_loop"]
        == "draw-study-data-directly-from-the-fitted-prior-predictive-distribution"
    )
    assert (
        built_in["posterior"]
        == "joint-gaussian-conditioning-updates-all-correlated-stochastic-parameters"
    )
    assert (
        built_in["fixed_parameters"]
        == "sd_outcome-is-finite-strictly-positive-constant-and-excluded-from-stochastic-covariance"
    )
    assert (
        built_in["inner_loop"]
        == "exactly-n_inner_loops-joint-posterior-draws-per-outer-data-set"
    )
    assert (
        built_in["covariance_failures"]
        == "raise-InputError-or-NumericalError-never-raw-linear-algebra-errors"
    )


def test_v2_evsi_signature_and_custom_callback_contract_are_explicit() -> None:
    """The breaking signature and paired callback protocol must be reviewable."""
    method = _json(V2_CONTRACT_PATH)["methods"]["evsi"]
    custom = method["custom_two_loop"]

    assert method["signature"] == (
        "evsi(model_func, psa_prior, trial_design, population=None, "
        'discount_rate=None, time_horizon=None, method="two_loop", '
        'n_outer_loops=100, n_inner_loops=1000, metamodel="linear", *, '
        "seed=None, trial_simulator=None, posterior_sampler=None)"
    )
    runtime_signature = inspect.signature(evsi)
    assert list(runtime_signature.parameters) == [
        "model_func",
        "psa_prior",
        "trial_design",
        "population",
        "discount_rate",
        "time_horizon",
        "method",
        "n_outer_loops",
        "n_inner_loops",
        "metamodel",
        "seed",
        "trial_simulator",
        "posterior_sampler",
    ]
    for name in ("seed", "trial_simulator", "posterior_sampler"):
        assert runtime_signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert custom["callbacks"] == (
        "trial_simulator-and-posterior_sampler-required-together"
    )
    assert custom["parameter_kind"] == "keyword-only"
    assert custom["posterior_sampler"].endswith("->exactly-n_draws-joint-ParameterSet")
    assert custom["coherence"] == (
        "callbacks-must-define-one-prior-sampling-likelihood-and-posterior-model"
    )


def test_v2_rng_contract_is_local_and_versioned() -> None:
    """v2 must not inherit v1's process-global unseeded random stream."""
    contract = _json(V2_CONTRACT_PATH)
    numerics = contract["numerics"]
    rng = contract["methods"]["evsi"]["rng"]

    assert numerics["seed_type"] == "unsigned-64-bit-integer"
    assert numerics["random_stream"] == "local-numpy-generator"
    assert numerics["global_rng_access"] == "forbidden"
    assert numerics["same_seed_same_dependency_set_same_platform"] == "bitwise"
    assert rng == {
        "seed_parameter": "keyword-only-uint64-or-none",
        "seeded": "one-local-generator-reproducible-for-the-declared-dependency-set",
        "unseeded": "fresh-local-generator",
        "global_state": "never-read-or-mutated",
    }


def test_v2_normal_normal_evsi_contract_is_analytical_and_complete() -> None:
    """The Rust analytical EVSI surface must declare its full study model."""
    method = _json(V2_CONTRACT_PATH)["methods"]["normal_normal_two_arm_evsi"]

    assert method["status"] == "stable"
    assert method["implementation"] == "rust"
    assert method["signature"] == (
        "normal_normal_two_arm_evsi(*, prior_mean, "
        "prior_standard_deviation, outcome_standard_deviation, "
        "total_sample_size, net_benefit_slope, net_benefit_intercept)"
    )
    assert method["randomness"] == "none-analytical"
    assert method["study_model"] == {
        "prior": "incremental-effect-normal",
        "likelihood": (
            "two-independent-normal-arms-with-common-known-outcome-standard-deviation"
        ),
        "allocation": "equal",
        "total_sample_size": ("positive-even-integer-within-supported-u32-range"),
        "net_benefit": ("net_benefit_slope*incremental_effect+net_benefit_intercept"),
    }


def test_v2_policy_retains_semver_and_records_the_evsi_major_change() -> None:
    """Subsequent v2 changes remain governed by the established policy."""
    v1 = _json(V1_POLICY_PATH)
    v2 = _json(V2_POLICY_PATH)

    assert v2["policy_version"] == "2.0.0"
    assert v2["applies_from"] == "2.0.0"
    assert v2["supersedes"] == "../v1/compatibility-policy.json"
    assert v2["semantic_versioning"] == v1["semantic_versioning"]
    assert v2["deprecation"] == v1["deprecation"]
    assert v2["excluded_stabilities"] == v1["excluded_stabilities"]
    assert v2["promotion_requirements"] == v1["promotion_requirements"]
    assert v2["compatibility_evidence"] == v1["compatibility_evidence"]
    assert v2["surfaces"]["python"]["additive_parameters"] == "keyword-only"
    assert v2["v1_to_v2"]["classification"] == "major-breaking-change"
    for changed_behavior in (
        "signature",
        "prior semantics",
        "posterior semantics",
        "random-number generation",
        "model-evaluation counts",
        "numerical results",
    ):
        assert changed_behavior in v2["v1_to_v2"]["reason"]


def test_v2_api_reference_and_migration_guide_cover_the_boundary() -> None:
    """Reader documentation must explain the machine-readable contract."""
    reference = API_REFERENCE_PATH.read_text(encoding="utf-8")
    migration = MIGRATION_GUIDE_PATH.read_text(encoding="utf-8")
    normalized_reference = " ".join(reference.split())

    for required in (
        "published v1.0 API remains frozen",
        "normative v2 release contract",
        "one joint multivariate-normal prior",
        "never read or mutate NumPy's process-global random state",
        "without silently replacing a negative Monte Carlo estimate with zero",
        "versioned as v2",
    ):
        assert required in normalized_reference

    for required in (
        "## Migrating from 1.x to 2.0",
        "### Signature boundary",
        "### Behaviour boundary",
        "### Analytical normal--normal EVSI",
        "### Custom two-loop studies",
        "### Binding migration",
        "`seed` becomes keyword-only",
        "fresh local `numpy.random.Generator`",
        "returned untruncated",
    ):
        assert required in migration
