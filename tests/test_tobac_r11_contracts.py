"""Consumer contract and independent-oracle witnesses for master R11.1-R11.5."""

from __future__ import annotations

import builtins
from copy import deepcopy
from fractions import Fraction
import hashlib
from itertools import permutations
import json
from pathlib import Path
import socket
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
from pydantic import ValidationError
import pytest

from voiage.contracts.adapters import adapt_backend
from voiage.contracts.analysis import (
    AnalysisSpec,
    NumericalPolicy,
    ParameterSpec,
    RunContext,
    thaw_json,
)
from voiage.contracts.kernel import EvpiKernel, run_evpi
from voiage.contracts.normalized_input import (
    DatasetManifest,
    FieldManifest,
    NormalizedInputBundle,
    SourceProvenance,
    TableManifest,
)
from voiage.exceptions import InputError
from voiage.main_backends import get_backend
from voiage.methods.basic import evpi
from voiage.methods.information_source_portfolio import (
    information_source_portfolio_value,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO_FIXTURE = (
    ROOT
    / "specs/frontier/information-source-portfolio/v1/fixtures/normative/input.json"
)
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def _load_portfolio_template() -> dict[str, Any]:
    return json.loads(PORTFOLIO_FIXTURE.read_text(encoding="utf-8"))


def _tobac_spec(
    *,
    unit: str = "normalised_decision_units",
    upper_bound: float = 1.0,
    distribution_mean: float = 0.5,
    method_contract_version: str = "1.0.0",
    method_options: dict[str, Any] | None = None,
    normalized_input_digest: str = DIGEST_A,
    binding_profile_digest: str = DIGEST_B,
    numerical_policy: NumericalPolicy | None = None,
) -> AnalysisSpec:
    return AnalysisSpec(
        analysis_id="tobac-m2-retail-evpi",
        decision_problem_id="tobac-retail-partial-replacement",
        method_family="evpi",
        method_contract_version=method_contract_version,
        strategy_names=(
            "reference",
            "retail_restriction",
            "restriction_plus_support",
            "expanded_cessation",
        ),
        parameters=(
            ParameterSpec(
                parameter_id="lawful_replacement_share",
                label="Lawful replacement share",
                role="uncertain",
                dtype="float64",
                unit=unit,
                lower_bound=0.0,
                upper_bound=upper_bound,
                distribution={
                    "family": "beta",
                    "mean": distribution_mean,
                    "covariance": [[0.04]],
                },
                extensions={"source": "tobac.m2.retail.v1"},
            ),
            ParameterSpec(
                parameter_id="crime_weight",
                label="Crime weight",
                role="decision",
                dtype="float64",
                unit="weight",
                lower_bound=0.0,
                upper_bound=2.0,
            ),
        ),
        method_options=(
            {"prior_kind": "uniform", "quit_nodes": 4}
            if method_options is None
            else method_options
        ),
        input_artifact_ids=("m2-retail-payoffs",),
        normalized_input_digest=normalized_input_digest,
        binding_profile_digest=binding_profile_digest,
        numerical_policy=numerical_policy or NumericalPolicy(),
        tags=frozenset({"tobac", "r11"}),
        extensions={"consumer_schema": "tobac.m2.retail.v1"},
    )


def _payoff_bundle(
    payoffs: np.ndarray, *, dataset_id: str = "tobac-payoffs"
) -> NormalizedInputBundle:
    table = pa.table(
        {
            f"strategy_{index}": pa.array(payoffs[:, index], type=pa.float64())
            for index in range(payoffs.shape[1])
        }
    )
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        tables=(
            TableManifest(
                table_id="payoffs",
                fields=tuple(
                    FieldManifest(
                        field_id=f"strategy_{index}",
                        dtype="float64",
                        unit="normalised_decision_units",
                    )
                    for index in range(payoffs.shape[1])
                ),
            ),
        ),
        provenance=SourceProvenance(
            provider_id="tobac-synthetic-fixture",
            source_uri="file:///redacted/tobac/payoffs",
            descriptor_digest=DIGEST_C,
            governance={"schema": "tobac.m2.retail.v1"},
        ),
    )
    return NormalizedInputBundle(manifest=manifest, tables={"payoffs": table})


def _portfolio_payload(
    template: dict[str, Any],
    *,
    owned: bool = False,
    reverse_states: bool = False,
    reverse_sources: bool = False,
) -> dict[str, Any]:
    likelihood = ((0.8, 0.2), (0.2, 0.8))
    payload = deepcopy(template)
    payload["schema_version"] = "2.0.0"
    payload["value_context"].update(
        value_unit="unit", cost_unit="unit", delay_cost_per_time=0.0
    )
    payload["actions"] = ["a", "b"]
    payload["sources"] = payload["sources"][:2]
    for source, direct_cost in zip(payload["sources"], (0.1, 0.2), strict=True):
        source.update(
            cost=direct_cost,
            cost_unit="unit",
            latency=0.0,
            asset_ids=["lab"],
            must_precede=[],
            excludes=[],
        )
    payload["shared_costs"] = {
        "assets": [{"asset_id": "lab", "cost": 0.5, "cost_unit": "unit"}],
        "existing_assets": ["lab"] if owned else [],
    }
    payload["states"] = [
        {
            "state_id": f"s{state}-r{report}",
            "probability": 0.5 * likelihood[report][state],
            "action_values": {
                "a": 10.0 if state == 0 else 0.0,
                "b": 10.0 if state == 1 else 0.0,
            },
            "source_observations": {
                "survey": str(report),
                "registry": str(report),
            },
        }
        for state in range(2)
        for report in range(2)
    ]
    if reverse_states:
        payload["states"].reverse()
    if reverse_sources:
        payload["sources"].reverse()
    payload["constraints"]["required_coverage"] = []
    payload["constraints"]["max_sources"] = 2
    return payload


def _near_tie_portfolio_payload(
    template: dict[str, Any],
    *,
    order: Sequence[int],
    offset: float,
    labels: Sequence[int],
) -> dict[str, Any]:
    epsilon = Fraction(1, 2**32)
    weights = (Fraction(1, 4), Fraction(3, 4))
    payoffs = ((1 + epsilon, 1), (1, 1 + epsilon))

    payload = deepcopy(template)
    source = deepcopy(payload["sources"][0])
    source.update(
        cost=0.0,
        cost_unit="unit",
        latency=0.0,
        privacy_cost=0.0,
        must_precede=[],
        excludes=[],
    )
    source_id = source["source_id"]
    payload["value_context"].update(
        value_unit="unit", cost_unit="unit", delay_cost_per_time=0.0
    )
    payload["actions"] = ["a", "b"]
    payload["sources"] = [source]
    payload["states"] = [
        {
            "state_id": f"state-{state_index}",
            "probability": float(weights[state_index]),
            "action_values": {
                "a": float(payoffs[state_index][0] + offset),
                "b": float(payoffs[state_index][1] + offset),
            },
            "source_observations": {source_id: str(label)},
        }
        for state_index, label in zip(order, labels, strict=True)
    ]
    payload["constraints"].update(max_sources=1, required_coverage=[])
    return payload


def _sha256_bytes(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def test_r11_1_parameter_round_trip_defaults_and_fail_closed_rejections() -> None:
    """R11.1: preserve parameter semantics on round-trip and reject invalid inputs."""
    spec = _tobac_spec()
    canonical = spec.canonical_json()
    restored = AnalysisSpec.model_validate_json(canonical)

    assert restored == spec
    assert restored.contract_digest() == spec.contract_digest()
    assert restored.parameters[0].unit == "normalised_decision_units"
    assert restored.parameters[0].lower_bound == 0.0
    assert restored.parameters[0].upper_bound == 1.0
    assert restored.parameters[0].distribution == {
        "family": "beta",
        "mean": 0.5,
        "covariance": ((0.04,),),
    }
    assert thaw_json(restored.parameters[0].distribution) == {
        "family": "beta",
        "mean": 0.5,
        "covariance": [[0.04]],
    }
    assert restored.parameters[0].extensions == {"source": "tobac.m2.retail.v1"}
    assert restored.normalized_input_digest == DIGEST_A
    assert restored.binding_profile_digest == DIGEST_B

    default_policy = NumericalPolicy()
    assert spec.schema_version == "2.0.0"
    assert default_policy.dtype == "float64"
    assert default_policy.execution_mode == "deterministic"
    assert default_policy.allow_fallback is False
    assert default_policy.nan_policy == "raise"

    # Extra fields must fail closed across all contract models.
    with pytest.raises(ValidationError):
        ParameterSpec.model_validate(
            {
                "parameter_id": "x",
                "role": "uncertain",
                "dtype": "float64",
                "unexpected": 1,
            }
        )
    with pytest.raises(ValidationError):
        NumericalPolicy.model_validate({"unexpected": True})
    with pytest.raises(ValidationError):
        AnalysisSpec.model_validate(
            {**spec.model_dump(mode="python"), "unexpected": "bad"}
        )

    # Inverted bounds and non-finite floats must fail closed.
    with pytest.raises(ValidationError, match="lower_bound"):
        ParameterSpec(
            parameter_id="bad_bounds",
            role="uncertain",
            dtype="float64",
            lower_bound=2.0,
            upper_bound=1.0,
        )
    for bad_float in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError):
            ParameterSpec(
                parameter_id="bad_float",
                role="uncertain",
                dtype="float64",
                lower_bound=bad_float,
            )

    # Unsupported schema versions, malformed digests, and duplicate IDs must fail.
    for bad_version in ("1.0.0", "3.0.0"):
        with pytest.raises(ValidationError):
            AnalysisSpec.model_validate(
                {**spec.model_dump(mode="python"), "schema_version": bad_version}
            )
    for bad_digest in ("not-a-digest", "a" * 63, "A" * 64):
        with pytest.raises(ValidationError):
            _tobac_spec(normalized_input_digest=bad_digest)
        with pytest.raises(ValidationError):
            _tobac_spec(binding_profile_digest=bad_digest)
    with pytest.raises(ValidationError, match="strategy_names must be unique"):
        AnalysisSpec.model_validate(
            {
                **spec.model_dump(mode="python"),
                "strategy_names": ("dup", "dup"),
            }
        )
    duplicate_param = ParameterSpec(
        parameter_id="lawful_replacement_share", role="uncertain", dtype="float64"
    )
    with pytest.raises(ValidationError, match="parameter identifiers must be unique"):
        AnalysisSpec.model_validate(
            {
                **spec.model_dump(mode="python"),
                "parameters": (duplicate_param, duplicate_param),
            }
        )

    # Digest tracks parameter semantics and method options deterministically.
    base_digest = spec.contract_digest()
    assert _tobac_spec(unit="NZD").contract_digest() != base_digest
    assert _tobac_spec(upper_bound=0.9).contract_digest() != base_digest
    assert _tobac_spec(distribution_mean=0.6).contract_digest() != base_digest
    assert (
        _tobac_spec(
            method_options={"prior_kind": "uniform", "quit_nodes": 8}
        ).contract_digest()
        != base_digest
    )


def test_r11_2_scenario_isolation_immutability_and_order_invariance() -> None:
    """R11.2: keep scenario execution isolated, inputs immutable, and order-invariant."""
    template = _load_portfolio_template()
    spec = _tobac_spec()
    policy = NumericalPolicy(backend_preference=("numpy",))

    # Contract models and nested JSON structures cannot be mutated.
    with pytest.raises(ValidationError):
        spec.analysis_id = "mutated"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        spec.parameters[0].unit = "NZD"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        policy.dtype = "float32"  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.method_options["quit_nodes"] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        spec.parameters[0].extensions["source"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        spec.parameters[0].distribution["mean"] = 0.9  # type: ignore[index]

    payoffs_a = np.array(
        [
            [0.0, 0.2, 0.1, 0.05],
            [0.0, -0.1, 0.3, 0.2],
            [0.0, 0.4, 0.0, 0.1],
            [0.0, 0.1, 0.2, 0.25],
        ],
        dtype=np.float64,
    )
    payoffs_b = np.array(
        [
            [1.0, 2.5, 1.5, 0.5],
            [2.0, 0.5, 3.0, 1.0],
            [1.5, 1.0, 0.5, 2.0],
            [0.5, 2.0, 2.5, 1.5],
        ],
        dtype=np.float64,
    )
    digest_a_before = _sha256_bytes(payoffs_a)
    digest_b_before = _sha256_bytes(payoffs_b)

    spec_a = spec.model_copy(update={"analysis_id": "scenario-a"})
    spec_b = spec.model_copy(update={"analysis_id": "scenario-b"})

    fwd_a = run_evpi(payoffs_a, spec=spec_a, policy=policy)
    fwd_b = run_evpi(payoffs_b, spec=spec_b, policy=policy)
    rev_b = run_evpi(payoffs_b, spec=spec_b, policy=policy)
    rev_a = run_evpi(payoffs_a, spec=spec_a, policy=policy)
    rep_a = run_evpi(payoffs_a, spec=spec_a, policy=policy)

    assert fwd_a.payload.value == rev_a.payload.value == rep_a.payload.value
    assert fwd_b.payload.value == rev_b.payload.value
    assert evpi(payoffs_a) == fwd_a.payload.value
    assert evpi(payoffs_b) == fwd_b.payload.value
    assert _sha256_bytes(payoffs_a) == digest_a_before
    assert _sha256_bytes(payoffs_b) == digest_b_before

    # Portfolio scenarios must be isolated in forward/reverse order with unchanged inputs.
    portfolio_a = _portfolio_payload(template, owned=False)
    portfolio_b = _portfolio_payload(template, owned=True)
    snap_a = deepcopy(portfolio_a)
    snap_b = deepcopy(portfolio_b)

    res_fwd_a = information_source_portfolio_value(portfolio_a).to_contract_dict()
    res_fwd_b = information_source_portfolio_value(portfolio_b).to_contract_dict()
    res_rev_b = information_source_portfolio_value(portfolio_b).to_contract_dict()
    res_rev_a = information_source_portfolio_value(portfolio_a).to_contract_dict()

    assert portfolio_a == snap_a
    assert portfolio_b == snap_b
    assert res_fwd_a["optimum"] == res_rev_a["optimum"]
    assert res_fwd_a["evaluated_sequences"] == res_rev_a["evaluated_sequences"]
    assert res_fwd_b["optimum"] == res_rev_b["optimum"]
    assert res_fwd_b["evaluated_sequences"] == res_rev_b["evaluated_sequences"]

    # State and source reversals preserve portfolio gross, cost, and net values.
    reversed_portfolio_a = _portfolio_payload(
        template, owned=False, reverse_states=True, reverse_sources=True
    )
    res_reversed = information_source_portfolio_value(
        reversed_portfolio_a
    ).to_contract_dict()
    assert res_reversed["optimum"]["gross_value"] == pytest.approx(
        res_fwd_a["optimum"]["gross_value"]
    )
    assert res_reversed["optimum"]["total_source_cost"] == pytest.approx(
        res_fwd_a["optimum"]["total_source_cost"]
    )
    assert res_reversed["optimum"]["net_value"] == pytest.approx(
        res_fwd_a["optimum"]["net_value"]
    )


def test_r11_3_io_independent_calculation_and_exact_fraction_oracles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R11.3: match independent exact Fraction oracles with zero filesystem or network I/O."""
    template = _load_portfolio_template()
    spec = _tobac_spec(
        method_options={"prior_kind": "weighted_finite", "case": "near_tie"}
    ).model_copy(update={"strategy_names": ("a", "b")})
    policy = NumericalPolicy(backend_preference=("numpy",))
    kernel = EvpiKernel()
    numpy_backend = adapt_backend(get_backend("numpy"))
    context = RunContext(
        run_id="io-free-check",
        spec_digest=spec.contract_digest(),
        selected_backend="numpy",
        device="cpu",
        capabilities=frozenset({"dense-array", "deterministic"}),
        package_version="3.1.0",
        python_version="3.15.2",
        platform="test",
    )

    epsilon = Fraction(1, 2**32)
    weights = (Fraction(1, 4), Fraction(3, 4), Fraction(0))
    payoffs = ((1 + epsilon, 1), (1, 1 + epsilon), (100, -100))
    baseline = max(
        sum(w * row[a] for w, row in zip(weights, payoffs, strict=True))
        for a in range(2)
    )
    expected_near_tie = (
        sum(w * max(row) for w, row in zip(weights, payoffs, strict=True)) - baseline
    )
    assert expected_near_tie == Fraction(1, 2**34)

    uniform_equivalent = np.array(
        [
            [float(1 + epsilon), 1.0],
            [1.0, float(1 + epsilon)],
            [1.0, float(1 + epsilon)],
            [1.0, float(1 + epsilon)],
        ],
        dtype=np.float64,
    )
    portfolio_unowned = _portfolio_payload(template, owned=False)
    portfolio_owned = _portfolio_payload(template, owned=True)
    near_tie_payloads = [
        (
            labels,
            expected_near_tie if labels == order else Fraction(0),
            _near_tie_portfolio_payload(
                template, order=order, offset=offset, labels=labels
            ),
        )
        for order in permutations(range(2))
        for offset in (0.0, 8.0)
        for labels in (order, (0, 0))
    ]
    zero_prob_payload = deepcopy(near_tie_payloads[0][2])
    zero_prob_payload["states"].append(
        {
            "state_id": "zero-mass",
            "probability": 0.0,
            "action_values": {"a": 100.0, "b": -100.0},
            "source_observations": {zero_prob_payload["sources"][0]["source_id"]: "2"},
        }
    )
    # Warm up schema validator before disabling filesystem/network I/O.
    _ = information_source_portfolio_value(portfolio_unowned)

    def _forbid_io(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("unexpected filesystem or network I/O during calculation")

    monkeypatch.setattr(builtins, "open", _forbid_io)
    monkeypatch.setattr(Path, "open", _forbid_io)
    monkeypatch.setattr(socket, "create_connection", _forbid_io)
    # _package_version reads dist-info metadata; patch it so run_evpi stays I/O-free.
    monkeypatch.setattr("voiage.contracts.kernel._package_version", lambda: "3.1.0")

    # 1) Direct kernel, evpi(), and run_evpi() on the near-tie exact binary fraction oracle.
    kernel_payload = kernel.calculate(
        spec,
        uniform_equivalent,
        backend=numpy_backend,
        policy=policy,
        context=context,
    )
    assert Fraction(kernel_payload.value) == expected_near_tie
    assert Fraction(evpi(uniform_equivalent)) == expected_near_tie
    assert (
        Fraction(run_evpi(uniform_equivalent, spec=spec, policy=policy).payload.value)
        == expected_near_tie
    )

    for offset in (0.0, 8.0):
        for order in permutations(range(4)):
            permuted = uniform_equivalent[list(order)] + offset
            observed = run_evpi(permuted, spec=spec, policy=policy).payload.value
            assert Fraction(observed) == expected_near_tie

    # 2) Partitioned weighted finite-state near-tie oracle across state orders x offsets.
    for _labels, reference, payload in near_tie_payloads:
        result = information_source_portfolio_value(payload).to_contract_dict()
        assert Fraction(result["optimum"]["gross_value"]) == reference

    # Portfolio contract rejects zero-probability states explicitly (strict > 0 schema).
    with pytest.raises(InputError, match="probability"):
        information_source_portfolio_value(zero_prob_payload)

    # 3) Dependent duplicate 80%-accurate reports + once-only shared-asset oracle.
    # States (0.5, 0.5), payoff (10, 0)/(0, 10) => baseline 5, posterior 8 => EVSI = 3.
    expected_gross = Fraction(3)
    for owned, expected_cost in ((False, Fraction(4, 5)), (True, Fraction(3, 10))):
        payload = portfolio_owned if owned else portfolio_unowned
        result = information_source_portfolio_value(payload).to_contract_dict()
        seq = next(
            item
            for item in result["evaluated_sequences"]
            if item["source_sequence"] == ["survey", "registry"]
        )
        assert seq["gross_value"] == pytest.approx(float(expected_gross))
        assert seq["total_source_cost"] == pytest.approx(float(expected_cost))
        assert seq["net_value"] == pytest.approx(float(expected_gross - expected_cost))


def test_r11_3_optional_jax_float64_near_tie_oracle() -> None:
    """R11.3: verify optional JAX float64 backend matches the exact binary oracle."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from voiage.main_backends import JaxBackend

    jax.config.update("jax_enable_x64", True)
    backend = JaxBackend()
    epsilon = Fraction(1, 2**32)
    expected = Fraction(1, 2**34)
    base = np.array(
        [
            [float(1 + epsilon), 1.0],
            [1.0, float(1 + epsilon)],
            [1.0, float(1 + epsilon)],
            [1.0, float(1 + epsilon)],
        ],
        dtype=np.float64,
    )
    for offset in (0.0, 8.0):
        for order in permutations(range(4)):
            jax_array = jnp.asarray(base[list(order)] + offset, dtype=jnp.float64)
            assert jax_array.dtype == jnp.float64
            observed = backend.evpi(jax_array)
            assert Fraction(observed) == expected


def test_r11_4_bounded_extension_preserves_reference_behavior() -> None:
    """R11.4: running an extended scenario leaves reference behavior and digests unchanged."""
    template = _load_portfolio_template()
    ref_spec = _tobac_spec()
    ref_digest_before = ref_spec.contract_digest()
    ref_payoffs = np.array([[10.0, 0.0], [0.0, 10.0]], dtype=np.float64)
    ref_bundle = _payoff_bundle(ref_payoffs, dataset_id="ref-payoffs")
    ref_bundle_digest_before = ref_bundle.content_digest
    ref_evpi_before = run_evpi(
        ref_payoffs,
        spec=ref_spec.model_copy(update={"strategy_names": ("a", "b")}),
    ).payload.value

    # Extended spec with an added parameter and an added 3rd strategy/column.
    ext_spec = _tobac_spec().model_copy(
        update={
            "analysis_id": "tobac-m2-retail-evpi-extended",
            "strategy_names": ("a", "b", "c"),
            "parameters": (
                *ref_spec.parameters,
                ParameterSpec(
                    parameter_id="enforcement_intensity",
                    role="design",
                    dtype="float64",
                    unit="inspections_per_year",
                    lower_bound=0.0,
                    upper_bound=500.0,
                ),
            ),
        }
    )
    ext_payoffs = np.array([[10.0, 0.0, 4.0], [0.0, 10.0, 4.0]], dtype=np.float64)
    ext_result = run_evpi(ext_payoffs, spec=ext_spec)
    assert ext_result.payload.value == pytest.approx(5.0)

    # Extended portfolio with an additional shared asset runs alongside reference.
    ref_portfolio = _portfolio_payload(template, owned=False)
    ref_portfolio_before = information_source_portfolio_value(
        ref_portfolio
    ).to_contract_dict()
    ext_portfolio = deepcopy(ref_portfolio)
    ext_portfolio["shared_costs"]["assets"].append(
        {"asset_id": "customs_audit", "cost": 0.4, "cost_unit": "unit"}
    )
    ext_portfolio["sources"][0]["asset_ids"].append("customs_audit")
    ext_portfolio_result = information_source_portfolio_value(
        ext_portfolio
    ).to_contract_dict()
    ext_joint = next(
        item
        for item in ext_portfolio_result["evaluated_sequences"]
        if item["source_sequence"] == ["survey", "registry"]
    )
    assert ext_joint["total_source_cost"] == pytest.approx(1.2)
    assert ext_portfolio_result["optimum"]["source_sequence"] == ["registry"]
    assert ext_portfolio_result["optimum"]["total_source_cost"] == pytest.approx(0.7)

    ref_evpi_after = run_evpi(
        ref_payoffs,
        spec=ref_spec.model_copy(update={"strategy_names": ("a", "b")}),
    ).payload.value
    ref_portfolio_after = information_source_portfolio_value(
        ref_portfolio
    ).to_contract_dict()

    assert ref_spec.contract_digest() == ref_digest_before
    assert ref_bundle.content_digest == ref_bundle_digest_before
    assert ref_evpi_after == ref_evpi_before
    assert ref_portfolio_after == ref_portfolio_before


def test_r11_5_result_identity_separates_parameters_schema_and_execution() -> None:
    """R11.5: result provenance distinguishes input, contract version, and execution changes."""
    payoffs_v1 = np.array([[10.0, 0.0], [0.0, 10.0]], dtype=np.float64)
    payoffs_v2 = np.array([[12.0, 0.0], [0.0, 10.0]], dtype=np.float64)
    bundle_v1 = _payoff_bundle(payoffs_v1, dataset_id="tobac-payoffs-v1")
    bundle_v2 = _payoff_bundle(payoffs_v2, dataset_id="tobac-payoffs-v1")

    base_policy = NumericalPolicy(
        backend_preference=("numpy",), absolute_tolerance=1e-10, seed=7
    )
    alt_policy = NumericalPolicy(
        backend_preference=("numpy",), absolute_tolerance=1e-12, seed=19
    )

    base_spec = _tobac_spec(
        method_contract_version="1.0.0",
        normalized_input_digest=bundle_v1.content_digest,
        numerical_policy=base_policy,
    ).model_copy(update={"strategy_names": ("a", "b")})

    changed_input_spec = base_spec.model_copy(
        update={"normalized_input_digest": bundle_v2.content_digest}
    )
    changed_version_spec = base_spec.model_copy(
        update={"method_contract_version": "1.1.0"}
    )

    res_base = run_evpi(payoffs_v1, spec=base_spec, policy=base_policy)
    res_input = run_evpi(payoffs_v2, spec=changed_input_spec, policy=base_policy)
    res_version = run_evpi(payoffs_v1, spec=changed_version_spec, policy=base_policy)
    res_exec = run_evpi(payoffs_v1, spec=base_spec, policy=alt_policy)

    # 1) Parameter/input change updates input and calculation digests, not contract version/backend.
    assert bundle_v2.content_digest != bundle_v1.content_digest
    assert res_input.run_context.input_digest != res_base.run_context.input_digest
    assert (
        res_input.provenance.details["calculation_input_digest"]
        != res_base.provenance.details["calculation_input_digest"]
    )
    assert res_input.method_contract_version == res_base.method_contract_version
    assert (
        res_input.run_context.selected_backend == res_base.run_context.selected_backend
    )

    # 2) Model/schema version change updates spec_digest and contract version, preserving input digest.
    assert res_version.method_contract_version == "1.1.0"
    assert res_version.run_context.spec_digest != res_base.run_context.spec_digest
    assert res_version.run_context.input_digest == res_base.run_context.input_digest
    assert (
        res_version.provenance.details["calculation_input_digest"]
        == res_base.provenance.details["calculation_input_digest"]
    )

    # 3) Execution-setting change updates spec_digest and seed, preserving input digest and EVPI.
    assert res_exec.run_context.spec_digest != res_base.run_context.spec_digest
    assert res_exec.provenance.seed == 19
    assert res_base.provenance.seed == 7
    assert (
        res_exec.provenance.details["calculation_input_digest"]
        == res_base.provenance.details["calculation_input_digest"]
    )
    assert res_exec.payload.value == res_base.payload.value
