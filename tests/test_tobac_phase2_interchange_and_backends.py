"""Phase 2 interchange and optional backend qualification tests for tobac."""

from __future__ import annotations

from fractions import Fraction
from importlib import import_module
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
import pytest

from voiage.contracts.analysis import AnalysisSpec, NumericalPolicy, ParameterSpec
from voiage.contracts.kernel import run_evpi
from voiage.contracts.normalized_input import NormalizedInputBundle
from voiage.ingestion.dataframe import from_dataframe
from voiage.methods.basic import evpi
from voiage.schema import ValueArray

if TYPE_CHECKING:
    from pathlib import Path

    import pyarrow as pa

_STRATEGIES = (
    "reference",
    "retail_restriction",
    "restriction_plus_support",
    "expanded_cessation",
)
_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


def _buffer_addresses(column: pa.ChunkedArray) -> tuple[tuple[int | None, ...], ...]:
    return tuple(
        tuple(
            buffer.address if buffer is not None else None for buffer in chunk.buffers()
        )
        for chunk in column.chunks
    )


def _synthetic_regulatory_payoffs(
    n_draws: int = 4096,
    *,
    seed: int = 20261126,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=0.5, scale=0.25, size=(n_draws, 1))
    offsets = np.array([[0.0, 0.04, 0.06, 0.02]], dtype=np.float64)
    noise = rng.normal(loc=0.0, scale=0.12, size=(n_draws, len(_STRATEGIES)))
    return np.asarray(base + offsets + noise, dtype=np.float64)


def _long_paired_draw_frame(payoffs: np.ndarray) -> pl.DataFrame:
    n_draws, n_strategies = payoffs.shape
    draw_ids: list[int] = []
    strategy_ids: list[str] = []
    perspectives: list[str] = []
    subgroups: list[str] = []
    net_benefits: list[float] = []
    draw_weights: list[float] = []
    weight = 1.0 / float(n_draws)
    for draw_idx in range(n_draws):
        perspective = "payer" if draw_idx % 2 == 0 else "societal"
        subgroup = "adult_retail" if draw_idx % 3 == 0 else "youth_retail"
        for strat_idx in range(n_strategies):
            draw_ids.append(draw_idx)
            strategy_ids.append(_STRATEGIES[strat_idx])
            perspectives.append(perspective)
            subgroups.append(subgroup)
            net_benefits.append(float(payoffs[draw_idx, strat_idx]))
            draw_weights.append(weight)
    return pl.DataFrame(
        {
            "draw_id": pl.Series("draw_id", draw_ids, dtype=pl.Int64),
            "strategy_id": pl.Series("strategy_id", strategy_ids, dtype=pl.String),
            "perspective": pl.Series("perspective", perspectives, dtype=pl.String),
            "subgroup_id": pl.Series("subgroup_id", subgroups, dtype=pl.String),
            "net_benefit": pl.Series("net_benefit", net_benefits, dtype=pl.Float64),
            "draw_weight": pl.Series("draw_weight", draw_weights, dtype=pl.Float64),
        }
    )


def _load_paired_draw_bundle(
    frame: pa.Table | pl.DataFrame,
    *,
    dataset_id: str = "tobac-phase2-paired",
) -> NormalizedInputBundle:
    experimental = import_module("voiage.experimental")
    return cast(
        "NormalizedInputBundle",
        experimental.paired_draw_bundle(
            frame,
            dataset_id=dataset_id,
            strategy_ids=_STRATEGIES,
            weight_field="draw_weight",
        ),
    )


def _evpi_spec(
    *,
    strategy_names: tuple[str, ...] = _STRATEGIES,
    numerical_policy: NumericalPolicy | None = None,
) -> AnalysisSpec:
    return AnalysisSpec(
        analysis_id="tobac-phase2-evpi-qualification",
        decision_problem_id="tobac-retail-partial-replacement",
        method_family="evpi",
        method_contract_version="1.0.0",
        strategy_names=strategy_names,
        parameters=(
            ParameterSpec(
                parameter_id="lawful_replacement_share",
                label="Lawful replacement share",
                role="uncertain",
                dtype="float64",
                unit="normalised_decision_units",
                lower_bound=0.0,
                upper_bound=1.0,
            ),
        ),
        normalized_input_digest=_DIGEST_A,
        binding_profile_digest=_DIGEST_B,
        numerical_policy=numerical_policy or NumericalPolicy(),
    )


def test_polars_paired_draw_bundle_ipc_parquet_round_trip_and_zero_copy(
    tmp_path: Path,
) -> None:
    payoffs = _synthetic_regulatory_payoffs(n_draws=64)
    polars_frame = _long_paired_draw_frame(payoffs)
    arrow_source = polars_frame.to_arrow(compat_level=pl.CompatLevel.newest())

    direct_zero_copy = from_dataframe(
        arrow_source,
        dataset_id="tobac-phase2-direct",
        table_id="net_benefits",
        allow_copy=False,
    )
    assert direct_zero_copy.table("net_benefits").num_rows == polars_frame.height
    assert any(
        diag.code == "dataframe_interchange.copy.zero_copy"
        for diag in direct_zero_copy.manifest.diagnostics
    )

    arrow_bundle = _load_paired_draw_bundle(arrow_source, dataset_id="tobac-arrow")
    arrow_table = arrow_bundle.table("net_benefits")
    assert arrow_table.equals(arrow_source)
    for col_name in arrow_source.column_names:
        assert _buffer_addresses(arrow_table[col_name]) == _buffer_addresses(
            arrow_source[col_name]
        )

    polars_bundle = _load_paired_draw_bundle(polars_frame, dataset_id="tobac-polars")
    assert any(
        diag.code == "dataframe_interchange.copy.zero_copy"
        for diag in polars_bundle.manifest.diagnostics
    )
    polars_table = polars_bundle.table("net_benefits")
    assert polars_table.equals(arrow_source)
    for col_name in arrow_source.column_names:
        assert _buffer_addresses(polars_table[col_name]) == _buffer_addresses(
            arrow_source[col_name]
        )

    ipc_path = tmp_path / "tobac_phase2.arrow"
    parquet_path = tmp_path / "tobac_phase2.parquet"
    polars_bundle.write_ipc(ipc_path)
    polars_bundle.write_parquet(parquet_path)

    restored_ipc = NormalizedInputBundle.read_ipc(ipc_path)
    restored_parquet = NormalizedInputBundle.read_parquet(parquet_path)

    for restored in (restored_ipc, restored_parquet):
        assert restored.content_digest == polars_bundle.content_digest
        restored_table = restored.table("net_benefits")
        assert restored_table.equals(arrow_source)
        for col_name in (
            "draw_id",
            "strategy_id",
            "perspective",
            "subgroup_id",
            "net_benefit",
            "draw_weight",
        ):
            assert (
                restored_table[col_name].to_pylist()
                == arrow_source[col_name].to_pylist()
            )
        payoff_binding, weight_binding = restored.manifest.bindings
        assert payoff_binding.strategy_names == _STRATEGIES
        assert payoff_binding.sample_id_field_id == "draw_id"
        assert payoff_binding.strategy_field_id == "strategy_id"
        assert weight_binding.field_ids == ("draw_weight",)


def test_value_array_numpy_vs_polars_evpi_parity() -> None:
    payoffs = _synthetic_regulatory_payoffs(n_draws=4096)
    strategies = list(_STRATEGIES)
    polars_wide = pl.DataFrame(
        {name: payoffs[:, idx] for idx, name in enumerate(strategies)}
    )

    va_numpy = ValueArray.from_numpy(payoffs, strategy_names=strategies)
    va_polars = ValueArray.from_polars(polars_wide)

    assert va_numpy == va_polars
    assert va_numpy.strategy_names == va_polars.strategy_names == strategies
    assert np.array_equal(va_numpy.numpy_values, payoffs)
    assert np.array_equal(va_polars.numpy_values, payoffs)

    evpi_numpy = evpi(va_numpy)
    evpi_polars = evpi(va_polars)
    assert evpi_numpy == evpi_polars


def test_evpi_parity_core_contract_and_jax_x64_on_regulatory_grid() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from voiage.main_backends import JaxBackend

    jax.config.update("jax_enable_x64", True)
    payoffs = _synthetic_regulatory_payoffs(n_draws=4096)
    spec = _evpi_spec()

    core_value = evpi(payoffs)
    contract_numpy = run_evpi(
        payoffs,
        spec=spec,
        policy=NumericalPolicy(backend_preference=("numpy",)),
    ).payload.value
    contract_jax = run_evpi(
        payoffs,
        spec=spec,
        policy=NumericalPolicy(backend_preference=("jax",)),
    ).payload.value
    jax_value = JaxBackend().evpi(jnp.asarray(payoffs, dtype=jnp.float64))

    assert abs(core_value - contract_numpy) <= 1e-12
    assert abs(core_value - jax_value) <= 1e-12
    assert abs(contract_numpy - jax_value) <= 1e-12
    assert abs(core_value - contract_jax) <= 1e-12
    assert abs(contract_jax - jax_value) <= 1e-12


def test_exact_binary_fraction_near_tie_oracle_parity_with_jax_x64() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from voiage.main_backends import JaxBackend

    jax.config.update("jax_enable_x64", True)
    epsilon = Fraction(1, 2**32)
    expected = Fraction(1, 2**34)
    payoffs = np.array(
        [
            [float(1 + epsilon), 1.0],
            [1.0, float(1 + epsilon)],
            [1.0, float(1 + epsilon)],
            [1.0, float(1 + epsilon)],
        ],
        dtype=np.float64,
    )
    spec = _evpi_spec(strategy_names=("a", "b"))

    core_value = evpi(payoffs)
    contract_value = run_evpi(
        payoffs,
        spec=spec,
        policy=NumericalPolicy(backend_preference=("numpy",)),
    ).payload.value
    jax_array = jnp.asarray(payoffs, dtype=jnp.float64)
    assert jax_array.dtype == jnp.float64
    jax_value = JaxBackend().evpi(jax_array)

    expected_float = float(expected)
    assert Fraction(core_value) == expected
    assert Fraction(contract_value) == expected
    assert Fraction(jax_value) == expected
    assert abs(core_value - expected_float) <= 1e-12
    assert abs(contract_value - expected_float) <= 1e-12
    assert abs(jax_value - expected_float) <= 1e-12
