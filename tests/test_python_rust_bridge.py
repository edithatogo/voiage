"""Contracts for the private production Python-to-Rust bridge."""

from __future__ import annotations

import hashlib
from importlib import import_module
from importlib.machinery import EXTENSION_SUFFIXES
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI
    import tomli as tomllib

import numpy as np
import pytest
import rfc8785

import voiage
from voiage.methods.ceaf import CEAFResult
from voiage.methods.dominance import DominanceResult

ROOT = Path(__file__).resolve().parents[1]
PYTHON_CRATE = ROOT / "rust/crates/voiage-python/Cargo.toml"
WORKSPACE_MANIFEST = ROOT / "rust/Cargo.toml"


def _cargo_version() -> str:
    workspace = tomllib.loads(WORKSPACE_MANIFEST.read_text(encoding="utf-8"))
    return workspace["workspace"]["package"]["version"]


def _payload_digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(rfc8785.dumps(payload)).hexdigest()


def _build_id(info: dict[str, object]) -> str:
    values = (
        info["build_id_algorithm"],
        info["source_revision"],
        info["source_tree_git_oid"],
        str(info["source_dirty"]).lower(),
        info["source_state_sha256"],
        info["target_triple"],
        info["rustc_version"],
        info["build_profile"],
        info["cargo_lock_sha256"],
        "" if info["source_date_epoch"] is None else str(info["source_date_epoch"]),
    )
    encoded = "".join(f"{len(value)}:{value}" for value in values).encode()
    return hashlib.sha256(encoded).hexdigest()


def _ceaf_result() -> CEAFResult:
    return CEAFResult(
        wtp_thresholds=np.array([0.0, 1.0]),
        optimal_strategy_indices=np.array([0, 1]),
        optimal_strategy_names=["A", "B"],
        acceptability_probabilities=np.array([0.75, 0.8]),
        probability_lower=np.array([0.6, 0.7]),
        probability_upper=np.array([0.9, 0.95]),
        expected_net_benefit=np.array([5.0, 6.0]),
        reporting={"standard": "CHEERS 2022"},
    )


def _dominance_result() -> DominanceResult:
    return DominanceResult(
        strategy_names=["A", "B"],
        costs=np.array([1.0, 2.0]),
        effects=np.array([1.0, 2.0]),
        frontier_indices=[0, 1],
        strongly_dominated_indices=[],
        extended_dominated_indices=[],
        status=["frontier", "frontier"],
        incremental_costs=np.array([1.0]),
        incremental_effects=np.array([1.0]),
        icers=np.array([1.0]),
        reporting={"standard": "CHEERS 2022"},
    )


def _ceaf_native_kwargs() -> dict[str, object]:
    return {
        "analysis_id": "analysis-001",
        "decision_problem_id": "decision-001",
        "wtp_thresholds": [0.0, 1.0],
        "optimal_strategy_indices": [0, 1],
        "optimal_strategy_names": ["A", "B"],
        "acceptability_probabilities": [0.75, 0.8],
        "probability_lower": [0.6, 0.7],
        "probability_upper": [0.9, 0.95],
        "expected_net_benefit": [5.0, 6.0],
        "reporting": {"standard": "CHEERS 2022"},
    }


def _dominance_native_kwargs() -> dict[str, object]:
    return {
        "analysis_id": "analysis-001",
        "decision_problem_id": "decision-001",
        "strategy_names": ["A", "B"],
        "costs": [1.0, 2.0],
        "effects": [1.0, 2.0],
        "frontier_indices": [0, 1],
        "strongly_dominated_indices": [],
        "extended_dominated_indices": [],
        "status": ["frontier", "frontier"],
        "incremental_costs": [1.0],
        "incremental_effects": [1.0],
        "icers": [1.0],
        "reporting": {"standard": "CHEERS 2022"},
    }


def test_private_native_extension_reports_runtime_provenance() -> None:
    native = import_module("voiage._core")
    info = native.runtime_info()

    assert any(str(native.__file__).endswith(suffix) for suffix in EXTENSION_SUFFIXES)
    assert info["engine"] == "rust"
    assert info["bridge"] == "pyo3"
    assert info["runtime_info_schema"] == 3
    assert info["core_version"] == _cargo_version()
    assert info["abi_version"] == 1
    assert re.fullmatch(r"[0-9a-f]{40}", info["source_revision"])
    assert isinstance(info["source_dirty"], bool)
    assert re.fullmatch(r"[0-9a-f]{64}", info["cargo_lock_sha256"])
    assert re.fullmatch(r"[0-9a-f]{40}", info["source_tree_git_oid"])
    assert re.fullmatch(r"[0-9a-f]{64}", info["source_state_sha256"])
    assert info["source_state_algorithm"] == "git-diff-and-untracked-sha256-v1"
    assert re.fullmatch(r"[0-9a-f]{64}", info["build_id"])
    assert info["build_id_algorithm"] == "length-prefixed-sha256-v2"
    assert info["build_id"] == _build_id(info)
    assert info["digest_algorithm"] == "rfc8785-sha256-v1"
    assert re.fullmatch(
        r"[A-Za-z0-9_]+-[A-Za-z0-9_]+-[A-Za-z0-9_.-]+", info["target_triple"]
    )
    assert re.fullmatch(r"rustc \d+\.\d+\.\d+(?: .+)?", info["rustc_version"])
    assert info["build_profile"] in {"debug", "dev", "release"}
    assert info["source_date_epoch"] is None or isinstance(
        info["source_date_epoch"], int
    )
    assert set(info["operations"]) == {
        "serialize_ceaf_result",
        "serialize_dominance_result",
    }
    for operation in info["operations"].values():
        assert set(operation) == {
            "calls",
            "native_entries",
            "successes",
            "failures",
            "digest_algorithm",
            "last_payload_sha256",
        }
        assert operation["digest_algorithm"] == info["digest_algorithm"]
        assert operation["calls"] == operation["successes"]
        assert operation["native_entries"] == (
            operation["successes"] + operation["failures"]
        )
        assert operation["last_payload_sha256"] is None or re.fullmatch(
            r"[0-9a-f]{64}", operation["last_payload_sha256"]
        )


def test_python_adapter_is_a_private_leaf_crate() -> None:
    workspace = tomllib.loads((ROOT / "rust/Cargo.toml").read_text(encoding="utf-8"))
    assert "crates/voiage-python" in workspace["workspace"]["members"]

    adapter = tomllib.loads(PYTHON_CRATE.read_text(encoding="utf-8"))
    dependencies = adapter["dependencies"]
    assert dependencies["pyo3"]["features"] == ["abi3-py310"]
    assert "voiage-domain" in dependencies
    assert "voiage-diagnostics" in dependencies
    assert "voiage-serialization" in dependencies
    assert "voiage-numerics" in dependencies

    for manifest_path in (ROOT / "rust/crates").glob("*/Cargo.toml"):
        if manifest_path == PYTHON_CRATE:
            continue
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        assert "pyo3" not in manifest.get("dependencies", {})
        assert "voiage-python" not in manifest.get("dependencies", {})


def test_maturin_maps_the_private_mixed_project_module() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    maturin = config["tool"]["maturin"]

    assert config["build-system"] == {
        "requires": ["maturin>=1.9,<2.0"],
        "build-backend": "maturin",
    }
    assert maturin == {
        "manifest-path": "rust/crates/voiage-python/Cargo.toml",
        "python-source": ".",
        "module-name": "voiage._core",
        "bindings": "pyo3",
        "features": ["pyo3/extension-module"],
        "locked": True,
        "include": [
            {
                "path": "rust/crates/voiage-python/source-provenance.txt",
                "format": "sdist",
            }
        ],
        "sbom": {"rust": False},
    }


def test_bridge_does_not_expand_the_frozen_public_api() -> None:
    assert "_core" not in voiage.__all__
    assert "_runtime" not in voiage.__all__

    stable_api = json.loads(
        (ROOT / "specs/v1/stable-api.json").read_text(encoding="utf-8")
    )
    declared_symbols = {
        symbol for group in stable_api["symbols"].values() for symbol in group
    }
    assert declared_symbols.isdisjoint(
        {"_core", "_runtime", "runtime_info", "runtime_info_schema"}
    )


@pytest.mark.parametrize(
    ("operation", "kwargs_factory"),
    [
        ("serialize_ceaf_result", _ceaf_native_kwargs),
        ("serialize_dominance_result", _dominance_native_kwargs),
    ],
)
def test_native_payload_digest_and_success_counters_prove_lineage(
    operation: str,
    kwargs_factory: Callable[[], dict[str, object]],
) -> None:
    native = import_module("voiage._core")
    before = dict(native.runtime_info()["operations"][operation])

    payload = getattr(native, operation)(**kwargs_factory())

    after = native.runtime_info()["operations"][operation]
    assert after["calls"] == before["calls"] + 1
    assert after["native_entries"] == before["native_entries"] + 1
    assert after["successes"] == before["successes"] + 1
    assert after["failures"] == before["failures"]
    assert after["last_payload_sha256"] == _payload_digest(payload)


@pytest.mark.parametrize(
    "value",
    [1e20, -1e20, 1e-7, -0.0, float(2**53), float(2**53 + 2)],
)
@pytest.mark.parametrize(
    ("operation", "kwargs_factory", "field"),
    [
        ("serialize_ceaf_result", _ceaf_native_kwargs, "expected_net_benefit"),
        ("serialize_dominance_result", _dominance_native_kwargs, "costs"),
    ],
)
def test_native_lineage_preserves_float_types_at_jcs_number_boundaries(
    operation: str,
    kwargs_factory: Callable[[], dict[str, object]],
    field: str,
    value: float,
) -> None:
    native = import_module("voiage._core")
    kwargs = kwargs_factory()
    values = list(kwargs[field])
    values[0] = value
    kwargs[field] = values

    payload = getattr(native, operation)(**kwargs)

    assert isinstance(payload[field][0], float)
    assert native.runtime_info()["operations"][operation][
        "last_payload_sha256"
    ] == _payload_digest(payload)


@pytest.mark.parametrize(
    ("operation", "kwargs_factory", "malformed_field"),
    [
        ("serialize_ceaf_result", _ceaf_native_kwargs, "optimal_strategy_indices"),
        ("serialize_dominance_result", _dominance_native_kwargs, "frontier_indices"),
    ],
)
def test_native_rejections_increment_only_attempt_and_failure_counters(
    operation: str,
    kwargs_factory: Callable[[], dict[str, object]],
    malformed_field: str,
) -> None:
    native = import_module("voiage._core")
    before = dict(native.runtime_info()["operations"][operation])
    kwargs = kwargs_factory()
    kwargs[malformed_field] = [-1]

    with pytest.raises(native.InputError):
        getattr(native, operation)(**kwargs)

    after = native.runtime_info()["operations"][operation]
    assert after["calls"] == before["calls"]
    assert after["native_entries"] == before["native_entries"] + 1
    assert after["successes"] == before["successes"]
    assert after["failures"] == before["failures"] + 1
    assert after["last_payload_sha256"] == before["last_payload_sha256"]


def test_ceaf_serialization_is_rust_owned_and_counted(monkeypatch) -> None:
    runtime = import_module("voiage._runtime")
    native = import_module("voiage._core")
    before = native.runtime_info()["operations"]["serialize_ceaf_result"]["calls"]

    def sentinel(**_: object) -> dict[str, object]:
        raise RuntimeError("Rust CEAF serializer invoked")

    monkeypatch.setattr(runtime, "serialize_ceaf_result", sentinel)
    with pytest.raises(RuntimeError, match="Rust CEAF serializer invoked"):
        _ceaf_result().to_dict(
            analysis_id="analysis-001",
            decision_problem_id="decision-001",
        )

    monkeypatch.undo()
    _ceaf_result().to_dict(
        analysis_id="analysis-001",
        decision_problem_id="decision-001",
    )
    after = native.runtime_info()["operations"]["serialize_ceaf_result"]["calls"]
    assert after == before + 1


def test_dominance_serialization_is_rust_owned_and_counted(monkeypatch) -> None:
    runtime = import_module("voiage._runtime")
    native = import_module("voiage._core")
    before = native.runtime_info()["operations"]["serialize_dominance_result"]["calls"]

    def sentinel(**_: object) -> dict[str, object]:
        raise RuntimeError("Rust dominance serializer invoked")

    monkeypatch.setattr(runtime, "serialize_dominance_result", sentinel)
    with pytest.raises(RuntimeError, match="Rust dominance serializer invoked"):
        _dominance_result().to_dict(
            analysis_id="analysis-001",
            decision_problem_id="decision-001",
        )

    monkeypatch.undo()
    _dominance_result().to_dict(
        analysis_id="analysis-001",
        decision_problem_id="decision-001",
    )
    after = native.runtime_info()["operations"]["serialize_dominance_result"]["calls"]
    assert after == before + 1


@pytest.mark.parametrize("malformed_index", [-1, 2**63, 2**200])
@pytest.mark.parametrize(
    ("operation", "kwargs_factory", "field"),
    [
        ("serialize_ceaf_result", _ceaf_native_kwargs, "optimal_strategy_indices"),
        ("serialize_dominance_result", _dominance_native_kwargs, "frontier_indices"),
        (
            "serialize_dominance_result",
            _dominance_native_kwargs,
            "strongly_dominated_indices",
        ),
        (
            "serialize_dominance_result",
            _dominance_native_kwargs,
            "extended_dominated_indices",
        ),
    ],
)
def test_exported_bridge_rejects_all_malformed_indices_with_stable_error(
    operation: str,
    kwargs_factory: Callable[[], dict[str, object]],
    field: str,
    malformed_index: int,
) -> None:
    native = import_module("voiage._core")
    kwargs = kwargs_factory()
    kwargs[field] = [malformed_index]

    with pytest.raises(native.InputError) as error:
        getattr(native, operation)(**kwargs)

    assert error.value.args[0] == "invalid_input"


@pytest.mark.parametrize(
    ("operation", "kwargs_factory"),
    [
        ("serialize_ceaf_result", _ceaf_native_kwargs),
        ("serialize_dominance_result", _dominance_native_kwargs),
    ],
)
def test_exported_bridge_maps_reporting_json_failures_to_stable_error(
    operation: str,
    kwargs_factory: Callable[[], dict[str, object]],
) -> None:
    native = import_module("voiage._core")
    kwargs = kwargs_factory()
    kwargs["reporting"] = {"unsupported": object()}

    with pytest.raises(native.SerializationError) as error:
        getattr(native, operation)(**kwargs)

    assert error.value.args[0] == "serialization_failure"
