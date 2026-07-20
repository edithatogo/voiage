"""Black-box contracts for a wheel installed outside the source checkout."""

from __future__ import annotations

import hashlib
from importlib.metadata import requires, version
import os
from pathlib import Path
import re

import numpy as np
import rfc8785

import voiage
import voiage._core as native
from voiage.methods.ceaf import CEAFResult
from voiage.methods.dominance import DominanceResult


def _digest(payload: dict[str, object]) -> str:
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


def _clean_source_state(info: dict[str, object]) -> str:
    hasher = hashlib.sha256()
    for value in (
        info["source_state_algorithm"],
        info["source_tree_git_oid"],
    ):
        encoded = value.encode()
        hasher.update(len(encoded).to_bytes(8, byteorder="big"))
        hasher.update(encoded)
    return hasher.hexdigest()


def test_imports_resolve_inside_the_wheel_environment() -> None:
    """Reject source-shadowed imports when CI supplies the wheel environment."""
    environment = os.environ.get("WHEEL_VENV")
    if environment is None:
        return

    root = Path(environment).resolve()
    assert Path(voiage.__file__).resolve().is_relative_to(root)
    assert Path(native.__file__).resolve().is_relative_to(root)


def test_installed_wheel_metadata_keeps_jax_optional() -> None:
    """Verify the built artifact, rather than only source TOML metadata."""
    if os.environ.get("WHEEL_VENV") is None:
        return
    requirements = requires("voiage") or []
    jax_requirements = [
        item.lower() for item in requirements if item.lower().startswith("jax")
    ]

    assert jax_requirements
    assert all("extra ==" in item for item in jax_requirements)
    assert any(
        'extra == "jax"' in item or "extra == 'jax'" in item
        for item in jax_requirements
    )


def test_installed_native_provenance_matches_built_artifact() -> None:
    if os.environ.get("WHEEL_VENV") is None:
        return
    info = native.runtime_info()
    assert info["core_version"] == version("voiage")
    assert info["source_revision"] == os.environ["EXPECTED_SOURCE_REVISION"]
    assert info["source_tree_git_oid"] == os.environ["EXPECTED_SOURCE_TREE_GIT_OID"]
    assert info["source_dirty"] is False
    assert info["runtime_info_schema"] == 3
    assert info["digest_algorithm"] == "rfc8785-sha256-v1"
    assert info["build_id_algorithm"] == "length-prefixed-sha256-v2"
    assert info["source_state_algorithm"] == "git-diff-and-untracked-sha256-v1"
    assert re.fullmatch(r"[0-9a-f]{64}", info["source_state_sha256"])
    assert info["source_state_sha256"] == _clean_source_state(info)
    assert re.fullmatch(r"[0-9a-f]{40}", info["source_tree_git_oid"])
    assert re.fullmatch(r"[0-9a-f]{64}", info["build_id"])
    assert info["build_id"] == _build_id(info)
    assert re.fullmatch(r"[0-9a-f]{64}", info["cargo_lock_sha256"])
    expected_platform = os.environ.get("EXPECTED_PLATFORM_SUFFIX")
    if expected_platform:
        target = info["target_triple"]
        if "linux" in expected_platform:
            assert "linux" in target
            assert "x86_64" in target
        elif "macos" in expected_platform:
            assert "apple-darwin" in target
            assert "aarch64" in target
        elif "win" in expected_platform:
            assert "windows" in target
            assert "x86_64" in target


def test_installed_private_diagnostics_do_not_expand_public_api() -> None:
    assert {"_core", "_runtime", "runtime_info", "runtime_info_schema"}.isdisjoint(
        voiage.__all__
    )


def test_installed_native_serializers_have_exact_payload_lineage() -> None:
    """Exercise both Rust-owned serializers from the installed artifact."""
    ceaf_before = dict(native.runtime_info()["operations"]["serialize_ceaf_result"])
    dominance_before = dict(
        native.runtime_info()["operations"]["serialize_dominance_result"]
    )
    ceaf = CEAFResult(
        wtp_thresholds=np.array([0.0]),
        optimal_strategy_indices=np.array([0]),
        optimal_strategy_names=["A"],
        acceptability_probabilities=np.array([1.0]),
        probability_lower=np.array([1.0]),
        probability_upper=np.array([1.0]),
        expected_net_benefit=np.array([1e20]),
        reporting={"standard": "CHEERS 2022"},
    ).to_dict(analysis_id="wheel-test", decision_problem_id="wheel-test")
    dominance = DominanceResult(
        strategy_names=["A", "B"],
        costs=np.array([1e20, 2e20]),
        effects=np.array([1.0, 2.0]),
        frontier_indices=[0, 1],
        strongly_dominated_indices=[],
        extended_dominated_indices=[],
        status=["frontier", "frontier"],
        incremental_costs=np.array([1.0]),
        incremental_effects=np.array([1.0]),
        icers=np.array([1.0]),
        reporting={"standard": "CHEERS 2022"},
    ).to_dict(analysis_id="wheel-test", decision_problem_id="wheel-test")

    assert ceaf == {
        "analysis_id": "wheel-test",
        "decision_problem_id": "wheel-test",
        "analysis_type": "ceaf",
        "wtp_thresholds": [0.0],
        "optimal_strategy_indices": [0],
        "optimal_strategy_names": ["A"],
        "acceptability_probabilities": [1.0],
        "probability_lower": [1.0],
        "probability_upper": [1.0],
        "expected_net_benefit": [1e20],
        "reporting": {"standard": "CHEERS 2022"},
    }
    assert dominance == {
        "analysis_id": "wheel-test",
        "decision_problem_id": "wheel-test",
        "analysis_type": "dominance",
        "strategy_names": ["A", "B"],
        "costs": [1e20, 2e20],
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
    assert isinstance(ceaf["expected_net_benefit"][0], float)
    assert all(isinstance(value, float) for value in dominance["costs"])

    ceaf_after = native.runtime_info()["operations"]["serialize_ceaf_result"]
    dominance_after = native.runtime_info()["operations"]["serialize_dominance_result"]
    for before, after, payload in (
        (ceaf_before, ceaf_after, ceaf),
        (dominance_before, dominance_after, dominance),
    ):
        assert after["calls"] == before["calls"] + 1
        assert after["native_entries"] == before["native_entries"] + 1
        assert after["successes"] == before["successes"] + 1
        assert after["failures"] == before["failures"]
        assert after["last_payload_sha256"] == _digest(payload)
