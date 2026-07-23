"""Contracts for the reproducible VOI/VOP software and method census."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import subprocess
import sys

from jsonschema import Draft202012Validator

ROOT = Path(__file__).parents[1]
LANDSCAPE = ROOT / "specs" / "software-landscape"
REGISTRY = LANDSCAPE / "registry.json"
SCHEMA = LANDSCAPE / "schema.json"
METHODS = LANDSCAPE / "methods.json"


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def test_software_landscape_validates_against_schema() -> None:
    """The checked-in census must obey its versioned public contract."""
    registry = _read_json(REGISTRY)
    schema = _read_json(SCHEMA)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(registry)


def test_required_ecosystems_and_seed_tools_are_present() -> None:
    """The initial census must span every requested discovery channel."""
    registry = _read_json(REGISTRY)
    assert isinstance(registry, dict)
    tools = registry["tools"]
    ecosystems = {tool["ecosystem"] for tool in tools}
    ids = {tool["id"] for tool in tools}

    assert {
        "r",
        "python",
        "rust",
        "julia",
        "mojo",
        "web",
        "commercial",
    } <= ecosystems
    assert {
        "r-voi",
        "r-bcea",
        "r-dampack",
        "savi",
        "analytica",
        "pyro-oed",
        "botorch",
    } <= ids
    assert len(ids) == len(tools)


def test_method_and_feature_references_are_complete() -> None:
    """Every observed feature must map to a canonical method or disposition."""
    registry = _read_json(REGISTRY)
    methods = _read_json(METHODS)
    assert isinstance(registry, dict)
    assert isinstance(methods, dict)
    method_ids = {method["id"] for method in methods["methods"]}
    assert len(method_ids) == len(methods["methods"])

    for tool in registry["tools"]:
        assert tool["features"], tool["id"]
        for feature in tool["features"]:
            assert set(feature["method_ids"]) <= method_ids
            assert feature["parity_state"] in {
                "native",
                "equivalent",
                "adapter",
                "planned",
                "excluded",
                "not-reproducible",
            }
            assert feature["evidence"]


def test_method_taxonomy_covers_core_vop_and_ml_families() -> None:
    """Core decision VOI, perspective, and information design stay distinct."""
    methods = _read_json(METHODS)
    assert isinstance(methods, dict)
    by_id = {method["id"]: method for method in methods["methods"]}

    assert {
        "evpi",
        "evppi",
        "evsi",
        "enbs",
        "directional-evop",
        "perspective-frontier",
        "perspective-sample-information",
        "expected-information-gain",
        "bayesian-oed",
        "active-learning",
        "knowledge-gradient",
        "llm-routing-voi",
        "rag-acquisition-voi",
        "agent-information-voi",
    } <= by_id.keys()
    for method in by_id.values():
        assert method["class"] in {
            "estimand",
            "estimator",
            "workflow",
            "visualization",
            "related-analysis",
            "application",
        }
        assert method["moscow"] in {"must", "should", "could", "wont"}
        assert method["voiage_state"] in {
            "native",
            "equivalent",
            "planned",
            "excluded",
        }
        assert method["maturity"] in {
            "stable",
            "experimental",
            "planned",
        }

    assert by_id["expected-information-gain"]["class"] == "estimand"
    assert by_id["llm-routing-voi"]["class"] == "application"


def test_search_snapshot_is_bounded_and_refreshable() -> None:
    """The snapshot records queries and expires before becoming silent lore."""
    registry = _read_json(REGISTRY)
    assert isinstance(registry, dict)
    searched_on = date.fromisoformat(registry["searched_on"])
    review_due = date.fromisoformat(registry["review_due"])

    assert searched_on < review_due
    assert (review_due - searched_on).days <= 93
    assert len(registry["searches"]) >= 9
    for search in registry["searches"]:
        assert search["channel"]
        assert search["query"]
        assert search["source_url"].startswith("https://")
        assert search["result"] in {"candidates-found", "no-direct-package-found"}


def test_generated_feature_matrix_is_current() -> None:
    """The human matrix must be a deterministic projection of the registries."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_voi_feature_matrix.py"),
            "--check",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
