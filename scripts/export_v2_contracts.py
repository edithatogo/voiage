#!/usr/bin/env python3
"""Deterministically export the canonical VOIAGE v2 contract artifacts."""

# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Protocol, cast

import numpy as np

from voiage.contracts.adapters import analysis_spec_from_inputs
from voiage.contracts.analysis import (
    AnalysisResult,
    DiagnosticEnvelope,
    NumericalPolicy,
    Provenance,
    RunContext,
)
from voiage.contracts.perspective import (
    PerspectivePayload,
    adapt_perspective_result,
)
from voiage.methods.perspective import METHOD_CONTRACT_VERSION, value_of_perspective
from voiage.schema import ParameterSet, ValueArray

SCHEMA_ROOT = Path("specs/core-api/schemas/v2")
EXAMPLE_ROOT = Path("specs/core-api/examples/v2")


def _json(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


class _Arguments(Protocol):
    root: Path
    write: bool
    check: bool


def _example_spec():
    values = ValueArray.from_numpy_perspectives(
        np.array(
            [
                [[10.0, 0.0], [0.0, 20.0]],
                [[10.0, 0.0], [0.0, 20.0]],
            ]
        ),
        strategy_names=["A", "B"],
        perspective_names=["payer", "societal"],
    )
    parameters = ParameterSet.from_numpy_or_dict({"prevalence": np.array([0.1, 0.2])})
    policy = NumericalPolicy(
        deterministic_fixture_mode=True,
        backend_preference=("numpy",),
    )
    spec = analysis_spec_from_inputs(
        analysis_id="perspective-v2-example",
        decision_problem_id="screening-program-001",
        method_family="value_of_perspective",
        method_contract_version=METHOD_CONTRACT_VERSION,
        values=values,
        parameters=parameters,
        numerical_policy=policy,
    )
    return spec, values


def contract_artifacts() -> dict[str, str]:
    """Return every generated artifact as canonical UTF-8 text."""
    spec, values = _example_spec()
    legacy = value_of_perspective(values)
    context = RunContext(
        run_id="contract-fixture-run",
        spec_digest=spec.contract_digest(),
        input_digest="0" * 64,
        requested_backend="numpy",
        selected_backend="numpy",
        backend_version="fixture",
        device="cpu",
        capabilities=frozenset({"dense-array", "deterministic"}),
        package_version="0.0.0",
        python_version="3.14",
        platform="contract-fixture",
    )
    result = AnalysisResult[PerspectivePayload](
        analysis_id=spec.analysis_id,
        decision_problem_id=spec.decision_problem_id,
        method_family=spec.method_family,
        method_contract_version=spec.method_contract_version,
        method_maturity="fixture-backed",
        numerical_policy=spec.numerical_policy,
        payload=adapt_perspective_result(legacy),
        run_context=context,
        diagnostics=DiagnosticEnvelope(
            analysis_id=spec.analysis_id,
            backend="numpy",
        ),
        provenance=Provenance(
            backend="numpy",
            method_family=spec.method_family,
            package_version="0.0.0",
            fixture_id="perspective-v2-example",
            details={"adapter": "legacy-perspective-result"},
        ),
    )
    return {
        (SCHEMA_ROOT / "analysis-spec.schema.json").as_posix(): _json(
            type(spec).model_json_schema()
        ),
        (SCHEMA_ROOT / "perspective-result.schema.json").as_posix(): _json(
            AnalysisResult[PerspectivePayload].model_json_schema()
        ),
        (EXAMPLE_ROOT / "analysis-spec.example.json").as_posix(): _json(
            spec.model_dump(mode="json")
        ),
        (EXAMPLE_ROOT / "perspective-result.example.json").as_posix(): _json(
            result.model_dump(mode="json")
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Write generated artifacts or verify the committed copies."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group(required=True)
    _ = mode.add_argument("--write", action="store_true")
    _ = mode.add_argument("--check", action="store_true")
    args = cast("_Arguments", cast("object", parser.parse_args(argv)))
    root = args.root.resolve()
    findings: list[str] = []
    for relative_path, rendered in contract_artifacts().items():
        path = root / relative_path
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_text(rendered, encoding="utf-8", newline="\n")
        elif not path.is_file() or path.read_text(encoding="utf-8") != rendered:
            findings.append(relative_path)
    if findings:
        print("stale v2 contract artifacts: " + ", ".join(findings), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
