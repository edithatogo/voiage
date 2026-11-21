# SPDX-License-Identifier: Apache-2.0
"""Run the predeclared ML00 feasibility experiment against an installed wheel."""

from __future__ import annotations

from datetime import UTC, datetime
from fractions import Fraction
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable

Array = NDArray[Any]
Stats = tuple[int, int, int, int]
RTOL = 1e-10
ATOL = 1e-12
U = 2.0**-53
PROTOCOL = Path(__file__).with_name("protocol.json")


def fixture(case: dict[str, Any]) -> Array:
    """Generate one registered workload with its declared layout."""
    rows, cols = case["rows"], case["strategies"]
    rng = np.random.Generator(np.random.PCG64(20261121))
    kind = case["kind"]
    if kind == "foundational":
        return ((np.arange(rows)[:, None] + 17 * np.arange(cols)) % 101).astype(
            np.float64
        )
    if kind in {"integers", "offset"}:
        values = rng.integers(-10_000, 18_000, size=(rows, cols)).astype(np.float64)
        return values + 2**20 if kind == "offset" else values
    if kind == "strided":
        return rng.standard_normal((rows, cols * 2))[:, ::2]
    return rng.standard_normal((rows, cols))


def independent_value(a: Array, *, exact: bool = False) -> float:
    """Calculate EVPI by mean opportunity loss, using exact arithmetic for witnesses."""
    if exact:
        values = [tuple(map(Fraction.from_float, map(float, row))) for row in a]
        maxima = [max(row) for row in values]
        return float(
            min(
                sum(best - item for best, item in zip(maxima, col, strict=True))
                for col in zip(*values, strict=True)
            )
            / len(values)
        )
    maxima = np.max(a, axis=1)
    return min(math.fsum(map(float, maxima - col)) for col in a.T) / len(a)


def matches(first: float, second: float) -> bool:
    """Apply the frozen G05 tolerances."""
    return math.isclose(first, second, rel_tol=RTOL, abs_tol=ATOL)


def prepare(a: Array) -> tuple[Array, int] | None:
    """Choose a power-of-two scale and constrain all int64 partial sums."""
    maximum = max(abs(float(np.min(a))), abs(float(np.max(a))))
    if maximum > 1_000_000 or len(a) > (1 << 32) - 1:
        return None
    limit = (1 << 61) // len(a)
    exponent = (
        min(44, math.floor(math.log2(limit) - math.log2(maximum)) - 1)
        if maximum
        else 44
    )
    if exponent < 0:
        return None
    scaled = np.ldexp(a, exponent)
    np.rint(scaled, out=scaled)
    integers = scaled.astype(np.int64)
    if max(abs(int(integers.min())), abs(int(integers.max()))) > limit:
        return None
    return integers, 1 << exponent


def certify(stats: Stats, n: int, scale: int) -> tuple[float, float, bool]:
    """Bound the error between the candidate, ideal EVPI, and the Rust reduction."""
    best, current, abs_best, abs_current = stats
    value = (best - current) / (n * scale)
    quantization = 1 / scale
    magnitude = (abs_best + abs_current) / (n * scale) + quantization
    gamma = (n + 1) * U / (1 - (n + 1) * U)
    # Allow for division and summation rounding, subnormals, and the last
    # subtraction and division. Round the combined bound upward.
    bound = quantization + gamma * magnitude + 2 * n * 2.0**-1022
    bound = math.nextafter(bound * (1 + 32 * U) + 4 * U * abs(value), math.inf)
    tolerance = max(ATOL, RTOL * max(0.0, abs(value) - bound))
    return value, bound, value >= 0 and bound <= tolerance


def upload(values: Array, backend: str) -> Any:
    """Stage the quantized array and synchronize on its device."""
    if backend == "mlx":
        import mlx.core as mx

        with mx.stream(mx.gpu):
            a = mx.array(values, dtype=mx.int64)
            mx.eval(a)
    else:
        import torch

        a = torch.as_tensor(values, device="mps", dtype=torch.int64)
        torch.mps.synchronize()
    return a


def reduce_device(a: Any, backend: str) -> Stats:
    """Return exact integer sums and magnitude witnesses from Metal."""
    if backend == "mlx":
        import mlx.core as mx

        with mx.stream(mx.gpu):
            top = mx.max(a, axis=1)
            result = (
                mx.sum(top),
                mx.max(mx.sum(a, axis=0)),
                mx.sum(mx.abs(top)),
                mx.max(mx.sum(mx.abs(a), axis=0)),
            )
            mx.eval(*result)
    else:
        import torch

        top = torch.amax(a, dim=1)
        result = (
            torch.sum(top),
            torch.amax(torch.sum(a, dim=0)),
            torch.sum(torch.abs(top)),
            torch.amax(torch.sum(torch.abs(a), dim=0)),
        )
        torch.mps.synchronize()
        assert all(
            item.device.type == "mps" and item.dtype == torch.int64 for item in result
        )
    return tuple(int(item.item()) for item in result)  # type: ignore[return-value]


def candidate(a: Array, backend: str) -> tuple[float | None, float | None, str]:
    """Evaluate eligibility and certify a candidate value."""
    prepared = prepare(a)
    if prepared is None:
        return None, None, "input_range"
    q, scale = prepared
    value, bound, admitted = certify(
        reduce_device(upload(q, backend), backend), len(a), scale
    )
    return (value, bound, "gpu") if admitted else (None, bound, "error_bound")


def workflow(a: Array, backend: str) -> tuple[float, str]:
    """Measure public Rust EVPI or the corresponding experimental workflow."""
    from voiage.methods.basic import evpi

    if backend == "rust" or a.ndim != 2 or len(a) < 16384 or a.shape[1] < 2:
        return evpi(a), "rust" if backend == "rust" else "size_or_shape"
    from voiage.analysis import DecisionAnalysis
    from voiage.core.utils import check_input_array

    # Exercise the installed API's normalization and analysis construction.
    if not np.all(np.isfinite(np.asarray(a, dtype=float))):
        return evpi(a), "nonfinite"
    analysis = DecisionAnalysis(nb_array=a)
    values = analysis.nb_array.numpy_values
    check_input_array(values, expected_ndim=2, name="nb_array", allow_empty=True)
    value, _, route = candidate(values, backend)
    return (value, route) if value is not None else (analysis.evpi(), route)


def _timed[T](function: Callable[..., T], *args: Any) -> tuple[int, T]:
    start = perf_counter_ns()
    value = function(*args)
    return perf_counter_ns() - start, value


def cold_case(case: dict[str, Any], backend: str) -> dict[str, Any]:
    """Measure the first call and memory in a fresh process."""
    a = fixture(case)
    elapsed, (value, route) = _timed(workflow, a, backend)
    memory = None
    if route == "gpu" and backend == "mlx":
        import mlx.core as mx

        memory = int(mx.get_peak_memory())
    if route == "gpu" and backend == "mps":
        import torch

        memory = int(torch.mps.driver_allocated_memory())
    return {
        "first_call_ns": elapsed,
        "value": value,
        "route": route,
        "workflow_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "device_bytes": memory,
    }


def cold_kernel(case: dict[str, Any], backend: str) -> dict[str, Any]:
    """Separate preparation, upload and first reduction in a fresh process."""
    a = fixture(case)
    setup_ns, prepared = _timed(prepare, a)
    assert prepared is not None
    upload_ns, gpu = _timed(upload, prepared[0], backend)
    first_ns, stats = _timed(reduce_device, gpu, backend)
    value, _, admitted = certify(stats, a.shape[0], prepared[1])
    assert admitted
    return {
        "value": value,
        "setup_ns": setup_ns,
        "upload_ns": upload_ns,
        "cold_ns": first_ns,
    }


def python_for(run: Path, lane: str) -> Path:
    """Select the isolated interpreter for a cold timing lane."""
    env = {"rust": ".cpu-venv", "mlx": ".mlx-venv", "mps": ".venv"}[lane]
    return run / env / "bin" / "python"


def validation(backends: list[str]) -> dict[str, Any]:
    """Run the untimed analytical and input-contract witnesses."""
    from voiage.analysis import DecisionAnalysis
    from voiage.main_backends import AppleMetalBackend
    from voiage.methods.basic import evpi

    first = np.array([[10.0, 12.0], [11.0, 9.0], [13.0, 14.0]])
    g05 = np.array([[1.0e300, 1.1e300], [1.2e300, 1.05e300]])
    largest = 0.75 * sys.float_info.max
    valid = {
        "analytic_2_over_3": first,
        "shifted_1e9": first + 1e9,
        "g05_affine_base": g05,
        "g05_affine_shift": np.array(
            [
                [1.000_000_000_1e300, 1.100_000_000_1e300],
                [1.200_000_000_1e300, 1.050_000_000_1e300],
            ]
        ),
        "g05_extreme": np.array([[largest, 0.0], [0.5 * largest, largest]]),
        "tie": np.array([[2.0, 2.0], [3.0, 3.0]]),
        "near_tie": np.array([[1.0, 1.0 + 2**-38], [1.0 + 2**-38, 1.0]]),
        "tiny": first * 2**-45,
        "float32_fortran_order": np.asfortranarray(first, dtype=np.float32),
        "permuted": first[::-1, ::-1],
    }
    outcomes = {}
    mps = AppleMetalBackend()
    for key, a in valid.items():
        analytic = independent_value(a, exact=True)
        rust = evpi(a)
        gpu = {}
        for backend in backends:
            value, bound, reason = candidate(a.astype(np.float64), backend)
            if value is not None:
                assert matches(value, analytic), (key, backend, value, analytic)
                assert matches(value, rust), (key, backend, value, rust)
            gpu[backend] = {"value": value, "bound": bound, "route": reason}
        baseline: dict[str, Any] = {"status": "outside_float32_range"}
        if np.max(np.abs(a)) < 2**100:
            try:
                baseline = {"status": "returned", "value": mps.evpi(a)}
            except (ValueError, RuntimeError) as error:
                baseline = {"status": "rejected", "error": str(error)}
        outcomes[key] = {
            "analytic": analytic,
            "rust": rust,
            "candidates": gpu,
            "existing_mps_float32": baseline,
        }

    import mlx.core as mx

    with mx.stream(mx.gpu):
        shifted = valid["shifted_1e9"]
        demo = {}
        for name, a in [("naive", shifted), ("host_centered", shifted - 1e9)]:
            values = mx.array(a, dtype=mx.float32)
            result = mx.mean(mx.max(values, axis=1)) - mx.max(mx.mean(values, axis=0))
            mx.eval(result)
            demo[name] = result.item()

    def plain_result(a: Array, name: str) -> tuple[str, float | str]:
        try:
            return "value", workflow(a, name)[0]
        except (TypeError, ValueError) as error:
            return "error", type(error).__name__

    edges = {
        "empty": np.empty((0, 2)),
        "single": first[:, :1],
        "invalid_shape": np.array([1.0, 2.0]),
        "nan": np.full((16384, 2), np.nan),
        "inf": np.array([[0.0, np.inf]]),
    }
    edge_results = {}
    for key, a in edges.items():
        reference = plain_result(a, "rust")
        assert all(plain_result(a, backend) == reference for backend in backends), key
        edge_results[key] = reference
    scaling_inputs = (100.0, 3.0, 0.05)
    scaled_expected = 2 / 3 * 100 * sum(1 / 1.05**t for t in range(1, 4))
    scaled_reference = evpi(first, *scaling_inputs)
    assert matches(scaled_expected, scaled_reference)
    scaled = {}
    analysis = DecisionAnalysis(nb_array=first)
    for backend in backends:
        score = outcomes["analytic_2_over_3"]["candidates"][backend]["value"]
        assert score is not None
        scaled[backend] = analysis._scale_to_population(score, *scaling_inputs, "EVPI")
        assert matches(scaled[backend], scaled_reference)
    return {
        "witnesses": outcomes,
        "edge_cases": edge_results,
        "mlx_float32": demo,
        "population_scaling": {
            "analytic": scaled_expected,
            "rust": scaled_reference,
            **scaled,
        },
    }


def _read_json_process(args: list[str | Path]) -> dict[str, Any]:
    # The executable is the experiment's local driver or isolated interpreter.
    return json.loads(subprocess.check_output(args, text=True))  # noqa: S603


def measure_case(
    case: dict[str, Any], backends: list[str], run: Path, settings: dict[str, Any]
) -> dict[str, Any]:
    """Collect raw kernel and paired workflow timings for one fixture."""
    a = fixture(case)
    filename = run / "inputs" / (case["id"] + ".f64le")
    np.ascontiguousarray(a, dtype="<f8").tofile(filename)
    warmups, count = settings["warmups"], settings["repetitions"]
    kernel_args = [run / "rust_evpi", filename, str(a.shape[0]), str(a.shape[1])]
    rust_kernel = _read_json_process([*kernel_args, str(warmups), str(count)])
    independent = independent_value(a)
    assert matches(rust_kernel["value"], independent), (
        case["id"],
        rust_kernel["value"],
        independent,
    )

    kernels = {"rust": rust_kernel}
    lanes = ["rust", *backends]
    results, warm, cold = {}, {lane: [] for lane in lanes}, {}
    for lane in lanes:
        elapsed, value = _timed(
            _read_json_process,
            [
                python_for(run, lane),
                "-I",
                "-B",
                __file__,
                "cold",
                json.dumps(case),
                lane,
            ],
        )
        cold[lane] = {"process_ns": elapsed, **value}
    for lane in backends:
        value, bound, reason = candidate(a, lane)
        results[lane] = {
            "bound": bound,
            "candidate_value": value,
            "numerical_route": reason,
        }
        if value is not None and a.shape[0] >= 16384:
            assert matches(value, independent), case["id"]
            assert matches(value, rust_kernel["value"]), case["id"]
            frozen_cold = _read_json_process(
                [
                    python_for(run, lane),
                    "-I",
                    "-B",
                    __file__,
                    "cold-kernel",
                    json.dumps(case),
                    lane,
                ]
            )
            assert frozen_cold["value"] == value
            prepared = prepare(a)
            assert prepared is not None
            gpu = upload(prepared[0], lane)
            stats = reduce_device(gpu, lane)
            assert certify(stats, a.shape[0], prepared[1])[0] == value
            for _ in range(warmups):
                reduce_device(gpu, lane)
            kernels[lane] = {
                **frozen_cold,
                "warm_ns": [_timed(reduce_device, gpu, lane)[0] for _ in range(count)],
            }
    for repetition in range(warmups + count):
        order = lanes if repetition % 2 == 0 else list(reversed(lanes))
        for lane in order:
            elapsed, (value, route) = _timed(workflow, a, lane)
            assert matches(value, independent), (case["id"], lane, value, independent)
            assert route == cold[lane]["route"]
            assert matches(value, cold[lane]["value"])
            if repetition >= warmups:
                warm[lane].append(elapsed)
    with filename.open("rb") as input_file:
        fixture_hash = hashlib.file_digest(input_file, "sha256").hexdigest()
    return {
        "fixture": case,
        "strides": a.strides,
        "value": independent,
        "sha256": fixture_hash,
        "kernel": kernels,
        "cold": cold,
        "workflow_warm_ns": warm,
        "candidates": results,
    }


def p95(samples: list[int]) -> int:
    """Return the preregistered nearest-rank 95th percentile."""
    return sorted(samples)[math.ceil(0.95 * len(samples)) - 1]


def evaluate(
    cases: list[dict[str, Any]], lane: str, limits: dict[str, Any]
) -> dict[str, Any]:
    """Apply the preregistered acceptance gates to one candidate."""
    warm = [case["workflow_warm_ns"] for case in cases]
    mixed = {
        name: statistics.median(
            map(sum, zip(*(runs[name] for runs in warm), strict=True))
        )
        for name in ("rust", lane)
    }
    eligible = sum(case["cold"][lane]["route"] == "gpu" for case in cases) / len(cases)
    regressions = [
        case["fixture"]["id"]
        for case in cases
        if p95(case["workflow_warm_ns"][lane])
        > p95(case["workflow_warm_ns"]["rust"])
        * (1 + limits["maximum_per_fixture_p95_slowdown_fraction"])
        + limits["per_fixture_p95_allowance_ms"] * 1e6
    ]
    rss = (
        max(
            case["cold"][lane]["workflow_rss_bytes"]
            - case["cold"]["rust"]["workflow_rss_bytes"]
            for case in cases
        )
        / 2**20
    )
    device = max(case["cold"][lane]["device_bytes"] or 0 for case in cases) / 2**20
    gates = {
        "eligible_share": eligible >= limits["minimum_eligible_fraction"],
        "mixed_median_speedup": mixed["rust"] / mixed[lane]
        >= limits["minimum_warm_mixed_median_speedup"],
        "p95": not regressions,
        "rss": rss <= limits["maximum_incremental_peak_workflow_rss_mib"],
        "device": device <= limits["maximum_peak_device_allocation_mib"],
    }
    decision = (
        ("blocked" if lane == "mps" else "go") if all(gates.values()) else "no-go"
    )
    return {
        "decision": decision,
        "gates": gates,
        "eligible_fraction": eligible,
        "warm_mixed_speedup": mixed["rust"] / mixed[lane],
        "warm_mixed_median_ns": mixed,
        "p95_regression_fixtures": regressions,
        "peak_extra_rss_mib": rss,
        "device_mib": device,
        "device_memory_measure": "allocator_peak"
        if lane == "mlx"
        else "retained_driver_allocation",
    }


def main() -> None:
    """Run an untimed witness check or the frozen benchmark."""
    if sys.argv[1] == "cold":
        print(
            json.dumps(cold_case(json.loads(sys.argv[2]), sys.argv[3]), allow_nan=False)
        )
        return
    if sys.argv[1] == "cold-kernel":
        print(
            json.dumps(
                cold_kernel(json.loads(sys.argv[2]), sys.argv[3]), allow_nan=False
            )
        )
        return
    run = Path(sys.argv[1]).resolve()
    protocol = json.loads(PROTOCOL.read_text())
    assert sys.version_info[:3] == (3, 15, 2)
    assert platform.machine() == "arm64"
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
    mps = json.loads((run / "mps-device-probe.json").read_text())
    backends = ["mlx", "mps"] if mps["int64_exact"] else ["mlx"]
    witnesses = validation(backends)
    if sys.argv[2] == "check":
        print(json.dumps(witnesses, indent=2, allow_nan=False))
        return
    started_at = datetime.now(UTC).isoformat()
    load_before = os.getloadavg()
    print("Numerical witnesses passed. Running registered fixtures.", flush=True)
    cases = []
    for case in protocol["fixtures"]:
        cases.append(measure_case(case, backends, run, protocol["timing"]))
        print(
            case["id"],
            {lane: cases[-1]["cold"][lane]["route"] for lane in backends},
            flush=True,
        )
    result = {
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "timing_started_utc": started_at,
        "timing_completed_utc": datetime.now(UTC).isoformat(),
        "load_before": load_before,
        "load_after": os.getloadavg(),
        "host": platform.platform(),
        "python": sys.version,
        "versions": {
            name: metadata.version(name)
            for name in ("mlx", "mlx-metal", "torch", "numpy", "voiage")
        },
        "mlx_device": json.loads((run / "mlx-device-probe.json").read_text()),
        "mps_device": mps,
        "validation": witnesses,
        "cases": cases,
        "outcomes": {
            lane: evaluate(cases, lane, protocol["go_thresholds"]) for lane in backends
        },
    }
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
