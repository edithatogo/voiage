# SPDX-License-Identifier: Apache-2.0
"""Profile macOS process memory during the frozen ML00 workflow."""

from __future__ import annotations

import argparse
from importlib import import_module, util
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

EVIDENCE = Path(__file__).resolve().parent


def _snapshot(phase: str) -> dict[str, str | int | bool]:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = subprocess.check_output(  # noqa: S603
        [
            "/usr/bin/footprint",
            "--noCategories",
            "--format",
            "bytes",
            "--pid",
            str(os.getpid()),
        ],
        text=True,
        timeout=10,
    )
    footprint = re.search(r"(?m)^\s*phys_footprint:\s+(\d+) B$", report)
    if footprint is None:
        message = "macOS footprint did not report physical footprint in bytes"
        raise ValueError(message)
    return {
        "phase": phase,
        "peak_rss_bytes": peak_rss,
        "physical_footprint_bytes": int(footprint[1]),
        "torch_imported": "torch" in sys.modules,
    }


def main() -> None:
    """Measure the selected workflow in a fresh process and verify its value."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture")
    parser.add_argument("backend", choices=("rust", "mps", "torch-import"))
    arguments = parser.parse_args()
    if sys.platform != "darwin" or sys.version_info[:3] != (3, 15, 2):
        parser.error("the profile requires macOS and Python 3.15.2")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("set PYTORCH_ENABLE_MPS_FALLBACK=0")

    spec = util.spec_from_file_location("ml00", EVIDENCE / "ml00.py")
    assert spec is not None
    assert spec.loader is not None
    ml00 = util.module_from_spec(spec)
    spec.loader.exec_module(ml00)
    baseline = json.loads((EVIDENCE / "results.json").read_text())
    case = next(
        (
            item
            for item in baseline["cases"]
            if item["fixture"]["id"] == arguments.fixture
        ),
        None,
    )
    if case is None:
        parser.error(f"unknown ML00 fixture: {arguments.fixture}")
    values = ml00.fixture(case["fixture"])
    phases = [_snapshot("fixture")]
    import_module("voiage.methods.basic")

    phases.append(_snapshot("public-api-import"))
    if arguments.backend == "torch-import":
        import_module("torch")

        phases.append(_snapshot("torch-import"))
        value, route = None, None
    else:
        import_module("voiage.analysis")

        phases.append(_snapshot("analysis-import"))

        def observe(function: Callable[..., Any], phase: str) -> Callable[..., Any]:
            def measured(*args: Any) -> Any:
                result = function(*args)
                phases.append(_snapshot(phase))
                return result

            return measured

        if arguments.backend == "mps":
            ml00.prepare = observe(ml00.prepare, "prepare")
            ml00.upload = observe(ml00.upload, "upload")
            ml00.reduce_device = observe(ml00.reduce_device, "reduction")
        value, route = ml00.workflow(values, arguments.backend)
        phases.append(_snapshot("workflow"))
        assert route == case["cold"][arguments.backend]["route"]
        assert ml00.matches(value, case["value"])

    print(
        json.dumps(
            {
                "fixture": arguments.fixture,
                "backend": arguments.backend,
                "value": value,
                "route": route,
                "phases": phases,
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
