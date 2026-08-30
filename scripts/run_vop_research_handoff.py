#!/usr/bin/env python3
"""Replay the pinned VOP scenario across two supported Python environments.

Export in VOP's locked environment; evaluate the hash-bound CSV with the public
voiage wheel in a separate environment. This agent-assisted computational replay
does not establish independent adoption or new human research use.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

VOP_REVISION = "2c46db2fe5f907d894bb07f1127c008f10ee462e"
PARAMETER_SHA256 = "f1a5c0c6463aca131ac94fcc3f5a8fda1e97752dd2ebb60b0b2c4047d7657608"
EXPORT_SCHEMA = "voiage.vop-research-export.v1"
CSV_NAME = "hpv_vaccination_net_benefit.csv"
HEADERS = ["standard_care", "hpv_vaccination"]
WTP = 50_000.0


def sha256(data: bytes) -> str:
    """Hash the exact bytes exchanged between environments."""
    return hashlib.sha256(data).hexdigest()


def write_receipt(path: Path, record: dict[str, Any]) -> None:
    """Create evidence without replacing any earlier run."""
    with path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def environment() -> dict[str, Any]:
    """Record installed versions without importing the other environment."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "prefix": sys.prefix,
        "distributions": {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
        },
    }


def generate_draws(
    params: dict[str, Any],
    run_cea: Callable[..., dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> NDArray[np.float64]:
    """Retain the original scenario's sampling order and perturbations."""
    if type(draws) is not int or draws < 2:
        raise ValueError("draws must be an integer of at least two")
    rng = np.random.default_rng(seed)
    net_benefit = np.empty((draws, 2), dtype=float)
    for index in range(draws):
        draw = copy.deepcopy(params)
        for option in ("new_treatment", "standard_care"):
            draw["costs"]["health_system"][option] = [
                float(value * rng.lognormal(mean=0.0, sigma=0.10))
                for value in draw["costs"]["health_system"][option]
            ]
        for option in ("new_treatment", "standard_care"):
            draw["qalys"][option] = [
                float(value * rng.lognormal(mean=0.0, sigma=0.03))
                for value in draw["qalys"][option]
            ]
        result = run_cea(draw, perspective="health_system")
        net_benefit[index] = (
            WTP * result["qalys_standard_care"] - result["cost_standard_care"],
            WTP * result["qalys_new_treatment"] - result["cost_new_treatment"],
        )
    if not np.isfinite(net_benefit).all():
        raise ValueError("VOP produced non-finite net benefits")
    return net_benefit


def export(vop_root: Path, output: Path, *, draws: int, seed: int) -> None:
    """Run only VOP and export the pinned scenario's sampled net benefits."""
    root = vop_root.resolve()
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to verify the VOP checkout")
    # Fixed read-only argument arrays; never interpreted by a shell.
    revision = subprocess.check_output(  # noqa: S603
        [git, "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    dirty = subprocess.check_output(  # noqa: S603
        [git, "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
        text=True,
        timeout=30,
    ).strip()
    parameter_path = root / "src/vop_poc_nz/parameters.yaml"
    if revision != VOP_REVISION or dirty:
        raise ValueError("VOP checkout must be clean at the pinned revision")
    if sha256(parameter_path.read_bytes()) != PARAMETER_SHA256:
        raise ValueError("VOP parameter SHA-256 mismatch")

    # The explicit, verified source root is the only VOP source used. There is
    # deliberately no voiage import in this process.
    sys.path.insert(0, str(root / "src"))
    from vop_poc_nz.cea_model_core import run_cea
    from vop_poc_nz.pipeline.analysis import load_parameters

    matrix = generate_draws(
        load_parameters(str(parameter_path))["hpv_vaccination"],
        run_cea,
        draws=draws,
        seed=seed,
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / CSV_NAME).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADERS)
        writer.writerows(matrix.tolist())
    write_receipt(
        output / "export.json",
        {
            "schema_version": EXPORT_SCHEMA,
            "source_repository": "https://github.com/edithatogo/vop_poc_nz",
            "source_revision": revision,
            "source_parameter_sha256": PARAMETER_SHA256,
            "draws": draws,
            "seed": seed,
            "willingness_to_pay_nzd_per_qaly": WTP,
            "net_benefit_csv_sha256": sha256((output / CSV_NAME).read_bytes()),
            "environment": environment(),
        },
    )


def load_handoff(
    receipt: Path, expected_sha256: str
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Check the receipt and exact CSV bytes before numerical evaluation."""
    receipt_bytes = receipt.read_bytes()
    if sha256(receipt_bytes) != expected_sha256:
        raise ValueError("export receipt SHA-256 mismatch")
    record = json.loads(receipt_bytes)
    if not isinstance(record, dict) or (
        record.get("schema_version") != EXPORT_SCHEMA
        or record.get("source_revision") != VOP_REVISION
        or record.get("source_parameter_sha256") != PARAMETER_SHA256
        or type(record.get("draws")) is not int
        or record["draws"] < 2
        or type(record.get("seed")) is not int
        or record["seed"] < 0
        or record.get("willingness_to_pay_nzd_per_qaly") != WTP
    ):
        raise ValueError("invalid export contract")
    csv_bytes = (receipt.parent / CSV_NAME).read_bytes()
    if sha256(csv_bytes) != record.get("net_benefit_csv_sha256"):
        raise ValueError("net-benefit CSV SHA-256 mismatch")
    rows = list(csv.reader(io.StringIO(csv_bytes.decode("utf-8"))))
    if not rows or rows[0] != HEADERS or any(len(row) != 2 for row in rows[1:]):
        raise ValueError("CSV must have the exact two strategy columns")
    matrix = np.array(rows[1:], dtype=float)
    if matrix.shape != (record["draws"], 2) or not np.isfinite(matrix).all():
        raise ValueError("CSV must contain the declared finite draw matrix")
    return matrix, record


def evaluate(receipt: Path, expected_sha256: str, output: Path) -> None:
    """Evaluate only voiage and compare EVPI with its mathematical definition."""
    matrix, record = load_handoff(receipt, expected_sha256)
    import voiage
    from voiage.methods.basic import evpi

    module_path = Path(voiage.__file__).resolve()
    if voiage.__version__ != "2.2.0" or not module_path.is_relative_to(
        Path(sys.prefix).resolve()
    ):
        raise ValueError(
            "evaluation requires the installed public voiage 2.2.0 environment"
        )
    value = float(evpi(matrix))
    reference = float(np.mean(np.max(matrix, axis=1)) - np.max(np.mean(matrix, axis=0)))
    if not np.isfinite(value) or not np.isclose(
        value, reference, rtol=1e-12, atol=1e-8
    ):
        raise ValueError("voiage EVPI disagrees with the independent NumPy calculation")
    write_receipt(
        output,
        {
            "schema_version": "voiage.vop-research-evaluation.v1",
            "export_receipt_sha256": expected_sha256,
            "net_benefit_csv_sha256": record["net_benefit_csv_sha256"],
            "voiage_version": voiage.__version__,
            "voiage_module": str(module_path),
            "evpi_nzd_per_cohort": value,
            "numpy_reference_evpi_nzd_per_cohort": reference,
            "environment": environment(),
            "scope": "Agent-assisted replay of the historical same-author workflow; not new human research use, independent adoption, in-process integration, clinical advice, or a policy estimate.",
        },
    )


def main() -> None:
    """Select one side of the explicit data handoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser("export")
    exporter.add_argument("--vop-root", type=Path, required=True)
    exporter.add_argument("--output", type=Path, required=True)
    exporter.add_argument("--draws", type=int, default=500)
    exporter.add_argument("--seed", type=int, default=20260727)
    consumer = commands.add_parser("evaluate")
    consumer.add_argument("--export-receipt", type=Path, required=True)
    consumer.add_argument("--export-sha256", required=True)
    consumer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export(args.vop_root, args.output, draws=args.draws, seed=args.seed)
    else:
        evaluate(args.export_receipt, args.export_sha256, args.output)


if __name__ == "__main__":
    main()
