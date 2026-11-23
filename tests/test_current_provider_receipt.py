"""Keep source-only provider evidence bound to the declarations it tested."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT / "conductor/tracks/remaining_backlog_delivery_20260831/"
    "provider-declarations-20261124-spack-make.json"
)
INPUTS = {
    "scripts/check_current_arrow_providers.py",
    "scripts/check_current_build_providers.py",
    "scripts/check_current_spack_providers.py",
    "packaging/current/arrow-provider-sources.json",
    "packaging/current/runtime-build-provider-sources.json",
    "packaging/current/polars-provider-sources.json",
    "packaging/current/spack/packages/arrow/package.py",
    "packaging/current/spack/packages/py-libcst/package.py",
    "packaging/current/spack/packages/py-pyarrow/package.py",
    "packaging/current/spack/packages/py-cython/package.py",
    "packaging/current/spack/packages/py-pydantic-core/package.py",
    "packaging/current/spack/packages/py-polars/package.py",
    "packaging/current/spack/packages/py-polars-runtime-32/package.py",
}


def test_provider_receipt_authenticates_current_inputs_and_source_identities() -> None:
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["python"] == "3.15.2"
    assert set(receipt["input_files_sha256"]) == INPUTS
    for name, expected in receipt["input_files_sha256"].items():
        data = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == expected
    expected_sources = {}
    for name in (
        "arrow-provider-sources.json",
        "runtime-build-provider-sources.json",
        "polars-provider-sources.json",
    ):
        packet = json.loads((ROOT / "packaging/current" / name).read_text())
        for provider in packet["providers"].values():
            source = provider["source"]
            expected_sources[source["filename"]] = source["sha256"]
        if name == "arrow-provider-sources.json":
            arrow = packet["providers"]["arrow"]
            expected_sources["spack-builtin-arrow-package.py"] = arrow[
                "builtin_recipe_sha256"
            ]
            for resource_name in ("mimalloc", "simdjson"):
                source = arrow[resource_name]
                filename = f"{resource_name}-{source['version']}.tar.gz"
                expected_sources[filename] = source["sha256"]
            uriparser = arrow["uriparser"]
            expected_sources[f"uriparser-{uriparser['version']}.tar.bz2"] = uriparser[
                "sha256"
            ]
    observed_sources = {
        row["filename"]: row["sha256"] for row in receipt["source_artifacts"]
    }
    assert len(observed_sources) == len(receipt["source_artifacts"])
    assert observed_sources == expected_sources


def test_provider_receipt_does_not_claim_native_acceptance() -> None:
    reports = json.loads(RECEIPT.read_text())["reports"]
    assert all(
        report["source_declarations_qualified"] is True for report in reports.values()
    )
    assert reports["arrow"]["full_native_solver_executed"] is False
    assert reports["arrow"]["native_build_executed"] is False
    assert reports["arrow"]["installed_module_acceptance"] is False
    assert reports["arrow"]["profiles"][1]["allocator_archive_verified"] is False
    assert reports["build"]["full_hpc_matrix_qualified"] is False
    assert reports["polars"]["native_solver_executed"] is False
    assert reports["polars"]["native_build_executed"] is False
    assert reports["polars"]["installed_module_acceptance"] is False
