"""Capability boundaries for experimental expected-utility information pricing."""

from __future__ import annotations

import json
from pathlib import Path
import re

from typer.main import get_command

from voiage import cli
from voiage.methods.utility_information import expected_utility_information_value

ROOT = Path(__file__).parents[1]
CAPABILITY_PATH = (
    ROOT
    / "specs/frontier/expected-utility-information-pricing/v1/capabilities.json"
)


def _capability() -> dict[str, object]:
    return json.loads(CAPABILITY_PATH.read_text(encoding="utf-8"))


def test_expected_utility_capability_dispositions_are_exact() -> None:
    capability = _capability()

    assert capability == {
        "schema_version": "expected-utility-information-capabilities-v1",
        "method_family": "expected_utility_information_pricing",
        "method_maturity": "experimental",
        "execution_authority": "rust",
        "fixture_manifest": (
            "specs/frontier/expected-utility-information-pricing/v1/fixtures/"
            "manifest.json"
        ),
        "python_modules": ["voiage/methods/utility_information.py"],
        "bindings": [
            {
                "language": "rust",
                "status": "executable",
                "surface": "voiage_numerics::expected_utility_information",
                "evidence": (
                    "rust/crates/voiage-numerics/tests/"
                    "expected_utility_information.rs"
                ),
                "fixture_evidence": True,
            },
            {
                "language": "python",
                "status": "executable",
                "surface": (
                    "voiage.methods.utility_information."
                    "expected_utility_information_value"
                ),
                "adapter": "pyo3",
                "evidence": "tests/test_expected_utility_information.py",
                "fixture_evidence": True,
            },
            {
                "language": "r",
                "status": "unsupported",
                "reason": (
                    "No expected-utility information-pricing C ABI or R facade "
                    "is implemented."
                ),
            },
            {
                "language": "julia",
                "status": "unsupported",
                "reason": (
                    "No expected-utility information-pricing C ABI or Julia "
                    "facade is implemented."
                ),
            },
            {
                "language": "mojo",
                "status": "external_boundary",
                "reason": (
                    "The repository has no local Mojo binding; any future surface "
                    "depends on upstream Rust interop."
                ),
            },
        ],
    }


def test_declared_executable_surfaces_and_evidence_exist() -> None:
    capability = _capability()
    bindings = {
        entry["language"]: entry for entry in capability["bindings"]  # type: ignore[index]
    }

    rust_source = (
        ROOT / "rust/crates/voiage-numerics/src/utility_information.rs"
    ).read_text(encoding="utf-8")
    rust_evidence = (ROOT / bindings["rust"]["evidence"]).read_text(encoding="utf-8")
    assert "pub fn expected_utility_information(" in rust_source
    assert callable(expected_utility_information_value)
    assert (
        expected_utility_information_value.__module__
        == "voiage.methods.utility_information"
    )
    assert all(
        (ROOT / entry["evidence"]).is_file()
        for entry in (bindings["rust"], bindings["python"])
    )
    assert "committed_normative_fixtures_drive_rust_conformance" in rust_evidence
    assert "affine-clairvoyant.json" in rust_evidence
    assert "log-buy-sell-asymmetry.json" in rust_evidence
    assert (ROOT / capability["fixture_manifest"]).is_file()  # type: ignore[index]
    assert all(
        entry["fixture_evidence"] is True
        for entry in (bindings["rust"], bindings["python"])
    )
    assert all(
        "fixture_evidence" not in entry
        for entry in (bindings["r"], bindings["julia"], bindings["mojo"])
    )


def test_unsupported_bindings_do_not_extend_the_stable_abi() -> None:
    stable_binding_sources = (
        ROOT / "rust/crates/voiage-ffi/include/voiage_v1.h",
        ROOT / "rust/crates/voiage-ffi/src/lib.rs",
        ROOT / "specs/abi/v1/symbols.txt",
        ROOT / "r-package/voiageR/NAMESPACE",
        ROOT / "r-package/voiageR/R/voiageR.R",
        ROOT / "bindings/julia/src/Voiage.jl",
    )
    forbidden_surface = re.compile(
        r"expected[_ -]?utility|clairvoyance|value[_ -]?of[_ -]?clairvoyance|\bvoc\b",
        re.IGNORECASE,
    )

    for path in stable_binding_sources:
        assert not forbidden_surface.search(path.read_text(encoding="utf-8")), path
    assert not (ROOT / "bindings/mojo").exists()


def test_voc_is_a_presentation_not_a_duplicate_kernel_or_cli() -> None:
    rust_kernel = (
        ROOT / "rust/crates/voiage-numerics/src/utility_information.rs"
    ).read_text(encoding="utf-8")
    python_adapter = (ROOT / "voiage/_runtime.py").read_text(encoding="utf-8")
    cli_commands = set(get_command(cli.app).commands)

    assert rust_kernel.count("pub fn expected_utility_information(") == 1
    assert not re.search(r"pub fn (?:value_of_clairvoyance|voc)\s*\(", rust_kernel)
    assert "compute_expected_utility_information" in python_adapter
    assert "compute_value_of_clairvoyance" not in python_adapter
    assert "compute_voc" not in python_adapter
    assert "calculate-expected-utility-information" in cli_commands
    assert "calculate-voc" not in cli_commands
