"""Tests for accelerator guidance and contract wording."""

from pathlib import Path


def test_accelerator_guidance_mentions_fpga_and_asic_lanes() -> None:
    """The accelerator guidance should keep FPGA and ASIC lanes explicit."""
    root = Path.cwd()
    accelerator_text = (
        root / "docs" / "developer_guide" / "rust_accelerators.rst"
    ).read_text()
    performance_text = (
        root / "docs" / "user_guide" / "performance_guide.md"
    ).read_text()
    roadmap_text = (
        root / "docs" / "developer_guide" / "hpc_native_roadmap.rst"
    ).read_text()
    contract_text = (
        root / "docs" / "developer_guide" / "hpc_acceleration_abstraction_contract.rst"
    ).read_text()

    assert "FPGA implementation lane" in accelerator_text
    assert "ASIC implementation lane" in accelerator_text
    assert "FPGA: separate execution lane" in performance_text
    assert "ASIC: contract-gated custom-circuit lane" in performance_text
    assert "FPGA implementation" in roadmap_text
    assert "ASIC / custom-circuit implementation" in roadmap_text
    assert "ASIC/custom-circuit feasibility" in contract_text
