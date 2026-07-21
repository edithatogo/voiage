"""Tests for accelerator guidance and contract wording."""

from pathlib import Path


def test_accelerator_guidance_mentions_fpga_and_asic_lanes() -> None:
    """The accelerator guidance should keep FPGA and ASIC lanes explicit."""
    root = Path.cwd()
    docs = root / "docs" / "astro-site" / "src" / "content" / "docs"
    accelerator_text = (docs / "developer-guide" / "rust-accelerators.mdx").read_text()
    performance_text = (docs / "user-guide" / "performance-guide.mdx").read_text()
    roadmap_text = (docs / "developer-guide" / "hpc-native-roadmap.mdx").read_text()
    contract_text = (
        docs / "developer-guide" / "hpc-acceleration-abstraction-contract.mdx"
    ).read_text()

    assert "FPGA implementation lane" in accelerator_text
    assert "ASIC implementation lane" in accelerator_text
    assert "FPGA: separate execution lane" in performance_text
    assert "ASIC: contract-gated custom-circuit lane" in performance_text
    assert "FPGA implementation" in roadmap_text
    assert "ASIC / custom-circuit implementation" in roadmap_text
    assert "ASIC/custom-circuit feasibility" in contract_text
