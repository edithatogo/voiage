from pathlib import Path


def test_hpc_native_roadmap_sequence_and_baselines() -> None:
    root = Path.cwd()
    roadmap_text = (root / "roadmap.md").read_text()
    guide_text = (root / "docs" / "developer_guide" / "hpc_native_roadmap.rst").read_text()
    contract_text = (root / "docs" / "developer_guide" / "hpc_distribution_contract.rst").read_text()
    accelerator_text = (root / "docs" / "developer_guide" / "rust_accelerators.rst").read_text()
    notes_text = (root / "conductor" / "tracks" / "apple-metal-integrated-gpu-optimization_20260511" / "working-notes.md").read_text()

    assert (
        "Phase 13: HPC Native Enablement Roadmap ✅/🔄 **SETUP COMPLETE, SPEEDUP EVIDENCE-GATED**"
        in roadmap_text
    )
    assert "Phase 14: HPC Capability Implementation Program ✅ **SETUP COMPLETE**" in roadmap_text
    assert "Apple Integrated GPU Optimization" in roadmap_text
    assert "Discrete GPU Acceleration" in roadmap_text
    assert "TPU Feasibility" in roadmap_text
    assert "ASIC / Custom-Circuit Feasibility" in roadmap_text

    assert "scalar_cpu_baseline" in guide_text
    assert "memory_throughput_baseline" in guide_text
    assert "Apple integrated GPUs are the first accelerator target" in guide_text
    assert "Apple deployment requirements" in guide_text
    assert "macOS Apple Silicon hosts for build and validation" in guide_text
    assert "CPU fallback coverage preserved in CI" in guide_text
    assert "Benchmark status" in guide_text
    assert "The current Apple Metal track has established the committed CPU baselines" in guide_text

    assert "CPU-first portability matters more than speculative accelerator work." in contract_text
    assert "starts with Apple integrated-GPU optimization through Metal-backed execution" in contract_text

    assert "Apple Metal adapter strategy" in accelerator_text
    assert "CPU fallback remains authoritative" in accelerator_text
    assert "should not require public API changes" in accelerator_text

    assert "Metal Path Draft" in notes_text
    assert "target the committed scalar EVPI baseline and memory/throughput baseline" in notes_text
    assert "keep the CPU fallback authoritative for every workload in scope" in notes_text
    assert "Apple Deployment Requirements" in notes_text
    assert "macOS Apple Silicon hosts for build and validation" in notes_text
    assert "Metal-capable system libraries available at runtime" in notes_text
    assert "Benchmark Status" in notes_text
    assert "device-backed code path to time or compare against the baseline" in notes_text
