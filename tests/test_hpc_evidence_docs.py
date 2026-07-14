from __future__ import annotations

import json
from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_hpc_docs_reference_colab_gpu_tpu_evidence_without_speedup_claim() -> None:
    docs = "\n".join(
        _read(path)
        for path in (
            "docs/developer_guide/hpc_native_roadmap.rst",
            "docs/developer_guide/hpc_acceleration_abstraction_contract.rst",
            "docs/developer_guide/hpc_distribution_contract.rst",
            "docs/developer_guide/rust_accelerators.rst",
        )
    )
    normalized_docs = " ".join(docs.split())

    for needle in (
        "colab_gpu_accelerator_evidence.json",
        "colab_tpu_accelerator_evidence.json",
        "colab_accelerator_evidence_manifest.json",
        'jax_devices == ["cuda:0"]',
        'jax_platforms == ["gpu"]',
        'jax_platforms == ["tpu"]',
        "cpu_evpi == jax_evpi == 1.25",
        "do not prove production speedup",
    ):
        assert needle in docs
    assert (
        "not sufficient by themselves to claim HPC-native acceleration"
        in normalized_docs
    )
    assert "future GPU/TPU/FPGA/ASIC decisions" in docs
    assert "Colab GPU/TPU visibility and parity evidence" in docs
    assert "not timing, warm-up, or throughput review packets" in docs


def test_roadmap_tracks_completed_setup_separately_from_speedup_gates() -> None:
    roadmap = _read("roadmap.md")

    for needle in (
        "Setup Complete, Speedup Evidence-Gated",
        "READINESS COMPLETE, LIVE CHECKS REFRESHABLE",
        "compact Colab v5e runtime validation has passed",
        "Completion decision: the umbrella setup program is complete and archived.",
    ):
        assert needle in roadmap

    for stale in (
        "active implementation now runs through the `hpc-capability-implementation-program_20260511`",
        "no TPU implementation until contract-safe",
        "Phase 14: HPC Capability Implementation Program 📋 **IN PROGRESS**",
    ):
        assert stale not in roadmap


def test_conductor_feasibility_records_reference_colab_evidence() -> None:
    records = "\n".join(
        _read(path)
        for path in (
            "conductor/archive/discrete-gpu-acceleration_20260511/working-notes.md",
            "conductor/archive/discrete-gpu-acceleration_20260511/handoff/feasibility_decision.json",
            "conductor/archive/tpu-acceleration-feasibility_20260511/working-notes.md",
            "conductor/archive/tpu-acceleration-feasibility_20260511/handoff/feasibility_decision.json",
            "conductor/archive/asic-acceleration-feasibility_20260511/working-notes.md",
            "conductor/archive/asic-acceleration-feasibility_20260511/handoff/feasibility_decision.json",
        )
    )

    for needle in (
        "Colab T4 evidence",
        "colab_gpu_accelerator_evidence.json",
        "Colab v5e evidence",
        "colab_tpu_accelerator_evidence.json",
        "production-sized workload",
        "production-scale profile",
    ):
        assert needle in records


def test_conductor_tracking_statuses_match_completed_spec_tracks() -> None:
    for path in (
        Path("conductor/archive/core-api-spec-foundation/metadata.json"),
        Path("conductor/archive/canonical-schemas-core-contracts/metadata.json"),
    ):
        metadata = json.loads(path.read_text(encoding="utf-8"))

        assert metadata["status"] == "completed"
        assert metadata["completed_at"] == "2026-04-18T00:00:00Z"


def test_asic_feasibility_does_not_contradict_pre_silicon_track_completion() -> None:
    tracks = _read("conductor/tracks.md")
    notes = _read(
        "conductor/archive/asic-acceleration-feasibility_20260511/working-notes.md"
    )
    normalized_notes = " ".join(notes.split())

    assert "ASIC Implementation" in tracks
    assert (
        "free CI pre-silicon evidence path is implemented; Tiny Tapeout, "
        "SkyWater MPW, and fabricated-silicon runtime remain future external gates"
        in tracks
    )
    assert (
        "The separate ASIC implementation track has completed the free CI "
        "pre-silicon evidence scope" in normalized_notes
    )
    assert "remains open only for explicit" not in tracks
    assert "remains open only for explicit" not in notes
    assert "no ASIC implementation track is open" not in tracks
    assert "No implementation track will be opened" not in notes


def test_registry_tracking_keeps_external_gates_outside_completed_tracks() -> None:
    plan = _read(
        "conductor/archive/binding-registry-live-verification_20260511/plan.md"
    )
    spack = _read(
        "conductor/archive/spack-registry-readiness_20260511/working-notes.md"
    )
    easybuild = _read(
        "conductor/archive/easybuild-registry-readiness_20260511/working-notes.md"
    )

    assert "- [x] Keep external conda-forge/CRAN/Julia" in plan
    assert "Protocol in workflow.md" in plan
    assert "workflow in workflow.md" not in plan
    assert "External maintainer gate" in spack
    assert "this completed readiness track" in spack
    assert "External maintainer gate" in easybuild
    assert "this completed readiness" in easybuild
    assert "External blocker" not in spack
    assert "External blocker" not in easybuild
