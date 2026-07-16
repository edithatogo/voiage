from __future__ import annotations

import json
from pathlib import Path
import re

TRACK_IDS = (
    "voi-frontier-architecture-dependency-governance_20260625",
    "conductor-commit-note-checkpoint-hardening_20260625",
    "strict-ci-cd-quality-gates_20260625",
    "dataset-registry-and-example-corpus_20260625",
    "rust-frontier-numerics-migration-completion_20260625",
    "bayesian-experimental-design-and-amortized-voi_20260625",
    "external-registry-publication-program_20260625",
    "conda-forge-feedstock-publication_20260625",
    "r-cran-runiverse-publication_20260625",
    "julia-general-registry-publication_20260625",
    "spack-package-merge-followthrough_20260625",
    "easybuild-easyconfig-merge-followthrough_20260625",
    "hpsf-curation-submission-followthrough_20260625",
    "e4s-inclusion-followthrough_20260625",
    "frontier-stable-promotion-program_20260625",
    "perspective-voi-stable-promotion_20260625",
    "preference-voi-stable-promotion_20260625",
    "validation-threshold-voi-stable-promotion_20260625",
    "distributional-implementation-voi-stable-promotion_20260625",
    "adjacent-frontier-runtime-completion_20260625",
    "hpc-production-speedup-evidence-program_20260625",
    "cpu-cluster-production-benchmark-evidence_20260625",
    "apple-metal-production-speedup-evidence_20260625",
    "discrete-gpu-production-speedup-evidence_20260625",
    "tpu-production-scale-colab-evidence_20260625",
    "accelerator-evidence-automation_20260625",
    "fpga-physical-board-runtime-evidence_20260625",
    "asic-mpw-shuttle-and-silicon-evidence_20260625",
    "custom-circuit-production-acceleration-review_20260625",
    "causal-identification-transportability-voi-mature-stable_20260625",
    "data-quality-measurement-privacy-linkage-voi-mature-stable_20260625",
    "computational-model-refinement-voi-mature-stable_20260625",
    "expert-elicitation-evidence-synthesis-voi-mature-stable_20260625",
    "dynamic-real-options-voi-mature-stable_20260625",
    "perspective-uncertainty-voi-mature-stable_20260625",
    "monitoring-surveillance-voi-mature-stable_20260625",
    "implementation-strategy-comparison-voi-mature-stable_20260625",
    "equity-information-voi-mature-stable_20260625",
    "explainability-transparency-voi-mature-stable_20260625",
    "interoperability-standardization-voi-mature-stable_20260625",
    "ambiguity-distribution-shift-voi-mature-stable_20260625",
    "adaptive-learning-bandit-voi-mature-stable_20260625",
    "capacity-budget-constrained-voi-mature-stable_20260625",
    "federated-privacy-preserving-voi-mature-stable_20260625",
    "ai-assisted-evidence-triage-voi-mature-stable_20260625",
    "regulatory-market-access-voi-mature-stable_20260625",
    "replication-reproducibility-voi-mature-stable_20260625",
    "evidence-obsolescence-refresh-voi-mature-stable_20260625",
    "strategic-behavior-game-theoretic-voi-mature-stable_20260625",
)

CROSS_CUTTING_TRACKS = TRACK_IDS[:6]
REGISTRY_TRACKS = TRACK_IDS[6:14]
FRONTIER_TRACKS = TRACK_IDS[14:20]
HPC_TRACKS = TRACK_IDS[20:26]
CUSTOM_CIRCUIT_TRACKS = TRACK_IDS[26:29]
METHOD_MATURE_STABLE_TRACKS = TRACK_IDS[29:]


def _read(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _resolve_track_root(track_id: str) -> Path:
    """Resolve a track's root directory, checking active tracks first
    then falling back to the archive directory."""
    active = Path("conductor/tracks") / track_id
    if active.is_dir():
        return active
    archived = Path("conductor/archive") / track_id
    if archived.is_dir():
        return archived
    msg = f"Track directory not found for {track_id} in conductor/tracks/ or conductor/archive/"
    raise FileNotFoundError(msg)


def _track_roots(track_id: str) -> tuple[Path, bool]:
    """Returns (root_path, is_archived) for a track."""
    active = Path("conductor/tracks") / track_id
    if active.is_dir():
        return active, False
    archived = Path("conductor/archive") / track_id
    if archived.is_dir():
        return archived, True
    msg = f"Track directory not found for {track_id} in conductor/tracks/ or conductor/archive/"
    raise FileNotFoundError(msg)


def _track_text(track_id: str) -> str:
    track_root, _ = _track_roots(track_id)
    return "\n".join(
        (
            _read(track_root / "spec.md"),
            _read(track_root / "plan.md"),
            _read(track_root / "metadata.json"),
        )
    )


def test_registry_exactly_matches_active_track_directories() -> None:
    """Every active directory is registered and every active link resolves."""
    registry = _read("conductor/tracks.md")
    active_ids = {
        path.name for path in Path("conductor/tracks").iterdir() if path.is_dir()
    }
    registered_ids = set(re.findall(r"\./tracks/([^/]+)/", registry))

    assert registered_ids == active_ids
    assert active_ids.isdisjoint(
        path.name for path in Path("conductor/archive").iterdir() if path.is_dir()
    )


def test_followthrough_tracks_have_required_conductor_artifacts() -> None:
    """Each follow-through track should be registered or archived and complete."""
    registry = _read("conductor/tracks.md")

    for track_id in TRACK_IDS:
        track_root, is_archived = _track_roots(track_id)
        metadata = json.loads(_read(track_root / "metadata.json"))
        spec = _read(track_root / "spec.md")
        plan = _read(track_root / "plan.md")
        index = _read(track_root / "index.md")

        assert track_root.is_dir()
        assert metadata["track_id"] == track_id
        if not is_archived:
            assert metadata["status"] in {"new", "in_progress", "blocked_external"}
        assert metadata["type"] == "feature"
        assert "# Track Specification:" in spec
        assert "# Track Implementation Plan:" in plan
        assert "[Specification](./spec.md)" in index
        if is_archived:
            assert f"./tracks/{track_id}/" not in registry
        elif f"./tracks/{track_id}/" in registry:
            assert f"*Link: [./tracks/{track_id}/]" in registry


def test_registry_encodes_execution_and_external_blocked_queues() -> None:
    """Only genuine work is numbered for automatic dependency-order execution."""
    registry = _read("conductor/tracks.md")
    executable = re.findall(
        r"## \[[ ~]\] Track: .*?\r?\n"
        r"\*Link: \[\./tracks/([^/]+)/\].*?\r?\n"
        r"\*Execution order: (\d{2}) of 32\*",
        registry,
    )
    assert len(executable) == 24
    assert [order for _, order in executable] == [
        f"{index:02d}" for index in range(9, 33)
    ]

    blocked = re.findall(
        r"## \[!\] Track: .*?\r?\n"
        r"\*Link: \[\./tracks/([^/]+)/\].*?\r?\n"
        r"\*Status: blocked_external — ([^*]+)\.\*",
        registry,
    )
    assert len(blocked) == 10
    assert all(gate.strip() for _, gate in blocked)

    assert "./archive/adjacent-frontier-runtime-completion_20260625/" in registry
    assert "./archive/perspective-voi-stable-promotion_20260625/" in registry


def test_followthrough_plans_encode_commit_notes_and_phase_checkpoints() -> None:
    """New tracks should follow the stricter Conductor checkpoint workflow."""
    for track_id in TRACK_IDS:
        track_root, _ = _track_roots(track_id)
        plan = _read(track_root / "plan.md")
        text = _track_text(track_id)

        assert "git note" in plan
        assert "short commit SHA" in plan
        assert "commit the plan update" in plan
        assert "GitHub Actions" in text
        assert "Conductor - User Manual Verification" in plan
        assert plan.count("Protocol in workflow.md") >= 3


def test_cross_cutting_tracks_cover_architecture_ci_datasets_rust_and_sota() -> None:
    """The earlier roadmap-expansion tracks should also be active."""
    combined = "\n".join(_track_text(track_id) for track_id in CROSS_CUTTING_TRACKS)

    for needle in (
        "maturity taxonomy",
        "commit notes",
        "mutation",
        "synthetic datasets",
        "Rust numerics core",
        "expected information gain",
        "amortized EVSI",
    ):
        assert needle in combined


def test_registry_followthrough_tracks_distinguish_live_states() -> None:
    """Registry tracks must distinguish readiness from external publication."""
    required_states = (
        "readiness",
        "submitted",
        "published",
        "indexed",
        "approved",
        "blocked",
        "not-found",
    )

    registry_docs = _read("docs/release/binding-submission-checklist.md")
    assert all(state in registry_docs for state in required_states)

    for track_id in REGISTRY_TRACKS:
        text = _track_text(track_id)
        assert "external" in text.lower()
        assert "GitHub Actions" in text
        assert "gh" in text
        assert "approval" in text.lower() or "maintainer" in text.lower()
        assert "blocked" in text.lower()


def test_frontier_followthrough_tracks_gate_stable_promotion() -> None:
    """Frontier methods should require parity before stable labels move."""
    frontier_docs = _read("docs/sota_voi_frontier.md")
    assert "cross-language parity" in frontier_docs
    assert "Rust-kernel parity" in frontier_docs

    for track_id in FRONTIER_TRACKS:
        text = _track_text(track_id)
        assert "stable" in text.lower()
        assert "cross-language" in text.lower()
        assert "fixture" in text.lower()
        assert "Rust" in text
        assert "changelog" in text.lower() or "release-note" in text.lower()


def test_hpc_followthrough_tracks_require_production_speedup_packets() -> None:
    """HPC tracks should require benchmark packets, not visibility alone."""
    hpc_docs = _read("docs/developer_guide/hpc_native_roadmap.rst")
    assert "hpc-production-speedup-evidence-program_20260625" in hpc_docs
    assert "tpu-production-scale-colab-evidence_20260625" in hpc_docs
    assert "production-sized benchmark packets" in hpc_docs

    for track_id in HPC_TRACKS:
        text = _track_text(track_id)
        assert "benchmark" in text.lower()
        assert "warm-up" in text.lower()
        assert "throughput" in text.lower()
        assert "CPU fallback" in text
        assert "speedup" in text.lower()


def test_fpga_asic_followthrough_tracks_preserve_external_gates() -> None:
    """Physical board and silicon work should remain external-gated."""
    for track_id in CUSTOM_CIRCUIT_TRACKS:
        text = _track_text(track_id)
        assert "external gate" in text.lower()
        assert "production" in text.lower()
        assert "CPU" in text
        assert "benchmark" in text.lower()

    fpga = _track_text("fpga-physical-board-runtime-evidence_20260625")
    asic = _track_text("asic-mpw-shuttle-and-silicon-evidence_20260625")
    review = _track_text("custom-circuit-production-acceleration-review_20260625")

    assert "physical board" in fpga
    assert "bitstream" in fpga
    assert "Tiny Tapeout" in asic
    assert "SkyWater MPW" in asic
    assert "fabricated silicon" in asic
    assert "go/no-go" in review


def test_conductor_setup_records_strict_followthrough_policy() -> None:
    """Conductor setup should enforce >90% and evidence-bound follow-through."""
    workflow = _read("conductor/workflow.md")
    tech_stack = _read("conductor/tech-stack.md")
    roadmap = _read("roadmap.md")

    assert ">90%" in workflow
    assert ">80%" not in workflow
    assert "Follow-through registry, frontier-promotion, HPC-speedup" in workflow
    assert "Colab CLI (`colab`)" in tech_stack
    assert "Google Cloud CLI (`gcloud`)" in tech_stack
    assert "Follow-Through Expansion (created June 25, 2026)" in roadmap


def test_dedicated_method_tracks_cover_runtime_to_stable_paths() -> None:
    """Each requested frontier method should have its own mature/stable path."""
    expected_keywords = {
        "causal-identification-transportability-voi-mature-stable_20260625": (
            "causal-identification",
            "transportability",
            "external-validity",
        ),
        "data-quality-measurement-privacy-linkage-voi-mature-stable_20260625": (
            "data-quality",
            "measurement-error",
            "privacy",
            "linkage",
        ),
        "computational-model-refinement-voi-mature-stable_20260625": (
            "computational VOI",
            "model refinement",
            "multi-fidelity",
        ),
        "expert-elicitation-evidence-synthesis-voi-mature-stable_20260625": (
            "expert-elicitation",
            "evidence-synthesis",
            "calibration",
        ),
        "dynamic-real-options-voi-mature-stable_20260625": (
            "dynamic real-options",
            "irreversibility",
            "policy lock-in",
        ),
        "perspective-uncertainty-voi-mature-stable_20260625": (
            "value of perspective",
            "perspective uncertainty",
            "stakeholder weights",
        ),
    }

    for track_id, needles in expected_keywords.items():
        text = _track_text(track_id)
        assert "mature/stable" in text
        assert "runtime implementation" in text
        assert "Cross-language conformance" in text
        assert "Rust parity" in text
        assert "release notes" in text
        for needle in needles:
            assert needle in text


def test_recommended_method_tracks_are_recorded() -> None:
    """Recommended VOI extensions should be represented as active tracks."""
    expected_tracks = (
        "monitoring-surveillance-voi-mature-stable_20260625",
        "implementation-strategy-comparison-voi-mature-stable_20260625",
        "equity-information-voi-mature-stable_20260625",
        "explainability-transparency-voi-mature-stable_20260625",
        "interoperability-standardization-voi-mature-stable_20260625",
        "ambiguity-distribution-shift-voi-mature-stable_20260625",
        "adaptive-learning-bandit-voi-mature-stable_20260625",
        "capacity-budget-constrained-voi-mature-stable_20260625",
        "federated-privacy-preserving-voi-mature-stable_20260625",
        "ai-assisted-evidence-triage-voi-mature-stable_20260625",
    )
    for track_id in expected_tracks:
        text = _track_text(track_id)
        assert "mature/stable" in text
        assert "stable promotion" in text
        assert "CLI" in text
        assert "property" in text


def test_extended_recommended_method_tracks_are_recorded() -> None:
    """Additional recommended VOI extensions should be explicit tracks."""
    expected_keywords = {
        "adaptive-learning-bandit-voi-mature-stable_20260625": (
            "adaptive learning",
            "bandit VOI",
            "sequential allocation",
        ),
        "capacity-budget-constrained-voi-mature-stable_20260625": (
            "capacity-constrained",
            "budget-constrained",
            "resource constraints",
        ),
        "federated-privacy-preserving-voi-mature-stable_20260625": (
            "federated",
            "privacy-preserving",
            "secure aggregation",
        ),
        "ai-assisted-evidence-triage-voi-mature-stable_20260625": (
            "AI-assisted",
            "evidence triage",
            "human-in-the-loop",
        ),
        "regulatory-market-access-voi-mature-stable_20260625": (
            "regulatory",
            "market-access",
            "approval probability",
        ),
        "replication-reproducibility-voi-mature-stable_20260625": (
            "replication",
            "reproducibility",
            "reanalysis",
        ),
        "evidence-obsolescence-refresh-voi-mature-stable_20260625": (
            "evidence obsolescence",
            "refresh VOI",
            "living evidence",
        ),
        "strategic-behavior-game-theoretic-voi-mature-stable_20260625": (
            "strategic behavior",
            "game-theoretic VOI",
            "equilibrium",
        ),
    }
    for track_id, needles in expected_keywords.items():
        text = _track_text(track_id)
        assert "mature/stable" in text
        assert "stable promotion" in text
        assert "Cross-language conformance" in text
        assert "Rust parity" in text
        for needle in needles:
            assert needle in text
