from __future__ import annotations

import json
from pathlib import Path


EVIDENCE_ROOT = Path(
    "conductor/tracks/hpc-acceleration-abstraction-contract_20260511/handoff"
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_colab_gpu_and_tpu_evidence_payloads_are_persisted() -> None:
    gpu = _load_json(EVIDENCE_ROOT / "colab_gpu_accelerator_evidence.json")
    tpu = _load_json(EVIDENCE_ROOT / "colab_tpu_accelerator_evidence.json")

    assert gpu["jax_platforms"] == ["gpu"]
    assert gpu["jax_devices"] == ["cuda:0"]
    assert gpu["has_gpu"] is True
    assert gpu["has_tpu"] is False

    assert tpu["jax_platforms"] == ["tpu"]
    assert tpu["jax_devices"] == ["TPU_0(process=0,(0,0,0,0))"]
    assert tpu["has_gpu"] is False
    assert tpu["has_tpu"] is True

    for payload in (gpu, tpu):
        assert payload["cpu_evpi"] == payload["jax_evpi"] == 1.25
        assert payload["fpga_is_placeholder"] is True
        assert payload["asic_is_placeholder"] is True
        assert payload["available_execution_adapters"] == [
            "process",
            "thread",
            "dask",
            "ray",
            "fpga",
            "asic",
        ]


def test_colab_accelerator_evidence_manifest_links_payloads() -> None:
    manifest = _load_json(EVIDENCE_ROOT / "colab_accelerator_evidence_manifest.json")
    runtime_evidence = manifest["runtime_evidence"]

    assert manifest["notebook"] == "examples/colab_accelerator_validation.ipynb"
    assert manifest["contract_checks"] == {
        "evpi_parity": "passed",
        "gpu_visibility": "passed",
        "tpu_visibility": "passed",
        "fpga_placeholder_visibility": "passed",
        "asic_placeholder_visibility": "passed",
    }
    assert isinstance(runtime_evidence, dict)
    assert runtime_evidence["gpu"]["status"] == "passed"
    assert runtime_evidence["tpu"]["status"] == "passed"

    for runtime in ("gpu", "tpu"):
        artifact_path = Path(str(runtime_evidence[runtime]["artifact_path"]))
        assert artifact_path.exists()
        assert artifact_path.parent == EVIDENCE_ROOT

        payload = _load_json(artifact_path)
        expected_platform = "gpu" if runtime == "gpu" else "tpu"
        assert payload["jax_platforms"] == [expected_platform]
