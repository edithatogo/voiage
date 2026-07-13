# Working Notes: Apple Metal Backend Prototype

## Scope Control for Closure

Phase 3 closes only when the review artifact proves both workloads are
measurable, reproducible, and ready for handoff to
`apple-metal-integrated-gpu-optimization_20260511` with no scope drift.

## Closure Deliverables

1. `conductor/archive/apple-metal-backend-prototype_20260510/handoff/phase_3_handoff_bundle.json`
   - Contains both workloads (`scalar_cpu_baseline`, `memory_throughput_baseline`).
   - Contains at least two independent runs per workload.
   - Captures review metadata (`review.phase == "phase_3"`) and payload schema
     (`payload_version == "1.0.0"`).
2. `conductor/archive/apple-metal-backend-prototype_20260510/handoff/phase_3_cpu_reference.json`
   - CPU-only reproducibility packet for environments without MPS.
3. `conductor/archive/apple-metal-backend-prototype_20260510/handoff/phase_3_runtime_freeze.txt`
   - Exact `python -m pip freeze` output captured at handoff time.
4. One handoff note to the integrated optimization track that references:
   - workload ordering recommendation,
   - known bottlenecks,
   - follow-up optimization questions.

## Execution Commands (Exact)

### Required CPU-only packet (no Metal hardware needed)

```bash
mkdir -p conductor/archive/apple-metal-backend-prototype_20260510/handoff && \
python3 - <<'PY'
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from voiage.main_backends import (
    NumpyBackend,
    benchmark_evpi,
    benchmark_memory_throughput,
    benchmark_mps_vs_cpu,
)

ARTIFACT_DIR = Path("conductor/archive/apple-metal-backend-prototype_20260510/handoff")
NB = np.array([[10.0, 1.0], [2.0, 8.0]], dtype=float)

cpu_ref = {
    "phase": "phase_3",
    "run_context": "cpu_reference_only",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "workload_signature": {
        "shape": list(NB.shape),
        "dtype": str(NB.dtype),
        "size": int(NB.size),
        "nbytes": int(NB.nbytes),
        "sha256": hashlib.sha256(NB.tobytes()).hexdigest(),
    },
    "backend": "NumpyBackend",
    "scalar_workload": {
        "evpi_payload": benchmark_evpi(NumpyBackend(), NB, repeats=10, warmup_runs=1),
        "memory_payload": benchmark_memory_throughput(
            NumpyBackend(),
            NB,
            repeats=10,
            warmup_runs=1,
        ),
    },
}

cpu_ref["scalar_workload"]["comparison_packet"] = benchmark_mps_vs_cpu(
    NB,
    repeats=10,
    warmup_runs=1,
    benchmark=benchmark_evpi,
)

cpu_ref["memory_workload"] = {"comparison_packet": benchmark_mps_vs_cpu(
    NB,
    repeats=10,
    warmup_runs=1,
    benchmark=benchmark_memory_throughput,
)}

bundle_path = ARTIFACT_DIR / "phase_3_cpu_reference.json"
bundle_path.write_text(json.dumps(cpu_ref, indent=2, sort_keys=True))
print(f"wrote {bundle_path}")
PY
python3 -m pip freeze > conductor/archive/apple-metal-backend-prototype_20260510/handoff/phase_3_runtime_freeze.txt
```

### Required full comparison packet (Metal-backed path when available)

```bash
python3 - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np

from voiage.main_backends import (
    benchmark_evpi,
    benchmark_memory_throughput,
    benchmark_mps_vs_cpu,
)

ARTIFACT_DIR = Path("conductor/archive/apple-metal-backend-prototype_20260510/handoff")
NB = np.array([[10.0, 1.0], [2.0, 8.0]])
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def run_dual_compare(workload_name: str, benchmark) -> dict[str, object]:
    outputs = []
    for run in range(1, 3):
        outputs.append(
            {
                "run": run,
                "packet": benchmark_mps_vs_cpu(
                    NB,
                    repeats=10,
                    warmup_runs=1,
                    benchmark=benchmark,
                ),
            }
        )
    return {
        "workload_name": workload_name,
        "run_count": len(outputs),
        "workload_hash_sha256": hashlib.sha256(NB.tobytes()).hexdigest(),
        "outputs": outputs,
    }


bundle = {
    "phase": "phase_3",
    "workload_matrix_shape": list(NB.shape),
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "runs": [
        run_dual_compare("scalar_evpi", benchmark_evpi),
        run_dual_compare("memory_throughput", benchmark_memory_throughput),
    ],
}

path = ARTIFACT_DIR / "phase_3_handoff_bundle.json"
path.write_text(json.dumps(bundle, indent=2, sort_keys=True))
print(f"wrote {path}")
PY
```

### Verification command for closure review

```bash
python3 - <<'PY'
import json
from pathlib import Path

ARTIFACT = Path(
    "conductor/archive/apple-metal-backend-prototype_20260510/"
    "handoff/phase_3_handoff_bundle.json"
)

payload = json.loads(ARTIFACT.read_text())
assert payload["phase"] == "phase_3", payload.get("phase")
assert isinstance(payload["runs"], list) and len(payload["runs"]) == 2

run0 = payload["runs"][0]["outputs"][0]["packet"]
assert run0["review"]["phase"] == "phase_3"
assert run0["payload_version"] == "1.0.0"
for run_entry in payload["runs"]:
    assert run_entry["run_count"] == 2, run_entry["workload_name"]
    for entry in run_entry["outputs"]:
        packet = entry["packet"]
        assert packet["review"]["phase"] == "phase_3", packet["review"]["phase"]
        assert packet["workload"]["shape"] == [2, 2]
        assert packet["payload_version"] == "1.0.0"
print("phase_3 artifact passes the baseline structure checks")
PY
```

## Phase 3 Completion Checklist

- [x] Both committed workloads have CPU vs Metal comparison packets.
- [x] At least two comparison runs are persisted for each workload.
- [x] Runtime and package freeze artifacts are versioned in `handoff/`.
- [x] Handoff evidence packet includes `workload`, `runtime`, `payload_version`,
  `review`, and reproducibility fields.
- [x] The optimization target selection statement is passed to the
  integrated GPU track.
- [x] Track artifacts are present in `conductor/archive/apple-metal-backend-prototype_20260510/handoff/`.

## Closure Conditions

The track can be marked complete only if all are true:

- Every payload records `review.phase == "phase_3"` and `payload_version == "1.0.0"`.
- CPU and Metal comparison payloads are deterministically reproducible with
  the same workload hash for both runs.
- EVPI contract anchors in outputs remain stable (`result == 3.0` and
  `summary.evpi == 3.0` where present).
- Runtime metadata captured in both packets includes:
  - `runtime.platform`
  - `runtime.system`
  - `runtime.backend.torch`
  - `runtime.backend.apple_metal_capability`
- Next-step workload ordering, open questions, and optimization priorities are
  communicated to `apple-metal-integrated-gpu-optimization_20260511`.
