# SPDX-License-Identifier: Apache-2.0
"""Evaluate a verified VOP handoff with an explicitly identified local candidate wheel."""

from hashlib import file_digest
from importlib import import_module
import json
from pathlib import Path
import runpy
import sys

import numpy as np

root, handoff_path, wheel = map(Path, sys.argv[1:4])
expected_wheel = sys.argv[4]
handoff = runpy.run_path(str(handoff_path))
receipt = root / "export.json"
with receipt.open("rb") as stream:
    receipt_hash = file_digest(stream, "sha256").hexdigest()
# Compare with the previously published reference before following the CSV handoff.
# The fresh export receipt also contains environment-specific values.
record = json.loads(receipt.read_bytes())
published = {
    "net_benefit_csv_sha256": "509411114afe90617f0f60a7743ad11f1160fb1288bb4ef144b572e5294050d9",
    "source_revision": "2c46db2fe5f907d894bb07f1127c008f10ee462e",
    "draws": 500,
    "seed": 20260727,
}
for key, wanted in published.items():
    if record.get(key) != wanted:
        raise ValueError(f"VOP published witness mismatch: {key}")
matrix, record = handoff["load_handoff"](receipt, receipt_hash)
verify_wheel = runpy.run_path(str(Path(__file__).with_name("wheel_identity.py")))[
    "verify_wheel"
]
verified = verify_wheel("voiage", "voiage", "2.2.0", expected_wheel, wheel)
# Import the candidate only after validating its installed files.
evpi = import_module("voiage.methods.basic").evpi

observed = float(evpi(matrix))
reference = float(np.max(matrix, axis=1).mean() - np.max(matrix.mean(axis=0)))
assert abs(observed - reference) < 1e-8
assert "vop_poc_nz" not in sys.modules
print(
    json.dumps(
        {
            "matrix_sha256": record["net_benefit_csv_sha256"],
            "rows": len(matrix),
            "seed": record["seed"],
            "export_revision": record["source_revision"],
            "candidate_wheel_sha256": expected_wheel,
            "wheel_members_verified": verified,
            "installed_voiage_evpi": observed,
            "reference_evpi": reference,
            "export_receipt_sha256": receipt_hash,
            "scope": "Pinned VOP source export, installed candidate evaluation; public wheel gate separate",
        },
        indent=2,
    )
)
