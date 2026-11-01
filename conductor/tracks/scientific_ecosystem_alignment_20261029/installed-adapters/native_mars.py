# SPDX-License-Identifier: Apache-2.0
"""Replay an installed MARS fitted artifact explicitly through its compiled predictor."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from pymars import EarthRegressor
from pymars_runtime import predict_json
from pymars_runtime import pymars_runtime as extension

x = np.linspace(-1.0, 1.0, 33)[:, None]
model = EarthRegressor().fit(x, x.ravel())
spec = model.earth_.get_model_spec()
serialized = json.dumps(spec, allow_nan=False, sort_keys=True)
result = np.asarray(predict_json(serialized, x.tolist()))
np.testing.assert_allclose(result, x.ravel(), rtol=1e-10, atol=1e-12)
binary = Path(extension.__file__).resolve()
assert binary.is_relative_to(Path(sys.prefix))
with binary.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
print(
    json.dumps(
        {
            "points": len(result),
            "max_absolute_error": float(np.max(np.abs(result - x.ravel()))),
            "extension_sha256": digest,
            "scope": "Rust prediction from fitted MARS model; no Rust training claim",
        },
        indent=2,
    )
)
