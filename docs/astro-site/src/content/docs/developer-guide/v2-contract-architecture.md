---
title: V2 contract architecture
description: Additive, generated contracts for reproducible VOIAGE analyses.
---

VOIAGE v2 contracts are an opt-in interoperability layer around the established
dataclass, NumPy, and xarray runtime APIs. They do not replace those APIs. The
Pydantic models capture declarative analysis inputs, numerical policy,
diagnostics, provenance, and JSON-safe result envelopes while compatibility
adapters preserve existing numerical behaviour.

The calculation kernel selects backends from explicit capability descriptors.
Unsupported method, dtype, device, determinism, or acceleration requirements
fail closed. A caller must explicitly allow fallback, and any fallback is
recorded in diagnostics and run context.

The current perspective adapter executes through the established NumPy path.
Requests for JAX, JIT, or another unsupported perspective capability therefore
fail closed unless the caller explicitly permits fallback to NumPy. The result
records that fallback as degraded rather than attributing NumPy work to another
backend.

Canonical schemas and examples live under `specs/core-api/schemas/v2/` and
`specs/core-api/examples/v2/`. They are generated deterministically from the
models. Run `python scripts/export_v2_contracts.py --check` to detect drift or
use `--write` when intentionally updating the committed artifacts.

Every result also carries an interchange identity. The contract interchange
adapter writes that result as schema-bearing Arrow IPC or Zstandard-compressed
Parquet, with the full canonical JSON envelope retained in `result_json`.
Focused conformance tests read both formats with PyArrow and Polars.

Concern and evidence models use the pinned VOP governance record shape. Local
path evidence must be marked `local_private`, which keeps it outside GitHub
projection payloads governed by the mirrored schema policy.

The focused governance gate is `uv run nox -s contracts`. Equivalent task
entries are available through taskipy (`task contracts`) and Pixi
(`pixi run contracts`). The gate verifies generated artifacts, runs focused
contract tests, and applies Ruff, ty, and BasedPyright to the contract boundary.

Scalene profiling and mutation testing are deliberately bounded and opt in.
Use `uv run nox -s contract_profile` or `uv run nox -s contract_mutation` when
needed. CI runs these expensive hooks only on the weekly schedule or an explicit
manual dispatch with expensive checks enabled, so normal pull requests retain a
fast deterministic contract gate.
