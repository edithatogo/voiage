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

Canonical schemas and examples live under `specs/core-api/schemas/v2/` and
`specs/core-api/examples/v2/`. They are generated deterministically from the
models. Run `python scripts/export_v2_contracts.py --check` to detect drift or
use `--write` when intentionally updating the committed artifacts.

The focused governance gate is `uv run nox -s contracts`. Equivalent task
entries are available through taskipy (`task contracts`) and Pixi
(`pixi run contracts`). The gate verifies generated artifacts, runs focused
contract tests, and applies Ruff, ty, and BasedPyright to the contract boundary.

Scalene profiling and mutation testing are deliberately bounded and opt in.
Use `uv run nox -s contract_profile` or `uv run nox -s contract_mutation` when
needed. CI runs these expensive hooks only on the weekly schedule or an explicit
manual dispatch with expensive checks enabled, so normal pull requests retain a
fast deterministic contract gate.
