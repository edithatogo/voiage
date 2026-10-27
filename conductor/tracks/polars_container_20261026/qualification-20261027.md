# Python 3.15.1 and Polars qualification: 27 October 2026

Acceptance remains pending the full local gate and current hosted PR checks.
The initial diagnostic report is preserved in `qualification-20261027-initial.md`.
Historical performance and binding evidence remains unchanged.

## Current candidate

The actual repo-local `.venv` and bootstrap `.venv-py315` interpreters report
Python 3.15.1. A frozen clean environment installation completed. Active package
constraints, workflows, tox, nox, helper scripts, documentation and the Rust
Python ABI now select that runtime. Historical v1/v2 profiles and published
2.2.0 HPC recipes retain their original release requirements; the HPC test
uses SHA256-verified metadata extracted from its pinned published sdist.

`uv lock --upgrade` completed, and the strict current dependency audit passed
(`dependency-frontier-python315.json`). This audits policy rather than proving
wheel/runtime behavior. The supported core dependencies installed successfully;
Polars 2.1.0 is the floor candidate. Polars 1.42.1 and 1.44.2 remain unqualified
because their timezone Series construction fails on this runtime.

The maintainer explicitly approved marking the optional PyTorch engine
unavailable while stable CPython 3.15 wheels are absent. Its API remains and
raises an explicit diagnostic; no engine is silently substituted. The optional
extra contains no installable PyTorch requirement. The native extension built
successfully using the CPython 3.15 ABI.

## Performance and validation

The matched comparison loads committed, hash-verified pre-container source
and runs both implementations with identical inputs and dependencies on
Python 3.15.1. The first run exposed construction regressions. Replacing
per-construction query plans with column null metadata and nested-array null
checks retained validation and restored the original performance bounds.
`benchmark-python315.json` records the exact source hashes and measurements:
2D construction 0.098 ms versus 0.208 ms; 3D 0.337 ms versus 0.261 ms;
extraction 0.071 ms; wide-to-long 0.100 ms versus 5.721 ms. All retained bounds
passed. Scientific source evidence pins were refreshed for annotation-only
modernization with original receipts preserved under `pre-python315-evidence/`.

## Resource controls and remaining gates

Every resumed acceptance job uses `scripts/guard_verification.py`: 4 GiB sampled
process-group RSS, one hour, 2 GiB minimum free workspace storage, external-SSD
temporary files and one native thread. Coverage uses two workers and tox uses
two environments. Limit exits are failures. Five control tests passed, including
termination and descendant cleanup. The earlier interrupted full diagnostic
exhausted temporary storage and is not acceptance evidence.

Full default tox, aggregate and changed-code coverage, clean package consumers,
current hosted checks, signed commits and protection-policy readback remain
required before merge. No accepted migration or release claim is made yet.

The pandas-free I/O track remains separate: only its T1.1 inventory is complete.
The translation reference remains pending the accepted container.
