# Dynamic Real-Options VOI v1 Fixtures

This directory contains the deterministic fixture set for the experimental
dynamic real-options VOI contract.

## Layout

- `normative/`: deterministic fixtures that anchor the planned contract and
  serve as the first conformance target for future language bindings.
- `evidence.json`: hash-pinned implementation evidence and explicit open-data
  and parity gates.

The committed fixture is intentionally small and uses the staged-evidence
surface already documented in the v1 example files. The dynamic real-options
API remains experimental until the wider cross-language conformance story is
complete, but this directory now provides the fixture backbone for promotion
work.
