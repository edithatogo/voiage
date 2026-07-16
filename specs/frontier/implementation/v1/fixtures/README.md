# Implementation-Adjusted VOI v1 Fixtures

This directory contains the deterministic fixture set for the experimental
implementation-adjusted VOI contract.

## Layout

- `normative/`: deterministic fixtures that anchor the contract and document
  the current Python behavior.
- `open-data/`: a small OWID/WHO-UNICEF HPV coverage snapshot with provenance
  and explicit uptake-proxy limitations for experimental analysis.

The committed fixture is intentionally small and keeps the implementation
friction surface aligned with the existing experimental example.
