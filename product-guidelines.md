# voiage - Product Guidelines

## Code Style and Conventions

- **Modules**: lowercase with underscores, for example
  `health_economics.py`.
- **Classes**: PascalCase, for example `DecisionAnalysis`.
- **Functions and Methods**: snake_case, for example `calculate_evpi`.
- **Constants**: upper snake case, for example `DEFAULT_POPULATION`.
- **Private Members**: prefix with an underscore.

## Documentation Standards

- Public APIs use NumPy-style docstrings.
- New or changed functions and methods must be type-hinted.
- User-facing changes should update docs, examples, or release notes when the
  behavior is visible outside tests.
- Sphinx remains in the local docs gate; Starlight/Astro docs-site work must
  preserve the same user-facing content boundaries.

## Error Handling

- Use domain-specific exceptions from `voiage.exceptions` where available.
- Validate inputs at method boundaries.
- Prefer actionable error messages that identify the invalid field or contract.
- Keep optional integrations fail-soft when the dependency or external format is
  unavailable, but do not hide failures in examples or CI checks.

## API Design Principles

- Preserve the stable `ValueArray`, `ParameterSet`, `TrialDesign`, result
  envelope, and diagnostics contracts.
- Keep functional and object-oriented APIs behaviorally aligned.
- Backend, accelerator, and binding implementations must preserve public API
  behavior before claiming parity.
- Keep experimental and evidence-gated surfaces explicitly labelled.

## CLI Guidelines

- Use Typer for CLI implementation.
- Core and frontier methods should expose reproducible CLI commands when they
  are user-facing.
- CLI examples should consume maintained fixtures, write generated outputs to
  temporary or user-selected paths, and fail fast on command failures.
- Help text should include working examples.

## Testing Standards

- Maintain the repository-wide coverage threshold at **>90%**.
- Add focused regression tests for cleanup rules, bug fixes, and new behavior.
- Keep generated fixtures deterministic.
- Use property-based tests where numerical invariants are more valuable than
  example-only checks.
- Run focused tests first, then the full tox gate before marking a slice done.

## Multi-Domain Support

- Keep core algorithms in `voiage.methods` and shared contracts in the schema
  and specs surfaces.
- Domain adapters should remain thin and artifact-first.
- Healthcare, financial, environmental, engineering, and future domain modules
  should preserve the same VOI result semantics.

## Release and External Gates

- `pyproject.toml` is the canonical Python package metadata source.
- `changelog.md` records user-facing changes.
- Registry, external curation, and hardware evidence gates must be recorded as
  external gates rather than repo-local completion.
- Binding and HPC release surfaces should keep generated artifacts separate
  from source-controlled contracts, fixtures, and workflows.
