Data Structures
===============

`voiage` uses a small set of explicit data structures to keep method inputs and
outputs serializable, inspectable, and stable across bindings.

The core contract treats xarray Datasets as the canonical in-memory
representation for the public data structures. The helper constructors and
copy methods preserve dataset coordinates so bindings can round-trip the same
labels without flattening them into tabular objects.

Key structures:

- `ValueArray`
- `DecisionAnalysis`
- `Intervention`
- `DecisionProblem`
- `TrialDesign`
- `ParameterSet`

`ValueArray` and `ParameterSet` provide explicit dataset round-trip helpers
(`from_dataset()` / `to_dataset()`) to make the xarray-based contract visible
to callers.

See also:

- `docs/api_reference/voiage.schema.rst`
- `docs/api_reference/voiage.analysis.rst`
