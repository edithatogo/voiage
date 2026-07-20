# Normative v1.0 contract

`stable-api.json` is the machine-readable source of truth for the public v1.0
surface. It narrows the current broad Python export set into five maturity
classes and records the future production contract before runtime migration:

`compatibility-policy.json` is the machine-readable SemVer and deprecation
contract for that stable surface across Rust, Python, the C ABI, bindings, CLI,
schemas, and serialized outputs.

- **Stable:** the core schemas, `DecisionAnalysis`, EVPI, EVPPI, EVSI, ENBS,
  CEAF and dominance.
- **Provisional:** convenience namespaces, ecosystem adapters and façade
  modules whose names may be narrowed before v1.0.
- **Experimental:** frontier and research method families that remain outside
  semantic-versioning guarantees until separately promoted.
- **Deprecated:** compatibility aliases retained only for the documented 0.x
  migration period.
- **Removed:** names intentionally absent from v1 and recorded for migration
  auditing.

The contract also governs Rust ownership of stable numerics, supported Python
and operating-system versions, method input shapes and results, missing-value
and infinity handling, deterministic seed behavior, standard errors, schemas,
diagnostics, reporting, provenance, plotting and CLI serialization.

The `implementation` fields describe the required v1.0 destination. They do
not assert that the current 0.x runtime already executes every method in Rust.
Migration evidence is supplied by later phases of the active Conductor
programme and must pass the shared fixtures before the v1.0 release.
