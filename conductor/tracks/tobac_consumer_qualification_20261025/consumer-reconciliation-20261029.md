# Consumer reconciliation starting point — 29 October 2026

Owner: chat `01a17ab7-148e-7042-be1e-03a9478643da`, worktree
`/Users/doughnut/.codex/worktrees/tobac-track-reconciliation/voiage`, branch
`codex/tobac-consumer-planning-20261025`. Scope: registration and initial
read-only contract mapping; no native service scope. Next: verify actual producer
and installed consumer behavior before implementing any demonstrated gap.

The observed tobac master is `5ff282d06d9b04477266b26907fe039572e11877`.
Its scientific ecosystem specification defines R11.1–R11.5. This is a
source-level starting point, not proof of acceptance for the named consumer.
The Voiage registration branch was based on main `16447de0d26041477660581b2c11c91f53e0e6ac`.

| Master requirement | Existing Voiage starting point | Evidence still needed |
| --- | --- | --- |
| R11.1 parameter validation and round trip | `ParameterSpec` defines parameter semantics and bounds; `AnalysisSpec` has a schema version and canonical JSON. | Demonstrate actual tobac values, units, covariance and provenance round trips, version rejection, migration and rollback. |
| R11.2 scenario isolation | Contract models are frozen and reject extra fields. | Run two actual consumer scenarios in both orders and repeatedly, verify inputs and random streams, and check for mutable state outside the models. |
| R11.3 calculation and adapter separation | `NumericalPolicy` records execution choices separately from parameter definitions. | Exercise the consumer calculation without I/O and separately verify adapters against the same model and inputs. |
| R11.4 bounded extension | Existing parameter and method contracts provide an initial interface. | Define a consumer-derived extension and independent reference; prove unchanged scenarios and qualified backend tolerances. |
| R11.5 result identity and replay | `AnalysisSpec` has input and binding digests; `RunContext` records backend and execution identity; normalized inputs carry source provenance. | Verify installed result and replay receipts distinguish changed parameters, model/schema identity and backend, with actual consumer artifacts. |

These references do not establish a complete parameter-set schema, installed
consumer parity, scientific applicability, or a need for a new interface.
The existing normalized Arrow digest/serialization review remains a separate
open dependency; do not infer its security or compatibility acceptance here.
