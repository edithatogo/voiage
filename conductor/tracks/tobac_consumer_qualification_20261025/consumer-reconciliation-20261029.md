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

## Structure witness lane

Chat `01a17ab7-148e-7042-be1e-03a9478643da` owns the isolated worktree
`/Users/doughnut/.codex/worktrees/tobac-structure-witness/voiage`, branch
`codex/tobac-structure-witness`, for a numerical counterexample in
`rust/crates/voiage-numerics/src/structural.rs`. No native service is in scope.
The selected native counterexample passed under the resource guard using one
build job and a 1 GiB sampled RSS limit. Next: determine a consumer-qualified
structure-only route without changing existing API behavior.
The case records the distinction between observing structure and observing
individual within-structure samples; it is not consumer acceptance evidence.

The existing finite-world portfolio evaluator was exercised with synthetic
worlds in both orders. A constant structure observation returned zero;
distinguishing equally likely structures returned five. With probabilities
0.8 and 0.2 the fixed-action baseline is eight and the structure-only value is
two. A four-world grouping with two outcomes per structure returned one,
compared with four for full outcome revelation. Eight focused Python cases
passed on Python 3.15.1 under the resource guard.
This is a bounded source witness, not an installed tobac qualification,
general weighted-sample qualification, or evidence for arbitrary models.
