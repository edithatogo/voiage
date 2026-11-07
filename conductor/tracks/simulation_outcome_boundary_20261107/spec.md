# Specification — optional simulation-outcome inputs

## Scope and baseline

At source revision `a5d87a736fa9f0f876e44872d946b2170d2a0898`, Voiage already
has Rust `voiage-domain` contracts for `ValueArray` and `ParameterSet`, and
Python `NormalizedInputBundle`/`VOIBinding` contracts for format-neutral input
and Arrow/Parquet interchange. Existing validation covers dimensions, finite
values, and declared bindings. That evidence does not establish the new paired
simulation-outcome profile proposed here.

Voiage remains a cross-domain value-of-information library. An optional adapter
will accept already-valued outcomes from simulation, game, HEOR, or other domain
producers through a versioned input profile. It reuses existing input and
numerical contracts rather than becoming a simulation or valuation engine.

## Ownership and dependency direction

| Owner | Responsibility |
|---|---|
| Voiage | VOI methods, validated decision inputs, numerical diagnostics, and optional outcome conversion |
| Kairos or another simulation producer | Simulation execution, event/state semantics, RNG, and outcome provenance |
| Generic game packages and their applications | Supported solving, specific games, strategic assumptions, and payoff construction |
| HEOR and domain producers | Valuation, units, perspective, health assumptions, currency/price basis, horizon, and discounting decisions |
| UOGTO and domain semantic owners | Their semantic identifiers and definitions; no local semantic fork |

No mandatory Kairos, game, or HEOR engine dependency enters Voiage or its default
installation. Producer-specific integration is opt-in. The proposed adapter
maps validated records to Voiage's Rust contracts and kernels; it does not add
a host-language simulation fallback or duplicate VOI/valuation algorithms.
Data conversion is not evidence that the producer model, valuation, or VOI
method is scientifically qualified.

## Proposed input contract

The public profile must declare its version and supported subset before runtime
implementation. Reuse compatible existing contracts; make any necessary
extension/version decision at the public-contract review boundary. This track's
creation changes no schema, IRI, ABI, dependency, or pin.

- Stable strategy identifiers and ordered strategy labels distinguish the
  decision alternatives. Every strategy has exactly one value for each declared
  draw key in the supported complete-pairing profile.
- Draw identity includes the producer's draw namespace and any needed scenario
  and replication identity. Row position or equal RNG seeds alone do not prove
  joint-draw correspondence. Preserve declared weights and their semantics;
  reject duplicate, missing, or inconsistent pairs rather than imputing them.
- Parameter samples, when supplied, join to outcomes by declared draw identity.
  Equal row counts are insufficient; reject unmatched or duplicated keys.
- Explicit units, perspective, value convention, and objective direction travel
  with each input. Where applicable, include currency/price year, population
  basis, horizon, and discounting status as producer declarations. Reject
  incompatible or absent required declarations before calling a VOI method.
- A maximized utility/net-benefit input is distinct from a minimized loss.
  Unsupported conventions fail explicitly. Do not infer a sign flip, currency
  conversion, willingness-to-pay threshold, health valuation, or perspective
  aggregation from a numeric column name.
- Retain producer/version, model/scenario revision, source/valuation provenance,
  artifact digest, and input contract version. Arrow/Parquet round trips must
  preserve these semantics and identities, not just array shapes.

The adapter may reorder records only through a declared deterministic key
mapping preserved in its receipt. Independently reordering each strategy's
samples changes their joint distribution and must fail the distinguishing test.
Missing values, non-finite values, unsupported weight semantics, and unknown
versions return explicit errors. The producer remains responsible for the
validity of any asserted coupling or common-random-number construction.

## Relationship to existing work

[Scientific ecosystem alignment](../scientific_ecosystem_alignment_20261029/index.md)
retains provider/pin ownership, Innovate/MARS adapter qualification, and existing
tobac/VOP installed numerical contracts. This new input-profile track supplies
a bounded contract and its tests to that owner; it neither replaces the track
nor promotes its unfinished gates. Reuse accepted receipts only when their
source, input profile, and tested artifact match the current candidate.

[HPC qualification](../hpc_source_build_qualification_20261029/index.md) remains
deferred except MLX/Metal under the existing 5 November disposition. No HPC
build, accelerator, compiler, or speedup work is activated here. Historical
completion labels and unsatisfied scientific/installed-consumer gates remain
unchanged.

## Acceptance and review

Eight bounded leaves in the plan distinguish protocol validation, identity,
value conventions, optional conversion, interchange, numerical equivalence,
and installed use. Public-contract and numerical boundaries receive coherent
independent specialist agent review; routine leaves use diff review and focused
tests. Existing required CI applies when implementation is delivered. Repeated
human approval is not a routine gate, and a planning document is not a completed
method or qualified integration.

Qualification must bind exact producer/consumer revisions, schema version,
artifact hashes, test commands/counts, declared tolerances, and observed results.
Use at least one health and one non-health already-valued fixture to demonstrate
the generic boundary, clearly labelling synthetic examples. Those fixtures do
not establish empirical validity or a complete real-producer integration.
Preserve negative tests and explicit unsupported cases. Roll back an adapter
through its prior qualified version; do not silently change values or fall back
to another engine.
