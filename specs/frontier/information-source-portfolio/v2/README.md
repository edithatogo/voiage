# Information-source portfolio v2: shared fixed costs

This experimental contract extends the finite-world v1 evaluator. It retains
the same decision-value model, direct source costs and ordering constraints,
and adds explicitly identified shared fixed assets. It does not model future
sample costs beyond the declared direct source costs, change the joint-world
law, or imply scientific qualification for a named consumer.

Each source declares `asset_ids`. `shared_costs.assets` declares a unique ID,
nonnegative finite cost and unit for each asset; `existing_assets` identifies
assets already owned. Every referenced ID must be declared and each cost unit
must match the value context. A selected sequence pays each direct cost and
each referenced, unowned asset once. The same union cost governs feasibility,
total and net value. The ordered marginal ledger charges an asset at its first
use in the sequence. It is not a Shapley cost allocation.

The v2 result records the asset declarations, ownership and direct source
costs in `cost_basis`; validation recomputes each sequence's cost and ordered
marginals from that basis. The selected result must match an evaluated
sequence. Schema version `2.0.0` identifies both input and output; existing
v1 schemas, inputs and results remain supported unchanged.

Schema files: [input](./schemas/input.schema.json) and
[result](./schemas/result.schema.json). The synthetic [input](./fixtures/normative/input.json)
and [independent cost expectations](./fixtures/normative/expected.json) cover
unowned and already-owned assets. Both this contract and its executable
implementation require package, native, hosted and consumer qualification;
source-level test success alone is not release or consumer acceptance.
