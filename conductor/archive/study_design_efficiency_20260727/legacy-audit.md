# Legacy surface audit

Audit date: 2026-07-31. Repository baseline: `25722c32`. These decisions are
normative for implementation under issue #571.

## Signed ENBS

`voiage.methods.sample_information.enbs` and the Rust `voiage-numerics` ENBS
kernel implement finite-validated, signed `EVSI - research_cost`. The stable
scalar function and closed v1 result schema remain unchanged. COSS may reuse
the Rust subtraction policy but receives a separate experimental result
envelope because adding curve fields to the stable schema would be breaking.

The NumPy, JAX and MPS `enbs_simple` compatibility helpers substitute EVPI for
EVSI and floor negative values at zero. Existing tests require that behavior.
They are clipped screening scores, are not suitable COSS inputs, and remain
unchanged until a separate deprecation and breaking-version decision.

## Clinical-trial optimizer

`VOIBasedSampleSizeOptimizer` evaluates a hard-coded step-10 grid, derives an
approximate total value from power/QALY heuristics, subtracts cost and relies
on `jnp.argmax`. It has no declared sampling model, exact feasible set,
feasibility reasons, deliberate tie policy, estimator uncertainty, optimum
uncertainty, boundary diagnostic or provenance. Its current output therefore
is neither EVSI nor governed COSS.

Its `voi_efficiency` field is `total_voi / total_cost` with an epsilon added to
the denominator. Adjacent `voi_per_dollar` and `cost_per_voi` fields are also
value/cost measures with units. They are not dimensionless EVSI/EVPI
efficiency. For backward compatibility the existing class and keys remain,
but documentation will label them legacy cost-efficiency outputs and the new
diagnostic will use an unambiguous name. No adapter may promote the heuristic
result to COSS unless genuine EVSI inputs and the governed common context are
provided.

## Plotting helper

`plot_evsi_vs_sample_size` accepts parallel sample-size, EVSI, optional ENBS
and cost arrays and can reuse a supplied axis. It remains a supported low-level
renderer. It does not establish COSS because it does not represent design
identity, units/scaling, feasibility, uncertainty, ties, selected optimum,
boundary state or the ENBS zero line.

A separate experimental COSS renderer will consume `CossResultV1` or
`CossPlotDataV1`, show infeasible designs and unavailable uncertainty, and
mark the selected/tied optimum and zero-ENBS reference. The versioned result
will not contain Matplotlib objects or backend-specific styling.

## CLI and public API

The stable `calculate-enbs` command, existing raw-array plotting command and
top-level stable ENBS export remain unchanged. COSS calculation/reporting and
EVSI/EVPI efficiency use separate experimental entry points and schemas. They
are not added to the stable top-level API until the scientific-review and
promotion gates are satisfied.

## Compatibility decisions

| Surface | Decision | Reason |
| --- | --- | --- |
| Stable scalar `enbs` | Preserve and reuse its signed Rust policy. | Scientifically compatible and already governed. |
| Stable ENBS schema | Preserve unchanged. | Closed schema; curve fields would break v1. |
| `enbs_simple` backends | Exclude from COSS; defer deprecation. | EVPI substitution and zero floor conflict with signed ENBS. |
| `VOIBasedSampleSizeOptimizer` | Preserve as legacy/provisional; do not call it COSS. | Heuristic value and implicit argmax semantics. |
| `voi_efficiency` | Preserve key but label legacy cost efficiency. | It is value divided by cost, not EVSI divided by EVPI. |
| `plot_evsi_vs_sample_size` | Preserve; add separate governed adapter. | Useful renderer but incomplete scientific contract. |
| Stable CLI/API | Preserve; add experimental entry points. | Avoid semantic and schema breakage. |
