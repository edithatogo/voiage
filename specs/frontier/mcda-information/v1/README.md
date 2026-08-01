# Finite additive MCDA perfect-information contract v1

This experimental contract freezes issue #560's finite compensatory additive
MCDA decision model. It is planned for v1.3.0 under canonical requirement M21.
Python provides strict portable validation and exact finite evaluation; stable
promotion and cross-language parity remain separate gates.

The input declares named alternatives, operationally distinct criteria with
raw units and directions, fixed ex-ante two-anchor linear value functions, a
finite joint state law, normalized default or state-specific weights, and
exactly one criterion, preference and joint information action. State-specific
weights and outcomes remain in one joint law so conditioning cannot erase their
dependence. Every action owns a disjoint cost record converted to the common
aggregate-value scale and basis.

The result contract retains baseline and conditional scores, full rankings and
complete choice ties; gross and signed net information value; the joint
interaction and both conditional increments; regret; fractional complete-tie
rank acceptability; and expected and statewise Pareto diagnostics on fixed,
direction-normalized criterion values. Exact-enumeration assurance does not
claim stable status or that any non-Python binding has been delivered.

The normative four-state fixture is intentionally correlated. Criterion-only
and preference-only information do not change the baseline choice, but their
joint refinement does, producing a positive interaction without double
counting. The pathology fixtures reject probability, normalization, partition
and unsupported aggregation violations.

Explicitly outside v1 are AHP elicitation, outranking and veto methods,
non-compensatory or nonlinear aggregation, post-information normalization,
ordinal-to-cardinal conversion, imperfect/sample information, social choice,
endogenous feasible sets, stable status and unverified language parity.
