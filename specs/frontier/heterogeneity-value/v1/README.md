# Static and dynamic heterogeneity value v1

This experimental finite contract separates the value of acting on a declared,
currently known subgroup structure from the value of resolving uncertainty in
subgroup effects. It reports four policy values: `C0` and `Cf` under current
information without and with subgroup-specific policies, and `P0` and `Pf`
under perfect information without and with subgroup-specific policies.

Static value is the direction-aware contrast between `Cf` and `C0`. Dynamic
value is the corresponding contrast between `Pf` and `P0`. The result verifies
`dynamic - static = EVPIf - EVPI0`. An optional finite signal model reports
population-common and subgroup-policy EVSI separately and verifies the
analogous sample-information identity. Study cost is reported only as a signed
net EVSI diagnostic; it never changes the gross decomposition.

The contract requires a prespecified partition, population weights, eligibility,
selection and multiplicity policies, fairness/privacy constraints, common units
and exact estimator assurance. It does not estimate subgroup effects, discover
segments, adjust selection bias, claim sparse-subgroup validity, or implement
stable/polyglot execution.

Each portable result embeds the complete strict input contract under
`assurance.input_contract`, commits to its canonical JSON with
`assurance.input_sha256`, and must exactly reproduce under standalone
re-evaluation. This binds policies, complete ties, subgroup declarations,
provenance, counts and every reported value to the evaluated model.
