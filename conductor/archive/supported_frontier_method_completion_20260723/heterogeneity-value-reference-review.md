# #599 heterogeneity-value reference review

## Scope and primary source

Issue #599 and children #786, #788 and #789 adopt the static/dynamic framework
in Espinoza, Manca, Claxton and Sculpher, *Medical Decision Making* 2014,
doi:10.1177/0272989X14538705, PMID 24944196. The source distinguishes the value
of acting on factors associated with heterogeneity using existing evidence
(static value) from the value of acquiring further subgroup-related evidence
under current uncertainty (dynamic value).

## Frozen finite construction

Let `C0` and `Cf` be current-information optimum values under one
population-common policy and subgroup-specific policies. Let `P0` and `Pf` be
the corresponding expected optima when the declared effect state is learned
perfectly. With direction-aware gains:

- static value is `Cf - C0` for maximization (`C0 - Cf` for minimization);
- dynamic value is `Pf - P0` for maximization (`P0 - Pf` for minimization);
- `EVPI0` compares `P0` with `C0`, while `EVPIf` compares `Pf` with `Cf`;
- `dynamic - static = EVPIf - EVPI0` is checked exactly.

For a declared finite imperfect signal, `S0` and `Sf` replace `P0` and `Pf`.
The evaluator reports `EVSI0`, `EVSIf` and the sample-informed segmentation
contrast separately. It does not relabel EVSI as dynamic value or subtract a
study cost from any gross estimand.

## Boundaries

The finite evaluator assumes a declared complete joint law and prespecified
partition. It cannot establish causal subgroup effects, discover a subgroup,
correct model selection or multiplicity, validate a sparse subgroup, discharge
fairness/privacy review, or support a stable method claim. The existing stable
`value_of_heterogeneity` summary remains a distinct descriptive current-
information surface and is not overwritten.
