# Value of Distribution-Family Information v1

This experimental contract values perfect resolution of a declared discrete
model-family index. It first integrates all remaining uncertainty inside each
family, then compares the current mixture-optimal alternative with the policy
that may select a different alternative after the family index is revealed.

The numerical identity is discrete-index EVPPI. The issue-facing VDI name does
not mean distributional-equity VOI, and the result is not full structural EVPI,
model-selection accuracy, a Bayes factor or model-discrimination EVSI.

The exact v1 input supplies one comparable conditional expected value for every
family/alternative pair. Each candidate family has a structured definition,
parameterization, within-family integration method, sources, data reference and
value transformation. The `conditional_value_assurance` record admits only
exact enumerated conditional expectations in v1; estimated or simulated tables
must not be relabelled as exact. Family probabilities are required, named and
checked without renormalization.

Comparability is also affirmative rather than descriptive: the request must
identify common population, horizon, discounting, value semantics and cost
location contracts, mark their verification complete and cite its evidence.
The result preserves these records and all numerical ties, reports gross VDI
and signed net VDI after the information cost, and labels exact enumeration as
exact rather than inventing a zero standard error.

The checked-in result schema is independently usable from a source distribution
or repository checkout. Wheels install the input schema and evaluator, and
return the same versioned result object, but v1 does not advertise a separately
installed result-schema resource.

Python execution now evaluates the same strict exact-enumeration contract and
returns complete ties plus a canonical presentation representative. Rust, R
and Julia remain unsupported until shared-fixture evidence exists; Mojo is
external. Scientific review of the terminology, family partition and
probability provenance is required before stable promotion.
