# Value of Distribution-Family Information v1

This experimental contract values perfect resolution of a declared discrete
model-family index. It first integrates all remaining uncertainty inside each
family, then compares the current mixture-optimal alternative with the policy
that may select a different alternative after the family index is revealed.

The numerical identity is discrete-index EVPPI. The issue-facing VDI name does
not mean distributional-equity VOI, and the result is not full structural EVPI,
model-selection accuracy, a Bayes factor or model-discrimination EVSI.

The exact v1 input supplies one comparable conditional expected value for every
family/alternative pair. Family probabilities are required, named and checked
without renormalization. The result preserves all numerical ties, reports gross
VDI and signed net VDI after the information cost, and labels exact enumeration
as exact rather than inventing a zero standard error.

Python execution is delivered separately in F557-3. Rust, R and Julia remain
unsupported until shared-fixture evidence exists; Mojo is external. Scientific
review of the terminology, family partition and probability provenance is
required before stable promotion.
