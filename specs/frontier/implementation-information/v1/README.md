# Implementation-information decomposition v1

This experimental contract enumerates uncertain states, intended actions and
realised actions. Current, specific and post-sample implementation are explicit
conditional distributions, so information and implementation are not assumed
independent.

The four primary cells are current/perfect information crossed with
current/perfect implementation. The result reports EVPIM, EVSIM, realizable
EVPI, EVP, implementation-adjusted EVSI, the interaction term, complete policy
ties, population/time scaling, costs and exact decomposition residuals.
`EVEIm` and `EVSEIm` are presentation candidates only, not new estimands.

Python is the only experimental runtime in v1. Rust, R, Julia and Mojo remain
explicitly unimplemented. This fixture-backed contract is not a stable-method
or release claim.
