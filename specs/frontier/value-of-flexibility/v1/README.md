# Value of Flexibility v1 experimental contract

This contract governs issue #559's initial timing-scenario estimator. It values
the ability to select a feasible policy for each declared timing scenario
against the best strategy committed before the scenario is known. It is an
adjacent option-value estimand, not EVPI, EVSI or a value-of-control alias.

The runtime remains experimental. Python is the first executable surface; Rust,
R and Julia execution are unsupported and Mojo remains outside the repository.
Transition-constrained sequential-period paths require a later reviewed
contract and are rejected rather than coerced into this version.
