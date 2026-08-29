# VoC alias and presentation review

Status: pass for experimental repository delivery at merged PR #712.

## Numerical authority

`voiage_numerics::expected_utility_information` is the sole numerical kernel.
Python, `DecisionAnalysis`, reporting and the
`calculate-expected-utility-information` CLI delegate to that result. There is
no `calculate-voc` command, second VoC kernel, R/Julia runtime adapter, or stable
v1 ABI addition.

## Presentation semantics

VoC presents the expected-utility value of the clairvoyant policy under the
declared utility, wealth/reference state, stakeholder scope and information
structure. The monetary EVPI label is emitted only after verifying a
positive-affine utility reduction. Nonlinear utility retains its utility-scale
and certainty-equivalent distinctions; it is not relabelled as monetary EVPI.

## Evidence

- `tests/test_expected_utility_information.py` covers the kernel and affine
  reduction.
- `tests/test_expected_utility_information_contract.py` covers the portable
  result and diagnostics.
- `tests/test_expected_utility_information_bindings.py` proves the single
  kernel/CLI disposition and explicit R, Julia and Mojo boundaries.
- PR #712 exact head `1048c4bc` passed 60 checks with 5 intentional skips and
  no failures, then merged as `b8395abf`.

Scientific review and stable promotion remain separate gates. This review does
not close #595 or promote the experimental family.
