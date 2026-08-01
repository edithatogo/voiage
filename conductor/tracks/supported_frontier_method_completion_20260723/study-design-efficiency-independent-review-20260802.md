# Independent study-design efficiency review

## Independence and reviewed boundary

The reviewer performed a read-only review and did not author or modify the
#571 implementation or this remediation. The initial review covered PR #679
exact head `ce5d712779897bdd7d398e367de6a7e0bc743692`, squash merge
`5d059a80447afc85cee63eb85971fc1c9e80f40c`, and their byte-identical tree
`3d7f52423d8135cec7366fea545bc4a4c463505a`.

The remediation re-review covered branch base
`5665b3e35b70c991954472a000634931080b2aeb` plus the current merge-resolution
working tree. Its exact immutable commit binding is recorded by the subsequent
umbrella evidence receipt after the signed merge commit.

## Initial findings

The initial verdict was zero Critical, zero High and two Medium findings:

1. Complete selection-probability normalization used declared absolute plus
   relative tolerance in `CossResultV1`, but the façade used only absolute
   tolerance and the leaf model imposed an unrelated `1e-12` mass cap.
2. `plot_coss` marked only the selected optimum and did not expose the complete
   tied set or selection-uncertainty availability.

## Remediation and independent re-review

The façade and result validator now apply the same probability-scale tolerance,
`absolute_tolerance + relative_tolerance`, while complete and partial maps are
validated in the context that supplies those tolerances. At absolute tolerance
zero and relative tolerance `0.1`, totals `0.95` and `1.05` are accepted and
round-trip through the result contract; `0.89` and `1.11` are rejected.

The plotting adapter now uses a distinct hollow-diamond encoding and an
explicit legend entry for every tied optimum. Full `CossResultV1` inputs also
state whether selection uncertainty is unavailable or available with its
declared method. An analytic two-way tie reproduced labels `Tied optima (a, b)`
and `Selection uncertainty available (analytic)`.

The independent re-review passed 81 focused Python tests and Ruff. Final
working-tree verdict: zero Critical, zero High and zero Medium findings. The
reviewer made no edits.

## Retained boundaries

This is engineering-review evidence only. The capability remains experimental;
R and Julia remain unsupported and Mojo remains external. Scientific review,
installed polyglot parity, stable promotion, release, parent #571 closure and
umbrella #318 closure remain pending.
