# CodeQL quality alert triage — 16 November 2026

The default-branch CodeQL inventory contained 69 open Python quality alerts
at commit `cdf9bc1427ded76df7e106f97148e60a8131a870`. None of those
records carried a CodeQL security-severity level. The inventory, individual
native readbacks and source-based review were made for this snapshot; later
scans can add or change findings.

## Individual dispositions

The following alerts were reviewed and dismissed individually, with a reason
and explanation recorded on each GitHub alert:

- #961, #994 and #1153: Python's standard-library `logging` and
  `voiage.logging` are distinct modules. Runtime module identity and paths
  confirmed the distinction.
- #1277, #1314 and #1315: ellipses are intentional placeholders in
  `typing.Protocol` declarations.
- #1001: `Protocol` is used as the base of `CalculationKernel`.
- #860 and #863: the imports intentionally probe optional JAX and
  scikit-learn availability.
- #968, #969, #995, #810 and #818: the imports are used in quoted type
  expressions or forward-referenced type bounds.
- #1294–#1299: the findings occur in unchanged, hash-bound upstream Spack
  catalogue sources retained as licence evidence. The repository parses those
  copies in `test_pinned_catalogue_license_sources_are_bound`; they are not
  the live overlay recipes.
- #811, #812, #916, #984, #1195, #1202, #1203, #1228, #1236, #1306 and
  #793: the test modules intentionally use the public import surface and a
  package or module object for export comparisons, reloads or monkeypatching.
- #194–#198, #200–#212, #865, #866 and #875: the reported locals are bound
  by an import before use. If the import fails, the corresponding `except`
  handler calls `pytest.skip`, which raises a skip outcome and stops the test
  before the reported use. Each import, handler and reported location was
  checked individually, and each alert records its own source-based reason.

These dispositions do not suppress the CodeQL workflow or its query suite.

## Source changes awaiting hosted confirmation

The associated branch proposes direct metamodel imports and explicit optional
capability checks to clarify the tests behind those dismissed findings. It also
removes redundant test import paths for #792, #876, #877 and #1313; strengthens
or clarifies the empty-handler paths for #677, #679, #801–#805, #857, #858 and
#874; and uses a stable Pythagorean calculation for #1123.

For #36, the observational example now rejects nonzero selection bias in the
model that does not implement it, and its demonstration passes only the bias
inputs that model supports. This does not add or validate a selection-bias
model.

These 16 source-addressed findings remain open on the default branch until
the change is integrated and GitHub reports the resulting CodeQL analysis.
A passing local check or a pull-request scan alone is not that readback.

## Finding retained open

[#864](https://github.com/edithatogo/voiage/security/code-scanning/864)
remains open. The adaptive-trial helper ignores its updated belief values when
computing net benefits. The prior
[source observation](adaptive-belief-observation-20261105.md) requires a
defined estimand, units and relationship between beliefs, outcomes and costs,
plus an independent analytic or published reference and a failing witness
before changing the calculation. Removing the assignment alone would obscure
that unresolved scientific issue. The simulator's documentation states this
limitation, and a strict expected-failure test records the observed lack of
dependence without prescribing an economic formula; both must be revisited
alongside the reference-backed repair.
