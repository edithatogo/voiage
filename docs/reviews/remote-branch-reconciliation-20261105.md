# Remote branch reconciliation — 5 November 2026

Remote heads were read with `git ls-remote --heads voiage` for this inventory.
The maintenance branch `2.0.x` is a release line and is
not a cleanup candidate. Local remote-tracking refs also contain stale names;
they are not evidence that those branches still exist on the server.

- `codex/backlog-evidence-20261103` at `0436e1bee` is the Arrow qualification
  branch; its changed source and handoff dispositions are in the delivery
  reconciliation. Preserve deferred HPC evidence.
- `codex/telemetry-deferral-integration` at `9a8b4ed30` predates the merged
  opt-in exporter and later deferral; preserve it pending ownership cleanup.
- `jules-555481459443272329-56b0afe3` at `3948d79c6` has no net diff from
  its merge base: its test addition was reverted on the branch.
- `jules-rf-rmse-test-fix-7210898025045590352` at `9115c40b1` adds unfitted
  random-forest error checks. A bounded equivalent was prepared on local
  branch `codex/random-forest-unfitted-20261105` at `bd627684e`; focused
  tests and Ruff pass. Delivery is pending in PR #1276.
- `fix-missing-rf-rmse-test-4086974318902082962` at `a1eaee432` includes
  similar random-forest coverage plus changes across twelve other files. Those
  additional changes have not been integrated; preserve the ref. A Python
  3.15 AST comparison found six script changes structurally unchanged and six
  files with changed ASTs (including three paper examples). A further limited
  comparison found those six trees equal after normalizing top-level import
  ordering, `timezone.utc` to `UTC`, and trailing whitespace in docstrings.
  This identifies the observed differences, not byte equality or a guarantee
  about import side effects; no paper or unrelated script changes were replayed.
- `test-create-financial-config-8805096995058823085` at `754d98e63` adds a
  call to a direct-execution block; current pytest already collects
  `test_create_financial_config`. This does not imply byte-identical trees.
- `test-to-dict-improvement-15134409924331881500` at `2bf371b68` adds a
  non-null model serialization witness; current
  `test_voi_analysis_config_to_dict_contract` covers object identity and full
  mapping. Its unrelated helper file has not been adopted.
- `test-condition-16538021912075585812` at `541124cc7` adds TinyGP protocol
  and conversion tests. Current tests cover the invalid protocol object and
  values-attribute conversion; the missing dense NumPy identity case is now
  prepared and tested on branch `codex/random-forest-unfitted-20261105` in
  PR #1276.

No branch was deleted or force-updated by this inventory. Patch-ID comparisons
alone did not establish integration, and no claim is made that all remaining
source differences are safe to merge.

## Subsequent delivery — 6 November 2026

PR #1276 merged the bounded random-forest and NumPy identity witnesses as
`256c7a209d39ea69a65cf276fddb3b8129ebb96e`. The other branch observations above
remain the dated 5 November inventory; they do not establish current ref state or
authorize deletion of branches with additional changes.
