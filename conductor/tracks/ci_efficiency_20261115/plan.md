# Implementation plan

- [x] Inspect current main, open PR ownership, required checks and job timings.
  Baseline: `cee30c63bc8c4edfe8968544315d21065fb34142`.
  Existing focused workflow contracts: 52 passed under Python 3.15.1.
  Standalone harness with `python -S`: zero findings across 37 workflows.
- [x] Remove unrelated dependency setup and add PR-only supersession to the two
  remaining selected workflows. Preserve all validation and artifact controls.
- [~] Verify the selected environments, workflow contracts and required local
  gate; retain any capacity or environment blocker explicitly.
  The 86 selected workflow/security/shard tests pass under Python 3.15.1;
  the standard-library-only harness reports zero findings. The locked tox
  bootstrap installs no project or scientific runtime. The first full gate
  stopped before execution with exit 125: 125,771,776 bytes free against its
  unchanged 2 GiB minimum. An isolated volume is being assessed.
- [~] Independently review the final diff and publish one focused PR.
  Independent review found no material regression. A structural comparison
  confirms all 28 affected job definitions, triggers and retained action pins
  are unchanged. Runtime C15 and hosted qualification remain pending.
- [ ] Verify current-revision hosted checks and record execution/queue timings.

The final local gate uses the existing resource guard and
`tox run-parallel -p 2`. The shared workspace initially has less than 2 GiB
free; no other task's files or caches will be removed to make that gate run.
