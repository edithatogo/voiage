CLI Reference
=============

The `voiage` command-line interface exposes calculation, plotting, and
configuration helpers for the core VOI workflows.

Representative commands:

- `voiage calculate-evpi`
- `voiage calculate-evppi`
- `voiage calculate-evsi`
- `voiage calculate-enbs`
- `voiage calculate-structural-evpi`
- `voiage calculate-structural-evppi`
- `voiage calculate-perspective`
- `voiage calculate-distributional-equity`
- `voiage calculate-implementation`
- `voiage calculate-dynamic-real-options`
- `voiage plot-ceac`
- `voiage plot-ceaf`
- `voiage plot-voi-curves`
- `voiage plot-dominance`

Sample inputs:

- `examples/cli_example.py`
- `examples/evpi_validation.ipynb`
- `examples/evppi_validation.ipynb`
- `examples/evsi_validation.ipynb`
- `examples/cli_samples/evpi_net_benefit.csv`
- `examples/cli_samples/evppi_parameters.csv`
- `examples/cli_samples/evsi_trial_design.json`

Common errors:

- Missing input files usually mean the CLI path arguments need to point at the
  repo-local examples or the generated CSV/JSON artifact.
- Invalid JSON input is rejected early by the CLI invariants tests.
- If a command reports a backend or approximation warning, the payload should
  include the method metadata and diagnostics fields described in
  `specs/core-api/`.

See also:

- `docs/user_guide/features/cli.md`
- `CONTRIBUTING.md`
