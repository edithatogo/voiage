# Working Notes: FPGA Implementation

FPGA is not currently implemented in the repository. This track makes that work
explicit so it can be planned and reviewed separately from ASIC work.

Current decision:
- Free CI-based pre-silicon evidence is acceptable first-pass progress for this
  track. The repo-owned path should run a deterministic fixed-point EVPI-style
  RTL kernel through Verilator, Yosys, and nextpnr using GitHub Actions with OSS
  CAD Suite as the default runner.
- GitHub Codespaces and Google Cloud Shell are acceptable manual fallback
  runners when GitHub Actions debugging is not enough.
- physical FPGA board runtime remains a separate future external gate and must
  not be implied by pre-silicon artifacts.

Later action:
- Use Chrome only if a browser sign-in or portal workflow is needed for an
  external FPGA service or cloud console. Pause for user action before any login,
  upload, or submission.
- Resume physical-board evidence only when a board or hosted FPGA runtime is
  available for reproducible CPU/FPGA comparison evidence.
