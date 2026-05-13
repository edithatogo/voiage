# Working Notes: FPGA Implementation

FPGA is not currently implemented in the repository. This track makes that work
explicit so it can be planned and reviewed separately from ASIC work.

Current blocker:
- No FPGA runtime or toolchain integration exists in the codebase yet, so the
  execution lane remains evidence-gated and cannot be marked complete until a
  real backend is added.

Later action:
- Resume this track only when an FPGA toolchain and a device-backed execution
  environment are available for reproducible CPU/FPGA comparison evidence.
