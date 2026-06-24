# Working Notes: ASIC Implementation

ASIC/custom-circuit capability is a separate implementation lane. It remains
the most conservative and should only proceed where the workload is regular
enough to preserve the same contract.

Current decision:
- Free CI-based pre-silicon evidence is acceptable first-pass progress for this
  track. The repo-owned path should run the deterministic fixed-point EVPI-style
  RTL kernel through Verilator, Yosys, and an OpenROAD/OpenLane/SKY130
  RTL-to-GDS evidence plan using GitHub Actions with Docker as the default
  runner.
- GitHub Codespaces and Google Cloud Shell are acceptable manual fallback
  runners when GitHub Actions debugging is not enough.
- Tiny Tapeout, SkyWater MPW, and fabricated-silicon runtime remain separate
  future external gates and must not be implied by pre-silicon artifacts.

Later action:
- Use Chrome only if a browser sign-in or portal workflow is needed for Tiny
  Tapeout, SkyWater MPW, Cloud Shell, or another external shuttle portal. Pause
  for user action before any login, upload, or submission.
- Resume fabricated-silicon evidence only when shuttle access or real silicon is
  available for reproducible CPU/ASIC comparison evidence.
