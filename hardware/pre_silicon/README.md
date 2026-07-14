# Pre-Silicon FPGA/ASIC Evidence Harness

This directory contains the repository-owned hardware kernel used to produce
first-pass FPGA and ASIC evidence without requiring physical hardware.

## Scope

- `rtl/evpi_fixed_point_kernel.v` implements a deterministic fixed-point
  EVPI-style difference kernel.
- `tb/evpi_fixed_point_kernel_tb.v` checks the RTL against the committed CPU
  fixture values.
- `fixtures/evpi_fixed_point_fixture.json` is the CPU reference bundle used by
  the manifest generator.

The harness is intentionally pre-silicon only. It can support Verilator, Yosys,
nextpnr, OpenROAD, and OpenLane evidence packets, but it does not claim physical
FPGA board runtime or fabricated ASIC runtime.

## Free Runner Order

1. GitHub Actions on standard Ubuntu runners.
2. GitHub Codespaces for interactive debugging.
3. Google Cloud Shell for manual fallback runs.

Browser automation is only appropriate for external sign-in flows such as
Cloud Shell, Tiny Tapeout, or SkyWater MPW portals.
