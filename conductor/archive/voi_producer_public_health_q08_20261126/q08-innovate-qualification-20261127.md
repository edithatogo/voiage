# Q08.2 Installed `innovate` Producer Qualification Receipt (27 November 2026)

- **Producer:** `innovate` `0.7.0a1` (`/Volumes/PortableSSD/codex-alert-44-tmp-01a1e448/wheel-installed/innovate/__init__.py`, `_native.cpython-315-darwin.so`, CPython `3.15.2` arm64)
- **Fixture:** `/Volumes/PortableSSD/GitHub/innovate/specs/ecosystem/voiage/uncertainty/diffusion_v1/{parameter_draws.csv,adoption_trajectories.csv}`
- **Workflow Executed:**
  1. Validated paired Bass trajectories via `innovate.paired_outcomes({"strategies": strategies})` (`24` rows, `time_unit="period"`, `provenance="voiage_diffusion_uncertainty_v1"`).
  2. Pivoted final-period adoption per `(draw_id, scenario_id)` into a `3 x 2` `float64` matrix and evaluated `voiage.methods.basic.evpi(nb)` -> `evpi=0.000000` (`exit 0`).
  3. Negative unit/weight/timebase and malformed-pairing guards verified in `tests/test_q08_producer_public_health.py` and `innovate/tests/native/test_paired_outcomes.py`.
