# Installed scientific adapters, 14 November 2026

The macOS arm64 checks used the published Voiage 3.1.0 wheel. Python 3.15.1
ran the released-package checks and the MARS source wheel. An isolated Python
3.15.2 environment qualified current Innovate and tobac sources. The
[receipt](installed-adapters-20261114.json) binds wheel bytes, source revisions,
reference results and the exact qualification scripts.

| Path | Verified result |
| --- | --- |
| Innovate 0.6.0 release | Two Bass scenarios round-tripped through Arrow 26.0.1. EVPI was 5.641720730739724 versus 5.641720730739725. Ordinary dependency resolution built the required statsmodels 0.14.6 and ruptures 1.1.9 from source after the wheel-only attempt failed. |
| Innovate 0.7.0a1 source | A locked Rust 1.99.1 build produced an installed native wheel. Bass and Logistic outputs survived the Polars/Arrow handoff. Equal-weight EVPI was 5.641720719927868. The 3-to-1 empirical case gave 3.731714419642895. Both matched the analytic references within the declared tolerance. |
| MARS 1.0.4 release and source | The installed Earth estimator returned 0.5151515151515149 against 17/33 using both wheels. The published wheel is pure Python. A separate locked source build supplied the compiled predictor and replayed 33 points with maximum error 4.44e-16. |
| tobac 0.2.0 source | The installed consumer passed 90 uniform, weighted and partition comparisons and six input-buffer ownership cases. Maximum absolute error was 2.84e-14. The default adapter accepted installed Voiage 3.1.0 and returned the exact 0.75 weighted control. An explicit mismatched version was rejected. |
| VOP 0.1.8 release | The installed exporter regenerated the frozen 500-draw HPV scenario CSV byte for byte. A separate Voiage environment verified the receipt and returned EVPI zero, as the same option wins every draw. A switching-decision control returned 5, and altered CSV bytes were rejected. |

The retained Innovate fixture also returned 1/600 after reversing draw and
strategy order. The probe rejected an altered fixture manifest and an import
outside the verified wheel. Dependency checks passed in the tested environments.
The guarded Innovate source build finished in 303.7 seconds with 1,799,824 KiB
sampled peak process-group RSS. MARS Cargo compilation took 16.54 seconds.

## Replay

Use the archived PyPI wheels identified in the receipt and verify their SHA-256
digests before installation. Use `uv pip install --python "$PY" <wheel>` in
isolated environments and `uv pip check --python "$PY"` after resolution.
Run the Innovate and tobac scripts from their receipt-pinned source revisions
with `python -I`. Pass `--wheel "$INNOVATE_WHEEL"` to current Innovate's script,
and `--wheel "$VOIAGE_WHEEL" --expected-version 3.1.0 --buffer-ownership --out
receipt.json` to tobac's. Build the native wheels from the named source
revisions with maturin 1.16.0, Rust 1.99.1 and Cargo's `--locked` option.
For the MARS compiled replay, use [native_mars.py](installed-adapters/native_mars.py).
On Python 3.15.1, load `verify_wheel` from
[wheel_identity.py](installed-adapters/wheel_identity.py) and pass the version
and digest from this receipt. Then call `mars()` and `innovate(fixture)`
from [probe.py](installed-adapters/probe.py) for the finite estimator and
frozen diffusion checks. The historical command line pins Voiage 2.2.0.

For VOP, use the published 0.1.8 wheel and PyYAML 6.0.3 in the producer
environment, and the published Voiage 3.1.0 wheel in the consumer environment.
Use `generate_draws` from [run_vop_research_handoff.py](../../../scripts/run_vop_research_handoff.py)
with the installed `vop_poc_nz.cea_model_core.run_cea` and the
`hpv_vaccination` entry in the wheel's `parameters.yaml`. Use 500 draws and
seed 20260727. Write the two strategy columns in the order specified by
`HEADERS` with Python's CSV writer. Compare the output SHA-256 with the
[frozen export receipt](../../../paper/research-use/v2.2.0/export.json).
In the consumer environment, place the regenerated CSV beside a copy of
that receipt. Read it with `load_handoff` and the recorded receipt SHA-256,
then call `voiage.methods.basic.evpi` on the verified matrix. The dated
research-use receipt describes the original run. The receipt for this
qualification identifies the fresh producer and consumer.

## Remaining acceptance

- Current VOP source declares Python below 3.15, Arrow 25 and Polars 1.x.
  Pip rejected the exact source wheel on Python 3.15.1. Independent resolution
  against Voiage 3.1.0 failed for both Arrow and Polars. The published 0.1.8
  export provides a reproducible file handoff for the frozen scenario.
  Upstream [Arrow](https://github.com/edithatogo/vop_poc_nz/pull/83) and
  [Polars](https://github.com/edithatogo/vop_poc_nz/pull/81) candidates are
  open with failing checks. Current-source adoption requires upstream Python
  3.15 packaging and installed numerical qualification with both libraries.
- The MARS compiled replay used a locally built wheel. Upstream delivery of
  the native/Python package split remains open. The Innovate source test
  covers one synthetic scenario grid on this platform. The wider tobac
  scientific and interoperability gates remain in its own track.
  [MARS PR #235](https://github.com/edithatogo/mars/pull/235).
  [tobac consumer track](../tobac_consumer_qualification_20261025/index.md).
- Repository integration is owned by the
  [programme's Voiage integrator](https://github.com/edithatogo/repository-standards/blob/7874f05132447c77f681fbca7ad5394066564aaf/conductor/tracks/scientific_ecosystem_alignment_20261029/handoff.md).
  Spack and EasyBuild installed acceptance and the wider HPC matrix remain
  deferred under the [5 November decision](../../reconciliation-20261105.md).
