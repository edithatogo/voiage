# Innovate E02 binding schema review

The source packet `3be50601acb77ecea95600a57acaf7bb1467618f`, based on merged
PR #1340, strengthens the Voiage decision-binding adapter and adds focused
regressions plus a synthetic installed-consumer probe. The original source
packet and its private local artifact locations are retained in the originating
worktree and the integrator's verified recovery bundle.

## Review findings and corrections

The previous adapter accepted an unsupported binding version; non-positive or
non-finite population or horizon; an invalid discount rate; non-finite net
benefits; blank or non-string perspective and strategy labels; and non-string
units whose representation happened to contain `net_benefit`. It now rejects
those inputs. The population, horizon and discount bounds match the existing
`DecisionAnalysis._scale_to_population` contract. Valid zero and one discount
rates and positive fractional population and horizon remain supported.

The first 19 negative cases and the later 10 label and unit cases failed on the
original implementation and passed after the corresponding repair. The existing
independent two-strategy EVPI case is 2.5. Another test uses a hand-authored
Innovate diffusion fixture and an explicitly synthetic valuation, with an
independent EVPI of 1/600. Neither test validates an empirical fit.

## Historical installed-candidate evidence

The source owner reported the following checks on a local development-profile
candidate; they are not release or current-main qualification:

- Innovate 0.7.0a1 wheel SHA-256
  `4547a4ba432ad63b89cc6f7e661df0ec55b3af53377330a146660ccd86c8440b`,
  built from source commit `69b674814513e09a683279c679b1584af8bb4214`.
- Voiage 3.1.0 wheel SHA-256
  `0859daa9571396cf385771e955f7107a58971948289c550442aeb8a44b6591d8`,
  built from the selected base plus the adapter patch with Python 3.15.2,
  maturin 1.16.0 and Rust/Cargo 1.100.0.
- The runtime and test patch SHA-256 was
  `49eae1ca0452763102deec78a452d3805736347b5c4fa53babd522c1b45b3746`.
  The original probe SHA-256 was
  `98fc163bc47174c4c48639f225d9ce658e93fabcc04568bc4cdd94ba568ebd0d`.
  The [integrated probe](installed-adapters/e02-native-binding-probe-20261117.py)
  uses a single module import and has SHA-256
  `401f672be5310ec81dd4f0d230f2438ec40b52b91e1f517207f00b87716fba93`.
- From outside the source checkout, 47 focused installed-candidate tests and
  `uv pip check` passed; imports resolved to the candidate environment.
  Focused Ruff and format checks also passed.
- The probe verified imported wheel members, ran native Innovate Bass and
  Logistic scenarios, round-tripped labelled values through Arrow, and
  evaluated the paired binding. The analytic EVPI was `5.6417207199278705`,
  versus `5.641720719927868` observed; reversing one strategy's draw order was
  rejected. The Arrow IPC SHA-256 was
  `10647b18d03acef07ab1432bf78ebc132938f9596da77141c39ddb9abab69bd5`.

The application mapping assigns one synthetic utility per adoption, uses an
alternative value of 16 and assumes no costs. `fit_status` is assumed for this
forward-model fixture and is not fit validation. The integrator independently
verified both retained wheel hashes and the original probe hash; the original
installed run was not repeated for this handoff or the import-only revision.

## Remaining gates

At the original checkpoint the free-space guard stopped before launching the
full tox gate. Current-source local and hosted verification is recorded with
the integrating PR, separately from the historical candidate evidence above.

The source DecisionBinding v1 record does not require a structured valuation
mapping, fit provenance, or product, region and time keys. Checking a unit
label alone cannot establish that raw counts were converted to valid utility.
Cross-repository contract, producer-export and empirical acceptance remain
open; this packet does not mark E02 complete.

Innovate E04 also remains open. At VOP main `7778459f1798e3b6631a285b292a3153ee233333`,
its Python, PyArrow and Polars constraints are incompatible with the selected
3.15.2/Arrow 26/Polars 2 stack. VOP PRs #81 and #83 had failed checks at the
original checkpoint. The earlier published VOP scenario receipt does not prove
current-source consumer migration or the independent perspective, regret and
equity acceptance.
