# Installed consumer observations — 29 October 2026

These are bounded local observations, not acceptance of the track, release
promotion, scientific calibration, or a replacement for the required gates.
The tested Voiage source tree is `b65109e087c0c18e91db5c7c921cc5b54f0540c3`,
which matches main commit `04f2e1beb05a01f5ff12b070f5f10f97673527d2`.
The installed wheel was built from clean checkout commit
`6ca17b12c9fa809c98faccf20111598549411802`, which has that identical tree.
The installed tobac source was archived from commit
`62a47123aae4186514f37b58d665da1ecae91998`. Python was 3.15.1 on macOS arm64.
Wheel identity, installed paths, dependency versions and numerical probe details
are recorded in `identity-receipt.json` and `consumer-receipt.json`. Local
paths in those receipts identify the original evidence environment, not a
portable location or a claim that it remains available.

The existing tobac qualification script passed 90 numerical comparisons
against its exact finite-state oracle and six buffer-ownership cases. Its
maximum absolute difference was `2.842170943040401e-14`. The environment
also contained separately installed legacy tabular libraries for that consumer probe;
their presence does not show that they are required by the Voiage base wheel.

The retained probe scripts ending in `.py.txt` are copies of the exact local
experiments, not pytest collection targets. `portfolio-receipt.json` records
two explicit correlated-report cases with shared costs and reversal of states
and sources. `scenario-receipt.json` records two real tobac regulatory
configuration scenarios, repeated in reverse order, with a parameter round
trip and input immutability. `interchange-receipt.json` records a correlated
likelihood and metadata transported through Arrow and a Polars sidecar, then
conditioned and evaluated by the explicitly selected installed candidate.
Their calculations use installed packages, and all input examples remain
synthetic or explicitly conditional; no empirical validity is inferred.

Six selected tobac parameter, scenario, no-I/O and replay tests passed in the
isolated environment. The full selected `test_evidence_interchange.py` run
had **35 passes and one failure**: the real-producer case calls tobac's
default adapter, which intentionally requires `2.2.0+tobac3151.2`, while this
candidate reports `2.2.0`. The explicit-candidate probe passed using the
adapter's `expected_version` argument and separately verified that the default
still rejects it. This failure is retained as a version-promotion boundary;
the candidate has not replaced the reference producer.

The wheel build, installation and tests used `scripts/guard_verification.py`
with a 10 GiB free-space floor. Sampled peak RSS was 418608 KiB for the build,
256992 KiB for the existing consumer probe, 125072 KiB for the regulatory
scenario probe and 208576 KiB for the interchange test run. The guard samples
RSS and is not a hard allocation limit. The original reports and artifacts are
in `/Volumes/PortableSSD/voiage-qualification-20261029-main/`.

To replay, build the specified Voiage tree into a wheel with Python 3.15.1,
install that wheel and the exact archived tobac source as non-editable packages
into a separate environment, and verify their installed paths. Copy these
probe files into an external work directory after removing the `.txt` suffix;
copy the Voiage v2 normative input fixture there as `portfolio-template.json`
and the exact tobac archive as `tobac-source/`. Run each probe under the
resource guard. The scenario probe creates `regulatory-parameters.json` with
exclusive creation, so use a fresh directory for each run. Pin and record the
same dependencies and compare results and hashes before using new evidence.

Still unproven include general covariance/schema migration and rollback,
comprehensive scenario isolation and stochastic behavior, the full consumer
contracts, native/coverage/security and hosted acceptance, and the separately
published producer required by tobac's release gate.
