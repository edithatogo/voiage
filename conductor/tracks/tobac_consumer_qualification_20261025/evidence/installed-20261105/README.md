# Explicit installed consumer qualification — 5 November 2026

The frozen tobac source revision `b57b4e12e551c7c7d6db757b8d0f404eb8691146`
(tree `9f116d5669b8d63d3a1992790503ecf77abc3410`) produced an independently
installed `tobac-regulatory-research` 0.2.0 wheel with SHA-256
`982b7346a5f3d9eeb125b503eaa74feeeab984b9575fa3583ddbe9f3c70689b0`;
42 installed members matched the wheel, excluding the rewritten RECORD.

The published voiage 3.1.0 macOS arm64 wheel was downloaded from PyPI and
checked against its registry SHA-256
`ca1a625ea2e0f6c63ffd306f19cb62c1c69242643164553d5ca633e94146eb5b`.
The consumer verifier checked 198 installed members and the native extension,
then executed 90 comparisons with maximum absolute difference
`2.842170943040401e-14` and six buffer ownership cases. Python was 3.15.1;
pandas and xarray were absent in the isolated environment. The exact
[receipt](receipt.json) has SHA-256
`3a6ae7dd93090f7589125a701778d19c328f70edcc326b50732eeb051c10f0ed`.

This qualifies the explicit `--expected-version 3.1.0` consumer pathway for
its tested cases. The default adapter still requires `2.2.0+tobac3151.2` and
was not promoted; other consumer contracts, scientific calibration and track
closeout remain separate. The relative interpreter invocation produced prefix
warnings; the process returned zero and the verifier's installed-member and
native-extension identity checks passed.
