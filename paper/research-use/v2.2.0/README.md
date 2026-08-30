# Released-package research-workflow replay

On 2026-08-30, an agent-assisted replay of the historical same-author
HPV-vaccination workflow evaluated 500 draws with the public voiage 2.2.0
wheel. VOP and voiage ran in separate supported environments. This is
computational compatibility evidence, not new human research use, independent
adoption, clinical advice, or a validated policy estimate. It does not replace
or recertify the July record in `paper/joss-developer-research-use.json`.

## Environment and result

The VOP checkout was clean at
`2c46db2fe5f907d894bb07f1127c008f10ee462e`. Its frozen environment used
Python 3.12.13, pandas 3.0.3 and SciPy 1.18.0. The separate PyPI installation
used voiage 2.2.0, Python 3.12.13, pandas 2.3.3 and SciPy 1.16.3. Both passed
`uv pip check`; neither used `--no-deps`. Installed versions and observed
local paths are retained in `export.json` and `evaluation.json`.

VOP's current pandas and SciPy requirements are outside voiage's stable
bounds. The CSV handoff therefore demonstrates file-based use only, not a
supported combined installation. Neither process imports the other package.

The seed, perturbations, willingness to pay and sampling order match the
historical workflow. The CSV SHA-256 is
`509411114afe90617f0f60a7743ad11f1160fb1288bb4ef144b572e5294050d9`,
identical to the historical 500-row result. EVPI was 0.0 NZD per model cohort,
matching a separate NumPy calculation of the EVPI definition. This zero
applies only to the specified scenario and perturbations.

## Reproduction

Use separate environments: VOP's frozen lock at the pinned revision and a
fresh normal PyPI installation of `voiage==2.2.0`. Run outside the voiage
checkout. Substitute absolute paths for the placeholders and choose a new
output directory; the script refuses to overwrite existing evidence.

```console
/path/to/vop/.venv/bin/python -I /path/to/voiage/scripts/run_vop_research_handoff.py export --vop-root /path/to/vop --output /path/to/new-run/export
shasum -a 256 /path/to/new-run/export/export.json
/path/to/consumer/.venv/bin/python -I /path/to/voiage/scripts/run_vop_research_handoff.py evaluate --export-receipt /path/to/new-run/export/export.json --export-sha256 <observed-export-sha256> --output /path/to/new-run/evaluation.json
```

The export receipt digest differs across machines because it includes the
observed environment. Pass the digest actually reviewed for that run; do not
substitute the archived receipt's digest. The consumer validates the receipt
and CSV hashes, pinned source and parameter identity, exact strategy columns,
draw count, finite values, installed voiage version and numerical reference.
The July one-environment script remains historical and is not the supported
v2.2.0 reproduction route.

The exact CSV is retained locally under
`.conductor/local/v2-2-research-handoff/hpv_vaccination_net_benefit.csv`.
It can also be regenerated from the pinned public VOP parameters. No new
venue communication or human attestation was produced by this replay.
