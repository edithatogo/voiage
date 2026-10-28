# Current-runtime overlay candidates

These are the only current-runtime recipe declarations. They target Python
3.15.1 and the canonical core policy. The Spack development recipe pins the
immutable migrated source candidate; the EasyBuild recipe requires a matching
qualified source archive and SHA-256 before it can be evaluated. It has no
source-download or dependency-resolution fallback.

No native solver, catalogue, toolchain, build or installed-module acceptance is
claimed. `runtime.json` retains the three native gates as false. Source-only
qualification is separately true and bound to the checksum of its accepted
receipt and archive. The declaration checker authenticates that receipt; it
does not run a native solver or build. Do not substitute the published older
sdist or treat the local candidate as a new release.

Before using EasyBuild, run the source qualifier against the actual supplied
archive on the selected Python 3.15.1 interpreter, with this repository retaining
the immutable source revision:

```sh
.venv/bin/python scripts/guard_verification.py --work-dir .conductor/local/hpc-source -- \
  .venv/bin/python scripts/qualify_hpc_source.py /path/to/voiage-2.2.0.tar.gz \
  --revision 3df3cb49e5c927075c0609a48a2535b4457424d4 \
  --sha256 3748d8c5d155dc90db2b2c9c35abac5bb27958cb34c747e60e1ef8f84212a989
```

Only after that succeeds, supply the same archive and hash through
`VOIAGE_QUALIFIED_SOURCE_ARCHIVE` and `VOIAGE_QUALIFIED_SOURCE_SHA256`. Keep the
qualifier output with the native execution receipt. The Spack candidate uses
the same immutable source revision. Full Linux builds and the required foss
compiler generations remain separate acceptance gates.

The older source-bound recipes, provider maps, licences, patches and logs remain
at their original paths for historical validation only. `overlay-lifecycle.json`
pins every retained byte. Their older runtimes and package sets describe those
sources; they are not current installation instructions. The old HPC smoke and
native-qualification helpers are historical replay tools, not current defaults.
Read-only historical validators remain available to audit their retained evidence.
