# Current-runtime overlay candidates

These are the only current-runtime recipe declarations. They target Python
3.15.1 and the canonical core policy. The Spack development recipe pins the
immutable migrated source candidate; the EasyBuild recipe requires a matching
qualified source archive and SHA-256 before it can be evaluated. It has no
source-download or dependency-resolution fallback.

No native solver, catalogue, toolchain, build or installed-module acceptance is
claimed. `runtime.json` retains the three native gates as false. Source-only
qualification is separately true and bound to the checksum of its accepted
receipt, archive and exact runtime policy. The declaration checker authenticates
that receipt; it
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

The current Polars providers follow the split 2.1.0 distribution. The frontend
requires its exact native runtime; optional integrations remain optional. The
native source retains upstream full/nightly defaults and requires the dated
`nightly-2026-09-01` compiler. `polars-provider-sources.json` binds both upstream
archives and their licence metadata. Check the actual downloaded archives with:

```sh
.venv/bin/python scripts/guard_verification.py --work-dir .conductor/local/polars-providers -- \
  .venv/bin/python scripts/check_current_spack_providers.py /path/to/source-archives
```

A successful check proves source/declaration agreement. The dated compiler and
bootstrap source candidates are present; their actual compiler builds remain
unqualified. The unmodified full solve, native build and installed-module checks
remain required. No optional integrations or source features are silently
substituted to make the catalogue solve.

## Arrow source prerequisites

The current LibCST source prerequisite selects Python 3.15.1, its native parser,
the matching runtime dependency markers and upstream MIT/PSF notices. Check the
actual source archive using `.venv/bin/python -m scripts.check_current_arrow_providers`
with the selected repo-local Python 3.15.1 interpreter. The `cpp` profile additionally
requires `--builtin-arrow /path/to/pinned/catalogue/arrow/package.py`; it authenticates
the immutable Arrow C++ source and preserves its upstream allocator declaration.
These are source checks; the complete Arrow profile and native installed matrix
remain unqualified.

Use a private Spack `config:build_stage` directory for source staging. Do not use
`spack stage --path` for this multi-resource candidate: Spack 1.2.2 assigns that
same custom directory to the root and resource, preventing normal resource
placement. Keep checksums enabled. The allocator source is pinned as a Spack
resource and supplied to CMake through its documented source URL override; it
is not disabled to avoid the fetch.

The obsolete physical-space estimate has been removed from the EasyBuild helper
and receipt schema. The completed GitHub Polars source-build pilot measured
9.76 GiB of disk growth (approximately 10 GB):
https://github.com/edithatogo/voiage/actions/runs/40691333092.
Retain resource guards and measure the actual workload rather than imposing the
retired estimate. This completed pilot used a binary compiler SDK; it does not
by itself qualify every compiler-generation or installed-module matrix gate.
