# Current-runtime overlay candidates

These are the only current-runtime recipe declarations. They target Python
3.15.1 and the canonical core policy. The Spack development recipe pins the
immutable migrated source candidate; the EasyBuild recipe requires a matching
qualified source archive and SHA-256 before it can be evaluated. It has no
source-download or dependency-resolution fallback.

No native solver, catalogue, toolchain, build or installed-module acceptance is
claimed. `runtime.json` retains those gates as false. Before using EasyBuild,
verify the supplied archive's source-identity metadata against `source_commit`,
check its project metadata against the runtime policy, and retain the archive
hash with the native receipt. Do not substitute the published older sdist.

The older source-bound recipes, provider maps, licences, patches and logs remain
at their original paths for historical validation only. `overlay-lifecycle.json`
pins every retained byte. Their older runtimes and package sets describe those
sources; they are not current installation instructions. The old HPC smoke and
native-qualification helpers are historical replay tools, not current defaults.
Read-only historical validators remain available to audit their retained evidence.
