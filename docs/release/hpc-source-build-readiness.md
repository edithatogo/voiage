# HPC source-native build readiness

The retained HPC packaging strategy builds the version 2.0.0 Python package
from immutable source commit `5e92151fc87afefbb411c992fb9f82fc4b8c049f`.
The Spack and EasyBuild recipes build the Rust-native Python extension with
Rust and maturin rather than using an unverified binary cache.

Both recipes retain a CPU-compatible path and validate the installed command
with `voiage --help`. They are local recipe evidence only: an upstream Spack or
EasyBuild pull request, review, merge, and any HPSF/E4S curation remain
external decisions.

Run `scripts/validate_hpc_recipes.sh` to solve the Spack recipe in an isolated
configuration and perform the EasyBuild style check. The script prefers Lmod
when it is installed, and uses ephemeral EasyBuild framework, easyblocks, and
style-check dependencies, so Homebrew's Environment Modules version interface
cannot be mistaken for an EasyBuild-compatible validation environment.

On this macOS host, the Lmod-backed EasyBuild 5.3.1 style check passed for the
retained recipe. The warnings about an unavailable `foss/2023a` hierarchy are
expected for a standalone syntax check without the upstream easyconfig tree;
they do not establish a build. A complete toolchain build, upstream pull
request, review, merge, and any HPSF/E4S curation remain external decisions.
