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
configuration. It runs the EasyBuild style check only when a modules tool is
available, so a workstation configuration cannot be mistaken for recipe proof.
On this macOS host, Environment Modules 5.6 is installed locally, but the
available EasyBuild module adapter cannot parse its version interface. Treat a
successful EasyBuild run as pending until an adapter-compatible environment is
available.
