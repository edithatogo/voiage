# HPC source-native build readiness

The retained HPC packaging strategy builds the version 2.0.0 Python package
from immutable source commit `5e92151fc87afefbb411c992fb9f82fc4b8c049f`.
The Spack and EasyBuild recipes build the Rust-native Python extension with
Rust and maturin rather than using an unverified binary cache.

Both recipes retain a CPU-compatible path and validate the installed command
with `voiage --help`. They are local recipe evidence only: an upstream Spack or
EasyBuild pull request, review, merge, and any HPSF/E4S curation remain
external decisions.
