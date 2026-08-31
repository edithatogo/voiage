# HPC source-native build readiness

The current Spack and EasyBuild candidates target the version 2.2.0 Python
package from its checksum-pinned PyPI source distribution. They declare Rust
and Maturin to build the native Python extension. See the
[distribution handoff](hpc-distribution-handoff.md) for the source identity,
candidate dependency stacks and recorded validation limits.

Both recipe families describe a CPU path and include an installed-command
check with `voiage --help`. They are local candidates only: an upstream Spack or
EasyBuild pull request, review, merge, and any HPSF/E4S curation remain
external decisions.

Run `bash scripts/validate_hpc_recipes.sh --spec` to attempt Spack
concretization in an isolated configuration, followed by EasyBuild style
checks and robot dry runs. This mode fails if any preceding step fails,
including missing dependency recipes or a modules tool; it does not silently
skip those checks. A failed Spack solve prevents the later EasyBuild checks
from running.

The default command, or explicit `--syntax`, only parses recipe syntax.
Neither syntax nor a successful spec proves a package build. The explicit
`--build` mode additionally requests real builds on a prepared host. The
recorded source-wheel smoke passed on macOS, while the Spack dependency graph
remains blocked and no Linux HPC build or module-load result is claimed.
