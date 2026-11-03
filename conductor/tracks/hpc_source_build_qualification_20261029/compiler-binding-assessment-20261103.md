# Compiler binding assessment — 3 November 2026

This is a source-level assessment against main `602f2102a`; it does not qualify
an installed package-manager compiler or promote the historical HPC candidate.
The current lane is chat `01a1a104-80d7-7501-86bd-85be63cb2cab`, managed
worktree `/Users/doughnut/.codex/worktrees/hpc-arrow-qualification/voiage`,
branch `codex/hpc-arrow-qualification-20261103`. Its initial ownership is this
assessment and a handoff entry. It owns no shared native environment or running
operation. Earlier assignments remain historical. Next operation: qualify the
actual resolved source paths and offline package-manager execution before any
new build; then capture each package manager's effective compiler configuration.

The graph and source/vendor checkpoints in PRs #1260 and #1261 and the baseline
source-compiler checkpoint in PR #1262 are distinct. The latter built and tested
stable and nightly compiler outputs but did not retain their binaries. Its
measurements cannot be treated as installed Spack or EasyBuild compiler outputs.

| Input | Baseline native driver | Current package-manager declarations |
| --- | --- | --- |
| Platform and stages | Linux x86-64 host/target; explicitly builds LLVM and stage 2, installs stage 2 | Spack invokes `x.py build` and `x.py install` without an explicit stage in the recipe. Both EasyBuild generations use `EB_Rust`; its effective commands must be captured on the selected Linux version. |
| Host C toolchain | Hosted Ubuntu baseline | EasyBuild selects GCCcore 12.3.0 and 13.3.0 respectively. Spack's concrete selected compiler must be read from the qualified graph and actual execution. These are not assumed equivalent. |
| Source and bootstrap | Stable 1.99.1 with 1.98.0 SDK; dated nightly with beta SDK, authenticated in the existing receipts | Current recipes pin the same Rust roles and source identities; actual resolved archives and installed compiler provenance still require checks for each execution. |
| Download and vendor | Disables downloaded LLVM and rustc; vendor and locked dependencies enabled | Spack explicitly disables LLVM/rustc downloads and enables vendor. EasyBuild easyconfigs explicitly disable downloaded rustc and enable vendor; effective LLVM and network controls must be verified in `EB_Rust`, not inferred from those fields. |
| Features and resources | Explicit optimized build, no debug info, selected tools, four jobs and one LLVM link job | Spack selects tools from variants and `make_jobs`; EasyBuild settings depend on its easyblock and configuration. Effective flags, backend set, job limits and installed linkage are not yet bound to the baseline receipt. |

The Arrow 26.0.0 SDK-backed component pilot is separately retained in the
capacity plan. It proves its stated source build and installed probes, not the
Spack or EasyBuild Arrow cells or use of the source-built compilers. The current
package-manager dependency graphs use the historical qualified Voiage source
candidate; main has since released newer Voiage versions. Do not silently swap
that candidate or infer current-release acceptance from the old graph. Freeze
and qualify a new source binding before making a current-release claim.

For each matrix cell, capture the exact package-manager version, graph and source
identities, resolved archive paths and hashes, offline-denial controls, effective
Rust/LLVM configuration, compiler and C toolchain versions, stage and install
commands, process and storage envelope, and installed compiler/link probes.
Reusing a compiler requires exact output and provenance identity. Where settings
or toolchains differ, build and verify that configuration separately. Preserve
all failures and keep P2–P5 open until their corresponding witnesses pass.
