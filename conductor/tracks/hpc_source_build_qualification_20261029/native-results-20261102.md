# Baseline native compiler results — 2 November 2026

Both cold Linux x86-64 profiles passed at source `24ae6cb6` on GitHub-hosted
Ubuntu 24.04, using actual Python 3.15.1. This accepts the bounded compiler
checkpoint, not the full HPC installed matrix.

| Role | Installed Rust | Source LLVM | Driver minutes | Peak sampled descendant RSS (GiB) | Minimum free disk (GiB) |
| --- | --- | --- | ---: | ---: | ---: |
| Stable | 1.99.1 | 23.1.1 | 137.3 | 5.81 | 56.58 |
| Nightly 2026-09-01 | 1.100.0-nightly | 23.1.0 | 138.3 | 6.06 | 57.60 |

- Stable: [run 41436471133](https://github.com/edithatogo/voiage/actions/runs/41436471133).
- Dated nightly: [run 41436475314](https://github.com/edithatogo/voiage/actions/runs/41436475314).
- [Receipt manifest](evidence/native-rust-llvm-20261102/manifest.json).

Both completed all 12 stages with owned-process cleanup. LLVM was built from
source with all 15 default backends and four experimental targets. Stage2 Rust,
std, Cargo and rustdoc were source-built and installed. Each installed Cargo
compiled and ran the deterministic `EVPI=5` fixture using its installed rustc;
`ldd` reported resolved linkage. Upstream lock hashes stayed unchanged.

The stable bootstrap exception is the authenticated Rust 1.98.0 SDK; nightly
uses authenticated 1.99.0-beta.1. The dated 1.100.0 binary SDK is a metadata/vendor
helper only. These SDKs are never accepted as source-built compiler outputs.
Compilation and installed probes ran unprivileged in a network namespace after
authenticated prefetch. Actual egress-denial controls and all command logs remain.

Artifact ZIP digests matched GitHub's SHA-256 values before extraction. Retained
files include stored and original SHA-256 hashes; large logs/metadata are
losslessly compressed. Failed attempts and 51 Linux controls remain alongside
successful runs. The final candidate reuses the ten exact native input objects.

RSS is a two-second sum that double-counts shared pages and can miss peaks.
GNU time retains cumulative command totals and a single-process maximum; it is
not the sampled combined RSS. Optional I/O permission gaps are explicit. Free
space includes unrelated runner activity, and directory snapshots are not peaks.
These are observed capacities for these runs, not guaranteed runner capacity.

The earlier full local tox/affected rerun evidence covers unchanged runtime
inputs. The later full repeat stopped at its disk-space preflight; 24 focused
source/compiler tests and current hosted gates cover the qualification fixes.
No binary toolchain archive, installed Voiage stack, full Spack/EasyBuild cell,
generation-isolation proof, cgroup peak or hotspot attribution is claimed.
Those downstream and P4/P5 requirements remain open.
