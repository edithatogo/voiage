# Scientific ecosystem qualification, 1 November 2026

This candidate extends the HPC requirement review at `ef535f31f` on the
producer baseline `7516cb612`. The scientific packages retain independent
release identities. The repository integrator owns main, compiler promotion and
the combined Linux solve. The capacity owner retains native measurement work.

## Compiler and capacity

Voiage's stable HPC toolchain is 1.99.1. Rust's source archive and embedded
stage0 record bind the 1.98.0 bootstrap manifest. The full Linux x86-64 SDK
and the rustc, rust-std and Cargo archives matched their manifest digests.
The current Spack root and EasyBuild compiler selections are exact. The host
Spack 1.2.2 class probe used the pinned builtin catalogue at `fa170d11` and
resolved the active provider declarations to `rust@=1.99.1` and
`rust-bootstrap@=1.98.0`. The library MSRV is 1.85. Polars 2.1.0 still
requires nightly-2026-09-01 with its August beta bootstrap.

Three focused regressions failed before the provider change. After review, 69
provider and overlay tests passed, including source and SDK URL/hash checks for
both the retained 1.96.0 profile and the 1.99.1 candidate. Python 3.15.1
strict dependency-frontier verification passed against an isolated upgraded
lock. The active Python lock is unchanged.

The SDK-backed [Polars pilot](https://github.com/edithatogo/voiage/actions/runs/40691333092)
and [Arrow pilot](https://github.com/edithatogo/voiage/actions/runs/40718827238)
succeeded. Their inputs and measurements describe those component builds.
The combined Linux Spack graph, EasyBuild foss/2023a and foss/2024a,
source-built Rust and LLVM, and workload-specific capacity acceptance are open.

## Consumer evidence

The [tobac installed packet](https://github.com/edithatogo/voiage/pull/1252)
at `99f3d317` records 90 finite-state comparisons, 24 exact near-tie cases
per tested CPU backend, and the default producer pin boundary. That open PR
uses producer `7516cb612`. It does not demonstrate downstream deployment of
the new HPC compiler. The previous [VOP pilot](https://github.com/edithatogo/voiage/pull/1158)
provides functional and replay evidence for its original revisions.

## Native tests and installed wheel

On macOS arm64, Rust 1.99.1 (LLVM 23.1.1) passed 269 locked workspace tests
with all features. Rust 1.85.0 passed the locked all-targets, all-features
workspace check. Both commands used repo-local Python 3.15.1 and Rust tree
`f0425087d177b9ab48fe238c01a47c885fd74f91`. The guarded runs took 202.7 s and
94.3 s respectively. The 1.99.1 SDK wheel build at `89ab5f5ac` took 70.0 s
and reached 1,328,592 KiB sampled peak process-group RSS. Cargo's compiler
fingerprint reported `rustc 1.99.1 (58cf8f9c3 2026-10-13)`.
These measurements describe this host and build profile.

A fresh Python 3.15.1 environment loaded the installed wheel, whose SHA-256
is `8f46b93023c35962e7848c5e49215156395de9338aa7b0168119248e1eeaf231`.
The installed tobac oracle verified 837 archive members, passed 90
comparisons (largest absolute difference 2.842170943040401e-14) and passed
six buffer-ownership cases. This uses tobac's explicit candidate selection.

Fresh Innovate Rust Bass predictions from six draw/scenario pairs produced
24 trajectory points. Installed Voiage returned EVPI 0.6590049776341118,
within 1e-10 of the independent decimal calculation. A separate frozen
diffusion fixture yielded 1/600. The installed MARS Earth estimator yielded
EVPPI 0.5151515151515149 against 17/33. A 500-draw VOP source export produced
the published reference matrix digest and zero EVPI on the installed candidate.

The Innovate Python package's pinned dependency range has no compatible macOS
arm64 Python 3.15.1 binary Arrow wheel. VOP's Python package declares a
maximum Python version below 3.15. Its historical public Voiage wheel also
failed dependency installation on this host when the Arrow 25 source build
could not locate the required Arrow CMake package. The pinned source export
and installed candidate path are verified separately. The package owners
retain these dependency and release gates.

[Installed adapter replay and pinned inputs](installed-adapters/README.md) · [Verification receipts](native-verification-20261101.json)
