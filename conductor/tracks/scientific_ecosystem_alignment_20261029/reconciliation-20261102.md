# Release and qualification reconciliation — 2 November 2026

Innovate 0.6.0 and Voiage 3.0.0 are published on PyPI. A fresh macOS arm64
Python 3.15.1 environment installed both from PyPI with PyArrow 26.0.0.
All 59 installed packages passed the dependency check. Innovate Bass results
survived Arrow IPC with SHA-256
9ac859fb8e887dafe9a3131f1866898842b3ebb134e8f997cc97e342c8ee4b7b.
Installed Voiage EVPI was 5.641720730739724 against reference
5.641720730739725. This qualifies the documented two-scenario CPU case.

Release sources and delivery:
- Innovate release PR https://github.com/edithatogo/innovate/pull/566
- Voiage release PR https://github.com/edithatogo/voiage/pull/1254
- Voiage release compiler repair https://github.com/edithatogo/voiage/pull/1255
- Voiage attested publication https://github.com/edithatogo/voiage/actions/runs/41305487966

Authentication-only runs from main succeeded, discarded minted tokens and
skipped publication:
- Innovate PyPI https://github.com/edithatogo/innovate/actions/runs/41308417130
- Innovate TestPyPI https://github.com/edithatogo/innovate/actions/runs/41311052488
- Voiage both indexes https://github.com/edithatogo/voiage/actions/runs/41309109201

The earlier missing Innovate Arrow wheel/dependency-range and historical public
Voiage installation blockers are resolved for these releases. Historical receipts
retain their original versions. VOP's own Python constraint, optional adapter
migrations, broader scientific scenarios and native HPC acceptance remain open.

## Available compute and next task

Available scope is GitHub Actions, user-provided Colab access and this Mac.
Next implement a bounded GitHub Linux graph-qualification job: freeze the Spack
and two EasyBuild generation inputs, concretize/resolve each unchanged root,
retain terminal graph and resource receipts, and classify failures before builds.
This advances HPC P1/P2 without claiming cluster or scheduler acceptance.

GitHub Linux runners can qualify graph resolution, selected source builds,
installed CPU consumers and generation isolation within runner time, RAM and
storage limits. Longer compiler/full-graph cells need explicit measured budgets
and resumable artifacts. Colab can provide additional bounded Linux experiments
after its installed CLI/authentication/runtime contract is qualified; its session
lifetime and variable capacity require terminal receipts. This Mac supplies arm64
builds and independent CPU/Arrow checks. Scheduler, MPI multi-node and target
cluster performance acceptance need a future matching environment.

On 2 November this Mac reported 32 GiB RAM and only about 300 MiB free on the
system volume. Recover capacity through a separately scoped, verified cleanup
before dispatching local source builds. No Colab executable was found on PATH or
in the inspected standard CLI directories; this is a discovery gap, not evidence
that Colab is unavailable.

## Remaining acceptance

Keep the HPC, tobac, optional telemetry and venue gates open. The source and
installed-candidate evidence already delivered must be reused. Do not infer full
track completion from the public-package smoke or authentication checks.
