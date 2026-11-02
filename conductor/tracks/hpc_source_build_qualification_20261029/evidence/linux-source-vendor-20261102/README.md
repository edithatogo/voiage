# Linux source/vendor checkpoint — 2 November 2026

[Source/vendor run 41394052644](https://github.com/edithatogo/voiage/actions/runs/41394052644)
passed all 11 jobs: 14 negative controls, nine Cargo source cells and the
independent EasyBuild Git-archive cell. The reviewed source revision is
`f44fda7cc56d43c1de7510c7cd6170c5e4e0103a`. GitHub tested its synthetic merge
checkout `2e18d751a16b235f21bba47ca3c83bd534cc55be`; its Git tree was verified
identical to that revision. The manifest preserves both identities.

[Graph run 41393252994](https://github.com/edithatogo/voiage/actions/runs/41393252994)
passed Spack and both EasyBuild generations. Its package inputs, graph helpers
and workflow Git objects are unchanged in the source/vendor candidate. The
manifest separately binds that run, its checkout and all graph artifacts.

`manifest.json` hashes the retained receipts, exact input Git objects, original
bytes and any stored gzip representation. `final/` contains per-cell source,
lock and vendor identities, offline metadata, SDK identities and command logs.
`graphs/` contains the dependency graphs and terminal resource receipts.
`history/` retains the upstream Polars lock failure, EasyBuild Python selection
failure, Git checksum discovery gate and compiler identity-control regression.
`local/` preserves focused verification, hosted check/artifact readbacks and the
failed local tox disk-space preflight. No stopped or failed gate is a pass.

Polars' qualified workspace has 456 registry crates and three pinned Git crates.
The source compiler closures contain 961 stable and 941 nightly registry crates.
The stable source/vendor stage took 118.339 seconds, peaking at 403,592 KiB
sampled RSS. Nightly took 130.380 seconds, peaking at 399,312 KiB. These are source
and vendor measurements, not Rust/LLVM compilation costs. Sampling and disk
limitations are retained in the original resource reports.

All qualified Cargo cells replay all-feature metadata with `--locked --offline`,
an empty Cargo home and blocked proxy. Bootstrap SDKs are authenticated input.
No source-built compiler, library install, full HPC module or installed
scientific consumer is accepted by this packet. Those native gates stay open.
