# Full source-preflight review — 3 November 2026

The source-preflight packet retains installed acceptance as false. Agent review
identified and corrected four dispatch/integrity issues: sudo replacing the
Spack executable search path, Spack not hashing mirrored VCS archives, the
Spack script module path, and legitimate private stage-to-cache aliases.
Mirror/cache manifests now freeze authenticated online bytes before offline
replay, and staged archives bind those bytes to concrete DAG hashes.

Sixteen focused controls cover missing/changed/shadowed inputs, unsafe
ancestors and aliases, source/extension receipts and concrete DAG binding.
The real EasyBuild 5.4.2 probe on repo-local Python 3.15.1 accepted the correct
checksum with one receipt and rejected a wrong checksum (exit 42) with none.
The observed unresolved auxiliary config.guess is a hard failure. Ruff,
workflow validation, shell syntax and reference-policy checks pass.

Linux run 41640186214 passed the Spack cell with 145 staged source witnesses.
Both EasyBuild cells stopped at FlexiBLAS checksum mismatch. The existing
3.4.4 digest matches the upstream GitHub release asset; upstream GitLab package
237/file 511 metadata confirms the existing 3.3.1 digest. Source URL repairs
preserve both digests and software/toolchain versions. Run 41642667878 checks
the repaired endpoints; its terminal result remains required.

The latest dependency lock was resolved and strict frontier audit passed.
The exploratory lock/report are retained locally; the selected qualified lock
is unchanged, so this source correction does not promote dependency upgrades.

The local final gate stopped at its 8 GiB storage reserve during concurrent
filesystem consumption; it did not pass. Linux run 41641510378 passed fourteen
environments but exposed missing historical Git objects in coverage because
the added runner used a shallow checkout. Full history is now requested,
matching the existing authoritative coverage lane. Run 41643034180 retains
the corrected full-gate attempt: all fifteen environments passed on Python
3.15.1, with 5,826 passed, 19 skipped and 95.54% coverage. Its frozen source
is `8fc333d82`; subsequent source-hook changes have separate affected checks.
No failed checks or coverage gates are skipped.

Complete source fetching is a prerequisite, not a compiler/install receipt.
Bootstrap/vendor integration, effective per-generation compiler configuration,
full source builds, installed scientific probes and isolation remain open.

The next EasyBuild run passed both FlexiBLAS source checks, then rejected the
checksum-pinned Polars lock patch outside the cache roots. That exact declared
file is now allowed only with its source-vendor manifest digest. Run 41643861789
then exposed an unavailable GNU auxiliary download in 2023a; a fixed-revision
mirror matches EasyBuild's existing config.guess SHA-256. A real ConfigureMake
fetch accepts both source and staged helper without a download. Run 41644344038
runs only 2023a while the existing 2024a job retains its own operation.
