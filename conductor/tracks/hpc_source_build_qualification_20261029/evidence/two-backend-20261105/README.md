# Colab and WSL qualification checkpoint

The canonical uncompressed `manifest.json.gz` SHA-256 is `deed5253c2abda3a91ce3b3b3b8f2962088245a0b3df2449337cc900ccbc99b6`.
Report contents redact private workspace roots and process IDs. Log `content`
contains bounded excerpts (Colab leading, WSL error lines/trailing); byte counts and hashes
cover full original logs. The SDK frontend entry is derived metadata without an
original-file digest. Retained report hashes bind their private original bytes.
The frozen executable candidate is
`6ebc61dc70d9996de836bbcd44ceea998d9f0cc5`, merged by PR #1269 as
`dcc09c81059fbfb919d107d64cec633364919f8e`.

Both backends passed fifteen native isolation and cleanup controls. Exact-head
hosted full tox (run41987436781), native controls (run41987441122) and all seventeen
required PR checks passed. Those checks do not establish installed acceptance.

Colab's EasyBuild 2024a attempt stopped during online fetching because
`isl-0.26.tar.bz2` was unavailable from its configured mirrors. Its aggregate
report records exit41 and completed cleanup; compilation never started.
The complete source-packaging SDK had matched its pinned digest before use;
earlier truncated responses were rejected. The terminal receipts were retained
before releasing this owned Colab runtime to stop compute consumption.

WSL's Spack attempt stopped during fresh environment concretization with a fatal
Python 3.15.1 `_TAIL_CALL_CACHE: Executing a cache` error, exit134 and completed
cleanup. Readback showed JIT disabled. This is a genuine unresolved runtime
failure; no older interpreter or solver bypass was substituted.

Both WSL EasyBuild attempts subsequently stopped at online fetching:
2023a could not fetch `hwloc-2.9.1.tar.gz`, and 2024a could not fetch
`hwloc-2.10.0.tar.gz`. Both report exit41 and completed cleanup. Their initial
running snapshot is superseded by these terminal reports. No job remains active.
WSL lacks process I/O counters; reports explicitly mark I/O incomplete while
retaining CPU/RSS/disk/time guards.

The official hwloc download pages link to `download.open-mpi.org`, while the
pinned recipes use the old `www.open-mpi.org` download path. Both official release
archives were fetched completely from the current host and exactly matched the
unchanged recipe SHA-256 values. The manifest retains their sizes and digests.
Two recipe backports change only that URL; build options, dependencies, checksums
and upstream licences remain unchanged. A fresh source replay and installed
attempt are still required before acceptance.

A separate hosted source run41987444843 failed fetching `hwloc-2.10.0.tar.gz`
from upstream and EasyBuild mirrors. Checksums, dependency tests and network
isolation remain enforced. All three installed matrix cells remain open.

Hosted source run41991750912 on `7622c6f2d` accepted both hwloc archives, with
resolved authenticated receipts, then stopped at the same obsolete-host issue
for OpenMPI 4.1.5/5.0.3. Its two terminal reports and failure excerpts are retained
in the manifest. Full tox41991743917 passed that candidate.

The two OpenMPI backports likewise change only the source URL. Their ten patches
are copied verbatim from the pinned catalogue for local patch discovery and
checked against the unchanged recipe hashes. Full release downloads match both
original archive hashes. Existing build options and tests remain untouched.
Fresh replay of the complete repaired source graph is still required.

Exact-path Git attributes preserve these upstream patch bytes, including required
unified-diff context spaces and original line endings. Their full SHA-256 values
are checked against the pinned recipe dictionaries; they are not reformatted.

Subsequent source run41993727314 and full tox41993722034 both pass at
`63030a42e`. Both EasyBuild generations fetched their complete graphs and
replayed them with external networking denied. The manifest retains resource
limits, network denial, empty replay diffs and every identical online/offline
resolved-source witness. Installed acceptance remains false.

An additional nightly Rust vendor job41993722518 reached its 8GiB free-space
reserve (not a source-checksum or memory failure). Its original resource report
is retained. Only Rust-source jobs on fresh GitHub-hosted VMs now remove the
unused Android SDK and retain before/after capacity receipts; the reserve and
all dependency/source checks remain unchanged. Fresh hosted validation is needed.

WSL briefly rejected creation of a diagnostic directory as read-only. Subsequent
exact-target write/read/fsync and cleanup passed; current mounts are writable.
Kernel warnings record a previous I/O failure and recommend filesystem checking.
Host backing location and free space were checked privately; no manual repair,
remount or shutdown was performed. This derived operator readback is distinct
from original file receipts. Guarded private builds may continue with these
limitations retained. Fresh Spack concretization passed in both debug and default
allocator diagnostic contexts on Python 3.15.1, without clearing the original
fatal attempt or establishing installed acceptance.
