# Colab and WSL qualification checkpoint

The canonical uncompressed `manifest.json.gz` SHA-256 is `850895e96a552163d47f677ad4ebdbd19f7fd53f06da7cd11b69c4a824557332`.
Report contents redact private workspace roots and process IDs. Log `content`
contains bounded excerpts (Colab leading, WSL trailing); byte counts and hashes
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

The two WSL EasyBuild generations were still fetching authenticated source inputs
at the recorded snapshot. Their foreground attached sessions remain owned by
this qualification lane. These are running snapshots, with no terminal aggregate
report or installed acceptance. WSL lacks process I/O counters; its aggregate
reports explicitly mark I/O incomplete while retaining CPU/RSS/disk/time guards.

A separate hosted source run41987444843 failed fetching `hwloc-2.10.0.tar.gz`
from upstream and EasyBuild mirrors. Checksums, dependency tests and network
isolation remain enforced. All three installed matrix cells remain open.
