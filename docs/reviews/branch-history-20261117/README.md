# Historical branch evidence — 17 November 2026

This packet preserves records found while reconciling older local branches
against main at `f5f302dc28981609369a2c5fdd5fe252b8f855a7`. The records are
dated observations of their original candidates. They do not establish current
native, scientific, hosted, or release acceptance.

The source tips were:

| Branch | Tip | Disposition |
| --- | --- | --- |
| `codex/archive-backlog-gate-20261103` | `db9bc470a4f48127fc05635719b9d6be9a047495` | Its nine changed paths exist on main and have since changed; retain the later source and the deferred HPC gates. |
| `codex/archive-primary-20261107` | `a9ec5b1cfd67c5a313deb6877f33fb7443facfef` | The later opt-in exporter supersedes the branch's no-exporter assertion. Two earlier telemetry records are copied below as historical records only. |
| `codex/hpc-arrow-qualification-20261103` | `0436e1bee899af78914bb1541870092174a65684` | All 27 changed paths exist on main; 16 were byte-identical at this comparison and 11 had subsequently changed. HPC installation acceptance remains open. |
| `codex/hpc-bootstrap-compat-20261105` | `062081bb1d80af4cbee1e52c18bf87dbaa316b20` | Main retains 33 of 34 changed paths, with later updates to some; the missing tobac record and the distinct historical witnesses below are preserved. |

These path comparisons do not assert that every later file is byte-equivalent
to its branch version. The complete branch histories are separately retained in
the integrator's verified Git recovery bundle; current source and current track
status remain authoritative.

## Retained records

- [Provider witness continuation](g03-provider-continuation-20261105.md) and
  [shard hosted continuation](g04-shard-hosted-20261105.md) are the exact
  branch-only appended text from the bootstrap branch. They document
  historical candidates and hosted runs; later qualification does not follow
  automatically from these observations.
- [Agent-safe branch-only records](agent-safe-branch-only.jsonl) retain four
  complete JSON lines absent from the current evidence file: the earlier
  extracted-package witness, focused provider and coverage-identity readbacks,
  and the historical G05 hosted reconciliation. The focused readbacks were
  terminal observations, not independently retained test logs.
- [Telemetry branch-only records](telemetry-branch-only.jsonl) retain two
  complete JSON lines from the older primary branch. Their historical
  acceptance and no-exporter wording describes that earlier candidate, not
  the present telemetry implementation or its open qualification gates.
- The [tobac parameter-artifact observation](../../../conductor/tracks/tobac_consumer_qualification_20261025/evidence/parameter-artifact-20261105.md)
  is copied byte-for-byte from the bootstrap branch. It records a focused
  terminal readback without an independent log, and leaves the consumer gates
  open.

The copied text has SHA-256 values `50c2ac2de08fe3ff4164c2044e54609447f7ecf048b2d8f895456f326684e173`
(provider), `a94acb4d8313cee94814986d99d967c9f5787c9a54f41e303d133668528a54eb`
(shard), and `ec37687aa93793f74b16be319aed55a969119a611e01fce20b5f1e7d9036c3c5`
(tobac). The extracted JSONL files have SHA-256 values
`1368098b0916eb3951824e72ddd6a8f356c55fe9a8e2d7f3b316f973424300a2`
(agent-safe) and `273f79a45bbea86dd40e4fbdc33a9d066ac74dffdfc53f59bfdd2e2cd5f13d7a`
(telemetry).
