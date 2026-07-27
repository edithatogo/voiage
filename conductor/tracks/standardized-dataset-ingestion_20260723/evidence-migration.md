# Evidence Ledger Migration — 2026-07-27

The original `evidence.jsonl` is retained unchanged as
`evidence.legacy.jsonl` with SHA-256
`3f53b195f672333b1ac8c659a7a2dceeaba83c400dd1d467387e55edccce9656`.

Final Conductor review found that legacy entries 9 and 10 do not satisfy the
current `evidence_schema: "1.0"` contract: both list artifact paths without
the required SHA-256 digest, and entry 10 records a validation command with
exit code 1 as `passed`. Their historical hashes make an in-place repair
inappropriate.

The replacement `evidence.jsonl` begins a new, valid hash chain with a
migration receipt that binds this immutable legacy artifact. It records only
future evidence under the current schema; it does not silently reinterpret,
repair, or upgrade historical claims.
