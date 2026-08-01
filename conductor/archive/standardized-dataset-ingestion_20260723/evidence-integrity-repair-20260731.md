# Evidence Ledger Integrity Repair — 2026-07-31

The current `evidence.jsonl` is preserved byte-for-byte as
`evidence.pre-integrity-repair-20260731.jsonl` with SHA-256
`e007a9316bc9084c565d33a69b5684675c3ece6dc6bfafd03e0131cbe0341738`.

The Conductor `evidence_ledger` validator found that entry 5 has a stored
`entry_sha256` of
`53c270faaa692e51338ff97ecc375806e0fcdeb3d7d2a74940fdd3bbef55caca`,
but its canonical schema-1.0 hash is
`a2c63708ba7518627292e6a77bd6c6ccdc9a97a32e198095cf6a40d78521f0bc`.
Consequently entry 6 also references an invalid predecessor. In-place repair
would alter committed historical evidence and is therefore inappropriate.

The replacement `evidence.jsonl` starts a new valid chain with an integrity
repair receipt that binds the immutable pre-repair artifact. It preserves the
historical claims without treating the defective chain as validated, and it
records only future evidence under the current schema.
