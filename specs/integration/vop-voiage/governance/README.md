# VOP governance schema mirror

This directory vendors the VOP-owned concern-governance JSON Schemas for
offline VOIAGE conformance. `UPSTREAM.json` pins the canonical repository,
commit, source path, and SHA-256 digest of every mirrored byte. Files under
`schemas/` must not be edited independently; refresh them from the canonical
commit and update the provenance manifest in the same reviewed change.

The GitHub projection policy is descriptive and fail-closed. It defines stable
record markers and excludes `local_private` evidence, but contains no network
mutation capability. The local governance ledger remains the source of truth.
