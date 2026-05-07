# lifecourse v1 Schema Overlays

This directory is reserved for integration-specific schema overlays.

The preferred approach is to reuse the canonical `voiage` core API schemas and
only add overlays here if `lifecourse` needs producer metadata, bundle
metadata, or contract-specific validation that does not belong in the shared
core contract.

The initial scaffold now includes one overlay for the shared VOI result
envelope and its compatibility metadata:

- [voi-result-envelope.schema.json](./voi-result-envelope.schema.json)

No additional overlays are required for the initial scaffold.

The envelope overlay is intentionally narrow: it records the compatibility
anchors used by the shared fixtures, while the core VOI method outputs remain
owned by the canonical `voiage` schemas.
