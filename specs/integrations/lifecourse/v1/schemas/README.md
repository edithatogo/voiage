# lifecourse v1 Schema Overlays

This directory is reserved for integration-specific schema overlays.

The preferred approach is to reuse the canonical `voiage` core API schemas and
only add overlays here if `lifecourse` needs producer metadata, bundle
metadata, or contract-specific validation that does not belong in the shared
core contract.

No overlays are required for the initial scaffold.
