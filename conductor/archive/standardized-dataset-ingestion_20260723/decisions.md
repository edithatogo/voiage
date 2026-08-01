# Dependency and supported-profile decisions

## P4-T1 — Croissant parser dependency (2026-07-24)

### Decision

Keep the current built-in Croissant 1.1 CSV adapter as the only supported
runtime profile for now. Do not add `mlcroissant` to `voiage[croissant]` in
this task. The existing named extra remains intentionally empty until a later
provider implementation proves that every materialization operation is routed
through `SourceAccessPolicy` and emits VOIAGE materialization receipts.

### Evidence considered

- `uv lock --upgrade` completed and
  `python scripts/dependency_frontier.py . --strict` passed. Its only lockfile
  changes were unrelated updates to `hypothesis` and `types-setuptools`; they
  were deliberately not retained.
- An isolated Python 3.14 installation of `mlcroissant==1.1.0` resolved 24
  packages, including `pandas`, `scipy`, `rdflib`, `requests`, and `fsspec`.
  Its documented operation graph can download, extract, and cache resources.
- The track contract requires parser dependencies to be optional and requires
  all remote, archive, and live materialization to obey the injected source
  policy and produce immutable receipts.

### Supported profile after this decision

The provider accepts one local Croissant 1.1 descriptor with one CSV
distribution and one `RecordSet`, exact top-level field declaration, and no
transformations or archives. It explicitly rejects unsupported archive
references and transformations before resource access. It does not claim full
Croissant 1.1 conformance or `mlcroissant` parser-capability support.

### Promotion criteria

P4-T2 through P4-T5 must first add offline conformance/negative fixtures,
source-policy-preserving integration tests, explicit parser-capability
declarations, receipt/provenance handling, and base-install isolation evidence.
Only then may a later task add a version-bounded `mlcroissant` dependency to the
`croissant` extra and publish the separate enhanced-parser profile.

## P4-T6 — Frictionless parser dependency (2026-07-26)

### Decision

Keep the built-in Frictionless Data Package CSV adapter as the only supported
runtime profile for now. Do not add `frictionless` to `voiage[frictionless]`
in this task. The named extra remains intentionally empty until a provider
implementation proves that descriptor loading and resource materialization
cannot bypass `SourceAccessPolicy` and always emit VOIAGE materialization
receipts.

### Evidence considered

- `uv lock --upgrade` and `python scripts/dependency_frontier.py . --strict`
  completed. The lock refresh changed unrelated packages and was deliberately
  discarded.
- An isolated Python 3.14 resolver installation imported
  `frictionless==5.19.0` successfully. That proves only resolver compatibility,
  not that the library's loading, archive, cache, or remote-resource behaviour
  obeys the VOIAGE source policy.
- The track contract prohibits parser-controlled network access, archive
  extraction, implicit authenticated access, and unreceipted materialization.

### Supported profile after this decision

The provider accepts one local `datapackage.json` descriptor with one declared
CSV resource and explicit schema. Remote URLs, archives, implicit resource
selection, and unsupported formats are rejected before resource access. This
does not claim full Data Package or Table Schema conformance, nor a
`frictionless` parser-capability profile.

### Promotion criteria

P4-T7 through P4-T10 must first add offline conformance and negative fixtures,
source-policy-preserving integration tests, capability declarations,
receipt/provenance handling, and base-install isolation evidence. Only then may
a later task add a version-bounded `frictionless` dependency and publish a
separate enhanced-parser profile.

## P8-T5 — Registry-specific provider assessment (2026-07-27)

### Decision

Do not add Hugging Face or OpenML registry-specific providers in this track.
The built-in Croissant provider remains an offline, descriptor-relative CSV
adapter and the default source policy remains fail-closed for every network
scheme. A registry adapter would be a distinct network/materialization product
surface, not a thin alias for the current provider.

### Evidence considered

- Hugging Face's read-only Dataset Viewer returned a concrete Parquet-shard
  inventory for `stanfordnlp/imdb` at
  `https://datasets-server.huggingface.co/parquet?dataset=stanfordnlp%2Fimdb`.
  The corresponding documented Croissant endpoint returned HTTP 404 at the
  same assessment time. The resulting registry surface is therefore not a
  stable Croissant descriptor source that the current provider can claim to
  support.
- The OpenML metadata endpoint
  `https://www.openml.org/api/v1/json/data/61` returned HTTP 504 during the
  assessment. No OpenML Croissant contract or authoritative interoperability
  fixture was available to prove a safe, reproducible implementation.
- Both registry paths would require authenticated access handling, redirects,
  mutable live data, and remote artifact materialization. Those operations
  need checksum-pinned receipts, cache policy, and replay tests before they
  can cross the provider boundary.

### Follow-on criteria

Open a registry-specific provider only when an authoritative, versioned
descriptor contract and rights-cleared fixture exist, the source path is routed
through `SourceAccessPolicy`, every redirect and downloaded artifact is
receipt-bound and content-addressed, and an opt-in live interoperability test
can run without weakening offline defaults. Until then, users may download and
pin a supported local Croissant CSV descriptor before invoking VOIAGE.
