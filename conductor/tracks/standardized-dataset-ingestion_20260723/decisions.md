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
