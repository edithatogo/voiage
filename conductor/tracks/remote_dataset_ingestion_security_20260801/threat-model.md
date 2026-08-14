# Bounded Remote-Ingestion Threat Model

This record is a repository-owned design boundary, not approval to enable
remote I/O. The current Croissant and Frictionless providers remain strictly
local and fail closed for remote URLs, archives, transforms, and mutable
resources.

## Threats and required controls

| Threat | Required control before enablement | Current disposition |
| --- | --- | --- |
| SSRF and private-network access | Scheme/host allow-list, IP-range rejection, and post-resolution checks | Not enabled; local policy rejects remote references |
| DNS rebinding and redirects | Resolve and pin addresses, re-check every redirect, cap redirect count | Not enabled |
| Archive/decompression bombs | Byte, entry, nesting, compression-ratio and wall-clock quotas | Not enabled; archives rejected |
| Cache poisoning or mutable reuse | Policy-bound cache key, content digest, receipt, expiry and offline replay checks | Local cache only; remote cache contract absent |
| Credential leakage | No credentials in URLs or logs; scoped secret injection and redaction | No remote credentials accepted |
| Resource exhaustion | Response, row, batch, archive and time limits with cancellation | Local quotas exist; remote transport not enabled |
| Telemetry/privacy leakage | Redacted receipts and source identifiers; no payload capture by default | Local provenance only |

## Approval boundary

The controls above must be specified as a versioned transport/cache/receipt
profile and receive explicit security-policy and, where applicable,
infrastructure approval before any live probe or network-enabled provider is
implemented. This track does not authorize credentials, paid services, or live
dataset acquisition.
