# R1 bounded remote-ingestion threat model and source-policy contract

This document defines the minimum security contract for any future remote
Croissant, Frictionless, or normalized-input materialization. It does not
enable network access. The current provider profile remains strict-local and
offline-first until R2 records explicit security-policy approval.

## Trust boundaries

| Boundary | Untrusted input | Required control |
|---|---|---|
| Caller to provider | URL, headers, credentials, format hints, byte/row limits | Typed validation; reject credentials and unsupported schemes by default. |
| Resolver to network | DNS answers, redirects, proxy metadata, TLS peer | HTTPS only; host and port allow-list; resolve and connect to the same validated address; bounded redirects; no private/link-local/loopback ranges. |
| Network to materializer | Response status, MIME, length, encoding and bytes | Maximum response bytes/time; exact content hash; declared MIME/format agreement; no implicit decompression. |
| Archive to filesystem | Archive names, symlinks, compression ratio and nesting | Extract into a fresh sandbox; reject traversal, absolute names, symlinks, device files, excessive members, depth or expansion ratio. |
| Cache to replay | Cache key, policy version, URL, validators, bytes and receipts | Key includes source, policy and format; hash and receipt must match; stale or cross-policy entries are misses. |
| Diagnostics to operators | URLs, headers, payloads and exception text | Redact secret-like values; bounded logs; never persist authorization headers or response bodies by default. |

## Threat catalogue and required disposition

- **SSRF and DNS rebinding:** reject private, loopback, link-local, multicast,
  metadata-service and reserved destinations at resolution and connection;
  re-check after every resolution and redirect.
- **Redirect confusion:** follow no redirects unless explicitly enabled by an
  approved policy; constrain count, scheme, host and port; issue a new policy
  decision after each hop.
- **Credentials and exfiltration:** no ambient environment credentials,
  credential helpers, cookies or proxy authentication; explicit secret input is
  rejected in the default profile and is never included in cache keys/logs.
- **Mutable or misleading resources:** require a verified SHA-256 receipt,
  bounded byte count, expected MIME and format; ETag/Last-Modified alone never
  proves identity.
- **Archive and parser abuse:** enforce byte/member/depth/ratio/time quotas;
  reject malformed JSON, duplicate keys where detectable, invalid schemas,
  decompression bombs and unsafe paths before materialization.
- **Cache poisoning:** bind cache entries to canonical source, policy version,
  format, content hash and receipt; do not reuse entries after policy changes.
- **Denial of service:** bound DNS attempts, redirects, connections, bytes,
  decompression, rows, columns and wall-clock time; fail closed on quota
  exhaustion without partial normalized output.
- **Telemetry disclosure:** redact tokens, passwords, API keys, signed URLs and
  sensitive query parameters; retain only hashes and non-sensitive policy
  decisions in evidence.

## Source-policy contract

The future approved policy must specify, in versioned machine-readable form:

1. allowed schemes, hosts, ports and DNS result classes;
2. redirect, proxy and TLS rules;
3. connection, response, archive and parser quotas;
4. accepted MIME/format mappings and schema-validation mode;
5. credential prohibition or explicit secret-provider boundary;
6. cache key, freshness, invalidation and policy-version semantics;
7. receipt fields: canonical source, retrieval time, policy version, byte
   length, MIME, SHA-256, parser version, licence/citation metadata and
   redaction status;
8. offline replay command and the exact failure diagnostics for missing or
   mismatched receipts.

## Fail-closed acceptance rules

Remote I/O is disabled when policy approval is absent, a source is not
allow-listed, a receipt is absent/mismatched, any quota is exceeded, or a
security decision is indeterminate. A failed request produces no cache entry,
no partial `NormalizedInputBundle`, and no claim of provider interoperability.
Controlled live probes remain owned by track #752 and require their own
rights-cleared source packet and approval.

## References

- `voiage/ingestion/` and provider tests: current strict-local boundary.
- `conductor/archive/standardized-dataset-ingestion_20260723/spec.md`:
  normalized-input and receipt semantics.
- `conductor/tracks/controlled_live_dataset_interoperability_20260801/spec.md`:
  separate controlled-probe boundary.
- `conductor/workflow.md`: evidence, approval and external-gate rules.
