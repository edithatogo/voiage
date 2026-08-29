# Specification: Remote Standardized-Dataset Ingestion Security

## Overview

Define and, only after approval, implement the security policy required for
general remote standardized-dataset ingestion.

## Requirements

- Establish an approved threat model for SSRF, DNS rebinding, redirects,
  scheme/host allow-lists, credentials, archives, decompression, cache
  poisoning, mutable resources, quotas, and telemetry redaction.
- Define bounded transport, cache, receipt, and offline-replay contracts before
  enabling any remote source.
- Retain strict fail-closed behaviour as the default and preserve the canonical
  normalized-input and preparation path.

## Acceptance criteria

- **AC-01:** Threat model and security policy are reviewed and approved.
- **AC-02:** Remote resolution has exhaustive adversarial and policy tests.
- **AC-03:** Verified cache and receipts prevent mutable or cross-policy reuse.
- **AC-04:** Security review, performance limits, documentation, and hosted
  evidence support the enabled profile.

## External gates

Security-policy approval and, where applicable, infrastructure/network policy
approval are required before implementation. No live resource, credential, or
paid service is authorized by this track alone.

## Out of scope

Dataset-specific controlled probes are owned by #752. Parser feature parity is
separate work after a dependency and security review.
