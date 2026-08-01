# Blocker-Resolution Plan

## Options

1. **Strict-local closure (recommended):** complete the threat model,
   fail-closed tests, receipts/cache design, documentation, and evidence while
   keeping remote I/O disabled. This is safe to implement without credentials,
   network access, or infrastructure approval.
2. **Staged remote pilot:** implement a narrowly allow-listed transport behind
   an opt-in feature flag after security and infrastructure approval, then run
   controlled probes and hosted checks. This provides earlier interoperability
   evidence but introduces network, DNS, credential, and mutable-source risk.
3. **Defer the track:** leave the existing local-only behavior unchanged until
   an external owner supplies an approved policy. This minimizes change but
   leaves the threat-model and reproducibility work undocumented.

## Recommendation

Use option 1 now. It addresses repository-owned blockers without weakening the
default security boundary. Revisit option 2 only when a versioned transport,
cache, receipt, and offline-replay profile has explicit security-policy and
infrastructure approval.

## Contingencies and exit criteria

- If approval is denied, retain the local-only profile and mark remote support
  unsupported; do not add live probes.
- If approval is granted, add adversarial SSRF/DNS/redirect/archive/cache tests
  before implementation, then require hosted security checks and receipt-bound
  replay evidence.
- The track is not archive-eligible until R2 approval, R3 review, R4–R6
  evidence, and hosted checks are complete.
