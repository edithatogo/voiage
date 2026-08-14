# R2 security-policy approval packet

## Decision requested

Approve or reject the bounded policy in
`r1-threat-model.md` for a future implementation. Approval authorizes
repository implementation and offline adversarial testing only; it does not
authorize live datasets, credentials, paid services, production deployment,
or controlled probes owned by issue #752.

## Recommended option: strict local-first transport profile

Approve a profile with HTTPS and explicit host/port allow-lists, private-range
and metadata-service rejection, DNS/connection revalidation, zero ambient
credentials, bounded redirects and quotas, sandboxed archive extraction,
content-hash receipts, policy-versioned caches, redacted diagnostics and
offline replay. Keep the default disabled until a caller supplies an approved
policy and a verified receipt.

Trade-off: this maximizes safety and reproducibility but does not support
arbitrary public URLs or authenticated providers. Broader access can be
introduced later as a separately reviewed policy version.

## Alternative options

- **Reject remote I/O:** retain the current strict-local providers permanently;
  close this track as a reviewed exclusion after documenting migration guidance.
- **Approve broader live access:** permit additional schemes, hosts or
  credentials. This requires a new threat review, infrastructure/network
  approval, rights-cleared source packets and explicit changes to the policy
  contract; it is not recommended as the default.

## Approval record

Until an accountable maintainer/security authority records a dated approval,
R2 remains `pending`, remote I/O remains disabled, and no implementation,
hosted check or live-probe result may be represented as enabled security
evidence.

## Conductor decision workflow

1. An accountable security/infrastructure authority selects **approve**,
   **reject**, or **request-revision** in `r2-approval-record.json` (or an
   authoritative portal whose immutable receipt is recorded there).
2. The authority records identity/role, UTC timestamp, policy and threat-model
   SHA-256 values, allowed schemes/hosts/ports, quota profile, credential
   boundary, expiry/review date, and any infrastructure conditions. Secrets,
   tokens and personal data are excluded.
3. Conductor verifies the hashes and required fields. Only an `approve` record
   with valid hashes may move `remote-security-policy-approval` to satisfied.
   `reject` closes the track as a reviewed exclusion; `request-revision`
   returns to R1 without enabling I/O.
4. After approval, R3 review and R4 adversarial tests still precede transport
   implementation. Approval never authorizes controlled probes in issue #752.
