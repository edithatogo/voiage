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
