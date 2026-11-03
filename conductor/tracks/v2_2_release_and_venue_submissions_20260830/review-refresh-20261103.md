# Review: pyOpenSci template refresh — 3 November 2026

Reviewed local commit `246c22b2e` against its parent, following the Conductor
implementation and review workflows. The current template is pinned to upstream
`bdc87671b60fdef417f0fea1619a338c715bb42b`; its blob, byte count and SHA-256 were
verified. Previous template provenance and dated personal confirmations remain
historical evidence. The survey and actual human-written submission text are
pending, and every external-action flag remains false.

The review exposed a validator gap: rehashed drafts could check the new
disclosure boxes without evidence. Four red mutation tests reproduced it; the
fix rejects those confirmations. All 54 focused packet tests and scoped Ruff
checks passed on repo-local Python 3.15.1. The packet validator passed. These
unchanged results are reused rather than rerun solely for review metadata.

A further review found the machine-readable continuation handoff still pointed
to an absent file. It now resolves to the maintained venue execution handoff,
matching the corrected index. Earlier publication receipts remain unchanged.

The full tox attempt stopped before execution with insufficient free space;
its local receipt is `.conductor/local/venue-refresh-review/resource-report.json`.
The latest bounded capacity read observed 193 MiB free on the system volume.
No full local gate, current hosted delivery, merge or venue submission is claimed.
The local branch also contains earlier HPC work and must not be delivered as an
undifferentiated venue-only PR.

Disposition: bounded refresh reviewed with fixes applied; delivery verification
and R14–R23 remain open. AC-04 through AC-07 require actual pyOpenSci/JOSS/rOpenSci
and final-closeout evidence. The request to archive does not supply those receipts.
Keep the track active with its existing scope and outstanding criteria. No
human declaration, survey response or external communication was generated or sent.
