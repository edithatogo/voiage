# Scientific-review panel readiness — #841 / #843–#849

This packet prepares the role-based subagent panel described in
`scientific-review-panel.md`. It is deliberately fail-closed: PR #854 is now
merged, so the candidate can be bound to the merged exact head, but no panel
output can substitute for an accountable scientific or maintainer decision.

## Execution order

1. Freeze the exact candidate at the merged PR #854 exact head and invalidate
   this packet if `main` advances.
2. Hash the packet, fixtures, schemas, installed parity evidence, issue/project
   reconciliation, and prior review reports.
3. Commission separated reports from the four required panel roles before
   sharing the orchestrator synthesis. These are advisory agent reports, not
   independent human attestations.
4. Record findings, severity, dissent, remediation ownership, and re-review
   requirements without downgrading or deleting findings.
5. Produce an orchestrator recommendation limited to the four protocol verdicts.
6. Obtain separate qualified scientific and maintainer decisions; do not infer
   them from subagent reports, green CI, or Project status.

## Required local artifacts

- `scientific-review-packet-20260803.json` — scope and gate boundary.
- `artifact-manifest-preparation.json` — pending immutable packet member hashes.
- `reviewer-attestations-preparation.json` — pending identity, qualifications, conflicts and
  independence status for every role.
- `finding-dispositions-preparation.json` and `disagreement-register-preparation.json` — pending
  findings and dissent.
- `orchestrator-synthesis.md` — recommendation and unresolved conditions.
- `adjudication-preparation.json` and `scientific-approval-preparation.json` — templates only until
  accountable decisions exist.

The packet is now candidate-bound to the merged baseline, but remains
`preparation-only` until all reports are present. The resulting recommendation
must keep the methods experimental unless the separate accountable scientific
and promotion decisions are recorded.
