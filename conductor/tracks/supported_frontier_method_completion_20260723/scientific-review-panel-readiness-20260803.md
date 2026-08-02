# Scientific-review panel readiness — #841 / #843–#849

This packet prepares the role-based subagent panel described in
`scientific-review-panel.md`. It is deliberately fail-closed: the candidate is
not frozen while PR #854 hosted checks are still moving, and no panel output
can substitute for an accountable scientific or maintainer decision.

## Execution order

1. Freeze the exact candidate after PR #854 reaches a stable exact head.
2. Hash the packet, fixtures, schemas, installed parity evidence, issue/project
   reconciliation, and prior review reports.
3. Commission independent reports from all five required roles before sharing
   the orchestrator synthesis.
4. Record findings, severity, dissent, remediation ownership, and re-review
   requirements without downgrading or deleting findings.
5. Produce an orchestrator recommendation limited to the four protocol verdicts.
6. Obtain separate qualified scientific and maintainer decisions; do not infer
   them from subagent reports, green CI, or Project status.

## Required local artifacts

- `scientific-review-packet-20260803.json` — scope and gate boundary.
- `artifact-manifest.json` — immutable packet member hashes.
- `reviewer-attestations.json` — identity, qualifications, conflicts and
  independence status for every role.
- `finding-dispositions.json` and `disagreement-register.json` — append-only
  findings and dissent.
- `orchestrator-synthesis.md` — recommendation and unresolved conditions.
- `adjudication.json` and `scientific-approval.json` — templates only until
  accountable decisions exist.

The packet must remain `not_started` until the candidate is frozen and all
required reports are present. The current state is preparation-only.
