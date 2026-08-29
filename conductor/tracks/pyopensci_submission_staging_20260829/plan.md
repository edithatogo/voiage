# Track Plan: pyOpenSci Submission Staging

## Phases

- [x] **Phase 1: Template and Candidate Freeze** [checkpoint: `4ef33e87`]
  - [x] Pin and hash the current official pyOpenSci submission template. (`a78a84e8`)
  - [x] Compare `v2.0.0` and `v2.1.0` evidence and record the candidate decision or gate. (`1cb6303e`)
- [x] **Phase 2: Unposted Submission Packet** [checkpoint: `357e87ca`]
  - [x] Add failing contract tests for template provenance, candidate identity, and external-state boundaries. (`99cc902f`; formatting `7aac651b`)
  - [x] Prepare the local submission draft and machine-readable staging manifest. (`fa4006d6`)
  - [x] Implement the submission-staging validator and pass the focused tests. (`be03cf7f`)
  - [x] **Review Fixes:** Reject incomplete human-attestation and external-action key sets. (`e0e4a51d`)
  - [x] **Review Fixes:** Remove Markdown trailing spaces and rebind the draft digest. (`37373672`)
- [~] **Phase 3: Assurance and Delivery**
  - [x] Run submission, package, documentation, security, and distribution-identity preflight checks. (`3bcc1932`)
  - [x] **Review Fixes:** Correct the evidence-validator invocation and normalize the unpublished invalid hash chain with an explicit audit record. (`c0387fdc`)
  - [~] **Review Fixes:** Remove the specification's extra blank line at EOF.
  - [~] Run full project assurance and automated review.
  - [ ] Commit, open a pull request, and obtain green hosted checks without merging or submitting externally.
