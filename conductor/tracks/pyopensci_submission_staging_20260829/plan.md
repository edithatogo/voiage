# Track Plan: pyOpenSci Submission Staging

## Phases

- [x] **Phase 1: Template and Candidate Freeze** [checkpoint: `4ef33e87`]
  - [x] Pin and hash the current official pyOpenSci submission template. (`a78a84e8`)
  - [x] Compare `v2.0.0` and `v2.1.0` evidence and record the candidate decision or gate. (`1cb6303e`)
- [~] **Phase 2: Unposted Submission Packet**
  - [x] Add failing contract tests for template provenance, candidate identity, and external-state boundaries. (`99cc902f`; formatting `7aac651b`)
  - [x] Prepare the local submission draft and machine-readable staging manifest. (`fa4006d6`)
  - [x] Implement the submission-staging validator and pass the focused tests. (`be03cf7f`)
- [ ] **Phase 3: Assurance and Delivery**
  - [ ] Run submission, package, documentation, security, and distribution-identity preflight checks.
  - [ ] Run full project assurance and automated review.
  - [ ] Commit, open a pull request, and obtain green hosted checks without merging or submitting externally.
