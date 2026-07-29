# OpenSSF Best Practices Badge Readiness

## State boundary

`voiage` has not applied for, earned, or published an OpenSSF Best Practices
badge. Repository automation can prepare evidence, but it cannot truthfully
make the maintainer attestations required by the external badge service.

The live Scorecard `CIIBestPracticesID` finding therefore remains an external
gate until the repository maintainer reviews the current criteria, submits the
project record, and records the resulting public URL.

## Repository evidence map

| Evidence area | Repository evidence | Required external action |
|---|---|---|
| Project identity and licensing | `README.md`, `LICENSE`, `CITATION.cff`, `codemeta.json` | Confirm the project identity in the badge record |
| Contribution and governance | `CONTRIBUTING.md`, `AGENTS.md`, `CODE_OF_CONDUCT.md`, `GOVERNANCE.md` | Confirm that the published processes describe current practice |
| Security reporting | `SECURITY.md`, private vulnerability reporting, secret scanning and push protection | Confirm the supported-version and response statements |
| Quality assurance | `tox.ini`, Rust workspace tests, binding workflows, coverage and documentation checks | Review the badge criteria against current required checks |
| Supply chain | pinned Actions, Renovate policy, dependency review, SBOM and provenance workflows | Confirm any criterion that requires a human assertion |
| Releases | signed-tag validation, trusted publishing, staged artifact digests and clean-install tests | Confirm only after an immutable release exercises the workflow |

## Fail-closed application protocol

1. Use the current default branch and immutable release evidence.
2. Review every badge answer; do not infer an answer from green CI alone.
3. Record the accountable maintainer and review date.
4. Submit through the external OpenSSF badge service.
5. Record the public project URL and observed badge level in Conductor.
6. Add a badge to public documentation only after the external service reports
   it.

Registry availability, authentication, review, and badge issuance are external
states. Repository readiness does not imply submission or award.
