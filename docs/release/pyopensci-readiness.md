# pyOpenSci readiness and maintenance commitment

This document records the repository-owned baseline and the maintainer's
maintenance commitment for a pyOpenSci-first review followed, if accepted and
in scope, by the JOSS partner pathway. It is not an inquiry, submission,
acceptance, or publication claim.

## Selected review sequence

The selected sequence is:

1. prepare and refresh the pyOpenSci evidence package;
2. complete the private pre-review survey and human-written submission text
   before the already-authorized, human-led pyOpenSci submission;
3. complete pyOpenSci review and record its external acceptance evidence; and
4. only then, once JOSS eligibility and author prerequisites are evidenced,
   use the already-selected partner route and identify the accepted review.

The partnership does not guarantee JOSS scope or acceptance. The current
[pyOpenSci author guide](https://www.pyopensci.org/software-peer-review/how-to/author-guide.html)
and [JOSS partnership guidance](https://www.pyopensci.org/software-peer-review/partners/joss.html)
remain authoritative and must be refreshed immediately before external action.

## Maintenance commitment

The repository owner commits to maintaining `voiage` for at least two years
after pyOpenSci acceptance. The acceptance date and resulting minimum-support
end date must be recorded if acceptance occurs.

Maintenance covers:

- reproducible defects in supported releases and stable public interfaces;
- private handling and remediation of supported security vulnerabilities;
- Python packaging and installation compatibility across declared runtimes;
- compatibility of the Rust-backed stable core and retained bindings; and
- documentation needed to install and use supported behavior.

Complete, reproducible issues affecting a supported release have a best-effort
acknowledgement and triage target of 14 calendar days. Critical defects receive
priority investigation. Resolution timing depends on severity,
reproducibility, maintainer capacity, compatibility risk, and release safety.
These are service targets, not a contractual service-level agreement: there is
no guaranteed fix deadline or continuous-availability commitment.

If the current maintainer cannot continue, the project will document the
reduced-support state, seek co-maintainers or a successor where practical, and
coordinate with pyOpenSci if the package has been accepted. If sustainable
maintenance cannot be restored, the project will publish a clear archival or
sunset notice rather than silently implying active support.

Security reports continue to follow `SECURITY.md`; public support and defect
reports follow `SUPPORT.md`.

## Repository evidence matrix

The machine-readable matrix is
`specs/submission-readiness/pyopensci-evidence.json`. The submission-readiness
validator requires every repository-controlled criterion to have existing
local evidence. The maintenance commitment is repository evidence; the
personal form attestations and review communication remain human gates despite
the maintainer's authorization of the route and autonomous repository work.

The repository-wide venue inventory was refreshed on 2026-08-29, and the
pyOpenSci/JOSS guidance and official template were rechecked on 2026-08-30. The current
package provides published installation instructions, an importable wheel
test, online quick starts and API reference, contribution and conduct
documents, issue templates, CI-backed tests, release evidence, support and
governance documentation, and an AI-use disclosure. Scope, overlap, and method
provenance are documented in the README, methods documentation, and paper.

Refresh the official guide and the evidence matrix immediately before any
author-led inquiry. Historical AI-review confirmations do not certify the
current packet. The unposted draft includes explicit current human-review,
development-history, AI-scope and communication checks. Previous requests #271
and #272 were withdrawn and verified closed on 31 August 2026. The remaining
immediate human gates are the private pre-review survey, a human-written
submission body, and authenticated pyOpenSci submission. The maintainer's
commitment to write later review communication personally is separately
confirmed. Withdrawal of those requests resolves the observed contact overlap;
it does not establish editorial scope or capacity approval.
