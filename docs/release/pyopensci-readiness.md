# pyOpenSci readiness and maintenance commitment

This document records the repository-owned baseline and the maintainer's
maintenance commitment for a pyOpenSci-first review followed, if accepted and
in scope, by the JOSS partner pathway. It is not an inquiry, submission,
acceptance, or publication claim.

## Selected review sequence

The selected sequence is:

1. prepare and refresh the pyOpenSci evidence package;
2. after a separate maintainer instruction, open the pyOpenSci review;
3. complete pyOpenSci review and record its external acceptance evidence; and
4. only then, after another maintainer instruction, request JOSS fast-track
   handling and identify the accepted pyOpenSci review.

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
decision and authenticated action to open an inquiry or review remain a
separate human gate.

The repository-wide venue inventory was refreshed on 2026-07-27, and the
pyOpenSci/JOSS guidance received a focused refresh on 2026-08-27. The current
package provides published installation instructions, an importable wheel
test, online quick starts and API reference, contribution and conduct
documents, issue templates, CI-backed tests, release evidence, support and
governance documentation, and an AI-use disclosure. Scope, overlap, and method
provenance are documented in the README, methods documentation, and paper.

Refresh the official guide and the evidence matrix immediately before any
author-led inquiry. This commitment does not grant permission to contact
pyOpenSci or JOSS.
