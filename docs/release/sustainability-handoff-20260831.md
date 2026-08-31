# Sustainability and identifier handoff

Reviewed on 31 August 2026 for issue #1026. This is an unsubmitted preparation
packet. No fiscal account, affiliation, badge certification or new identifier
was created.

## Current decisions

| Route | Evidence and remaining requirement | Recommendation |
| --- | --- | --- |
| Open Source Collective | The repository is personally owned. OSC's current eligibility guidance requires organizational repositories and shared ownership. | Defer activation; do not transfer ownership merely to satisfy a checklist. |
| GitHub Sponsors or another individual funding service | No active receiving account or verified project payment link is recorded. | Consider separately if funding is needed; publish no placeholder donation URL. |
| NumFOCUS affiliation | Its current official page says new applications are paused. Distributed leadership and an active contributor community are also expected. | Retain the packet below and reassess when applications reopen and community evidence exists. |
| NumFOCUS fiscal sponsorship | Requires additional governance and legal arrangements beyond affiliation. | Defer; no fiscal sponsorship or tax status is claimed for voiage. |
| OpenSSF Passing badge | Public Project 13835 remains in progress; its page displays an unknown name and incomplete basics. | Prepare evidence, then obtain truthful criterion-by-criterion maintainer certification in the actual account. |
| RRID | The recorded SciCrunch suggestion awaits an assigned, resolving identifier. | Keep #298 open; add only a curator-issued RRID. |
| Software Heritage | The retained SWHID identifies the earlier snapshot containing v2.0.0. | Preserve that historical scope in both citation metadata formats. |
| Zenodo | No assigned Zenodo DOI is evidenced for this handoff. | Defer deposition until a distinct archival need is selected; do not invent a DOI. |

Official sources: [OSC eligibility](https://docs.oscollective.org/interested-in-joining-osc/is-osc-right-for-me),
[OSC fiscal hosting](https://oscollective.org/projects/),
[NumFOCUS requirements and application status](https://numfocus.org/projects-overview),
and [OpenSSF Project 13835](https://www.bestpractices.dev/en/projects/13835/passing).
Recheck these sources before any external action. Application availability and
platform status may change. The prior strategy's asserted paper-DOI requirement
for NumFOCUS affiliation is not established by the current official overview.

## OSC configuration packet, pending eligibility and maintainer choices

The project identity would be voiage, with its public GitHub repository and
documentation site, Apache-2.0 license, and the scope described in `README.md`.
OSC describes itself as a US 501(c)(6) fiscal host. That does not make voiage an
activated collective or confer a tax status on this repository.

Before applying, the maintainer must decide whether shared organizational
ownership is appropriate, identify real administrators and expense approvers,
define a budget and payment needs, and review the host's current agreement,
fees and disclosure terms. Keep personal payment and tax information outside
the repository. After host acceptance, verify the actual collective URL and
any GitHub Sponsors connection before adding payment links to `SUPPORT.md`.

## NumFOCUS Affiliated Project packet

| Application topic | Prepared repository evidence | Missing external or human evidence |
| --- | --- | --- |
| Scientific purpose | `README.md`, `paper.md`, canonical method registry and worked examples | Attributable independent research adoption; #471 remains open |
| Open licensing and access | `LICENSE`, public repository and documentation | No additional license choice is made here |
| Contribution process | `CONTRIBUTING.md`, issue templates, protected PR workflow | Actual sustained contributor participation |
| Leadership and continuity | `GOVERNANCE.md`, `SUPPORT.md`, maintenance policy | Distributed leadership; do not count agents as human contributors |
| Community conduct | `CODE_OF_CONDUCT.md` and its reporting process | Maintainer confirmation of accountable contact and practical handling |
| Roadmap and sustainability | `roadmap.md`, current Conductor tracks | A reviewed budget, named project contacts and funding priorities |
| Submission readiness | This dated packet and official eligibility page | Application reopening, maintainer application decision and actual submission receipt |

Affiliation is distinct from fiscal sponsorship. Do not promise fiscal services,
grant funding, acceptance, or a particular award amount. No application or
contact has been sent.

## OpenSSF questionnaire preparation

Use `README.md` and the public docs for description and interfaces; `LICENSE`
for licensing; `CONTRIBUTING.md`, `SUPPORT.md` and `SECURITY.md` for contribution,
support and vulnerability reporting; Git history and signed release receipts
for version control and delivery; and `tox.ini`, `.github/workflows/`, tests
and retained run evidence for quality controls. These are evidence pointers,
not completed responses to the whole questionnaire.

The maintainer must inspect each actual criterion, document any justified
non-applicability, and confirm personal knowledge and operational practices
where required. A configured scanner, an agent review, or a green pipeline
does not by itself establish every security practice. Verify the saved public
project identity and awarded status after submission. Keep the Passing badge
separate from issue #620's numerical Scorecard goal.
