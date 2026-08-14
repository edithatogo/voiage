# H8-D-C source-observation and remediation readiness

Issue [#867](https://github.com/edithatogo/voiage/issues/867) is a bounded
repository-preparation child of #853. It refreshes automated observations for
the six sources already frozen in the H8-C manifest and partitions all nineteen
challenge findings by their next evidence prerequisite. It does not change the
historical manifest, freeze a replacement packet, provide an independent
retrieval receipt or satisfy H8-D.

## Current observations

| Source | Automated observation | Drift interpretation | Authority boundary |
| --- | --- | --- | --- |
| Heath et al. | PMC HTML retrieved; bytes changed | Dynamic representation; semantic review pending | CC BY-NC 4.0 was observed, but use scope, redistribution and applicability remain unreviewed |
| Belmont Report | Official PDF observed through web retrieval; CLI returned 403 | No current digest; drift not assessable | Rights, independent retrieval and candidate applicability remain unverified |
| 45 CFR Part 46 | Official eCFR Versioner XML retrieved for 2026-07-24 | Prior HTML and current XML are not byte-comparable | Candidate jurisdiction and regulatory applicability remain unassessed |
| ICH E6(R3) | Official PDF digest is unchanged | Byte-stable for the same representation | ICH notice conditions and third-party exclusions still require accountable review |
| Camilleri et al. | NeurIPS abstract digest is unchanged | Byte-stable for the same representation | Rights and candidate applicability remain unverified |
| Bottero et al. | NeurIPS abstract digest is unchanged | Byte-stable for the same representation | Rights and candidate applicability remain unverified |

No source bytes are retained in the repository. Observation equality is not a
scientific, legal, ethics or regulatory conclusion. The automated record keeps
rights review, drift review, applicability review, independent source review,
publication authority and real-study authority false.

## Finding readiness

- Repository changes are implemented but still require independent re-review
  for `H8D-API-GOV-01`, `H8D-API-GOV-02` and `H8D-GP-01`.
- Independent source review is a prerequisite for `H8D-ED-04`,
  `H8D-EST-002`, `H8D-API-GOV-03`, `H8D-GP-02` and `H8D-DS-05`.
- Candidate context and qualified human review remain prerequisites for the
  other eleven findings, including Critical `H8D-DS-03`.

These groups are disjoint and cover the complete nineteen-finding register.
Every finding remains `pending`; implemented repository changes are not
finding dispositions.

## Replacement-packet decision

A replacement packet is not ready to freeze. It requires a declared candidate
context, independent source retrieval and rights review, jurisdiction-specific
applicability, eligible independent human reviewers, and disposition evidence
for every finding. The immutable H8-C packet remains the historical baseline.

## Hosted governance readback

GitHub issue #867 is a native child of #853. Project 28 records it as planned
v1.3.0 Must, In Progress, Human, Unverified, Mitigating, Critical, P1 and Clean,
with review due 2026-11-30 and owner role `Source assurance and scientific
review coordinator`. Repository validation and a green merged PR may complete
only this preparation slice.
