# Round 6 three-way source verification

Three AI-assisted reviewers split the 15 references into disjoint groups and
checked bibliographic fields and manuscript claims against publisher, DOI,
archive, CRAN, PyPI, or immutable repository records. This is machine-assisted
evidence, not the pending author source confirmation.

## Results

| Records | Result |
| --- | --- |
| `ades2004evsi`, `rothery2020voi` | Accurate general EVSI sources. The manuscript now distinguishes their broad methodological support from its repository-specific conjugate normal--normal derivation. |
| `claxton1999irrelevance` | Accurate foundational record. “Well established” was narrowed to a long methodological history. |
| `andronis2016implementation` | Accurate record supporting improved rather than complete implementation. |
| `voiage2026` | Accurate historical v1.0.0 release only. Version, date, signed-tag commit, and scope were made explicit. |
| `voiage_software_heritage` | Snapshot contains the signed v1.0.0 release. The exact release SWHID was added; the reviewed revision remains unarchived. |
| `voi_cran2024` | Accurate authors, date, DOI, version 1.0.3, and EVPI/EVPPI/EVSI/ENBS scope. |
| `dampack2024` | Accurate metadata. The manuscript now says analysis and visualisation of decision-model outputs. |
| `green2022bcea`, `strong2014evppi` | Accurate publisher records and appropriately bounded claims. |
| `savi2025` | Release and capability claims verified. Credited developers and University of Sheffield publisher metadata replaced the ambiguous institutional author. |
| `vop_poc_nz2026` | Immutable compatibility evidence verified. The CFF title, software version 0.2.2, and contract-bundle version 1.0.0 are now separate. |
| `adamczewski2022valueofinformation` | Commit is from 2022, not 2025. The key/year and simplified binary-decision wording were corrected. |
| `mordaunt2025trdcea` | Version and functional scope verified. The author is identified from the package's requested citation because PyPI's structured field is empty. |
| `tuffaha2021webtools` | Complete DOI record and direct support for the web-tool comparison. |

## Remaining boundary

All records remain `queued` in
`paper/joss-references.verification.json`. SourceRight structural
reconciliation and these three source reviews reduce uncertainty but do not
replace the author's final source-by-source confirmation. Software and web
records without DOIs remain valid and were not assigned invented identifiers.
