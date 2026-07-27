# Source-by-source citation confirmation

**Scope.** This ledger reviews every citation occurrence in `paper.md` at
source digest `a6f3898019ce0d13785e3e4f36eade6c392dc0a15d248af66cb8e8408b070d50`.
It is an agent-led fit check, not a claim that a human has read every source in
full. SourceRight remains read-only; the author must make the final human
confirmation before JOSS submission.

| Citation key | Manuscript claim(s) checked | Authoritative source inspected | Finding |
| --- | --- | --- | --- |
| `rothery2020voi` | Definitions of EVPI, EVPPI, EVSI and ENBS; ENBS combines population value and study cost; broader EVSI context | [PubMed/PMC record](https://pubmed.ncbi.nlm.nih.gov/32197720/) | **Supports.** Its abstract explicitly covers all four measures and good-practice EVPPI/EVSI method selection. |
| `claxton1999irrelevance` | VOI has a long methodological history in decision analysis and health economics | [DOI record](https://doi.org/10.1016/S0167-6296(98)00039-3) | **Supports.** A foundational health-technology decision-analysis paper; the manuscript makes only a historical claim. |
| `ades2004evsi` | EVSI in medical decision modelling; broader methodology for the normal-model illustration | [DOI record](https://doi.org/10.1177/0272989X04263162) | **Supports.** Directly concerns EVSI calculations in medical decision modelling. |
| `voi_cran2024` | `voi` implements EVPI, EVPPI, EVSI and ENBS | [CRAN record](https://CRAN.R-project.org/package=voi) | **Supports exactly.** CRAN lists all four measures and alternative computational methods. |
| `green2022bcea` | `BCEA` combines Bayesian cost-effectiveness analysis with VOI and graphical reporting | [JOSS article](https://doi.org/10.21105/joss.04206) and [CRAN record](https://CRAN.R-project.org/package=BCEA) | **Supports.** The package analyses cost/effect samples, makes graphical summaries and integrates `voi`. |
| `dampack2024` | `dampack` analyses and visualises decision-model outputs, sensitivity analysis and VOI | [CRAN record](https://CRAN.R-project.org/package=dampack) | **Supports.** It documents decision-model analysis/visualisation and deterministic, probabilistic and VOI vignettes. |
| `savi2025` | SAVI is a web application providing EVPI and regression-based EVPPI | [SAVI application](https://savi.shef.ac.uk/SAVI/) | **Supports exactly.** The site describes online EVPI and single/group EVPPI from PSA input. |
| `strong2014evppi` | SAVI's regression-based EVPPI method | [DOI record](https://doi.org/10.1177/0272989X13505910) and [SAVI method statement](https://savi.shef.ac.uk/SAVI/) | **Supports exactly.** SAVI identifies this paper as its non-parametric regression method for multiparameter EVPPI. |
| `tuffaha2021webtools` | Web tools differ in methods and required inputs | [DOI record](https://doi.org/10.1007/s40258-021-00662-4) | **Supports.** A review of web-based VOI tools; the manuscript makes the restrained comparative claim only. |
| `andronis2016implementation` | Implementation-adjusted methods consider incomplete change in practice | [DOI record](https://doi.org/10.1177/0272989X15614814) | **Supports exactly.** The paper adjusts expected VOI for implementation. |
| `adamczewski2022valueofinformation` | `value-of-information` estimates the value of a noisy signal in a simplified binary decision | [pinned source](https://github.com/tadamcz/value-of-information/tree/8d4ce9610effae941dc31eadf897f1fca5c3be60) and [project interface](https://valueofinfo.com/) | **Supports.** The project describes the expected benefit of a noisy signal for one uncertain quantity and a binary decision boundary. |
| `mordaunt2025trdcea` | `trd-cea-toolkit` situates VOI in a disease-specific health-economic workflow | [PyPI record](https://pypi.org/project/trd-cea-toolkit/0.4.0/) | **Supports.** A bounded self-citation to the named disease-specific toolkit, not a general methodological claim. |
| `vop_poc_nz2026` | Related same-author project publishes versioned formats designed to work with `voiage` | [pinned VOP commit](https://github.com/edithatogo/vop_poc_nz/commit/2c46db2fe5f907d894bb07f1127c008f10ee462e) | **Supports.** The revision contains the `vop-voiage` compatibility contract and versioned exchange bundle. The manuscript correctly denies independent adoption. |
| `voiage2026` | Source code and signed v2.0.0 release are public | [signed release](https://github.com/edithatogo/voiage/releases/tag/v2.0.0) | **Supports exactly.** First-party release evidence, not a methodological citation. |
| `voiage_software_heritage` | Software Heritage snapshot includes the signed release | [archived snapshot](https://archive.softwareheritage.org/swh:1:snp:31f89375852737bb9eb62ebc03fadfbc7ff70c2d) | **Supports subject to snapshot inspection.** The SWHID is the cited immutable archive; release inclusion is also checked against repository release evidence. |

## Disposition

All 18 citation occurrences resolve to these 15 records. No cited claim needs
wording change. Software capability descriptions are checked against official
package pages; first-party project/release statements are checked against pinned
first-party records. The author should review this ledger and explicitly
confirm the final source check; until then the readiness manifest records that
human gate as pending.
