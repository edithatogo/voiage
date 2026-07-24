# JOSS panel round 1: synthesis and disposition

Date: 24 July 2026

This is an internal simulated review, not a JOSS editorial decision. Eight
independent roles reviewed the manuscript, repository, release evidence, and
individual sentences against the shared 1,000-point rubric.

## Panel result

| Role | Score |
| --- | ---: |
| Applied-research accessibility reviewer | 688 |
| Sentence editor | 771 |
| Simulated Associate Editor-in-Chief | 780 |
| Domain and VOI-methods reviewer | 784 |
| Numerical verifier | 797 |
| Research-software reviewer | 815 |
| Reproducibility and packaging reviewer | 816 |
| Handling editor | 859 |

The fail-closed panel score is **688/1000**. Every reviewer recommended major
revision. The score is the minimum reviewer score, not an average.

## Convergent findings

All or nearly all reviewers identified the following:

1. The public EVSI claim was broader than the validated scientific evidence.
   The worked example used a separate normal-normal calculation, while generic
   public estimators did not declare a sampling likelihood and posterior update.
2. The installed R package did not execute its claimed Rust EVPI path.
3. The manuscript described author-created demonstrations as research uses.
4. Cross-language claims exceeded the EVPI-only shared surface.
5. The v1.0.0 release did not contain the claimed SBOM asset.
6. The state-of-the-field comparison omitted `dampack`, indirectly cited SAVI,
   and contained an incorrect `voi` authorship record.
7. The manuscript used a maturity label absent from the governed taxonomy.
8. The worked health example reported mechanics but not its results.
9. Engineering terminology displaced the applied decision problem and its
   consequences for non-developer readers.
10. Availability material was placed under References and did not identify an
    exact reviewed revision.

## Prioritised remediation

### P0: scientific and runtime blockers

- Add a Rust-native analytical normal-normal EVSI kernel with prespecified
  reference tests and expose it through Python.
- Correct the default two-loop posterior update and add simulation-recovery
  evidence.
- Mark compatibility EVSI estimators that do not declare a likelihood and
  posterior update as non-stable.
- Route the health-example calculation through the public package API.
- Repair and genuinely test the installed R-to-Rust EVPI path.

### P1: evidence and governance

- Reconcile method maturity with the governed promotion ladder and keep
  approximation/backend status on separate metadata axes.
- Produce revision-bound SBOM and release-evidence records without claiming
  assets absent from v1.0.0.
- Name the exact software revision reviewed by the JOSS manuscript.

### P2: manuscript and citations

- Reframe the Summary around the applied questions VOI answers.
- Explain the decision/result information preserved across tools in ordinary
  language.
- Limit R and Julia claims to capabilities demonstrated by installed tests.
- Add the health-example findings and classify both repository examples
  accurately.
- Add `dampack`, a direct SAVI source, and good-practice EVSI guidance; correct
  the `voi` authorship record.
- Replace unsupported domain and impact claims with bounded present evidence.
- Move availability evidence to a dedicated section and remove peripheral
  citation-padding language.

### P3: final verification

- Render and inspect the JOSS PDF.
- Run repository, citation, prose, package, native, and release-evidence checks.
- Re-run all eight panel roles against the revised paper and repository.
- Continue revision until every reviewer scores at least 996/1000 with no
  fail-closed blocker.

## External gates

The following cannot be manufactured by repository changes:

- attributable non-author installation or research-use evidence under issue
  #471;
- JOSS editorial screening, reviewer assignment, review, and acceptance;
- the acceptance-stage archival DOI;
- an arXiv identifier if the author retains arXiv-first sequencing.

These gates must be reported accurately and kept separate from manuscript
quality scores.
