# Round 5 reproducibility and packaging review

Date: 24 July 2026
Reviewed head: `07cf3f83098952439f7b94773c62a8cbf9c176c1`
Score: 754/1000
Disposition: major revision

This is an independent AI-assisted reproducibility review.

## Reproduced evidence

- A clean locked environment regenerated all three CSV files and both figure
  formats.
- Every tracked reproduction hash matched.
- Reported EVPI, EVPPI, EVSI, ENBS, intervals, and seeds matched.
- The JOSS validator and focused tests passed.
- The Open Journals workflow built a clean six-page PDF.
- The complete hosted matrix ultimately passed at the reviewed head.
- R and Julia source bindings returned the expected Rust EVPI result when
  supplied with the native library.
- Public Python v1 installed cleanly, but did not contain the revised EVSI
  contract.

## Deductions

| Area | Maximum | Awarded |
| --- | ---: | ---: |
| Numerical reproduction and traceability | 200 | 185 |
| Python packaging and installation | 170 | 112 |
| Rust, R, and Julia installation | 160 | 112 |
| CI, tests, and build evidence | 160 | 160 |
| Release, provenance, and archive | 160 | 58 |
| Licence, support, and documentation | 90 | 82 |
| Manuscript precision | 60 | 45 |

## Blocking findings

1. `voiage==2.0.0` did not exist on PyPI or GitHub.
2. The paper described functionality absent from its cited v1 release.
3. The Software Heritage identifier covered v1, not the reviewed revision.
4. CI evidence was revision-bound rather than release-bound.
5. Research-use evidence remained absent.

## Reproduction improvements

- Regenerate portable manuscript outputs in a clean directory in CI.
- Record source revision, lockfile digest, environment, command, inputs, seeds,
  and synthetic-data status in a structured manifest.
- Call the JOSS PDF rebuildable rather than byte-reproducible because build
  timestamps vary.
- Provide an output-directory or verification mode that does not overwrite
  tracked source artifacts.
- Keep the separate native-library requirement explicit for R and Julia.
