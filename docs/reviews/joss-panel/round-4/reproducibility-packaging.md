# Round 4 reproducibility and packaging review

Date: 24 July 2026  
Score: 874/1000  
Recommendation: major revision

This is an internal AI-assisted readiness simulation, not a formal JOSS review.

## Reproduced evidence

The reviewer independently built and installed the local v2 Python wheel,
regenerated the paper values and figures, tested the Rust kernels, installed
the R package against the native library, and loaded the Julia package against
that library. The numerical outputs reproduced.

## Blocking defects

- Public GitHub, PyPI, and crates.io releases remain at v1.0.0.
- The paper describes v2 functionality absent from v1.
- CFF, CodeMeta, and README metadata present v2 as released before live
  publication.
- The reviewed PDF predates the current source.
- Julia tests depend on a fixture outside the standalone Julia package.
- R and Julia require separately provisioned native libraries.
- The reviewed hosted compatibility-and-coverage job failed changed-line and
  changed-branch thresholds.

## Required corrections

1. Make the Julia package test fixture self-contained and test a standalone
   package copy.
2. Regenerate and commit all paper outputs with the final terminology.
3. Add one exact reproduction command and a checksum manifest.
4. Publish and verify the v2 release assets, provenance, SBOM, and evidence
   manifest.
5. Request a matching Software Heritage snapshot.
6. Replace prospective availability language with verified release facts.
