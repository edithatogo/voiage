# Round 5 research-software review

Date: 24 July 2026
Reviewed head: `07cf3f83098952439f7b94773c62a8cbf9c176c1`
Score: 673/1000
Disposition: major revision

This is an independent AI-assisted research-software review.

## Assessment

The Rust, Python, R, and Julia tests provide strong evidence of numerical
correctness and engineering maturity. The paper's central claim exceeded the
executable interfaces:

- common Python EVPI and EVPPI calls return scalars;
- Python has a separate opt-in provenance-bearing `run_evpi()` envelope;
- R and Julia accept matrices and return scalar EVPI;
- R and Julia do not carry units, assumptions, diagnostics, or provenance.

The three interfaces therefore demonstrate shared EVPI numerics, not a shared
semantic result envelope.

## Deductions

| Category | Maximum | Awarded |
| --- | ---: | ---: |
| Scope, significance, and use | 150 | 55 |
| Scientific communication | 150 | 122 |
| Claim-to-executable alignment | 150 | 80 |
| Functionality and numerics | 120 | 112 |
| Distribution and polyglot usability | 120 | 65 |
| Documentation and examples | 100 | 62 |
| Tests, CI, and reproduction | 100 | 90 |
| Governance and community pathways | 70 | 55 |
| Citation, archive, and metadata | 40 | 32 |

## Blocking findings

1. Demonstrate research use.
2. Either expose a shared semantic envelope through R and Julia or narrow the
   manuscript to the implemented scalar parity.
3. Publish and archive an exact installable release.
4. Correct broken examples, nonexistent APIs, and inaccurate architecture
   claims in public documentation.
5. Complete the AI attestation.

## Sentence-level findings

- Qualify every claim that the package retains context with the specific
  contract-based Python route.
- Say that Rust contains selected domain, serialization, and numerical
  components rather than implying every workflow is Rust-owned.
- State that current R and Julia bindings expose scalar EVPI.
- Move generic-estimator detail to method documentation.
- Replace v1/future-release language after release publication.
