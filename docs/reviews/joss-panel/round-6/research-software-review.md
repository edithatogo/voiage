# Round 6 research-software and reproducibility review

**Manuscript score at review snapshot: 891/1000.**
**Full-package score at review snapshot: 826/1000.**

This is an AI-assisted internal review, not a JOSS reviewer report.

The review verified the Rust EVPI implementation through Python, R, and Julia,
but found that the original evidence ledger overstated the R fixture test. The
shared R fixture used a pure-R replacement, while a separate native test
exercised one Rust-backed case.

Accepted revisions:

- state that Python and Julia use hand-calculated cases and R uses a native
  Rust-backed case;
- cite both fixture and native-path evidence where relevant;
- retain the separate-native-library limitation for R and Julia;
- run the hosted paper workflow for the exact reviewed source;
- keep v2 release and archive claims pending until those objects exist.

The review found strong packaging and test infrastructure but no remotely
reproducible evidence for the then-uncommitted manuscript revision.
