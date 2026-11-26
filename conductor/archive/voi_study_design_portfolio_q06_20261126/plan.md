# Implementation Plan: Q06 Research Design & Portfolio Decisions

- [x] **Q06.1** — Freeze study-design, discounting, recruitment-delay, and portfolio-dependence contracts in `voiage-domain` and `rust/crates/voiage-numerics/src/enbs.rs` / `study_design.rs`.
- [x] **Q06.2** — Implement exact small-portfolio enumeration and overlap/dependence guards (`V06`, `F06`), rejecting incompatible EVSI summation.
- [x] **Q06.3** — Harden selective frontier extensions (`V10`: structural uncertainty, distributional equity, and adaptive/sequential design interfaces) with regret-based test oracles.
- [x] **Q06.4** — Wire `voiage/experimental/study_design.py`, verify `cargo test --locked` + installed-wheel `pytest`, and archive track.
