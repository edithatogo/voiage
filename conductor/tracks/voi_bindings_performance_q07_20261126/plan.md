# Implementation Plan: Q07 Thin Bindings & Measured Computation

- [ ] **Q07.1** — Extend cross-language differential parity harnesses (`tests/test_q02_rust_authority_parity.py`, `bindings/julia/`, `r-package/voiageR/`) for newly promoted `Q04`–`Q06` methods (`V12`).
- [ ] **Q07.2** — Eliminate redundant `.tolist()` / copy crossings in PyO3 and Arrow interchange (`F04`), verifying shape/stride/lifetime invariants.
- [ ] **Q07.3** — Extend deterministic checkpoint/restart and cancellation boundaries (`F05`) to EVPPI/EVSI streaming reductions with exact-bit replay witnesses.
- [ ] **Q07.4** — Record cold/warm CPU benchmark and memory receipts and archive track.
