# Deterministic sensitivity analysis v1 experimental contract

This versioned contract governs issue #556 and delivery subissue #725. It
represents one-way, explicitly feasible two-way and named scenario evaluations
as normalized deterministic records. Every record retains the complete
coordinate vector, raw alternative outcomes and declared units so that a
consumer does not mistake a tornado range for a probability, EVPPI or global
sensitivity measure.

Python provides an experimental shared evaluator for callback and normalized
record inputs under #726. Rust, R and Julia remain unsupported; Mojo remains
outside the repository boundary. This executable claim does not imply stable
status, scientific approval or cross-language parity.
