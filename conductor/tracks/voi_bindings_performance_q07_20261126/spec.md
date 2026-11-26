# Specification: Q07 Thin Bindings & Measured Computation

Covers roadmap packages **V12** (installed cross-language bindings), **F04** (array & execution performance), and **F05** (resumable & observable research execution).

## Acceptance Criteria
1. Exercise positive, invalid, and unavailable-method cases across installed Rust, Python, R, and Julia consumers with zero silent Python fallback for native claims.
2. Profile `.tolist()` crossings, allocation/sorting, and serialization; verify buffer ownership, lifetimes, shape/stride handling, and copy counts before claiming zero-copy.
3. Benchmark deterministic block/streaming reductions (`F05`) with explicit thread count, cancellation, checkpoint/restart, and partial-result semantics.
4. Preserve Apple Silicon MLX/Metal no-go and HPC deferral; keep independent CPU acceptance.
