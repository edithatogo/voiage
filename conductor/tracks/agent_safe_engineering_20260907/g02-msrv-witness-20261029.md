# G02 MSRV witness continuation — 29 October 2026

Chat `01a17ab7-148e-7042-be1e-03a9478643da` owns the isolated worktree
`/Users/doughnut/.codex/worktrees/msrv-rejection-witness/voiage` on branch
`codex/msrv-rejection-witness`. The scope is the offline MSRV fixture script,
its tests, the Rust MSRV workflow and this track's G02 record. There is no
native service scope. The next operation is clean candidate qualification and
integration, following the existing owner and full-gate rules.

The previous policy test checked version comparisons but did not exercise
Cargo. On macOS, `rustup run 1.76.0 cargo` also found Homebrew `rustc` 1.99.1
on PATH; pinning both tools is necessary to avoid a false witness.

A guarded isolated run using Cargo and rustc 1.85.0 proved that the locked,
offline fixture succeeds with its optional backend disabled, rejects a backend
requiring 1.86, and succeeds with a compatible 1.85 backend. A separate
negative run rejected the mixed 1.85 Cargo / 1.99 compiler selection before
building. The Python policy/tool-selection tests and actionlint passed locally.
These are bounded local witnesses, not a complete feature matrix or track
acceptance. Full local and current-head hosted gates remain required.

Hosted review of the first candidate identified three failures. The MSRV job
selected Cargo 1.85.1 because the toolchain selector was `1.85`; it now selects
`1.85.0`, the explicitly verified compiler and Cargo pair. The repository harness
identified a frozen telemetry input modified only to rename a policy test; that
file is restored byte-for-byte. The R requalification record now binds the new
bindings workflow hash while retaining `hosted_pending`, without asserting new
R qualification. Guarded local harness, submission-readiness validator and 44
focused tests passed; current-head hosted evidence remains required.

The second hosted revision passed its MSRV, repository-harness and frontier
jobs but exposed two coverage failures: the binding lifecycle test still matched
the former `1.85` selector as a literal string. Its assertion now checks the
explicit `1.85.0` selector; the affected binding and tool-selection tests pass
under the resource guard. Current-head hosted and full local gates remain open.
