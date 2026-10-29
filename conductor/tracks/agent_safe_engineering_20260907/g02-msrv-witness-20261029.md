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
