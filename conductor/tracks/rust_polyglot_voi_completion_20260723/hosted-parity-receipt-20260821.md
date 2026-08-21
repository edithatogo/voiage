# Hosted installed-parity receipt — 2026-08-21

PR #992 tested exact head `21b6a073328d2ccde5b032b13c0928e926293c9e`
and merged as `69241d94efed35ad6511bd53d31725e0200df418`.
The retained-bindings workflow
[run 32429467957](https://github.com/edithatogo/voiage/actions/runs/32429467957)
completed successfully.

The run passed Rust at the declared 1.85 MSRV and current toolchain, R
package-development checks and clean installed native tests on Linux, Windows,
and macOS, Julia 1.10–1.12 on Linux, Windows, and macOS with an isolated depot,
and the shared cross-language differential-conformance job. The companion
[CI run 32429467998](https://github.com/edithatogo/voiage/actions/runs/32429467998)
passed Python 3.12–3.14, frontier contracts, and coverage. Across the exact head,
GitHub reported 60 successful checks, four intentional skips, one neutral
CodeQL wrapper result, no failures, and no pending checks before merge.

This closes the repository-owned installed parity evidence action for the
advertised Rust, Python, R, and Julia surfaces. Mojo remains explicitly
unsupported. M22–M31 remain experimental under the owner Option A decision;
this receipt does not assert stable promotion, independent review, publication,
registry acceptance, or broader scientific validity.
