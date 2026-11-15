# Proptest 1.12 compatibility hold

PR #1281 remains open as a deferred dependency update. It widens the Proptest
development dependency to admit version 1.12. The
[Proptest 1.12 manifest](https://github.com/proptest-rs/proptest/blob/v1.12.0/proptest/Cargo.toml)
requires Rust 1.88, while the
[Voiage workspace](https://github.com/edithatogo/voiage/blob/cee30c63bc8c4edfe8968544315d21065fb34142/rust/Cargo.toml)
retains a Rust 1.85 minimum.

Do not merge this update until the selected dependency set is shown to work at
the maintained minimum, or a separately reviewed compatibility-policy change
raises that minimum and its affected build, test, package and consumer gates
pass. This note records the hold; it does not approve an MSRV change.
