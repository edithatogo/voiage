# Track Specification: Rust Release Submission

## Overview

This track closes the Rust release submission path. It keeps the crates.io
publish flow aligned with the documented `rust-v*` tag pattern and GitHub
Release artifact generation.

## Functional Requirements

1. Keep the Rust crate version and `rust-v*` tag pattern aligned.
2. Keep the `cargo fmt`, `cargo clippy`, `cargo test`, `cargo doc`, and
   `cargo package` gates intact.
3. Keep the crates.io publish step explicit about its token requirement.
4. Keep the GitHub Release source-archive path documented.

## Non-Functional Requirements

1. Preserve the Rust crate as the canonical execution core.
2. Avoid runtime behavior changes in this track.
3. Keep the release path reproducible and explicit.

## Acceptance Criteria

1. The Rust release docs and checklist agree on the crates.io path.
2. The docs clearly state that publication happens on `rust-v*` tags when
   credentials are available.
3. The release assets remain attached to GitHub Releases.

## Out of Scope

1. Changing Rust runtime semantics.
2. Replacing the existing crates.io publish flow.
3. Claiming registry submission succeeded without verification.
