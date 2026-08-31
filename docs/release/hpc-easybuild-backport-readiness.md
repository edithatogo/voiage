# EasyBuild backport readiness

The [2024a foundation overlay](../../packaging/easybuild-overlay/README.md)
provides source-bound Python, scientific and CLI support candidates for
Voiage 2.2.0. Its real dependency dry-run and separate source-built support
consumer checks pass. Neither result establishes an installed EasyBuild stack.

The provider map avoids treating NumPy, SciPy and pandas as nonexistent
standalone catalogue modules. It preserves Voiage's SciPy and pandas upper
bounds and records the build-backend constraints found in downloaded sources.
The full native Arrow, Rust extension and JSON Schema closure is still
pending, as is the separate foss 2023a implementation.

Issue #1025 remains open until both requested toolchain variants have actual
build and installed-module evidence. A Linux ARM64 VM is a valid build host
for that architecture; it is not evidence of x86-64, scheduler or production
cluster behavior. Upstream submission remains a separate review boundary.
