# EasyBuild backport readiness

The [2024a foundation overlay](../../packaging/easybuild-overlay/README.md)
provides source-bound Python, scientific and CLI support candidates for
Voiage 2.2.0. The current candidate uses Python 3.12.14 throughout its resolved
build/test closure and OpenSSL 3.5.8 for the source fallback. Original
Python 3.12.3 receipts remain historical. Its real dependency dry-run and separate source-built support
consumer checks pass. Neither result establishes an installed EasyBuild stack.

The provider map avoids treating NumPy, SciPy and pandas as nonexistent
standalone catalogue modules. It preserves Voiage's SciPy and pandas upper
bounds and records the build-backend constraints found in downloaded sources.
Both foss 2023a and 2024a root recipes now resolve their complete declared
provider graphs with EasyBuild 5.4.0 against the pinned catalogue. Each
sanitized robot log contains 108 modules. The 2023a graph contains only Python
3.12.14. The 2024a graph keeps its root runtime and Voiage providers on Python
3.12.14 while inheriting one catalogue build-tool edge: libpciaccess 0.18.1
uses Meson 1.4.0, which uses Python 3.12.3. The validator binds that exception
to build dependencies so it cannot silently become a root runtime dependency.
The receipt is
[`root-graph-resolution.json`](../../packaging/easybuild/root-graph-resolution.json).

Native Arrow, Rust-extension, JSON Schema and full Voiage builds remain
pending for both generations. The robot evidence parses and resolves the graph;
it does not compile it or qualify installed modules.

Issue #1025 remains open until both requested toolchain variants have actual
build and installed-module evidence. A Linux ARM64 VM is a valid build host
for that architecture; it is not evidence of x86-64, scheduler or production
cluster behavior. Upstream submission remains a separate review boundary.
