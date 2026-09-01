# EasyBuild foss 2023a readiness

The [2023a foundation](../../packaging/easybuild-2023a-overlay/README.md)
prepares current Python 3.12.14 bindings and compatible scientific, support and
build-backend recipes without mixing GCC generations. Its source hashes,
ordered providers and dependency dry run are candidate evidence.

The packet does not establish a native build or a complete Voiage graph.
Four native build-only interpreter overrides preserve all other recipe bytes;
the inherited numerical test policy remains explicit. Issue #1025 stays open until the full
requested stacks and installed-module checks have passed; upstream submission
is a separate action.
