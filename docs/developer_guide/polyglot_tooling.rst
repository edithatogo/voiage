Polyglot Tooling and Observability
==================================

The Python package is the reference implementation, but the project now ships
release contracts for several language bindings. This page records the
developer-facing expectations that keep those bindings aligned.

Tooling parity
--------------

The Python reference package keeps the language-specific tools that have a
clear local benefit:

* ``scalene`` is kept for Python-only profiling.
* ``mutmut`` is kept for Python-only mutation testing.
* ``pytest-benchmark`` is kept for Python performance regression checks.
* ``ruff`` and ``ty`` remain the Python lint and type gates.

Those tools are not forced onto the non-Python bindings. The equivalent gates
for the other package ecosystems are the native build, lint, test, packaging,
and conformance checks documented in the release matrix:

* TypeScript: ``npm run check`` and ``npm pack --dry-run``
* Go: ``go test`` and ``go vet``
* Rust: ``cargo fmt``, ``cargo clippy``, ``cargo test``, ``cargo package``
* Julia: ``Pkg.test``
* .NET 11: ``dotnet build``, ``dotnet test``, ``dotnet pack``
* R: ``R CMD build`` and ``R CMD check --as-cran``

The release matrix in `docs/release/polyglot-bindings.md` describes the
package-manager targets and tag conventions that go with those gates.

The canonical version source and the manifest lockstep rule are documented in
`docs/developer_guide/versioning_and_release_policy.rst`. Use that policy page
when changing any release-manifest version.

Versioning contract
-------------------

Each binding follows the release conventions of its own ecosystem:

* Python tags trigger PyPI/TestPyPI publication, while the conda-forge recipe
  is updated separately through the feedstock flow.
* TypeScript uses ``typescript-v*`` tags and publishes to npm with provenance.
* Go uses ``bindings/go/v*`` tags that line up with the module proxy rules.
* Rust uses ``rust-v*`` tags and publishes to crates.io.
* Julia uses ``julia-v*`` tags with TagBot sync and the General registry.
* .NET uses ``dotnet-v*`` tags and publishes ``net11.0`` packages to NuGet.
* R uses ``r-v*`` tags and publishes source artifacts through GitHub Releases,
  with CRAN or r-universe still handled externally.

Logging policy
--------------

Library code should stay quiet by default. The CLI is the only surface that is
expected to emit user-facing status output, and even there the output must stay
stable for scripting.

The current policy is:

* ``--quiet`` suppresses status chatter.
* ``--format json`` and ``--format csv`` are the machine-readable output modes.
* Debug or verbose detail must remain opt-in and must not change stdout
  payloads that users rely on for automation.

When a binding does not have a direct equivalent for a Python-only tool, the
fallback is to keep the gap explicit rather than invent a weak substitute.
That keeps the quality gates honest and makes the release contract easier to
audit across languages.
