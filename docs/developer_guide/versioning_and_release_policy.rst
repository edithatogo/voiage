Versioning and Release Policy
=============================

``voiage`` uses semantic versioning and tag-driven releases. The Cargo
workspace version is authoritative, Maturin exposes it as dynamic Python
package metadata, and release tags are validated fail-closed against it. The
polyglot manifests remain explicit and synchronized to the same version.

Canonical version source
------------------------

The Python package declares a dynamic version in ``pyproject.toml``. Maturin
reads the authoritative Cargo workspace version when building package
metadata. The version-sync validator compares external binding manifests to
that version, and release validation rejects a ``v*`` tag that does not match
it exactly.

Binding manifests are expected to match the canonical version exactly:

* ``bindings/typescript/package.json``
* ``bindings/julia/Project.toml``
* ``bindings/rust/Cargo.toml``
* ``bindings/dotnet/src/Voiage.Core/Voiage.Core.csproj``
* ``r-package/voiageR/DESCRIPTION``

Release tags and package registries
-----------------------------------

The repository keeps ecosystem-specific tag and registry conventions:

* Python publishes from ``v*`` tags.
* TypeScript publishes from ``typescript-v*`` tags.
* Julia publishes from ``julia-v*`` tags.
* Rust publishes from ``rust-v*`` tags.
* .NET publishes from ``dotnet-v*`` tags.
* R publishes from ``r-v*`` tags.

The version numbers in those package manifests still need to line up with the
latest released Python tag. The tag prefixes and registry targets remain
ecosystem-specific; only the version value is shared.

Validation
----------

The repo-local validator checks the canonical version against the binding
manifests and fails if any manifest drifts:

.. code-block:: bash

   tox -e version-sync

or, equivalently:

.. code-block:: bash

   uv run python scripts/validate_version_sync.py

Release flow
------------

When preparing a release:

1. Update the Cargo workspace version and synchronized binding manifests.
2. Run the version-sync validator and fail-closed release-tag validation.
3. Cut the matching ``v<version>`` tag.
4. Let the tag-specific release workflows build and publish the Python package.

The validator keeps the repo honest before release automation starts.
