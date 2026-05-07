Versioning and Release Policy
=============================

``voiage`` uses semantic versioning and tag-driven releases, but the
repository needs one canonical version source so manifests and release
automation stay in lockstep.

Canonical version source
------------------------

The current release line treats ``pyproject.toml`` as the canonical source of
truth for the repository version. That version drives the Python package
metadata, the release tag checks, and the validation of the binding manifests.

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
canonical repository version. The tag prefixes and registry targets remain
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

When changing versions:

1. Update the canonical version in ``pyproject.toml``.
2. Update the binding manifests to the same version.
3. Run the version-sync validator.
4. Cut the matching tag and let the ecosystem-specific release workflows run.

The validator keeps the repo honest before release automation starts.
