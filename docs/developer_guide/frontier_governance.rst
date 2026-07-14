Frontier VOI Architecture and Dependency Governance
====================================================

This document defines the final frontier VOI architecture, backend boundaries,
dependency policy, method maturity taxonomy, and non-conflicting implementation
sequence. It is the authoritative governance reference for all downstream
frontier, Rust-core, HPC, and registry follow-through tracks.

The machine-readable companion lives in ``voiage/governance.py`` and is
enforced by ``tests/test_frontier_governance.py``.

Method Maturity Taxonomy
-------------------------

Every frontier VOI method family carries exactly one maturity label from the
ordered promotion ladder:

1. **planned** -- designed but no runtime, no CLI, no fixtures.
2. **experimental** -- runtime implementation exists with a public API but no
   deterministic fixtures or cross-language parity.
3. **fixture-backed** -- deterministic normative fixtures, a schema, and a
   frontier registry entry exist. Cross-language parity is not yet proven.
4. **stable** -- production-ready with cross-language parity, Rust-kernel
   parity (where applicable), full documentation, and approved stable promotion.

Promotion from one level to the next requires the criteria encoded in
``MATURITY_LEVELS`` inside ``voiage/governance.py``. No track may mark a method
``stable`` without evidence from the corresponding stable-promotion track.

Backend Boundary
-----------------

The architecture is split into five layers with strict ownership rules.
Backend selection dispatch belongs in ``backends``, never in ``methods``.
VOI semantics belong in ``methods``, never in ``backends`` or ``cli``.
The full ownership table is in ``BACKEND_OWNERSHIP`` inside
``voiage/governance.py``.

Dependency Policy
------------------

The base install stays lightweight and conflict-free. Heavy or bleeding-edge
backends (JAX, PyTorch, CuPy) are always optional extras. The split between
``base`` and ``optional`` is documented in ``DEPENDENCY_POLICY`` inside
``voiage/governance.py`` and enforced by ``validate_dependency_policy()``.

This ensures that ``pip install voiage`` never pulls in a multi-gigabyte
GPU stack. Optional dependencies are activated via extras
(e.g. ``pip install voiage[plotting]`` or ``voiage[deep_learning]``).

Non-Conflicting Implementation Sequence
----------------------------------------

Downstream tracks must follow this order to avoid conflicts:

1. Architecture and dependency governance (this track).
2. Dataset registry and example corpus.
3. Rust frontier numerics migration completion.
4. Frontier method runtime completion and stable promotion.
5. External registry publication and HPC production evidence.

Tracks that depend on maturity taxonomy decisions or backend boundary rules
must not proceed until this governance track lands. There are **no conflicts**
between the governance decisions and the existing frontier registry, fixture
schemas, or backend abstraction layer.
