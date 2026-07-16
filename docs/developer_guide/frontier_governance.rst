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

Stable-Promotion Evidence Matrix
---------------------------------

The machine-readable matrix is stored at
``specs/frontier/governance/promotion-matrix.json`` and mirrored by
``PROMOTION_MATRIX``. It keeps maturity labels separate from evidence states:
``experimental`` requires only a runtime and public API; ``fixture-backed``
adds deterministic fixtures, schemas, and a registry entry;
``cross-language-parity`` adds verified binding parity; and ``stable`` also
requires Rust-kernel parity where applicable, complete documentation,
changelog and migration-guide entries, an explicit promotion approval, and
public-API compatibility evidence.

The ``validate_promotion_evidence`` helper rejects incomplete evidence. In
particular, a family must not be described as stable merely because its
fixtures parse or because one language binding happens to agree. The current
frontier registry intentionally contains no stable frontier family until this
matrix is satisfied.

The family-level application of the matrix is maintained in
``specs/frontier/governance/promotion-checklist.json``. It enumerates every
fixture-registry family and records the repository owner, current state, next
gate, blocker state, and artifact paths. The checklist is explicit about
``stable_claim_allowed`` so downstream documentation and release tooling have
a safe default. A false value is intentional evidence that the family is not
yet stable; it is not a placeholder for an eventual approval.

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

Experimental Design Interfaces
-------------------------------

``voiage.experimental_design`` provides backend-neutral summaries for
expected information gain, cost-aware Bayesian optimal experimental design,
active learning, and amortized EVSI. These APIs consume simulated arrays or
precomputed candidate scores; fitting models with NumPyro, JAX, or
simulation-based inference remains an optional backend concern. The
interfaces are experimental and do not authorize stable claims until the
promotion matrix, deterministic fixtures, and cross-language parity gates
are satisfied.

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
