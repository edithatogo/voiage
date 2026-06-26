"""Frontier VOI architecture and dependency governance policy.

This module encodes the governance decisions that keep the frontier VOI
implementation conflict-free:

- :data:`MATURITY_LEVELS` / :data:`MATURITY_PROMOTION_ORDER` define the
  method maturity taxonomy and the ordered promotion ladder.
- :data:`BACKEND_OWNERSHIP` defines which architectural layer owns which
  responsibility so that cross-layer leaks are caught early.
- :data:`DEPENDENCY_POLICY` documents the split between the lightweight
  base install and the optional bleeding-edge backends.

The authoritative prose lives in
``docs/developer_guide/frontier_governance.rst``; this module is the
machine-readable companion that makes the policy testable.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------- #
# Method maturity taxonomy
# --------------------------------------------------------------------------- #

#: Ordered promotion ladder from least to most mature.
MATURITY_PROMOTION_ORDER: Final[tuple[str, ...]] = (
    "planned",
    "experimental",
    "fixture-backed",
    "stable",
)

#: Metadata for each maturity level.
MATURITY_LEVELS: Final[dict[str, dict[str, object]]] = {
    "planned": {
        "description": (
            "Method is designed but has no runtime implementation, no CLI, "
            "and no deterministic fixtures."
        ),
        "promotion_criteria": [
            "Runtime Python function exists and is importable.",
            "At least one unit test passes.",
        ],
    },
    "experimental": {
        "description": (
            "Method has a runtime implementation and a public API surface "
            "but lacks deterministic fixtures or cross-language parity."
        ),
        "promotion_criteria": [
            "Deterministic fixture-backed conformance contract exists.",
            "Frontier registry manifest lists the family.",
            "CLI entrypoint is wired if the method family is user-facing.",
        ],
    },
    "fixture-backed": {
        "description": (
            "Method has deterministic normative fixtures, a schema, and a "
            "frontier registry entry. Runtime is validated against fixtures "
            "but cross-language parity is not yet proven."
        ),
        "promotion_criteria": [
            "Cross-language conformance fixtures pass in Python and at least "
            "one binding.",
            "Rust-kernel parity evidence is recorded (or an explicit gate "
            "explains why Rust parity is deferred).",
            "Changelog or release note documents the fixture-backed surface.",
        ],
    },
    "stable": {
        "description": (
            "Method is production-ready: cross-language parity, Rust-kernel "
            "parity (where applicable), full documentation, and stable "
            "promotion approval."
        ),
        "promotion_criteria": [
            "All fixture-backed criteria are met and merged.",
            "Stable promotion track approves the maturity label change.",
            "Public API is frozen for the current major version.",
        ],
    },
}


def validate_maturity_label(label: str) -> None:
    """Raise :class:`ValueError` if *label* is not a governed maturity level.

    Parameters
    ----------
    label
        The maturity label to check.

    Raises
    ------
    ValueError
        If *label* is not one of :data:`MATURITY_PROMOTION_ORDER`.
    """
    if label not in MATURITY_PROMOTION_ORDER:
        raise ValueError(
            f"Unknown maturity label '{label}'. "
            f"Expected one of: {', '.join(MATURITY_PROMOTION_ORDER)}"
        )


# --------------------------------------------------------------------------- #
# Backend boundary ownership
# --------------------------------------------------------------------------- #

#: Maps each architectural layer to its owner and boundary.
BACKEND_OWNERSHIP: Final[dict[str, dict[str, str]]] = {
    "schema": {
        "owner": "voiage.schema / specs/",
        "boundary": (
            "Owns structured input/output contract objects, validation, and "
            "JSON schemas. Must not contain numerical computation or backend "
            "dispatch logic."
        ),
    },
    "methods": {
        "owner": "voiage.methods/",
        "boundary": (
            "Owns the numerical VOI algorithms (EVPI, EVPPI, EVSI, frontier "
            "families). May request a backend for array operations but must "
            "not perform backend selection dispatch or I/O."
        ),
    },
    "backends": {
        "owner": "voiage.backends / voiage.main_backends",
        "boundary": (
            "Owns NumPy/JAX/Metal/GPU/TPU backend selection, compilation, "
            "and benchmarking. Must not implement VOI semantics or CLI "
            "commands."
        ),
    },
    "cli": {
        "owner": "voiage.cli",
        "boundary": (
            "Owns command-line argument parsing, output formatting, and "
            "batch I/O. Must compose public analysis methods rather than "
            "reimplementing them."
        ),
    },
    "rust_core": {
        "owner": "bindings/rust/",
        "boundary": (
            "Owns the authoritative numerical kernels once the Rust-core "
            "migration lands. Python remains the public facade; the Rust "
            "core is called through a narrow ABI boundary."
        ),
    },
}

#: Responsibilities that are **not** owned by the ``methods`` layer.
_METHODS_FORBIDDEN_RESPONSIBILITIES: Final[frozenset[str]] = frozenset(
    {
        "backend selection dispatch",
        "backend dispatch",
        "cli argument parsing",
        "file i/o",
        "json schema generation",
    }
)


def validate_backend_boundary(layer: str, responsibility: str) -> None:
    """Raise :class:`ValueError` if *responsibility* leaks across *layer*.

    Parameters
    ----------
    layer
        The architectural layer (one of :data:`BACKEND_OWNERSHIP`).
    responsibility
        A lowercase description of the responsibility being checked.

    Raises
    ------
    ValueError
        If the layer does not own the given responsibility.
    """
    if layer not in BACKEND_OWNERSHIP:
        raise ValueError(f"Unknown architecture layer '{layer}'.")

    normalized = responsibility.lower().strip()
    if layer == "methods" and normalized in _METHODS_FORBIDDEN_RESPONSIBILITIES:
        raise ValueError(
            f"Layer 'methods' must not own '{responsibility}'. "
            "This responsibility belongs to 'backends' or 'cli'."
        )


# --------------------------------------------------------------------------- #
# Dependency policy
# --------------------------------------------------------------------------- #

#: Lightweight dependencies required for the base install.
DEPENDENCY_POLICY: Final[dict[str, list[str]]] = {
    "base": [
        "numpy",
        "scipy",
        "pandas",
        "xarray",
        "scikit-learn",
        "statsmodels",
        "typer",
        "defusedxml",
        "psutil",
        "typing_extensions",
    ],
    "optional": [
        "jax",
        "numpyro",
        "matplotlib",
        "seaborn",
        "torch",
    ],
}


def validate_dependency_policy() -> None:
    """Verify that no heavy optional dependency leaks into the base list.

    Raises
    ------
    ValueError
        If a known heavy backend appears in the base dependency list.
    """
    heavy = {"jax", "torch", "pytorch", "cupy", "numba"}
    for dep in DEPENDENCY_POLICY["base"]:
        if any(h in dep.lower() for h in heavy):
            raise ValueError(
                f"Heavy optional dependency '{dep}' must not be in the base "
                "install. Move it to the optional list."
            )
