Architecture
============

This guide describes the current `voiage` implementation shape so contributors
can extend it without breaking the public contract.

Module structure
-----------------

- ``voiage.schema`` holds the structured in-memory contract objects.
- ``voiage.analysis`` provides the orchestration layer and the
  ``DecisionAnalysis`` facade.
- ``voiage.methods`` contains the method implementations.
- ``voiage.plot`` contains plotting helpers that stay separate from
  calculation logic.
- ``voiage.cli`` exposes the command-line surface for the public workflows.

Backend abstraction
--------------------

The library keeps the computational backend behind a small dispatch layer.
NumPy remains the default runtime path, while JAX is an optional acceleration
backend rather than a user-facing contract requirement.

The key rule is that the backend should not change the analysis semantics.
Only performance and execution details are allowed to vary.

DecisionAnalysis flow
---------------------

The central flow is:

``input -> schema -> method -> result -> plot/CLI output``

``DecisionAnalysis`` accepts raw arrays or schema objects, normalizes them into
the shared data structures, and then routes calculations through the public
method surface. This keeps the user-visible API thin while preserving a single
source of truth for validation and result formatting.

Data flow and boundaries
------------------------

- Inputs should enter through schema objects where possible.
- Methods should return structured result objects for multi-field analyses.
- Plotting should consume result objects rather than recomputing analysis.
- CLI commands should compose the public analysis methods instead of
  reimplementing them.

When you add a new feature, keep the implementation in the narrowest layer that
owns the behavior. Contracts and validation belong in ``schema`` or the
relevant spec; orchestration belongs in ``analysis``; numerical logic belongs
in ``methods``.
