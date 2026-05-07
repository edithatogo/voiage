Implementing New Methods
========================

This guide is for contributors adding or extending the stable core VOI
methods in ``voiage``.

The current stable method surface includes:

* ``evpi``
* ``evppi``
* ``evsi``
* ``enbs``
* ``calculate_ceaf``
* ``calculate_dominance``
* ``value_of_heterogeneity``

Those methods are the baseline for the public API and should remain the model
for new stable work. Experimental frontier methods follow their own contract
pages and are intentionally not covered here.

Before you start
-----------------

Review the existing implementations before adding a new method:

* ``voiage/methods/basic.py`` for EVPI and EVPPI
* ``voiage/methods/sample_information.py`` for EVSI and ENBS
* ``voiage/methods/ceaf.py`` for CEAF summaries
* ``voiage/methods/dominance.py`` for dominance and frontier calculations
* ``voiage/methods/heterogeneity.py`` for subgroup-aware value calculations

The stable method surface should continue to use the shared schema objects in
``voiage.schema`` and the orchestration layer in
``voiage.analysis.DecisionAnalysis``.

Method shape
------------

Keep the function signature explicit, typed, and easy to route from the
analysis facade.

.. code-block:: python

   def calculate_example(
       net_benefit: ValueArray,
       *,
       population: int | None = None,
       time_horizon: float | None = None,
       discount_rate: float | None = None,
   ) -> ExampleResult:
       """Calculate the example summary."""

       ...

Prefer the same design choices used elsewhere in the stable code:

* accept structured inputs rather than raw arrays where possible
* return a dataclass or result object for multi-field summaries
* keep the public wrapper on ``DecisionAnalysis`` thin
* preserve compatibility with the existing CLI and plotting surfaces

Implementation checklist
------------------------

1. Add the method implementation under ``voiage/methods/``.
2. Use the existing schema objects or add a new one in ``voiage/schema.py``
   if the method needs structured input.
3. Add a ``DecisionAnalysis`` wrapper if the method belongs on the main
   analysis surface.
4. Export the new method from ``voiage/methods/__init__.py`` and any other
   curated public surface that should expose it.
5. Add unit tests that cover:
   * normal inputs
   * boundary conditions
   * validation failures
   * population scaling or discounting logic where relevant
6. Add or update user-guide examples if the method is part of the stable
   documentation surface.
7. If the method is CLI-visible, add the command or option wiring and test it.
8. Update the changelog and todo tracking entry for the completed slice.

Stable-method guidance
----------------------

EVPI and EVPPI
  Keep the formulas explicit and document the assumptions around net benefit
  inputs, parameter samples, and any population adjustment.

EVSI and ENBS
  Keep the study-design inputs clear. ``ENBS`` should remain the net value of
  the chosen EVSI estimate minus research cost.

CEAF and dominance
  Preserve the frontier ordering, dominance classification, and ICER
  calculations. Keep plotting logic separate from calculation logic.

Value of Heterogeneity
  Keep subgroup definitions, weights, and aggregated welfare outputs separate
  so the summary stays transparent.

When a method is experimental
-----------------------------

If a method is not yet part of the stable core surface, keep the contract
explicit and documented as experimental in the user-facing guide. This page is
reserved for the stable methods that contributors can extend without changing
their public contract shape.

