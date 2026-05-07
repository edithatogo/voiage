Getting Started
===============

This section provides a quick introduction to using `voiage`.

Installation
------------
First, ensure you have `voiage` installed. The recommended development flow
uses `uv`:

.. code-block:: bash

   uv sync

For a published release, install from PyPI with:

.. code-block:: bash

   pip install voiage

Basic Usage: Expected Value of Perfect Information (EVPI)
---------------------------------------------------------

The Expected Value of Perfect Information (EVPI) is the maximum amount one should be willing to pay to eliminate all uncertainty.

Here's a simple example of how to calculate EVPI using `voiage`.

.. code-block:: python

   import numpy as np
   from voiage.analysis import DecisionAnalysis
   from voiage.schema import ValueArray

   values = np.array([
       [20000.0, 25000.0],
       [21000.0, 24800.0],
       [20500.0, 25250.0],
   ])
   value_array = ValueArray.from_numpy(values, ["Standard care", "New treatment"])
   analysis = DecisionAnalysis(nb_array=value_array)
   print(f"The per-decision EVPI is: {analysis.evpi():.2f}")

   population_evpi = analysis.evpi(population=100000, time_horizon=10, discount_rate=0.03)
   print(f"The population-level EVPI is: {population_evpi:.2f}")

This example demonstrates how to set up your net benefit data and call the `evpi` function.

Basic Usage: Expected Net Benefit of Sampling (ENBS)
----------------------------------------------------

Once you have an EVSI (Expected Value of Sample Information) result and know the cost of research,
you can calculate the Expected Net Benefit of Sampling (ENBS).

.. code-block:: python

   from voiage.methods.sample_information import enbs

   dummy_evsi_result = 500000.0
   cost_of_research = 150000.0

   enbs_value = enbs(dummy_evsi_result, cost_of_research)
   print(f"The Expected Net Benefit of Sampling (ENBS) is: {enbs_value:.2f}")

   if enbs_value > 0:
       print("The research is potentially worthwhile as ENBS is positive.")
   else:
       print("The research may not be worthwhile as ENBS is non-positive.")

For more detailed usage and other VOI metrics (EVPPI, EVSI), please refer to the
:doc:`User Guide <user_guide/index>` and :doc:`API Reference <api_reference/index>` sections.

For runnable notebook walkthroughs, use the example index in
:doc:`examples/index`:

* `Getting Started <../examples/getting_started.ipynb>`_
* `EVPI Validation <../examples/evpi_validation.ipynb>`_
* `EVPPI Validation <../examples/evppi_validation.ipynb>`_
* `EVSI Validation <../examples/evsi_validation.ipynb>`_
* `Interactive Tutorial <../examples/interactive_tutorial.ipynb>`_
* `Visualization Gallery <../examples/visualization_gallery.ipynb>`_
* `Network Meta-Analysis <../examples/nma_validation.ipynb>`_
* `Structural VOI <../examples/structural_voi_validation.ipynb>`_
* `Advanced Methods <../examples/advanced_methods.ipynb>`_
* `Financial VOI <../examples/financial_voi.ipynb>`_
* `Environmental VOI <../examples/environmental_voi.ipynb>`_
* `Engineering VOI <../examples/engineering_voi.ipynb>`_
