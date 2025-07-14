Getting Started
===============

This section provides a quick introduction to using `voiage`.

Installation
------------
First, ensure you have `voiage` installed. For the v0.1 release, you can install it from TestPyPI:

.. code-block:: bash

   pip install --index-url https://test.pypi.org/simple/ --no-deps voiage

Basic Usage: Expected Value of Perfect Information (EVPI)
---------------------------------------------------------

The Expected Value of Perfect Information (EVPI) is the maximum amount one should be willing to pay to eliminate all uncertainty.

Here's a simple example of how to calculate EVPI using `voiage`.

.. code-block:: python

   import numpy as np
   from voiage.core.data_structures import NetBenefitArray
   from voiage.methods.basic import evpi

   # Simulate net monetary benefit (NMB) for two interventions across PSA samples
   # Let's assume 1000 PSA samples
   num_samples = 1000
   nmb_intervention_A = np.random.normal(loc=20000, scale=5000, size=num_samples)
   nmb_intervention_B = np.random.normal(loc=25000, scale=7000, size=num_samples)

   # Combine into a NetBenefitArray
   # The columns represent interventions, rows represent PSA samples
   nmb_data = np.vstack([nmb_intervention_A, nmb_intervention_B]).T
   net_benefit_array = NetBenefitArray(nmb_data)

   # Calculate EVPI
   # The `population`, `time_horizon`, and `discount_rate` parameters are optional
   # and used for scaling the EVPI to a population level. If not provided,
   # EVPI is calculated per-decision.

   # Example: Calculate per-decision EVPI
   evpi_value_per_decision = evpi(net_benefit_array)
   print(f"The per-decision EVPI is: {evpi_value_per_decision:.2f}")

   # Example: Calculate population-level EVPI
   population_size = 100000
   time_horizon_years = 10
   annual_discount_rate = 0.03

   evpi_value_population = evpi(
       net_benefit_array,
       population=population_size,
       time_horizon=time_horizon_years,
       discount_rate=annual_discount_rate,
   )
   print(f"The population-level EVPI is: {evpi_value_population:.2f}")

This example demonstrates how to set up your net benefit data and call the `evpi` function.

Basic Usage: Expected Net Benefit of Sampling (ENBS)
----------------------------------------------------

Once you have an EVSI (Expected Value of Sample Information) result and know the cost of research,
you can calculate the Expected Net Benefit of Sampling (ENBS).

.. code-block:: python

   from voiage.methods.sample_information import enbs

   # Assume a calculated EVSI value (e.g., from a previous EVSI analysis)
   # For this example, we'll use a dummy value.
   dummy_evsi_result = 500000.0 # Example EVSI value (e.g., population-level EVSI)
   cost_of_research = 150000.0 # Example cost of the research study

   # Calculate ENBS
   enbs_value = enbs(dummy_evsi_result, cost_of_research)
   print(f"The Expected Net Benefit of Sampling (ENBS) is: {enbs_value:.2f}")

   if enbs_value > 0:
       print("The research is potentially worthwhile as ENBS is positive.")
   else:
       print("The research may not be worthwhile as ENBS is non-positive.")

For more detailed usage and other VOI metrics (EVPPI, EVSI), please refer to the
`User Guide <user_guide/index.html>`_ and `API Reference <api_reference/index.html>`_ sections.
