Getting Started
===============

This guide will walk you through the basics of using `voiage` to perform a Value of Information analysis.

Installation
------------

To install `voiage`, you can use `pip`:

.. code-block:: bash

    pip install voiage

Basic Usage
-----------

The main entry point for `voiage` is the `DecisionAnalysis` class. This class encapsulates the model, data, and decision options, providing a fluent interface for performing VOI calculations.

Here's a simple example of how to use `voiage` to calculate the Expected Value of Perfect Information (EVPI):

.. code-block:: python

    import numpy as np
    from voiage import DecisionAnalysis, ValueArray

    # Create a ValueArray object containing the net benefit values
    nb_array = ValueArray(
        values=np.array([[100, 105], [110, 100], [90, 110], [120, 100], [95, 115]]),
        strategy_names=["Strategy A", "Strategy B"],
    )

    # Create a DecisionAnalysis object
    analysis = DecisionAnalysis(parameters=None, values=nb_array)

    # Calculate the EVPI
    evpi = analysis.evpi()

    print(f"EVPI: {evpi}")

This will output:

.. code-block:: text

    EVPI: 6.0
