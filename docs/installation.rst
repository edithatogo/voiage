Installation
============

voiage can be installed from PyPI using pip, or directly from the source code
if you want the latest development version or intend to contribute.

Prerequisites
-------------

*   Python 3.8 or higher.
*   `pip` and `setuptools` (usually included with Python).

We recommend using a virtual environment (e.g., via `venv` or `conda`) to
manage project dependencies.

Using pip (Recommended)
-----------------------

For the v0.1 release, `voiage` is available on TestPyPI. You can install it using:

.. code-block:: bash

   pip install --index-url https://test.pypi.org/simple/ --no-deps voiage

Once `voiage` is officially released on PyPI, you will be able to install it with:

.. code-block:: bash

   pip install voiage

To install with optional dependencies that might be needed for specific
features (e.g., advanced Bayesian modeling, specific EVSI methods, or
plotting), you might use:

.. code-block:: bash

   pip install voiage[all]  # Example, actual extras will be defined

Or for specific extras (e.g., `numpyro` for NumPyro integration, `plot` for plotting):
.. code-block:: bash

   pip install "voiage[numpyro,plot]"

A full development installation can be done by cloning the repository and installing the `dev` dependencies:
.. code-block:: bash

   git clone https://github.com/your-username/voiage.git
   cd voiage
   pip install -e ".[dev]"

Dependencies
------------
`voiage` relies on the following core libraries:
*   NumPy, SciPy, Pandas, and xarray for data structures and numerical computing.
*   NumPyro for Bayesian modeling.
*   scikit-learn and statsmodels for statistical modeling and machine learning.
*   Typer for the command-line interface.

