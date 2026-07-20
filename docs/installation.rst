Installation
============

voiage can be installed from PyPI using pip, or directly from the source code
if you want the latest development version or intend to contribute.

Prerequisites
-------------

*   Python 3.12-3.14.
*   `pip` and `setuptools` (usually included with Python).

We recommend using a virtual environment (e.g., via `venv` or `conda`) to
manage project dependencies.

Using pip (Recommended)
-----------------------

For the current published release, install it from PyPI using:

.. code-block:: bash

   pip install voiage

A full development installation can be done by cloning the repository and installing the `dev` dependencies:
.. code-block:: bash

   git clone https://github.com/edithatogo/voiage.git
   cd voiage
   uv sync --extra dev

Dependencies
------------
`voiage` relies on the following core libraries:
*   NumPy, SciPy, Pandas, and xarray for data structures and numerical computing.
*   NumPyro for Bayesian modeling.
*   scikit-learn and statsmodels for statistical modeling and machine learning.
*   Typer for the command-line interface.

Examples and validation notebooks are available under `examples/`.
