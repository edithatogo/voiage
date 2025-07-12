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

Or for specific extras (e.g., `pymc` for PyMC integration, `plot` for plotting):

.. code-block:: bash

   pip install voiage[pymc,plot]

(Note: The exact names and availability of extras will be detailed in the
`pyproject.toml` file and release notes.)

Installing from Source (for Development)
----------------------------------------

If you want to install the latest development version or contribute to voiage:

1.  **Clone the repository:**

    .. code-block:: bash

       git clone https://github.com/doughnut/voiage.git
       cd voiage

2.  **Install in editable mode with development dependencies:**

    This allows you to make changes to the source code and have them immediately
    reflected in your environment. It will also install tools needed for
    testing, linting, and building documentation.

    .. code-block:: bash

       pip install -e .[dev]

    The `[dev]` extra typically includes pytest, Ruff, MyPy, Tox, Sphinx, etc.
    Check `pyproject.toml` for the exact list of development dependencies.

Verifying Installation
----------------------

After installation, you can verify that voiage is correctly installed by
opening a Python interpreter and trying to import it:

.. code-block:: python

   import voiage
   print(voiage.__version__)

This should print the installed version of voiage without errors.

Dependencies
------------

voiage relies on several core Python libraries from the scientific computing stack.
The main dependencies usually include:

*   NumPy
*   SciPy
*   Pandas
*   xarray (for labeled multi-dimensional data structures)

Optional dependencies for advanced features or specific methods might include:

*   PyMC (or other PPLs like NumPyro/Pyro) for Bayesian modeling.
*   JAX for automatic differentiation and GPU/TPU acceleration.
*   scikit-learn / statsmodels for regression-based metamodels in EVPPI/EVSI.
*   SALib for sensitivity analysis methods.
*   Dask / Joblib for parallelism.
*   Matplotlib / Seaborn / ArviZ for plotting.
*   Typer / Click for Command-Line Interface (CLI) functionality.

Development dependencies (for contributors) usually include:

*   pytest, pytest-cov (for testing and coverage)
*   Ruff (for linting and formatting)
*   MyPy (for static type checking)
*   Tox (for automating tests in different environments)
*   pre-commit (for running checks before commits)
*   Bandit (for security analysis)
*   Sphinx, sphinx-rtd-theme (for documentation building)

Please refer to the `pyproject.toml` file in the repository for the most
up-to-date and detailed list of dependencies and their version specifications.
