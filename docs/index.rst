.. voiage documentation master file, created by
   sphinx-quickstart on Mon Dec 25 10:00:00 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to voiage's documentation!
===================================

**voiage** is a Python library for Value of Information (VOI) analysis.
VOI methods help quantify the economic value of acquiring additional
information to reduce uncertainty in decision-making, particularly in fields
like health economics, risk analysis, and decision sciences.

This library aims to provide a comprehensive suite of tools for calculating
various VOI metrics, including:

*   Expected Value of Perfect Information (EVPI)
*   Expected Value of Partial Perfect Information (EVPPI)
*   Expected Value of Sample Information (EVSI)
*   Expected Net Benefit of Sampling (ENBS)
*   And advanced VOI analyses for structural uncertainty, network meta-analysis,
    adaptive designs, research portfolios, sequential decisions, observational data,
    and model calibration.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   getting_started
   user_guide/index
   api_reference/index
   examples/index
   cross_domain_usage
   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


---

**Note**: This documentation is currently under development alongside the voiage library.
Content will be expanded as features are implemented.

Quick Links
-----------
* `GitHub Repository <https://github.com/doughnut/voiage>`_
* `TestPyPI Page <https://test.pypi.org/project/voiage/>`_ (for v0.1 release)


Getting Help
------------
If you encounter any issues or have questions, please `raise an issue on GitHub <https://github.com/doughnut/voiage/issues>`_.


.. include:: introduction.rst

.. include:: installation.rst

.. include:: getting_started.rst

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api_reference/index

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Development:

   contributing
   changelog
